"""
Integration tests for connecting to a BabelBrain remote job server (BabelBrain_Pytest_Usage.md,
Tables 4 and 8): the server's real HTTP front end (server._make_handler, its TLS context, job and
workspace managers) against the two clients BabelBrain uses,

    RemoteServers.test_connection   the check run when a server is added in the GUI
    client_functions                the requests the remote pipeline (RunServerCalculation) makes

over every connection option the server supports:

    http            plain HTTP (loopback / behind a TLS proxy)
    http + token    bearer token authentication
    https           TLS with a certificate from a private CA (client trusts it via 'cafile'),
                    or a self-signed one, or verification skipped ('insecure', testing only)
    https + token
    mutual TLS      the server also requires a client certificate signed by its CA

Each option is tried with the right client settings (must connect, and upload a file) and with the
wrong ones (must be refused with a readable message). The server runs in a thread on a free
loopback port; no pipeline runs, so no GPU, Qt session or test data is needed. TLS certificates
are made with openssl (helpers/tls_certs.py); TLS cases skip without it. The full pipeline on a
remote server is the E2E test e2e_test_remote_full_pipeline.
"""
import logging
import os
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from Tests.helpers.parameters import requires_feature
from Tests.helpers.tls_certs import make_certificates

pytestmark = [requires_feature('remote_servers'), pytest.mark.minor_release, pytest.mark.major_release]

TOKEN = 'pytest-secret-token'

# Server configurations: token, certificate, key and client CA (mutual TLS) by certificate name
SERVERS = {
    'http':          {'token': None,  'cert': None},
    'http_token':    {'token': TOKEN, 'cert': None},
    'https':         {'token': None,  'cert': 'server'},
    'https_token':   {'token': TOKEN, 'cert': 'server'},
    'self_signed':   {'token': None,  'cert': 'self_signed'},
    'mutual_tls':    {'token': TOKEN, 'cert': 'server', 'client_ca': 'ca'},
}

# (case id, server, client record fields, connects?, expected error text). Certificate fields name
# files from make_certificates.
CONNECTION_CASES = [
    ('http',                              'http',        {}, True, None),
    ('token_correct',                     'http_token',  {'token': TOKEN}, True, None),
    ('token_missing',                     'http_token',  {}, False, 'Unauthorized (401)'),
    ('token_wrong',                       'http_token',  {'token': 'wrong'}, False, 'Unauthorized (401)'),
    ('https_trusted_ca',                  'https',       {'https': True, 'cafile': 'ca'}, True, None),
    ('https_untrusted_ca',                'https',       {'https': True}, False, 'TLS error'),
    ('https_wrong_ca',                    'https',       {'https': True, 'cafile': 'rogue_ca'}, False, 'TLS error'),
    ('https_insecure',                    'https',       {'https': True, 'insecure': True}, True, None),
    ('https_token_correct',               'https_token', {'https': True, 'cafile': 'ca', 'token': TOKEN}, True, None),
    ('https_token_missing',               'https_token', {'https': True, 'cafile': 'ca'}, False, 'Unauthorized (401)'),
    ('self_signed_trusted',               'self_signed', {'https': True, 'cafile': 'self_signed'}, True, None),
    ('self_signed_untrusted',             'self_signed', {'https': True}, False, 'TLS error'),
    ('http_client_to_https_server',       'https',       {}, False, None),
    ('https_client_to_http_server',       'http',        {'https': True, 'insecure': True}, False, 'TLS error'),
    ('mutual_tls_client_certificate',     'mutual_tls',  {'https': True, 'cafile': 'ca', 'token': TOKEN,
                                                          'client_cert': 'client', 'client_key': 'client'}, True, None),
    ('mutual_tls_no_client_certificate',  'mutual_tls',  {'https': True, 'cafile': 'ca', 'token': TOKEN}, False, None),
    ('mutual_tls_rogue_client_certificate', 'mutual_tls', {'https': True, 'cafile': 'ca', 'token': TOKEN,
                                                           'client_cert': 'rogue_client',
                                                           'client_key': 'rogue_client'}, False, None),
]
CONNECTING_CASES = [case for case in CONNECTION_CASES if case[3]]


# ================================================================================================================================
# FIXTURES
# ================================================================================================================================
@pytest.fixture(scope='module')
def server_module():
    import server
    return server


@pytest.fixture(scope='module')
def certs(tmp_path_factory):
    return make_certificates(tmp_path_factory.mktemp('certs'))


def _needs_tls(server_name, record):
    return SERVERS[server_name]['cert'] is not None or record.get('https')


@pytest.fixture
def start_server(server_module, tmp_path, request):
    """start_server(name) runs the server configuration SERVERS[name] on a free loopback port and
    returns (port, workspace folder). Stopped after the test."""
    running = []

    def _start(name):
        settings = SERVERS[name]
        certs = request.getfixturevalue('certs') if settings['cert'] else {}
        certfile = certs.get(f"{settings['cert']}_pem")
        keyfile = certs.get(f"{settings['cert']}_key")
        client_ca = certs.get(f"{settings['client_ca']}_pem") if settings.get('client_ca') else None
        tls = server_module._build_tls_context(certfile, keyfile, client_ca)

        workspaces = tmp_path / f"workspaces_{name}"
        workspaces.mkdir()
        capabilities = {'server_version': 'pytest', 'features': ['workspaces', 'uploads'],
                        'tls': tls is not None, 'mutual_tls': client_ca is not None}
        no_sessions = SimpleNamespace(list_summaries=lambda: [], get=lambda session_id: None)
        handler = server_module._make_handler(server_module.JobManager(), settings['token'], capabilities, {}, {},
                                              server_module.WorkspaceManager(temp_base=str(workspaces)), no_sessions)
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        if tls is not None:
            httpd.socket = tls.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=httpd.serve_forever, name=f"pytest-server-{name}", daemon=True).start()
        running.append(httpd)
        port = httpd.server_address[1]
        logging.info(f"Started '{name}' server on 127.0.0.1:{port}: "
                     f"{'mutual TLS' if client_ca else 'TLS' if tls else 'plain HTTP'}"
                     f" ({settings['cert'] or 'no'} certificate), token {'required' if settings['token'] else 'off'}")
        return port, workspaces

    yield _start
    for httpd in running:
        httpd.shutdown()
        httpd.server_close()


def _client_record(fields, port, request):
    """RemoteServers record for `fields`, with certificate names replaced by their files."""
    record = {'name': 'pytest', 'host': '127.0.0.1', 'port': port, **fields}
    for key, suffix in (('cafile', '_pem'), ('client_cert', '_pem'), ('client_key', '_key')):
        if key in record:
            record[key] = request.getfixturevalue('certs')[record[key] + suffix]
    shown = {k: (os.path.basename(v) if k in ('cafile', 'client_cert', 'client_key') else
                 '***' if k == 'token' else v) for k, v in record.items()}
    logging.info(f"Client record: {shown}")
    return record


@pytest.fixture
def cf(monkeypatch):
    """client_functions, with its module settings restored after the test."""
    import client_functions
    for name in ('BASE', 'TOKEN', 'CA', 'CLIENT_CERT', 'CLIENT_KEY', 'INSECURE', '_SSL_CTX', '_SSL_KEY'):
        monkeypatch.setattr(client_functions, name, getattr(client_functions, name))
    return client_functions


def _ids(cases):
    return [case[0] for case in cases]


# ================================================================================================================================
# TESTS
# ================================================================================================================================
@pytest.mark.parametrize('case,server_name,fields,connects,error', CONNECTION_CASES, ids=_ids(CONNECTION_CASES))
def test_connection_check(start_server, request, case, server_name, fields, connects, error):
    """RemoteServers.test_connection, as run when a server is added in BabelBrain's GUI."""
    import RemoteServers

    if _needs_tls(server_name, fields):
        request.getfixturevalue('certs')
    port, _ = start_server(server_name)
    record = _client_record(fields, port, request)
    ok, info = RemoteServers.test_connection(record, timeout=5)
    logging.info(f"test_connection -> ok={ok}: {info}")

    assert ok is connects, f"Expected {'a connection' if connects else 'a refusal'}, got: {info}"
    if connects:
        assert info['health']['status'] == 'ok'
        assert info['capabilities']['tls'] is (SERVERS[server_name]['cert'] is not None)
        assert info['capabilities']['mutual_tls'] is ('client_ca' in SERVERS[server_name])
    elif error:
        assert error in info


@pytest.mark.parametrize('case,server_name,fields,connects,error', CONNECTING_CASES, ids=_ids(CONNECTING_CASES))
def test_workspace_upload(start_server, request, cf, tmp_path, case, server_name, fields, connects, error):
    """client_functions moves data over the connection: create a workspace, upload a file into a
    sub folder, list it, then delete the workspace (as the remote pipeline stages its inputs)."""
    port, workspaces = start_server(server_name)
    cf.bind_server(_client_record(fields, port, request))
    payload = os.urandom(256 * 1024)
    local = tmp_path / 'T1W.nii.gz'
    local.write_bytes(payload)

    ws = cf.create_workspace()
    logging.info(f"Created workspace {ws['workspace_id']} at {ws['root']}")
    server_path = cf.upload(ws['workspace_id'], str(local), 'm2m_pytest/T1W.nii.gz')
    logging.info(f"Uploaded {len(payload)} bytes to {server_path}")
    listing = cf._req('GET', f"/workspaces/{ws['workspace_id']}")
    assert listing['files'] == [os.path.join('m2m_pytest', 'T1W.nii.gz')]
    with open(server_path, 'rb') as f:
        assert f.read() == payload, "Uploaded file differs from the local one"
    assert os.path.commonpath([server_path, str(workspaces)]) == str(workspaces)

    assert cf.delete_workspace(ws['workspace_id'])['deleted']
    assert not os.path.exists(ws['root'])


@pytest.mark.parametrize('name', ['../outside.bin', 'm2m/../../outside.bin'])
def test_upload_cannot_leave_workspace(start_server, request, cf, tmp_path, name):
    """An upload name climbing out of the workspace is refused over the wire (400) and nothing is
    written outside it."""
    port, workspaces = start_server('http_token')
    cf.bind_server(_client_record({'token': TOKEN}, port, request))
    local = tmp_path / 'payload.bin'
    local.write_bytes(b'not allowed')
    ws = cf.create_workspace()
    with pytest.raises(urllib.error.HTTPError) as refused:
        cf.upload(ws['workspace_id'], str(local), name)
    logging.info(f"Upload as {name!r} -> HTTP {refused.value.code}")
    assert refused.value.code == 400
    written = [os.path.join(root, f) for root, _, files in os.walk(workspaces.parent) for f in files
               if f == 'outside.bin']
    assert not written, f"Upload escaped the workspace: {written}"


@pytest.mark.parametrize('fields,expected', [
    ({}, urllib.error.HTTPError),
    ({'token': 'wrong'}, urllib.error.HTTPError),
])
def test_requests_without_valid_token_are_refused(start_server, request, cf, fields, expected):
    """Every endpoint but /healthz needs the token, including the ones the pipeline uses."""
    port, _ = start_server('http_token')
    cf.bind_server(_client_record(fields, port, request))
    assert cf._req('GET', '/healthz', retries=1)['status'] == 'ok'
    for method, path, body in [('GET', '/capabilities', None), ('GET', '/workspaces', None),
                               ('POST', '/workspaces', {'mode': 'temp'}), ('POST', '/jobs', {'config': {}})]:
        with pytest.raises(expected) as refused:
            cf._req(method, path, body, retries=1)
        logging.info(f"{method} {path} without a valid token -> HTTP {refused.value.code}")
        assert refused.value.code == 401


def test_mutual_tls_rejects_client_without_certificate_at_handshake(start_server, request, cf):
    """With mutual TLS, a client without a certificate doesn't get as far as sending its token."""
    request.getfixturevalue('certs')
    port, _ = start_server('mutual_tls')
    cf.bind_server(_client_record({'https': True, 'cafile': 'ca', 'token': TOKEN}, port, request))
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)) as refused:
        cf._req('GET', '/capabilities', retries=1)
    logging.info(f"Refused at the TLS handshake: {refused.value!r}")
    assert not isinstance(refused.value, urllib.error.HTTPError)
