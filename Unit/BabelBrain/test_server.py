"""
Unit tests for server.py (BabelBrain_Test_Plan.md, Tables 3 and 7), the BabelBrain job server
behind "remote server" computing engines. These cover its connection security settings without
serving anything: which bind addresses count as local, the HTTPS / mutual TLS context, the refusal
to start an open server on a network address, upload name sanitising and job spec validation.
Serving requests over each protocol is covered by Integration/test_remote_server_connection.py.
"""
import logging
import os
import ssl
from types import SimpleNamespace

import pytest

from Tests.helpers.parameters import requires_feature
from Tests.helpers.tls_certs import make_certificates

pytestmark = requires_feature('remote_servers')

SERVER_ENV = ('BABEL_SERVER_TOKEN', 'BABEL_SERVER_CERTFILE', 'BABEL_SERVER_KEYFILE', 'BABEL_SERVER_CAFILE',
              'BABEL_SERVER_ALLOW_INSECURE')


@pytest.fixture(scope='module')
def server():
    import server as server_module
    return server_module


@pytest.fixture(scope='module')
def certs(tmp_path_factory):
    folder = tmp_path_factory.mktemp('certs')
    logging.info(f"Making test CA, server and client certificates in {folder}")
    return make_certificates(folder)


@pytest.fixture
def clean_env(monkeypatch):
    """No server settings inherited from the environment."""
    for name in SERVER_ENV:
        monkeypatch.delenv(name, raising=False)


# ================================================================================================================================
# BIND ADDRESS AND TLS
# ================================================================================================================================
@pytest.mark.parametrize('host,loopback', [
    ('127.0.0.1', True), ('localhost', True), ('LocalHost ', True), ('::1', True),
    ('0.0.0.0', False), ('', False), ('::', False), ('10.0.0.5', False), ('gpu-box.lab', False),
])
def test_is_loopback(server, host, loopback):
    logging.info(f"Checking bind address {host!r} is {'loopback' if loopback else 'not loopback'}")
    assert server._is_loopback(host) is loopback


def test_tls_context_none_without_certificate(server):
    logging.info("Checking no certificate gives no TLS context (plain http)")
    assert server._build_tls_context(None, None, None) is None


def test_tls_context_https(server, certs):
    logging.info("Building a TLS context from the server certificate and key, no CA")
    ctx = server._build_tls_context(certs['server_pem'], certs['server_key'], None)
    logging.info(f"HTTPS context: minimum {ctx.minimum_version}, verify_mode {ctx.verify_mode}")
    logging.info("Checking TLS >= 1.2 and no client certificate required")
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    assert ctx.verify_mode == ssl.CERT_NONE   # clients don't need a certificate


def test_tls_context_mutual_tls_requires_client_certificates(server, certs):
    logging.info("Building a TLS context from the server certificate and key plus a client CA")
    ctx = server._build_tls_context(certs['server_pem'], certs['server_key'], certs['ca_pem'])
    logging.info(f"Mutual TLS context: verify_mode {ctx.verify_mode}")
    logging.info("Checking client certificates are required")
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_tls_context_rejects_key_that_does_not_match(server, certs):
    logging.info("Checking the server certificate with the client's key raises SSLError")
    with pytest.raises(ssl.SSLError):
        server._build_tls_context(certs['server_pem'], certs['client_key'], None)


# ================================================================================================================================
# START UP REFUSALS (run_server returns 2 before serving anything)
# ================================================================================================================================
def _args(**kwargs):
    return SimpleNamespace(**{'serve_host': '127.0.0.1', 'serve_port': 0, 'serve_token': None, 'headless': True,
                              **kwargs})


def test_refuses_network_address_without_token(server, clean_env, capsys):
    """An open server on a network address would let anyone who can reach it read and write files."""
    logging.info("Starting the server on 0.0.0.0 without a token, expecting exit code 2")
    assert server.run_server(None, _args(serve_host='0.0.0.0')) == 2
    out = capsys.readouterr().out
    logging.info(out.strip())
    logging.info("Checking it printed the refusal")
    assert 'REFUSING to start' in out


def test_refuses_unreadable_certificate(server, clean_env, tmp_path, capsys):
    missing = str(tmp_path / 'missing.pem')
    logging.info(f"Starting the server with missing cert/key {missing}, expecting exit code 2")
    assert server.run_server(None, _args(serve_certfile=missing, serve_keyfile=missing)) == 2
    out = capsys.readouterr().out
    logging.info(out.strip())
    logging.info("Checking it printed the cert/key loading error")
    assert 'cannot load TLS cert/key' in out


def test_refuses_mismatched_certificate_from_environment(server, clean_env, certs, monkeypatch, capsys):
    monkeypatch.setenv('BABEL_SERVER_CERTFILE', certs['server_pem'])
    monkeypatch.setenv('BABEL_SERVER_KEYFILE', certs['client_key'])
    logging.info("Starting the server with the server certificate and the client key from the environment")
    assert server.run_server(None, _args()) == 2
    logging.info("Checking it refused (exit code 2) with the cert/key loading error")
    assert 'cannot load TLS cert/key' in capsys.readouterr().out


# ================================================================================================================================
# UPLOAD NAMES AND JOB SPECS
# ================================================================================================================================
@pytest.mark.parametrize('name,relpath', [
    ('T1W.nii.gz', 'T1W.nii.gz'),
    ('m2m_ernie/final_tissues.nii.gz', os.path.join('m2m_ernie', 'final_tissues.nii.gz')),
    ('m2m_ernie\\final_tissues.nii.gz', os.path.join('m2m_ernie', 'final_tissues.nii.gz')),
    ('/etc/passwd', os.path.join('etc', 'passwd')),           # absolute paths stay inside the workspace
    ('./a/./b.txt', os.path.join('a', 'b.txt')),
])
def test_upload_names_stay_in_workspace(server, name, relpath):
    logging.info(f"Checking upload name {name!r} maps to {relpath!r} in the workspace")
    assert server._safe_relpath(name) == relpath


@pytest.mark.parametrize('name', ['../outside.txt', 'm2m/../../outside.txt', '..\\outside.txt', '', '/', './'])
def test_upload_names_escaping_or_empty_are_rejected(server, name):
    logging.info(f"Checking upload name {name!r} is rejected")
    with pytest.raises(ValueError):
        server._safe_relpath(name)


@pytest.mark.parametrize('spec,error', [
    ({'config': {}}, None),
    ({'config': {}, 'steps': {'planning': []}, 'keep_alive': True}, None),
    ([], 'JSON object'),
    ({}, "'config'"),
    ({'config': {}, 'steps': []}, "'steps' must be an object"),
    ({'config': {}, 'standalone': {}, 'steps': {}}, "both 'standalone' and 'steps'"),
    ({'config': {}, 'keep_alive': 'yes'}, "'keep_alive' must be a boolean"),
])
def test_validate_job_spec(server, spec, error):
    logging.info(f"Validating job spec {spec}, expecting {'no error' if error is None else repr(error)}")
    result = server._validate_spec(spec)
    logging.info(f"Validation result: {result}")
    if error is None:
        assert result is None
    else:
        assert error in result
