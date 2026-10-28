"""
Unit tests for RemoteServers.py (BabelBrain_Pytest_Usage.md, Tables 3 and 7): the saved list of
remote job servers (~/.babelbrain/servers.json) and how a record turns into a URL and TLS settings.
The servers file is redirected to a temporary folder, so the user's own list is never touched.
Connecting to a real server is covered by Integration/test_remote_server_connection.py.
"""
import json
import logging
import ssl

import pytest

from Tests.helpers.parameters import requires_feature

pytestmark = requires_feature('remote_servers')


@pytest.fixture
def rs(tmp_path, monkeypatch):
    """RemoteServers with its servers file in tmp_path."""
    import RemoteServers
    monkeypatch.setattr(RemoteServers, 'CONFIG_DIR', str(tmp_path))
    monkeypatch.setattr(RemoteServers, 'SERVERS_FILE', str(tmp_path / 'servers.json'))
    logging.info(f"Servers file redirected to {tmp_path / 'servers.json'}")
    return RemoteServers


def test_record_defaults(rs):
    """A bare record gets the default host/port and every TLS option off, so http-only records
    saved by older versions keep working."""
    record = rs._norm({'name': ' lab '})
    logging.info(f"Normalised record: {record}")
    assert record == {'name': 'lab', 'host': '127.0.0.1', 'port': 8760, 'token': None, 'https': False,
                      'cafile': None, 'client_cert': None, 'client_key': None, 'insecure': False}


def test_record_blank_optional_fields_become_none(rs):
    record = rs._norm({'name': 'lab', 'host': '  ', 'port': '', 'token': '  ', 'cafile': ''})
    assert (record['host'], record['port'], record['token'], record['cafile']) == ('127.0.0.1', 8760, None, None)


def test_save_and_load_round_trip(rs):
    servers = [{'name': 'gpu-box', 'host': '10.0.0.5', 'port': 9000, 'token': 'secret'},
               {'name': 'tls-box', 'host': 'tls.lab', 'https': True, 'cafile': '/ca.pem'}]
    rs.save_servers(servers)
    with open(rs.servers_path()) as f:
        logging.info(f"servers.json:\n{f.read()}")
    loaded = rs.load_servers()
    assert [s['name'] for s in loaded] == ['gpu-box', 'tls-box']
    assert (loaded[0]['port'], loaded[0]['token'], loaded[0]['https']) == (9000, 'secret', False)
    assert (loaded[1]['https'], loaded[1]['cafile'], loaded[1]['port']) == (True, '/ca.pem', 8760)


@pytest.mark.parametrize('content', ['', 'not json', '{"name": "lab"}', '[1, "x", {"host": "no-name"}]'])
def test_load_ignores_missing_or_invalid_files(rs, content):
    """A missing, corrupt or wrongly shaped file gives an empty list instead of an error (entries
    without a name are dropped)."""
    if content:
        with open(rs.servers_path(), 'w') as f:
            f.write(content)
    assert rs.load_servers() == []


def test_add_update_get_and_remove(rs):
    rs.add_or_update({'name': 'a', 'host': 'host-a'})
    rs.add_or_update({'name': 'b', 'host': 'host-b'})
    rs.add_or_update({'name': 'a', 'host': 'host-a2', 'token': 't'})   # same name replaces
    logging.info(f"After adding a, b and updating a: {rs.load_servers()}")
    assert sorted(s['name'] for s in rs.load_servers()) == ['a', 'b']
    assert rs.get_server('a')['host'] == 'host-a2' and rs.get_server('a')['token'] == 't'
    assert rs.get_server('missing') is None
    rs.remove('a')
    assert [s['name'] for s in rs.load_servers()] == ['b']


@pytest.mark.parametrize('record,url', [
    ({'host': 'gpu-box', 'port': 8760}, 'http://gpu-box:8760'),
    ({'host': '10.0.0.5', 'port': '9000', 'https': True}, 'https://10.0.0.5:9000'),
])
def test_base_url(rs, record, url):
    assert rs.base_url(record) == url


def test_ssl_context_plain_http_has_none(rs):
    assert rs.ssl_context({'https': False}) is None


def test_ssl_context_verifies_by_default(rs):
    ctx = rs.ssl_context({'https': True})
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_ssl_context_insecure_skips_verification(rs):
    ctx = rs.ssl_context({'https': True, 'insecure': True})
    assert ctx.verify_mode == ssl.CERT_NONE and not ctx.check_hostname


def test_ssl_context_missing_cafile_raises(rs, tmp_path):
    with pytest.raises((FileNotFoundError, ssl.SSLError, OSError)):
        rs.ssl_context({'https': True, 'cafile': str(tmp_path / 'missing.pem')})


def test_connection_to_closed_port_reports_unreachable(rs):
    """No server listening: a readable error, not an exception."""
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]   # free port, closed again before connecting
    ok, info = rs.test_connection({'host': '127.0.0.1', 'port': port}, timeout=2)
    logging.info(f"test_connection to closed port {port}: {ok}, {info}")
    assert not ok and info.startswith(f"Cannot reach http://127.0.0.1:{port}")
