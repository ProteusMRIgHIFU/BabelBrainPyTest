"""
Unit tests for client_functions.py (BabelBrain_Pytest_Usage.md, Tables 3 and 7): the HTTP helpers
the remote pipeline (RunServerCalculation) and the example clients use. These check how a server
record configures them (URL, bearer token, TLS settings) without any network; requests to a real
server are covered by Integration/test_remote_server_connection.py.
"""
import logging
import ssl

import pytest

from Tests.helpers.parameters import requires_feature

pytestmark = requires_feature('remote_servers')

CLIENT_GLOBALS = ('BASE', 'TOKEN', 'CA', 'CLIENT_CERT', 'CLIENT_KEY', 'INSECURE', '_SSL_CTX', '_SSL_KEY')


@pytest.fixture
def cf(monkeypatch):
    """client_functions, with its module settings restored after the test."""
    import client_functions
    for name in CLIENT_GLOBALS:
        monkeypatch.setattr(client_functions, name, getattr(client_functions, name))
    return client_functions


def test_bind_server_plain_http(cf):
    cf.bind_server({'host': 'gpu-box', 'port': 9000, 'token': 'secret'})
    logging.info(f"BASE={cf.BASE} TOKEN={cf.TOKEN} CA={cf.CA} INSECURE={cf.INSECURE}")
    assert (cf.BASE, cf.TOKEN, cf.CA, cf.CLIENT_CERT, cf.INSECURE) == ('http://gpu-box:9000', 'secret', None, None, False)


def test_bind_server_tls_options(cf):
    cf.bind_server({'host': 'tls.lab', 'port': None, 'https': True, 'cafile': '/ca.pem',
                    'client_cert': '/c.pem', 'client_key': '/c.key', 'insecure': True})
    logging.info(f"BASE={cf.BASE} CA={cf.CA} CLIENT_CERT={cf.CLIENT_CERT} INSECURE={cf.INSECURE}")
    assert (cf.BASE, cf.CA, cf.CLIENT_CERT, cf.CLIENT_KEY, cf.INSECURE) == \
        ('https://tls.lab:8760', '/ca.pem', '/c.pem', '/c.key', True)


def test_headers_add_bearer_token_only_when_set(cf):
    cf.TOKEN = None
    assert cf._headers({'Content-Type': 'application/json'}) == {'Content-Type': 'application/json'}
    cf.TOKEN = 'secret'
    assert cf._headers() == {'Authorization': 'Bearer secret'}


def test_ssl_context_none_for_http(cf):
    cf.bind_server({'host': 'gpu-box', 'port': 8760})
    assert cf._ssl_context() is None


def test_ssl_context_is_reused_until_settings_change(cf):
    cf.bind_server({'host': 'tls.lab', 'port': 8760, 'https': True})
    first = cf._ssl_context()
    assert first.verify_mode == ssl.CERT_REQUIRED and first.check_hostname
    assert cf._ssl_context() is first
    cf.INSECURE = True
    insecure = cf._ssl_context()
    logging.info(f"Rebuilt after INSECURE=True: verify_mode={insecure.verify_mode}")
    assert insecure is not first and insecure.verify_mode == ssl.CERT_NONE and not insecure.check_hostname
    cf.BASE = 'https://other:8760'
    assert cf._ssl_context() is not insecure


@pytest.mark.parametrize('server_path,dirname,basename', [
    ('/srv/ws/m2m_ernie/final_tissues.nii.gz', '/srv/ws/m2m_ernie', 'final_tissues.nii.gz'),
    ('C:\\ws\\m2m_ernie\\final_tissues.nii.gz', 'C:\\ws\\m2m_ernie', 'final_tissues.nii.gz'),
])
def test_server_paths_split_on_either_separator(cf, server_path, dirname, basename):
    """Paths the server returns use its own separator, whatever the client's OS."""
    assert (cf.server_dirname(server_path), cf.server_basename(server_path)) == (dirname, basename)
