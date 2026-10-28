"""
Throwaway TLS certificates for the remote server tests, made with the openssl command line (the
cryptography package isn't a BabelBrain dependency). Tests needing them skip when openssl is missing.

make_certificates(folder) writes:

    ca.pem / ca.key                 test certificate authority
    server.pem / server.key         server certificate for localhost / 127.0.0.1, signed by the CA
    client.pem / client.key         client certificate signed by the CA (mutual TLS)
    rogue_ca.pem / rogue_ca.key     an unrelated CA
    rogue_client.pem / .key         client certificate signed by the unrelated CA
    self_signed.pem / .key          self-signed server certificate (no CA)
"""
import logging
import os
import shutil
import subprocess

import pytest

SUBJECT = "/O=BabelBrain pytest"


def openssl():
    """Path to the openssl executable, skipping the test when there isn't one."""
    path = shutil.which('openssl')
    if path is None:
        pytest.skip("openssl is needed to make TLS test certificates")
    return path


def _run(*args, cwd):
    result = subprocess.run([openssl(), *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"openssl {' '.join(args)} failed: {result.stderr}")


def _ca(folder, name):
    _run('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
         '-keyout', f"{name}.key", '-out', f"{name}.pem", '-subj', f"{SUBJECT}/CN={name}",
         '-addext', 'basicConstraints=critical,CA:TRUE', '-addext', 'keyUsage=critical,keyCertSign,cRLSign',
         cwd=folder)


def _signed(folder, name, ca, extensions):
    """Certificate `name` signed by `ca`, with extra X509v3 `extensions` (lines of an extfile)."""
    _run('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', f"{name}.key", '-out', f"{name}.csr",
         '-subj', f"{SUBJECT}/CN={name}", cwd=folder)
    ext = os.path.join(folder, f"{name}.ext")
    with open(ext, 'w') as f:
        f.write('\n'.join(extensions) + '\n')
    _run('x509', '-req', '-in', f"{name}.csr", '-CA', f"{ca}.pem", '-CAkey', f"{ca}.key",
         '-CAcreateserial', '-days', '2', '-out', f"{name}.pem", '-extfile', ext, cwd=folder)


def make_certificates(folder):
    """Write the test certificates into `folder` and return {name: path} for every .pem/.key."""
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Making TLS test certificates with {openssl()} in {folder}")
    _ca(folder, 'ca')
    _ca(folder, 'rogue_ca')
    _signed(folder, 'server', 'ca', ['subjectAltName=DNS:localhost,IP:127.0.0.1', 'extendedKeyUsage=serverAuth'])
    _signed(folder, 'client', 'ca', ['extendedKeyUsage=clientAuth'])
    _signed(folder, 'rogue_client', 'rogue_ca', ['extendedKeyUsage=clientAuth'])
    _run('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2', '-keyout', 'self_signed.key',
         '-out', 'self_signed.pem', '-subj', f"{SUBJECT}/CN=self_signed",
         '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1', cwd=folder)
    return {os.path.splitext(f)[0] + os.path.splitext(f)[1].replace('.', '_'): os.path.join(folder, f)
            for f in os.listdir(folder) if f.endswith(('.pem', '.key'))}
