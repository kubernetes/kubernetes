#!/usr/bin/env python
"""verify_util.py: CT signature verification utility.

Usage:

  verify_util.py <command> [flags] [cert_file]

Known commands:

  verify_sct: Verify Signed Certificate Timestamp over X.509 certificate.

  The cert_file must contain one or more PEM-encoded certificates.

  For example:

  verify_util.py verify_sct --sct=cert_sct.tls --log_key=log_key.pem cert.pem
"""

import sys
from ct.crypto import cert
from ct.crypto import pem
from ct.crypto import verify
from ct.proto import client_pb2
from ct.serialization import tls_message
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_string("sct", None, "TLS-encoded SCT file")
gflags.DEFINE_string("log_key", None, "PEM-encoded CT log key")

def exit_with_message(error_message):
    print error_message
    print "Use --helpshort or --help to get help."
    sys.exit(1)


def verify_sct(chain, sct_tls, log_key_pem):
    sct = client_pb2.SignedCertificateTimestamp()
    tls_message.decode(sct_tls, sct)

    log_key = pem.from_pem(log_key_pem, 'PUBLIC KEY')[0]
    key_info = verify.create_key_info_from_raw_key(log_key)

    lv = verify.LogVerifier(key_info)
    print lv.verify_sct(sct, chain)


def main(argv):
    if len(argv) <= 1 or argv[1][0] == "-":
        # No command. Parse flags anyway to trigger help flags.
        try:
            argv = FLAGS(argv)
            exit_with_message("No command")
        except gflags.FlagsError as e:
            exit_with_message("Error parsing flags: %s" % e)

    argv = argv[1:]

    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        exit_with_message("Error parsing flags: %s" % e)

    command, cert_file = argv[0:2]

    if command != "verify_sct":
        exit_with_message("Unknown command %s" % command)

    if not cert_file:
        exit_with_message("No certificate file given")

    chain = list(cert.certs_from_pem_file(cert_file, strict_der = False))

    verify_sct(chain,
               open(FLAGS.sct, 'rb').read(),
               open(FLAGS.log_key, 'rb').read())
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
