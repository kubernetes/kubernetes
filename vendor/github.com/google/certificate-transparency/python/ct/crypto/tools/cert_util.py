#!/usr/bin/env python
"""cert_util.py: X509 certificate parsing utility.

Usage:

  cert_util.py <command> [flags] [cert_file ...]

Known commands:

  print: print information about the certificates in given files

  Each file must contain either one or more PEM-encoded certificates,
  or a single DER certificate.

  For example:

  cert_util.py print cert.pem           - pretty-print the certificate(s)
  cert_util.py print c1.pem c2.pem      - pretty-print certificates from
                                          multiple files
  cert_util.py print cert.der           - both PEM and DER are accepted formats
                                          (use --filetype to force a format)
  cert_util.py print --debug cert.pem   - print full ASN.1 structure
  cert_util.py print --subject cert.pem - print the subject name
  cert_util.py print --issuer cert.pem  - print the issuer name
  cert_util.py print --fingerprint cert.pem
                                        - print the SHA-1 fingerprint
  cert_util.py print --fingerprint --digest="sha256" cert.pem
                                        - print the SHA-256 fingerprint
"""

import sys
from ct.crypto import cert
from ct.crypto import error
from ct.crypto import pem
from ct.crypto.asn1 import print_util
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_bool("subject", False, "Print option: prints certificate subject")
gflags.DEFINE_bool("issuer", False, "Print option: prints certificate issuer")
gflags.DEFINE_bool("fingerprint", False, "Print option: prints certificate "
                   "fingerprint")
gflags.DEFINE_string("digest", "sha1", "Print option: fingerprint digest to use")
gflags.DEFINE_bool("debug", False,
                   "Print option: prints full ASN.1 debug information")
gflags.DEFINE_string("filetype", "", "Read option: specify an input file "
                     "format (pem or der). If no format is specified, the "
                     "parser attempts to detect the format automatically.")
gflags.RegisterValidator("filetype", lambda value: not value or
                         value.lower() in {"pem", "der"},
                         message="--filetype must be one of pem or der")


def print_cert(certificate):
    if not FLAGS.subject and not FLAGS.issuer and not FLAGS.fingerprint:
        if FLAGS.debug:
            print "%r" % certificate
        else:
            print certificate
    else:
        if FLAGS.subject:
            print "subject:\n%s" % certificate.print_subject_name()
        if FLAGS.issuer:
            print "issuer:\n%s" % certificate.print_issuer_name()
        if FLAGS.fingerprint:
            # Print in a format familiar from OpenSSL.
            print "%s fingerprint: %s\n" % (
                FLAGS.digest.upper(), print_util.bytes_to_hex(
                    certificate.fingerprint(hashfunc=FLAGS.digest)))


def print_certs(cert_file):
    """Print the certificates, or parts thereof, as specified by flags."""
    # If no format is specified, try PEM first, and automatically fall back
    # to DER. The advantage is that usage is more convenient; the disadvantage
    # is that error messages are less helpful because we don't know the expected
    # file format.
    printed = False
    if not FLAGS.filetype or FLAGS.filetype.lower() == "pem":
        if not FLAGS.filetype:
            print "Attempting to read PEM"

        try:
            for c in cert.certs_from_pem_file(cert_file, strict_der=False):
                print_cert(c)
                printed = True
        except pem.PemError as e:
            if not printed:
                # Immediate error
                print "File is not a valid PEM file: %s" % e
            else:
                exit_with_message("Error while scanning PEM blocks: %s" % e)
        except error.ASN1Error as e:
            exit_with_message("Bad DER encoding: %s" % e)

    if not printed and FLAGS.filetype.lower() != "pem":
        if not FLAGS.filetype:
            print "Attempting to read raw DER"
        try:
            print_cert(cert.Certificate.from_der_file(cert_file,
                                                      strict_der=False))
        except error.ASN1Error as e:
            exit_with_message("Failed to parse DER from %s" % cert_file)


def exit_with_message(error_message):
    print error_message
    print "Use --helpshort or --help to get help."
    sys.exit(1)


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

    command, argv = argv[0], argv[1:]

    if command != "print":
        exit_with_message("Unknown command %s" % command)

    if not argv:
        exit_with_message("No certificate file given")
    for filename in argv:
        print_certs(filename)
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
