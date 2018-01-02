#!/usr/bin/env python
"""Extracts all Precertificates from the log."""

import os
import sys

import gflags

from ct.client import scanner
from ct.proto import client_pb2

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("multi", 2, "Number of cert fetching and parsing "
                      "processes to use, in addition to the main process.")

gflags.DEFINE_string("output_directory", None,
                     "Output directory for individual Precertificates.")

gflags.DEFINE_integer("start_entry", 0, "Log entry to start from.")



def _precert_matches(certificate, entry_type, extra_data, certificate_index):
    """Matcher function for the scanner. Returns a filename and certificate in
     PEM format if it's a precertificate, None otherwise."""
    if entry_type == client_pb2.PRECERT_ENTRY:
        return ("precert_%d.pem" % certificate_index, certificate.to_pem())
    return None


def write_matched_certificate(matcher_output):
    output_file, der_data = matcher_output
    with open(os.path.join(FLAGS.output_directory, output_file), "wb") as f:
        f.write(der_data)


def run():
    if not FLAGS.output_directory:
        raise Exception("Certificates output directory must be specified.")

    res = scanner.scan_log(
        _precert_matches, "https://ct.googleapis.com/pilot", FLAGS.multi,
        write_matched_certificate,
        start_entry=FLAGS.start_entry)

    print "Scanned %d, %d matched and %d failed strict or partial parsing" % (
        res.total, res.matches, res.errors)


if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    run()
