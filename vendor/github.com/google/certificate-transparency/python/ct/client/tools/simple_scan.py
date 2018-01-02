#!/usr/bin/env python

import os
import sys

import gflags

from ct.client import scanner

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("multi", 2, "Number of cert parsing processes to use in "
                      "addition to the main process and the network process.")
gflags.DEFINE_string("output", None,
                     "Output directory to write certificates to.")


def match(certificate, entry_type, extra_data, certificate_index):
    # Fill this in with your match criteria, e.g.
    #
    # return "google" in certificate.subject_name().lower()
    #
    # NB: for precertificates, issuer matching may not work as expected
    # when the precertificate has been issued by the special-purpose
    # precertificate signing certificate.
    return ("cert_%d.der" % certificate_index, certificate.to_der())

def write_matched_certificate(matcher_output):
    output_file, der_data = matcher_output
    with open(os.path.join(FLAGS.output, output_file), "wb") as f:
        f.write(der_data)


def run():
    if not FLAGS.output:
        raise Exception("Certificates output directory must be specified.")

    res = scanner.scan_log(
        match, "https://ct.googleapis.com/pilot", FLAGS.multi,
        write_matched_certificate)
    print "Scanned %d, %d matched and %d failed strict or partial parsing" % (
        res.total, res.matches, res.errors)


if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    run()
