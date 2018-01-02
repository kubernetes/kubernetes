#!/usr/bin/env python
"""Generates a list of hashes of EV certificates found in a log."""

import hashlib
import functools
import pickle
import os
import sys

import ev_metadata
import gflags

from ct.client import scanner
from ct.crypto import cert
from ct.proto import client_pb2

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("hash_trim", 8, "Number of bytes of the SHA-256 digest "
                      "to use in the whitelist.")

gflags.DEFINE_integer("multi", 2, "Number of cert fetching and parsing "
                      "processes to use, in addition to the main process.")

gflags.DEFINE_string("output_directory", None,
                     "Output directory for individual EV certificates. "
                     "If provided, individual EV certs will be written there.")


def calculate_certificate_hash(certificate):
    """Hash the input's DER representation and trim it."""
    hasher = hashlib.sha256(certificate.to_der())
    return hasher.digest()[0:FLAGS.hash_trim]


def find_matching_policies(certificate):
    """Returns the certificate's EV policy OID, if exists."""
    try:
        matching_policies = []
        for policy in certificate.policies():
            if policy['policyIdentifier'] in ev_metadata.EV_POLICIES:
                matching_policies.append(policy['policyIdentifier'])
        return matching_policies
    except cert.CertificateError:
        pass
    return []


def does_root_match_policy(policy_oid, cert_chain):
    """Returns true if the fingerprint of the root certificate matches the
    expected fingerprint for this EV policy OID."""
    if not cert_chain: # Empty chain
        return False
    root_fingerprint = hashlib.sha1(cert_chain[-1]).digest()
    return root_fingerprint in ev_metadata.EV_POLICIES[policy_oid]


def _write_cert_and_chain(
        output_dir, certificate, extra_data, certificate_index):
    """Writes the certificate and its chain to files for later analysis."""
    open(
        os.path.join(output_dir,
                     "cert_%d.der" % certificate_index), "wb"
        ).write(certificate.to_der())

    pickle.dump(
        list(extra_data.certificate_chain),
        open(os.path.join(output_dir,
                          "cert_%d_extra_data.pickle" % certificate_index),
             "wb"))

def _ev_match(
        output_dir, last_acceptable_entry_index, certificate, entry_type,
        extra_data, certificate_index):
    """Matcher function for the scanner. Returns the certificate's hash if
    it is a valid, non-expired, EV certificate, None otherwise."""
    # Only generate whitelist for non-precertificates. It is expected that if
    # a precertificate was submitted then the issued SCT would be embedded
    # in the final certificate.
    if entry_type != client_pb2.X509_ENTRY:
        return None
    # No point in including expired certificates.
    if certificate.is_expired():
        return None
    # Do not include entries beyond the last entry included in the whitelist
    # generated on January 1st, 2015.
    if certificate_index > last_acceptable_entry_index:
        return None

    # Only include certificates that have an EV OID.
    matching_policies = find_matching_policies(certificate)
    if not matching_policies:
        return None

    # Removed the requirement that the root of the chain matches the root that
    # should be used for the EV policy OID.
    # See https://code.google.com/p/chromium/issues/detail?id=524635 for
    # details.

    # Matching certificate
    if output_dir:
        _write_cert_and_chain(
            output_dir, certificate, extra_data, certificate_index)

    return calculate_certificate_hash(certificate)


def generate_ev_cert_hashes_from_log(
        log_url, num_processes, output_directory, last_acceptable_entry_index):
    """Scans the given log and generates a list of hashes for all EV
    certificates in it.

    Returns a tuple of (scan_results, hashes_list)"""
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    ev_hashes = set()
    def add_hash(cert_hash):
        """Store the hash. Always called from the main process, so safe."""
        ev_hashes.add(cert_hash)
    bound_ev_match = functools.partial(_ev_match, output_directory,
                                       last_acceptable_entry_index)
    res = scanner.scan_log(bound_ev_match, log_url, num_processes, add_hash)
    return (res, ev_hashes)

def main(output_file):
    """Scan and save results to a file."""
    res, hashes_set = generate_ev_cert_hashes_from_log(
        "https://ct.googleapis.com/pilot",
        FLAGS.multi,
        FLAGS.output_directory)
    print "Scanned %d, %d matched and %d failed strict or partial parsing" % (
        res.total, res.matches, res.errors)
    print "There are %d EV hashes." % (len(hashes_set))
    with open(output_file, "wb") as hashes_file:
        hashes_list = list(hashes_set)
        hashes_list.sort()
        for trimmed_hash in hashes_list:
            hashes_file.write(trimmed_hash)


if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: %s <output file>\n  <output file> will contain the "
            "sorted, truncated hashes list.\n" % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
