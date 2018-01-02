#!/usr/bin/env python
"""This utility fetches the proof for a single certificate by its hash."""

import struct
import sys

from ct.client import log_client
from ct.crypto import cert
from ct.crypto import merkle
from ct.proto import client_pb2
from ct.serialization import tls_message
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string("cert", None, "Certificate file (PEM format) to fetch a "
                     "proof for.")
gflags.DEFINE_string("sct", None,
                     "SCT file (ProtoBuf/binary) of said certificate.")
gflags.DEFINE_bool("binary_sct", False, "SCT is in binary format")
gflags.DEFINE_integer("timestamp", None,
                     "Timestamp from SCT of said certificate.")
gflags.DEFINE_string("log_url", "https://ct.googleapis.com/pilot",
                     "URL of CT log.")
gflags.DEFINE_bool("verbose", False, "Verbose output or not.")


def create_leaf(timestamp, x509_cert_bytes):
    """Creates a MerkleTreeLeaf for the given X509 certificate."""
    leaf = client_pb2.MerkleTreeLeaf()
    leaf.version = client_pb2.V1
    leaf.leaf_type = client_pb2.TIMESTAMPED_ENTRY
    leaf.timestamped_entry.timestamp = timestamp
    leaf.timestamped_entry.entry_type = client_pb2.X509_ENTRY
    leaf.timestamped_entry.asn1_cert = x509_cert_bytes
    return tls_message.encode(leaf)

def construct_leaf_from_file(cert_file, cert_sct_timestamp):
    """Creates a MerkleTreeLeaf from a given PEM certificate file."""
    cert_to_lookup = cert.Certificate.from_pem_file(cert_file)
    return create_leaf(cert_sct_timestamp, cert_to_lookup.to_der())

def read_sct_from_file(sct_file):
    cert_sct = client_pb2.SignedCertificateTimestamp()
    cert_sct.ParseFromString(open(sct_file, 'rb').read())
    return cert_sct

def fetch_single_proof(leaf_hash, log_url):
    """Fetch the proof for the supplied certificate."""
    client = log_client.LogClient(log_url)
    sth = client.get_sth()
    if FLAGS.verbose:
      print "The log contains %d certificates." % (sth.tree_size)
      print "Tree root hash: %s" % (sth.sha256_root_hash.encode("hex"))

    proof_from_hash = client.get_proof_by_hash(
            leaf_hash, sth.tree_size)
    return sth, proof_from_hash

def run():
    """Fetch the proof for the supplied certificate."""
    #TODO(eranm): Attempt fetching the SCT for this chain if none was given.
    if FLAGS.sct:
        cert_sct = client_pb2.SignedCertificateTimestamp()
        sct_data = open(FLAGS.sct, 'rb').read()
        if FLAGS.binary_sct:
            tls_message.decode(sct_data, cert_sct)
        else:
            cert_sct.ParseFromString(sct_data)
        sct_timestamp = cert_sct.timestamp
        print 'SCT for cert:', cert_sct
    else:
        sct_timestamp = FLAGS.timestamp

    constructed_leaf = construct_leaf_from_file(FLAGS.cert, sct_timestamp)
    leaf_hash = merkle.TreeHasher().hash_leaf(constructed_leaf)
    if FLAGS.verbose:
      print "Leaf hash: %s" % (leaf_hash.encode("hex"))

    (sth, proof) = fetch_single_proof(leaf_hash, FLAGS.log_url);
    if FLAGS.verbose:
      print "Leaf index in tree is %d, proof has %d hashes" % (
          proof.leaf_index, len(proof.audit_path))
      print "Audit path: %s" % ([t.encode('hex') for t in proof.audit_path])

    verifier = merkle.MerkleVerifier()
    if verifier.verify_leaf_inclusion(constructed_leaf, proof.leaf_index,
                                      proof.audit_path, sth):
      print 'Proof verifies OK.'

if __name__ == '__main__':
    sys.argv = FLAGS(sys.argv)
    run()
