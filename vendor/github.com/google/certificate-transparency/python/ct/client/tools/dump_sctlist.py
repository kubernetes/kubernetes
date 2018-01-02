#!/usr/bin/env python
""" This is a util to dump the SCTs contained within a
    SignedCertificateTimestampList structure.
    This structure is used to represent a collection of SCTs being passed over
    a TLS handshake.  See RFC6962 section 3.3 for more details. """
import sys
from ct.proto import client_pb2
from ct.serialization import tls_message

def dump_sctlist(sct_list):
    """Prints the proto representation of the SCTs contained in sct_list.
       Arguments:
       sct_list the packed SignedCertificateTransparencyList structure.
    """
    tr = tls_message.TLSReader(sct_list)
    sctlist = client_pb2.SignedCertificateTimestampList()
    tr.read(sctlist)
    for s in sctlist.sct_list:
        sct = client_pb2.SignedCertificateTimestamp()
        tls_message.decode(s, sct)
        print sct

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Usage: dump_sctlist.py <file_containing_sct_list>"
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = f.read()
    dump_sctlist(data)
