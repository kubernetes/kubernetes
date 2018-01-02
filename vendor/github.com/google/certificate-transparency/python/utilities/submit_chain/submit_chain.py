#!/usr/bin/env python
"""Submits a chain to a list of logs."""

import base64
import hashlib
import sys

import json
import logging
import gflags

from ct.client import log_client
from ct.crypto import cert
from ct.crypto import error
from ct.crypto import pem
from ct.crypto import verify
from ct.proto import client_pb2
from ct.serialization import tls_message

FLAGS = gflags.FLAGS

gflags.DEFINE_string("log_list", None, "File containing the list of logs "
                     "to submit to (see certificate-transparency.org/known-logs"
                     " for the format description).")
gflags.DEFINE_string("chain", None, "Certificate chain to submit (PEM).")
gflags.DEFINE_string("log_scheme", "http", "Log scheme (http/https)")
gflags.DEFINE_string("output", None, "output file for sct_list")
gflags.MarkFlagAsRequired("log_list")
gflags.MarkFlagAsRequired("chain")
gflags.MarkFlagAsRequired("output")

def _read_ct_log_list(log_list_file):
    """Parses the log list JSON, returns a log url to key map."""
    try:
        log_list_json = json.loads(log_list_file)
        log_url_to_key = {}
        for log_info in log_list_json['logs']:
            log_url_to_key[FLAGS.log_scheme + '://' + log_info['url']] = (
                base64.decodestring(log_info['key']))
        return log_url_to_key
    except (OSError, IOError) as io_exception:
        raise Exception('Could not read log list file %s: %s' %
                        (log_list_file, io_exception))

def _submit_to_single_log(log_url, full_chain):
    """Submits the chain to a single log specified by log_url."""
    ct_client = log_client.LogClient(log_url, connection_timeout=10)
    res = None
    try:
        res = ct_client.add_chain(full_chain)
    except log_client.HTTPError as err:
        logging.info('Skipping log %s because of error: %s\n', log_url, err)
    return res

def _map_log_id_to_verifier(log_list):
    """Returns a map from log id to verifier object from the log_list."""
    log_id_to_verifier = {}
    for log_key in log_list.values():
        key_info = verify.create_key_info_from_raw_key(log_key)
        key_id = hashlib.sha256(log_key).digest()
        log_id_to_verifier[key_id] = verify.LogVerifier(key_info)
    return log_id_to_verifier

def _submit_to_all_logs(log_list, certs_chain):
    """Submits the chain to all logs in log_list and validates SCTs."""
    log_id_to_verifier = _map_log_id_to_verifier(log_list)

    chain_der = [c.to_der() for c in certs_chain]
    raw_scts_for_cert = []
    for log_url in log_list.keys():
        res = _submit_to_single_log(log_url, chain_der)
        if res:
            raw_scts_for_cert.append(res)
        else:
            logging.info("No SCT from log %s", log_url)

    validated_scts = []
    for raw_sct in raw_scts_for_cert:
        key_id = raw_sct.id.key_id
        try:
            log_id_to_verifier[key_id].verify_sct(raw_sct, certs_chain)
            validated_scts.append(raw_sct)
        except error.SignatureError as err:
            logging.warning(
                    'Discarding SCT from log_id %s which does not validate: %s',
                    key_id.encode('hex'), err)
        except KeyError as err:
            logging.warning('Could not find CT log validator for log_id %s. '
                    'The log key for this log is probably misconfigured.',
                    key_id.encode('hex'))

    scts_for_cert = [tls_message.encode(proto_sct)
            for proto_sct in validated_scts
            if proto_sct]
    sct_list = client_pb2.SignedCertificateTimestampList()
    sct_list.sct_list.extend(scts_for_cert)
    return tls_message.encode(sct_list)

def run():
    """Submits the chain specified in the flags to all logs."""
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting up.")
    with open(FLAGS.log_list) as log_list_file:
        log_url_to_key = _read_ct_log_list(log_list_file.read())

    certs_chain = [c for c in cert.certs_from_pem_file(FLAGS.chain)]
    logging.info("Chain is of length %d", len(certs_chain))

    sct_list = _submit_to_all_logs(log_url_to_key, certs_chain)
    with open(FLAGS.output, 'wb') as sct_list_file:
        sct_list_file.write(sct_list)

if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    run()
