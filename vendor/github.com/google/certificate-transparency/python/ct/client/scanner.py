#!/usr/bin/env python

import collections
import functools
import multiprocessing
import traceback
import Queue

from ct.client import entry_decoder
from ct.client import log_client
from ct.crypto import cert
from ct.crypto import error
from ct.proto import client_pb2


# Messages types:
# Special queue messages to stop the subprocesses.
_WORKER_STOPPED = "WORKER_STOPPED"
_ERROR_PARSING_ENTRY = "ERROR_PARSING_ENTRY"
_ENTRY_MATCHING = "ENTRY_MATCHING"
_PROGRESS_REPORT = "PROGRESS_REPORT"


class QueueMessage(object):
    def __init__(self, msg_type, msg=None, certificates_scanned=1,
                 matcher_output=None):
        self.msg_type = msg_type
        self.msg = msg
        # Number of certificates scanned.
        self.certificates_scanned = certificates_scanned
        self.matcher_output = matcher_output


# This is only used on the entries input queue
_STOP_WORKER = "STOP_WORKER"

_BATCH_SIZE = 1000

def process_entries(entry_queue, output_queue, match_callback):
    stopped = False
    total_processed = 0
    while not stopped:
        count, entry = entry_queue.get()
        if entry == _STOP_WORKER:
            stopped = True
            # Each worker signals when they've picked up their
            # "STOP_WORKER" message.
            output_queue.put(QueueMessage(
                _WORKER_STOPPED,
                certificates_scanned=total_processed))
        else:
            entry_response = client_pb2.EntryResponse()
            entry_response.ParseFromString(entry)
            parsed_entry = entry_decoder.decode_entry(entry_response)
            ts_entry = parsed_entry.merkle_leaf.timestamped_entry
            total_processed += 1
            c = None
            if ts_entry.entry_type == client_pb2.X509_ENTRY:
                der_cert = ts_entry.asn1_cert
            else:
                # The original, signed precertificate.
                der_cert = (parsed_entry.extra_data.precert_chain_entry.pre_certificate)
            try:
                c = cert.Certificate(der_cert)
            except error.Error as e:
                try:
                    c = cert.Certificate(der_cert, strict_der=False)
                except error.Error as e:
                    output_queue.put(QueueMessage(
                        _ERROR_PARSING_ENTRY,
                        "Error parsing entry %d:\n%s" %
                        (count, e)))
                else:
                    output_queue.put(QueueMessage(
                        _ERROR_PARSING_ENTRY,
                        "Entry %d failed strict parsing:\n%s" %
                        (count, c)))
            except Exception as e:
                print "Unknown parsing failure for entry %d:\n%s" % (
                    count, e)
                traceback.print_exc()
                output_queue.put(QueueMessage(
                    _ERROR_PARSING_ENTRY,
                    "Entry %d failed parsing with an unknown error:\n%s" %
                    (count, e)))
            if c:
                match_result = match_callback(
                        c, ts_entry.entry_type, parsed_entry.extra_data, count)
                if match_result:
                    output_queue.put(QueueMessage(
                            _ENTRY_MATCHING,
                            "Entry %d:\n%s" % (count, c),
                            matcher_output=match_result))
            if not total_processed % _BATCH_SIZE:
                output_queue.put(QueueMessage(
                    _PROGRESS_REPORT,
                    "Scanned %d entries" % total_processed,
                    certificates_scanned=_BATCH_SIZE))

def _scan(entry_queue, log_url, range_description):
    range_start, range_end = range_description
    range_end -= 1
    client = log_client.LogClient(log_url)
    try:
        entries = client.get_entries(range_start, range_end)
        scanned = range_start
        for entry in entries:
            # Can't pickle protocol buffers with protobuf module version < 2.5.0
            # (https://code.google.com/p/protobuf/issues/detail?id=418)
            # so send serialized entry.
            entry_queue.put((scanned, entry.SerializeToString()))
            scanned += 1
    except Exception as e:
        print "Exception when fetching range %d to %d:\n%s" % (
            range_start, range_end, e)
        traceback.print_exc()

ScanResults = collections.namedtuple(
    'ScanResults', ['total', 'matches', 'errors'])


def _get_tree_size(log_url):
    client = log_client.LogClient(log_url)
    sth = client.get_sth()
    print "Got STH: %s" % sth
    return sth.tree_size


def _send_stop_to_workers(to_queue, num_instances):
    for _ in range(num_instances):
        to_queue.put((0, _STOP_WORKER))


def _process_worker_messages(
    workers_input_queue, workers_output_queue, scanners_done, num_workers,
    matcher_output_handler):
    total_scanned = 0
    total_matches = 0
    total_errors = 0
    scan_progress = 0

    workers_done = 0
    stop_sent = False
    while workers_done < num_workers:
        try:
            msg = workers_output_queue.get(block=True, timeout=3)
            if msg.msg_type == _WORKER_STOPPED:
                workers_done += 1
                total_scanned += msg.certificates_scanned
            elif msg.msg_type == _ERROR_PARSING_ENTRY:
                total_errors += 1
            elif msg.msg_type == _ENTRY_MATCHING:
                total_matches += 1
                if matcher_output_handler:
                    matcher_output_handler(msg.matcher_output)
            elif (msg.msg_type == _PROGRESS_REPORT and
                  msg.certificates_scanned > 0):
                scan_progress += msg.certificates_scanned
                print msg.msg, " Total: %d" % scan_progress
            else:
                print msg.msg
        except Queue.Empty:
            are_active = ""
            if scanners_done.value:
                are_active = "NOT"
            print "Scanners are %s active, Workers done: %d" % (
                are_active, workers_done)
        # Done handling the message, now let's check if the scanners
        # are done and if so stop the workers
        if scanners_done.value and not stop_sent:
            print "All scanners done, stopping."
            _send_stop_to_workers(workers_input_queue, num_workers)
            # To avoid re-sending stop
            stop_sent = True

    return ScanResults(total_scanned, total_matches, total_errors)

def scan_log(match_callback, log_url,total_processes=2,
             matcher_output_handler=None, start_entry=0):
    # (index, entry) tuples
    m = multiprocessing.Manager()
    R = 2
    assert total_processes >= R
    num_workers = total_processes // R
    num_scanners = total_processes - num_workers
    entry_queue = m.Queue(num_scanners * _BATCH_SIZE)
    output_queue = multiprocessing.Queue(10000)
    print "Allocating %d fetchers and %d processing workers" % (
        num_scanners, num_workers)

    tree_size = _get_tree_size(log_url)
    workers_done = multiprocessing.Value('b', 0)
    # Must use a flag rather than submitting STOP to the queue directly
    # since if the queue will be full there'll be a deadlock.
    def stop_workers_callback(_):
        workers_done.value = 1

    bound_scan = functools.partial(_scan, entry_queue, log_url)

    scan_start_range = range(start_entry, tree_size, _BATCH_SIZE)
    scan_range = zip(scan_start_range, scan_start_range[1:] + [tree_size])
    scanners_pool = multiprocessing.Pool(num_scanners)
    res = scanners_pool.map_async(bound_scan, scan_range,
                                  callback=stop_workers_callback)
    scanners_pool.close()

    workers = [
        multiprocessing.Process(
            target=process_entries,
            args=(entry_queue, output_queue, match_callback))
               for _ in range(num_workers)]
    for w in workers:
        w.start()

    try:
      res = _process_worker_messages(
          entry_queue, output_queue, workers_done, num_workers,
          matcher_output_handler)
    # Do not hang the interpreter upon ^C.
    except (KeyboardInterrupt, SystemExit):
        for w in workers:
            w.terminate()
        scanners_pool.terminate()
        m.shutdown()
        raise

    scanners_pool.join()
    for w in workers:
        w.join()
    m.shutdown()
    return res
