import gflags
import logging

from ct.client import entry_decoder
from ct.client import state
from ct.client import aggregated_reporter
from ct.crypto import error
from ct.crypto import merkle
from ct.client import db_reporter
from ct.client import text_reporter
from ct.proto import client_pb2
from twisted.internet import defer
from twisted.internet import threads

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("entry_write_batch_size", 1000, "Maximum number of "
                      "entries to batch into one database write")

class Monitor(object):
    def __init__(self, client, verifier, hasher, db, cert_db, log_key,
                 state_keeper):
        self.__client = client
        self.__verifier = verifier
        self.__hasher = hasher
        self.__db = db
        self.__state_keeper = state_keeper

        # TODO(ekasper): once consistency checks are in place, also load/store
        # Merkle tree info.
        # Depends on: Merkle trees implemented in Python.
        self.__state = client_pb2.MonitorState()
        self.__report = aggregated_reporter.AggregatedCertificateReport(
                (text_reporter.TextCertificateReport(),
                 db_reporter.CertDBCertificateReport(cert_db, log_key)))
        try:
            self.__state = self.__state_keeper.read(client_pb2.MonitorState)
        except state.FileNotFoundError:
            # TODO(ekasper): initialize state file with a setup script, so we
            # can raise with certainty when it's not found.
            logging.warning("Monitor state file not found, assuming first "
                            "run.")
        else:
            if not self.__state.HasField("verified_sth"):
                logging.warning("No verified monitor state, assuming first run.")

        # load compact merkle tree state from the monitor state
        self._verified_tree = merkle.CompactMerkleTree(hasher)
        self._unverified_tree = merkle.CompactMerkleTree(hasher)
        self._verified_tree.load(self.__state.verified_tree)
        self._unverified_tree.load(self.__state.unverified_tree)

    def __repr__(self):
        return "%r(%r, %r, %r)" % (self.__class__.__name__, self.__client,
                                   self.__verifier, self.__db)

    def __str__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__, self.__client,
                                   self.__verifier, self.__db)

    def __update_state(self, new_state):
        """Update state and write to disk."""
        # save compact merkle tree state into the monitor state
        self._verified_tree.save(new_state.verified_tree)
        self._unverified_tree.save(new_state.unverified_tree)
        self.__state_keeper.write(new_state)
        self.__state = new_state
        logging.info("New state is %s" % new_state)

    @property
    def servername(self):
        return self.__client.servername

    @property
    def data_timestamp(self):
        """Timestamp of the latest verified data, in milliseconds since epoch.
        """
        return self.__state.verified_sth.timestamp

    def __fired_deferred(self, result):
        """"Create a fired deferred to indicate that the asynchronous operation
        should proceed immediately, when the result is already available."""
        fire = defer.Deferred()
        fire.callback(result)
        return fire

    def _set_pending_sth(self, new_sth):
        """Set pending_sth from new_sth, or just verified_sth if not bigger."""
        logging.info("STH verified, updating state.")
        if new_sth.tree_size < self.__state.verified_sth.tree_size:
            raise ValueError("pending size must be >= verified size")
        if new_sth.timestamp <= self.__state.verified_sth.timestamp:
            raise ValueError("pending time must be > verified time")
        new_state = client_pb2.MonitorState()
        new_state.CopyFrom(self.__state)
        if new_sth.tree_size > self.__state.verified_sth.tree_size:
            new_state.pending_sth.CopyFrom(new_sth)
        else:
            new_state.verified_sth.CopyFrom(new_sth)
        self.__update_state(new_state)
        return True

    def _set_verified_tree(self, new_tree):
        """Set verified_tree and maybe move pending_sth to verified_sth."""
        self._verified_tree = new_tree
        old_state = self.__state
        new_state = client_pb2.MonitorState()
        new_state.CopyFrom(self.__state)
        assert old_state.pending_sth.tree_size >= new_tree.tree_size
        if old_state.pending_sth.tree_size == new_tree.tree_size:
            # all pending entries retrieved
            # already did consistency checks so this should always be true
            assert (old_state.pending_sth.sha256_root_hash ==
                    self._verified_tree.root_hash())
            new_state.verified_sth.CopyFrom(old_state.pending_sth)
            new_state.ClearField("pending_sth")
        self.__update_state(new_state)
        # we just set new verified tree, so we report all changes
        self.__report.report()

    def _update_unverified_data(self, unverified_tree):
        self._unverified_tree = unverified_tree
        new_state = client_pb2.MonitorState()
        new_state.CopyFrom(self.__state)
        self.__update_state(new_state)

    def __get_audited_sth(self, sth, verify_status):
        audited_sth = client_pb2.AuditedSth()
        audited_sth.sth.CopyFrom(sth)
        audited_sth.audit.status = verify_status
        return audited_sth

    def __verify_consistency_callback(self, proof, old_sth, new_sth):
        self.__db.store_sth(self.servername,
                            self.__get_audited_sth(new_sth,
                                                   client_pb2.UNVERIFIED))
        try:
            logging.debug("got proof for (%s, %s): %s",
                old_sth.tree_size, new_sth.tree_size,
                map(lambda b: b[:8].encode("base64")[:-2] + "...", proof))
            self.__verifier.verify_sth_consistency(old_sth, new_sth, proof)
        except error.VerifyError as e:
            # catches both ConsistencyError and ProofError. when alerts are
            # implemented, only the former should trigger an immediate alert;
            # the latter may have innocent causes (e.g. data corruption,
            # software bug) so we could give it a chance to recover before
            # alerting.
            self.__db.store_sth(self.servername,
                                self.__get_audited_sth(new_sth,
                                                       client_pb2.VERIFY_ERROR))
            logging.error("Could not verify STH consistency: %s vs %s!!!\n%s" %
                          (old_sth, new_sth, e))
            raise
        else:
            self.__db.store_sth(self.servername,
                                self.__get_audited_sth(new_sth,
                                                       client_pb2.VERIFIED))

    def _verify_consistency(self, old_sth, new_sth):
        """Verifies that old STH is consistent with new STH.

        Returns: Deferred that fires with boolean indicating whether updating
        succeeded.

        Deferred raises: error.ConsistencyError if STHs were inconsistent"""
        proof = self.__client.get_sth_consistency(old_sth.tree_size,
                                                  new_sth.tree_size)
        proof.addCallback(self.__verify_consistency_callback, old_sth, new_sth)
        return proof

    def __update_sth_errback(self, failure):
        """Fired if there was network error or log server sent invalid
        response"""
        logging.error("get-sth from %s failed: %s" % (self.servername,
                                                     failure.getErrorMessage()))
        return False

    def __update_sth_verify_consistency_before_accepting_errback(self, failure):
        """Errback for verify_consistency method which is called before setting
        sth as verified. If STH was invalid appropriate error message is
        already logged, so we only want to return false as update_sth failed."""
        failure.trap(error.VerifyError)
        return False

    def __handle_old_sth_errback(self, failure, sth_response):
        failure.trap(error.VerifyError)
        logging.error("Received older STH which is older and inconsistent "
                      "with current verified STH: %s vs %s. Error: %s" %
                      (sth_response, self.__state.verified_sth, failure))

    def __handle_old_sth_callback(self, result, sth_response):
        logging.warning("Rejecting received "
                        "STH: timestamp is older than current verified "
                        "STH: %s vs %s " %
                        (sth_response, self.__state.verified_sth))

    def __update_sth_callback(self, sth_response):
        # If we got the same response as last time, do nothing.
        # If we got an older response than last time, make sure that it's
        # consistent with current verified STH and then return False.
        # (If older response is consistent then, there is nothing wrong
        # with the fact that we recieved older timestamp - the log could be
        # out of sync - but we should not rewind to older data.)
        #
        # The client should always return an STH but best eliminate the
        # None == None case explicitly by only shortcutting the verification
        # if we already have a verified STH.
        if self.__state.HasField("verified_sth"):
            if sth_response == self.__state.verified_sth:
                logging.info("Ignoring already-verified STH: %s" %
                             sth_response)
                return True
            elif (sth_response.timestamp <
                    self.__state.verified_sth.timestamp):
                d = self._verify_consistency(sth_response,
                                            self.__state.verified_sth)
                d.addCallback(self.__handle_old_sth_callback, sth_response)
                d.addErrback(self.__handle_old_sth_errback, sth_response)
                return False
        try:
            self.__verifier.verify_sth(sth_response)
        except (error.EncodingError, error.VerifyError):
            logging.error("Invalid STH: %s" % sth_response)
            return False

        # Verify consistency to catch the log trying to trick us
        # into rewinding the tree.
        d = self._verify_consistency(self.__state.verified_sth, sth_response)
        d.addCallback(lambda result: self._set_pending_sth(sth_response))
        d.addErrback(self.__update_sth_verify_consistency_before_accepting_errback)
        return d

    def _update_sth(self):
        """Get a new candidate STH. If update succeeds, stores the new STH as
        pending. Does nothing if there is already a pending
        STH.

        Returns: Deferred that fires with boolean indicating whether updating
        succeeded."""
        if self.__state.HasField("pending_sth"):
            return self.__fired_deferred(True)
        logging.info("Fetching new STH")
        sth_response = self.__client.get_sth()
        sth_response.addCallback(self.__update_sth_callback)
        sth_response.addErrback(self.__update_sth_errback)
        return sth_response

    def _compute_projected_sth_from_tree(self, tree, extra_leaves):
        partial_sth = client_pb2.SthResponse()
        old_size = tree.tree_size
        partial_sth.tree_size = old_size + len(extra_leaves)
        # we only want to check the hash, so just use a dummy timestamp
        # that looks valid so the temporal verifier doesn't complain
        partial_sth.timestamp = 0
        extra_raw_leaves = [leaf.leaf_input for leaf in extra_leaves]
        new_tree = tree.extended(extra_raw_leaves)
        partial_sth.sha256_root_hash = new_tree.root_hash()
        return partial_sth, new_tree


    def _compute_projected_sth(self, extra_leaves):
        """Compute a partial projected STH.

        Useful for when an intermediate STH is not directly available from the
        server, but you still want to do something with the root hash.

        Args:
            extra_leaves: Extra leaves present in the tree for the new STH, in
                the same order as in that tree.

        Returns:
            (partial_sth, new_tree)
            partial_sth: A partial STH with timestamp 0 and empty signature.
            new_tree: New CompactMerkleTree with the extra_leaves integrated.
        """
        return self._compute_projected_sth_from_tree(self._verified_tree,
                                                     extra_leaves)

    @staticmethod
    def __estimate_time(num_new_entries):
        if num_new_entries < 1000:
            return "a moment"
        elif num_new_entries < 1000000:
            return "a while"
        else:
            return "all night"

    def _fetch_entries_errback(self, e, consumer):
        logging.error("get-entries from %s failed: %s" %
                      (self.servername, e))
        consumer.done(None)
        return True


    def _scan_entries(self, entries):
        """Passes entries to certificate report.

        Args:
            entries: array of (entry_index, entry_response) tuples.
        """
        der_certs = []
        for entry_index, entry in entries:
            parsed_entry = entry_decoder.decode_entry(entry)
            ts_entry = parsed_entry.merkle_leaf.timestamped_entry
            if ts_entry.entry_type == client_pb2.X509_ENTRY:
                der_cert = ts_entry.asn1_cert
                der_chain = parsed_entry.extra_data.certificate_chain
            else:
                der_cert = (
                    parsed_entry.extra_data.precert_chain_entry.pre_certificate)
                der_chain = (
                    parsed_entry.extra_data.
                    precert_chain_entry.precertificate_chain)
            der_chain = der_chain[:]
            der_certs.append((entry_index, der_cert, der_chain,
                              ts_entry.entry_type))
        self.__report.scan_der_certs(der_certs)

    def _scan_entries_errback(self, e):
        logging.error("Failed to scan entries from %s: %s",
                      self.servername, e)
        self._update_unverified_data(self._verified_tree)
        return e

    class EntryConsumer(object):
        """Consumer for log_client.EntryProducer.

        When everything is consumed, consumed field fires a boolean indicating
        success of consuming.
        """
        def __init__(self, producer, monitor, pending_sth, verified_tree):
            self._producer = producer
            self._monitor = monitor
            self._pending_sth = pending_sth
            self._query_size = self._producer._end - self._producer._start + 1
            self._end = self._producer._end
            self._start = self._producer._start
            self._next_sequence_number = self._start
            #unverified_tree is tree that will be built during consumption
            self._next_sequence_number = self._producer._start
            self._unverified_tree = verified_tree
            self.consumed = defer.Deferred()
            self._fetched = 0

        def done(self, result):
            if not result:
                self.consumed.callback(False)
                return False
            self.result = result
            if result < self._query_size:
                logging.error("Failed to fetch all entries: expected tree size "
                              "%d vs retrieved tree size %d" %
                              (self._end + 1, self._next_sequence_number))
                self.consumed.callback(False)
                return False
            # check that the batch is consistent with the eventual pending_sth
            d = self._monitor._verify_new_tree(self._partial_sth, self._unverified_tree, result)
            d.chainDeferred(self.consumed)
            return True

        def consume(self, entry_batch):
            self._fetched += len(entry_batch)
            logging.info("Fetched %d entries (total: %d from %d)" %
                         (len(entry_batch), self._fetched, self._query_size))

            scan = threads.deferToThread(
                    self._monitor._scan_entries,
                    enumerate(entry_batch, self._next_sequence_number))
            scan.addErrback(self._monitor._scan_entries_errback)
            # calculate the hash for the latest fetched certs
            # TODO(ekasper): parse temporary data into permanent storage.
            self._partial_sth, self._unverified_tree = \
                    self._monitor._compute_projected_sth_from_tree(
                        self._unverified_tree, entry_batch)
            self._next_sequence_number += len(entry_batch)
            self._monitor._update_unverified_data(self._unverified_tree)
            return scan

    def _fetch_entries(self, start, end):
        """Fetches entries from the log.

        Returns: Deferred that fires with boolean indicating whether fetching
        suceeded"""
        num_new_entries = end - start + 1
        logging.info("Fetching %d new entries: this will take %s..." %
                     (num_new_entries,
                      self.__estimate_time(num_new_entries)))
        producer = self.__client.get_entries(start, end)
        consumer = Monitor.EntryConsumer(producer, self,
                                         self.__state.pending_sth,
                                         self._unverified_tree)
        d = producer.startProducing(consumer)
        d.addCallback(consumer.done)
        d.addErrback(self._fetch_entries_errback, consumer)
        return consumer.consumed

    def _update_entries(self):
        """Retrieve new entries according to the pending STH.

        Returns: Deferred that fires with boolean indicating whether updating
        succeeded.
        """
        if not self.__state.HasField("pending_sth"):
            return self.__fired_deferred(True)
        # Default is 0, which is what we want.
        wanted_entries = self.__state.pending_sth.tree_size
        last_parsed_size = self._unverified_tree.tree_size

        if wanted_entries > last_parsed_size:
            d = self._fetch_entries(last_parsed_size, wanted_entries-1)
        else:
            d = self._verify_entries()

        d.addErrback(self._update_entries_errback)
        return d

    def _update_entries_errback(self, failure):
        logging.error("Updating entries from %s failed: %s",
                      self.servername, failure)
        return False

    def _verify_entries(self):
        projected_sth, new_tree = self._compute_projected_sth_from_tree(
                self._unverified_tree, ())

        new_entries_count = new_tree.tree_size - self._verified_tree.tree_size

        return self._verify_new_tree(projected_sth, new_tree, new_entries_count)

    def _verify_new_tree(self, partial_sth, new_tree, new_entries_count):
        d = self._verify_consistency(partial_sth, self.__state.pending_sth)

        def set_verified_tree(result, new_tree, new_entries_count):
            logging.info("Verified %d entries", (new_entries_count))
            self._set_verified_tree(new_tree)
            return True

        d.addCallback(set_verified_tree, new_tree, new_entries_count)
        d.addErrback(self._verify_new_tree_errback)
        return d

    def _verify_new_tree_errback(self, failure):
            failure.trap(error.VerifyError)
            self._update_unverified_data(self._verified_tree)
            return False

    def _update_result(self, updates_result):
        if not updates_result:
            logging.error("Update failed")
        return updates_result

    def update(self):
        """Update log view. Returns True if the update succeeded, False if any
        error occurred."""
        logging.info("Starting update for %s" % self.servername)
        d = self._update_sth()
        d.addCallback(lambda sth_result: self._update_entries() if sth_result
                      else False)
        d.addCallback(self._update_result)
        return d


