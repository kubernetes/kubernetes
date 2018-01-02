import copy

from ct.client import log_client
from ct.crypto import merkle
from ct.proto import client_pb2

DEFAULT_STH = client_pb2.SthResponse()
DEFAULT_STH.timestamp = 1234
DEFAULT_STH.tree_size = 1000
DEFAULT_STH.sha256_root_hash = "hash\x00"
DEFAULT_STH.tree_head_signature = "sig\xff"
DEFAULT_FAKE_PROOF = [(_c*32) for _c in "abc"]
DEFAULT_FAKE_ROOTS = [("cert-%d" % _i) for _i in range(4)]
DEFAULT_URI = "https://example.com"


class FakeHandlerBase(log_client.RequestHandler):
    """A fake request handler for generating responses locally."""

    def __init__(self, uri, entry_limit=0, tree_size=0):
        self._uri = uri
        self._entry_limit = entry_limit

        self._sth = copy.deepcopy(DEFAULT_STH)
        # Override with custom size
        if tree_size > 0:
            self._sth.tree_size = tree_size

    @classmethod
    def make_response(cls, code, reason, json_content=None):
        """Generate a response of the desired format."""
        raise NotImplementedError

    def get_sth(self):
        return self.make_response(200, "OK",
                                  json_content=sth_to_json(self._sth))

    def get_entries(self, start, end):
        end = min(end, self._sth.tree_size - 1)
        if start < 0 or end < 0 or start > end:
            return self.make_response(400, "Bad Request")

        if self._entry_limit > 0:
            end = min(start + self._entry_limit - 1, end)

        return self.make_response(200, "OK", json_content=entries_to_json(
            make_entries(start, end)))

    def get_sth_consistency(self, old_size, new_size):
        if not 0 <= old_size <= new_size <= self._sth.tree_size:
            return self.make_response(400, "Bad Request")

        return self.make_response(200, "OK",
                                  json_content=consistency_proof_to_json(
                                      DEFAULT_FAKE_PROOF))

    def get_roots(self):
        return self.make_response(200, "OK", json_content=roots_to_json(
            DEFAULT_FAKE_ROOTS))

    def get_entry_and_proof(self, leaf_index, tree_size):
        if (leaf_index >= tree_size or leaf_index < 0 or tree_size <= 0
            or tree_size > self._sth.tree_size):
            return self.make_response(400, "Bad Request")

        return self.make_response(200, "OK",
                                  json_content=entry_and_proof_to_json(
                                      make_entry(leaf_index),
                                      DEFAULT_FAKE_PROOF))

    def get_proof_by_hash(self, leaf_hash, tree_size):
        """If the hash is known, return a (fake) audit proof."""
        if (not leaf_hash or tree_size <= 0 or
            tree_size > self._sth.tree_size):
            return self.make_response(400, "Bad Request")

        hasher = merkle.TreeHasher()
        for i in range(self._sth.tree_size):
            entry = make_entry(i)
            if hasher.hash_leaf(entry.leaf_input) == leaf_hash:
                return self.make_response(200, "OK",
                                          json_content=proof_and_index_to_json(
                                              DEFAULT_FAKE_PROOF, i))

        # Not found
        return self.make_response(400, "Bad Request")

    def get_response(self, uri, params=None):
        """Generate a fake response."""
        if params is None:
            params = {}

        prefix = self._uri + "/ct/v1/"
        if not uri.startswith(prefix):
            return self.make_response(404, "Not Found")

        path = uri[len(prefix):]

        if path == "get-sth":
            return self.get_sth()

        elif path == "get-entries":
            start = int(params.get("start", -1))
            end = int(params.get("end", -1))
            return self.get_entries(start, end)

        elif path == "get-sth-consistency":
            old_size = int(params.get("first", -1))
            new_size = int(params.get("second", -1))
            return self.get_sth_consistency(old_size, new_size)

        elif path == "get-roots":
            return self.get_roots()

        elif path == "get-entry-and-proof":
            leaf_index = int(params.get("leaf_index", -1))
            tree_size = int(params.get("tree_size", -1))
            return self.get_entry_and_proof(leaf_index, tree_size)

        elif path == "get-proof-by-hash":
            leaf_hash = params.get("hash", "").decode("base64")
            tree_size = int(params.get("tree_size", -1))
            return self.get_proof_by_hash(leaf_hash, tree_size)

        else:
            # Bad path
            return self.make_response(404, "Not Found")


def make_entry(leaf_index):
    entry = client_pb2.EntryResponse()
    entry.leaf_input = "leaf_input-%d" % leaf_index
    entry.extra_data = "extra_data-%d" % leaf_index
    return entry


def make_entries(start, end):
    entries = []
    for i in range(start, end+1):
        entries.append(make_entry(i))
    return entries


def verify_entries(entries, start, end):
    if end - start + 1 != len(entries):
        return False
    for i in range(start, end+1):
        if make_entry(i) != entries[i]:
            return False
    return True


def sth_to_json(sth):
    return {"timestamp": sth.timestamp, "tree_size": sth.tree_size,
            "sha256_root_hash": sth.sha256_root_hash.encode("base64"),
            "tree_head_signature": sth.tree_head_signature.encode("base64")}


def entries_to_json(entries):
    return {"entries": [{"leaf_input": entry.leaf_input.encode("base64"),
                         "extra_data": entry.extra_data.encode("base64")}
                        for entry in entries]}


def consistency_proof_to_json(hashes):
    return {"consistency": [h.encode("base64") for h in hashes]}


def roots_to_json(roots):
    return {"certificates": [r.encode("base64") for r in roots]}


def entry_and_proof_to_json(entry, proof):
    return {"leaf_input": entry.leaf_input.encode("base64"),
            "extra_data": entry.extra_data.encode("base64"),
            "audit_path": [h.encode("base64") for h in proof]}


def proof_and_index_to_json(proof, leaf_index):
    return {"leaf_index": leaf_index,
            "audit_path": [h.encode("base64") for h in proof]}
