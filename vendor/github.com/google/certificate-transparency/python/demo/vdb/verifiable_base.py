import cPickle as pickle

from verifiable_log import VerifiableLog
from verifiable_map import VerifiableMap

# Extend this class, override _apply_operation and add your own API to:
# 1. append to log
# 2. read from map
class VerifiableBase:
  def __init__(self, log):
    # The log, such as a VerifiableLog
    self._log = log

    # Internal map that we use.  The mapper is responsible for mutating this
    # when triggered by log changes.
    self._map = VerifiableMap()

    # How many log changes have been processed
    self._ops_processed = 0

    # After we process a log operation, we capture the corresponding map
    # mutation index which may be higher or lower.
    self._log_sth_to_map_sth = {0: 0}

  # Called internally to poll the log and process all updates
  def _update_from_log(self):
    log_size = self._log.get_tree_head()['tree_size']
    ctr = 0
    while log_size > self._ops_processed:
      for entry in self._log.get_entries(self._ops_processed, log_size - 1):
        # Call mapper
        self._apply_operation(self._ops_processed, entry, self._map)
        self._ops_processed += 1
        self._log_sth_to_map_sth[self._ops_processed] = self._map.get_tree_head()['tree_size']

  # Called by the underlying map when new entries are sequenced by the log
  # Override me!
  def _apply_operation(self, idx, entry, map):
    pass

  # Get the value and proof for a key.  Tree size the number of entries in the log
  def get(self, key, tree_size):
    if tree_size > self._ops_processed:
      raise ValueError
    return self._map.get(key, self._log_sth_to_map_sth[tree_size])

  # Return the current tree head, this triggers fetching the latest entries
  # from the log (if needed) and this tree_size should be passed to corresponding
  # get() calls.
  def get_tree_head(self, tree_size=None):
    if tree_size is None or tree_size > self._ops_processed:
      self._update_from_log()
      if tree_size is None:
        tree_size = self._ops_processed
    if tree_size > self._ops_processed:
      raise ValueError
    rv = self._map.get_tree_head(self._log_sth_to_map_sth[tree_size])
    rv['tree_size'] = tree_size # override what the map says
    rv['log_tree_head'] = self._log.get_tree_head(tree_size)
    return rv

  def get_log_entries(self, start, end):
    return self._log.get_entries(start, end)

  def get_log_consistency(self, first, second):
    return self._log.consistency_proof(first, second)

  # Get the value and proof for a key.  Tree size the number of entries in the log
  def debug_dump(self, tree_size):
    return self._map._root.debug_dump(self._log_sth_to_map_sth[self.get_tree_head(tree_size)['tree_size']])
