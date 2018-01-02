import json, sys, cmd
import cPickle as pickle

from verifiable_base import VerifiableBase
from verifiable_log import VerifiableLog
from verifiable_map import VerifiableMap, recalc_tree_hash

# Example general purpose verifiable database
# Mutation opertions append to its log
# Its verifiable map then calls the callback (_apply_operation) to change the view.
class VerifiableDatabase(VerifiableBase):
  def __init__(self):
    VerifiableBase.__init__(self, VerifiableLog())

  # Private, call back for the underlying map when new entries are sequenced by the log
  def _apply_operation(self, idx, operation, map):
    op = json.loads(operation)
    if op['operation'] == 'set':
      map.put(str(op['key']), str(op['value']))
    elif op['operation'] == 'delete':
      map.put(str(op['key']), '')

  # Example database operation
  def set(self, key, value):
    self._log.append(json.dumps({'operation': 'set', 'key': key, 'value': value}))

  # Example database operation
  def delete(self, key):
    self._log.append(json.dumps({'operation': 'delete', 'key': key}))

  # Return a value for a key and given tree_size (as returned by get_tree_head)
  # Also returns proof
  def get(self, key, tree_size):
    val, proof = VerifiableBase.get(self, str(key), tree_size)
    val = str(val) if len(val) else None
    return val, proof

# Test right val is returned and inclusion proof checks out
def test(db, query, tree_size, exp_val):
  val, proof = db.get(query, tree_size)
  assert val == exp_val
  assert recalc_tree_hash(query, str(val) if val else '', proof) == db.get_tree_head(tree_size)['sha256_root_hash']


class ReplCmd(cmd.Cmd):
  def __init__(self):
    cmd.Cmd.__init__(self)
    self.prompt = '> '
    self.do_new()

  def do_sth(self, arg):
    try:
      if not len(arg.strip()):
        seq = None
      else:
        seq = int(arg)
    except:
      self.help_sth()
      return
    print self.db.get_tree_head(seq)

  def help_sth(self):
    print 'sth <integer> - Updates tree to sequence number and print STH. Leave blank for latest.'

  def do_new(self, arg=''):
    self.db = VerifiableDatabase()

  def help_new(self):
    print 'new - creates a new database, called by default upon launch'

  def do_save(self, arg):
    arg = arg.strip()
    if not len(arg):
      self.help_save()
      return
    pickle.dump(self.db, file(arg, 'wb'))

  def help_save(self):
    print 'save <path> - save state to a path'

  def do_load(self, arg):
    arg = arg.strip()
    if not len(arg):
      self.help_load()
      return
    self.db = pickle.load(file(arg, 'rb'))

  def help_load(self):
    print 'load <path> - load state from path'

  def do_new(self, arg=''):
    self.db = VerifiableDatabase()

  def help_new(self):
    print 'new - creates a new database, called by default upon launch'

  def do_set(self, arg):
    try:
      n, v = arg.split(' ')
    except:
      self.help_set()
      return
    n = n.strip()
    v = v.strip()
    self.db.set(n, v)
    self.do_get(n)

  def help_set(self):
    print 'set <key> <value> - set key (string) to the specified value (string)'

  def do_get(self, arg):
    try:
      n, v = arg.split(' ')
      n = n.strip()
      v = v.strip()
    except:
      n = arg.strip()
      v = self.db.get_tree_head(None)['tree_size']
    try:
      v = int(v)
    except:
      self.help_get()
      return
    try:
      val, proof = self.db.get(n, v)
    except ValueError:
      print 'Tree size does not exist.'
      return
    print 'Value:     ', val
    print 'Proof:     ', proof
    print 'Map hash:  ', self.db.get_tree_head(v)['sha256_root_hash']
    print 'Log hash:  ', self.db.get_tree_head(v)['log_tree_head']['sha256_root_hash']
    print 'Tree size: ', self.db.get_tree_head(v)['tree_size']

  def help_get(self):
    print 'get <key> <integer> - get value as of this sequence number.  Leave blank for latest.'

  def do_del(self, arg):
    n = arg.strip()
    self.db.delete(n)
    self.do_get(n)

  def help_del(self):
    print 'del <key> - delete key (string) from database'

  def do_dump(self, arg=''):
    try:
      if not len(arg.strip()):
        seq = None
      else:
        seq = int(arg)
    except:
      self.help_dump()
      return
    print 'Tree:'
    self.db.debug_dump(seq)

  def help_dump(self):
    print 'dump <integer> - dump the tree as of this sequence number.  Leave blank for latest.'

  def do_log(self, arg=''):
    for i, x in enumerate(self.db.get_log_entries(0, self.db.get_tree_head()['tree_size'] - 1)):
      print i, x

  def help_log(self):
    print 'log - dump all ops'


db = VerifiableDatabase()
db.set('foo', 'bar')
db.set('foo', 'baz')

db.delete('foo')
db.set('foo', 'bar')
db.get_tree_head()

test(db, 'foo', 0, None)
test(db, 'foo', 1, 'bar')
test(db, 'foo', 2, 'baz')
test(db, 'foo', 3, None)
test(db, 'foo', 4, 'bar')

if __name__ == '__main__':
  ReplCmd().cmdloop('Type "help" to get started.')
