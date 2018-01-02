import hashlib, base64, sys

from verifiable_log import VerifiableLog

# Making this shorter affects the Merkle tree size only, can be helpful for debugging.
SHA_LEN = 256

# Convenience
def hashit(x):
  return hashlib.sha256(x).digest()

def hashleaf(x):
  return hashit(chr(0) + x)

def hashparent(a, b):
  return hashit(chr(1) + a + b)

# Pre-compute defaults for empty leaves
DEFAULTS = [hashleaf('')]
for i in range(SHA_LEN):
  DEFAULTS.append(hashparent(DEFAULTS[-1], DEFAULTS[-1]))
DEFAULTS = DEFAULTS[::-1]


# Internal utility class for keep a "version-tracked" value
class SequencedData:
  def __init__(self, default=None):
    self._seqs = [0]
    self._vals = [default]

  # Get the value that was in effect at seq time.
  def get(self, seq):
    i = len(self._seqs) - 1  #TODO replace with bin search
    while self._seqs[i] > seq and i > 0:
      i -= 1
    return self._vals[i]

  # Set a new value in effect as of seq time.  seq must be greater than current max.
  def set(self, value, seq):
    if self.get(seq) == value:
      return False
    #elif seq == self._seqs[-1]:
    #  self._vals[-1] = value
    #  return True
    elif seq > self._seqs[-1]:
      self._seqs.append(seq)
      self._vals.append(value)
      return True
    else:
      raise

  # For debug
  def debug_dump(self):
    for s, v in zip(self._seqs, self._vals):
      print 's', repr(s), 'v', repr(v)

# Represent node in sparse merkle tree
class Node:
  def __init__(self, parent=None):
    # The parent - never changes, only None for root
    self._parent = parent

    # The depth - never changes
    self._depth = parent._depth + 1 if parent else 0

    # Hash for this node - changed by _redo_hash()
    self._hash = SequencedData(DEFAULTS[self._depth])

    # The left child node - initially None, then set, then unchanged.
    self._left = SequencedData()

    # The right child node - initially None, then set, then unchanged.
    self._right = SequencedData()

    # If _value is set, the contains the full path to that value.
    # As an optimization we store a value a high in the tree as is currently
    # unique.  We need this path to (a) calculate inclusion proofs correctly for
    # distant cousins and (b) so that we can push the value lower as more values
    # are added
    self._path = SequencedData()

    # The value for a node.  This should never be set if a left or right child
    # node is present.
    self._value = SequencedData()

    # The following is informational only...
    self._node_count = 1
    while parent:
      parent._node_count += 1
      parent = parent._parent

  # Return the hash of the left subtree, even if None
  def left_hash(self, seq):
    x = self._left.get(seq)
    return x.hash(seq) if x else DEFAULTS[self._depth + 1]

  # Return the hash of the left subtree, even if None
  def right_hash(self, seq):
    x = self._right.get(seq)
    return x.hash(seq) if x else DEFAULTS[self._depth + 1]

  def left(self, seq):
    return self._left.get(seq)

  def right(self, seq):
    return self._right.get(seq)

  def value(self, seq):
    return self._value.get(seq)

  def path(self, seq):
    return self._path.get(seq)

  def set_left(self, left, seq):
    self._left.set(left, seq)

  def set_right(self, right, seq):
    self._right.set(right, seq)

  # Change the value - will always recalc hash of all ancestors
  def set_value(self, value, seq):
    self._value.set(value, seq)
    self._redo_hash(seq)

  def set_path(self, path, seq):
    self._path.set(path, seq)

  # Recalc our hash, and all ancestors.  Even though we cheat and optimize
  # the storage of the tree into minimal nodes, we still hash 256 levels deep.
  def _redo_hash(self, seq):
    v = self._value.get(seq)
    if v is None:
      if self._left.get(seq) is None and self._right.get(seq) is None:
        h = DEFAULTS[self._depth]
      else:
        h = hashparent(self.left_hash(seq), self.right_hash(seq))
    else:
      h = hashleaf(v)
      p = self._path.get(seq)
      for i in range(SHA_LEN, self._depth, -1):
        if p[i - 1]:
          h = hashparent(DEFAULTS[i], h)
        else:
          h = hashparent(h, DEFAULTS[i])

    if self._hash.set(h, seq):
      if self._parent is not None:
        self._parent._redo_hash(seq)

  def hash(self, seq):
    return self._hash.get(seq)

  # Dump all info out for debugging
  def debug_dump(self, seq):
    print ('  ' * self._depth) + "H: " + base64.b64encode(self.hash(seq))
    v = self._value.get(seq)
    if v is None:
      x = self.left(seq)
      if x:
        print ('  ' * self._depth) + "L:"
        x.debug_dump(seq)
      else:
        print ('  ' * self._depth) + "L: " + base64.b64encode(self.left_hash(seq))
      x = self.right(seq)
      if x:
        print ('  ' * self._depth) + "R:"
        x.debug_dump(seq)
      else:
        print ('  ' * self._depth) + "R: " + base64.b64encode(self.right_hash(seq))
    else:
        print ('  ' * self._depth) + "V:" + repr(v)
        print ('  ' * self._depth) + "P:" + ''.join([str(int(x)) for x in self.path(seq)])

# Take a key as string and produce 256 boolean values indicating left (False) or right (True)
def construct_key_path(key):
  return [x == '1' for x in ''.join(('%8s' % bin(ord(x))[2:]) for x in hashlib.sha256(key).digest())[:SHA_LEN]]

# Take a proof array and replace any default values (ie empty trees) with the
# count of how many consecutive there are.  Finally base64 encode each other one.
# e.g. [xxxx, yyyyy, zzzzz, empty1, empty2, empty255] would convert to [xxx, yyy, zzz, 253]
def compress_proof(proof):
  counter = 0
  rv = []
  for y in ['' if x == DEFAULTS[idx + 1] else base64.b64encode(x) for idx, x in enumerate(proof)]:
    if y == '':
      counter += 1
    else:
      if counter > 0:
        rv.append(counter)
        counter = 0
      rv.append(y)
  if counter > 0:
    rv.append(counter)
  return rv

# Undo what compress_proof does.  Output is 256 raw byte arrays.
def decompress_proof(proof):
  pp = []
  for y in proof:
    if type(y) == int:
      pp += [''] * y
    else:
      pp.append(y)
  return [DEFAULTS[idx + 1] if x == '' else base64.b64decode(x) for idx, x in enumerate(pp)]

# Map that knows nothing about logs.
class VerifiableMap:
  def __init__(self):
    self._root = Node()

    # Every actual mutation bumps this by one.
    self._tree_size = 0

  # Internal method called recursively by put initially
  # cur is the current node, idx is what level inside of path we are at,
  # value is the value we want to place.
  def _place(self, cur, idx, path, value):
    if idx < SHA_LEN:
      vs = [value]
      ps = [path]

      prev_value = cur.value(self._tree_size)
      if prev_value is not None:
        prev_path = cur.path(self._tree_size)
        if prev_path != path:
          vs.append(prev_value)
          ps.append(prev_path)

        self._tree_size += 1
        cur.set_path(None, self._tree_size)

        self._tree_size += 1
        cur.set_value(None, self._tree_size)

      for v, p in zip(vs, ps):
        l = cur.left(self._tree_size)
        r = cur.right(self._tree_size)

        if l is None and r is None and len(vs) == 1:
          self._place(cur, SHA_LEN, p, v)
        else:
          if p[idx]: # right
            next = r
            if next is None:
              next = Node(cur)
              self._tree_size += 1
              cur.set_right(next, self._tree_size)
          else: # left
            next = l
            if next is None:
              next = Node(cur)
              self._tree_size += 1
              cur.set_left(next, self._tree_size)
          self._place(next, idx + 1, p, v)
    else:
      self._tree_size += 1
      cur.set_path(path, self._tree_size)

      self._tree_size += 1
      cur.set_value(value, self._tree_size)

  # Set key to this value. Will likely cause multiple tree mutations.
  def put(self, key, value):
    self._place(self._root, 0, construct_key_path(key), value)

  # Return the latest value for a key without any proof
  # Should only be used by friendly classes.
  def get_latest(self, key):
    cur = self._root
    path = construct_key_path(key)
    for ch in path:
      if cur is None or cur.path(self._tree_size) == path:
        break
      if ch: # right
        cur = cur.right(self._tree_size)
      else: # left
        cur = cur.left(self._tree_size)
    return cur.value(self._tree_size) if cur else ''

  # For a given key and seq return the value as of that time and a proof
  # proof is "compressed" and already base64 encoded.
  # Will not mutate the tree
  def get(self, key, seq):
    last = cur = self._root
    proof = []
    path = construct_key_path(key)
    for ch in path:
      if cur is None or cur.path(seq) == path:
        break
      else:
        last = cur
        if cur.value(seq) is not None:
          cur = None
        else:
          if ch: # right
            proof.append(cur.left_hash(seq))
            cur = cur.right(seq)
          else: # left
            proof.append(cur.right_hash(seq))
            cur = cur.left(seq)

    if len(proof) < SHA_LEN:
      v = last.value(seq)
      if v is not None:
        p = last.path(seq)
        while len(proof) < SHA_LEN and path[len(proof)] == p[len(proof)]:
          proof.append(DEFAULTS[len(proof) + 1])
        if len(proof) < SHA_LEN:
          h = hashleaf(v)
          for i in range(SHA_LEN, len(proof) + 1, -1):
            if p[i - 1]:
              h = hashparent(DEFAULTS[i], h)
            else:
              h = hashparent(h, DEFAULTS[i])
          proof.append(h)

    if len(proof) < SHA_LEN:
      proof += DEFAULTS[-(SHA_LEN - len(proof)):]

    return cur.value(seq) if cur else '', compress_proof(proof)

  # Return the tree size (# of mutations) and the root hash as of that time.
  def get_tree_head(self, seq=None):
    if seq is None:
      seq = self._tree_size
    rv = {
      'tree_size': seq,
      'sha256_root_hash': base64.b64encode(self._root.hash(seq)),
    }
    return rv

# Utility method for checking an inclusion proof
def recalc_tree_hash(key, value, proof):
  t = hashleaf(value)
  for ch, p in zip(construct_key_path(key), decompress_proof(proof))[::-1]:
    if ch:
      t = hashparent(p, t)
    else:
      t = hashparent(t, p)
  return base64.b64encode(t)
