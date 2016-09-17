package state

import (
	"fmt"
	"strings"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/go-memdb"
)

// KVs is used to pull the full list of KVS entries for use during snapshots.
func (s *StateSnapshot) KVs() (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("kvs", "id_prefix")
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Tombstones is used to pull all the tombstones from the graveyard.
func (s *StateSnapshot) Tombstones() (memdb.ResultIterator, error) {
	return s.store.kvsGraveyard.DumpTxn(s.tx)
}

// KVS is used when restoring from a snapshot. Use KVSSet for general inserts.
func (s *StateRestore) KVS(entry *structs.DirEntry) error {
	if err := s.tx.Insert("kvs", entry); err != nil {
		return fmt.Errorf("failed inserting kvs entry: %s", err)
	}

	if err := indexUpdateMaxTxn(s.tx, entry.ModifyIndex, "kvs"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// We have a single top-level KVS watch trigger instead of doing
	// tons of prefix watches.
	return nil
}

// Tombstone is used when restoring from a snapshot. For general inserts, use
// Graveyard.InsertTxn.
func (s *StateRestore) Tombstone(stone *Tombstone) error {
	if err := s.store.kvsGraveyard.RestoreTxn(s.tx, stone); err != nil {
		return fmt.Errorf("failed restoring tombstone: %s", err)
	}
	return nil
}

// ReapTombstones is used to delete all the tombstones with an index
// less than or equal to the given index. This is used to prevent
// unbounded storage growth of the tombstones.
func (s *StateStore) ReapTombstones(index uint64) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	if err := s.kvsGraveyard.ReapTxn(tx, index); err != nil {
		return fmt.Errorf("failed to reap kvs tombstones: %s", err)
	}

	tx.Commit()
	return nil
}

// KVSSet is used to store a key/value pair.
func (s *StateStore) KVSSet(idx uint64, entry *structs.DirEntry) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Perform the actual set.
	if err := s.kvsSetTxn(tx, idx, entry, false); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// kvsSetTxn is used to insert or update a key/value pair in the state
// store. It is the inner method used and handles only the actual storage.
// If updateSession is true, then the incoming entry will set the new
// session (should be validated before calling this). Otherwise, we will keep
// whatever the existing session is.
func (s *StateStore) kvsSetTxn(tx *memdb.Txn, idx uint64, entry *structs.DirEntry, updateSession bool) error {
	// Retrieve an existing KV pair
	existing, err := tx.First("kvs", "id", entry.Key)
	if err != nil {
		return fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Set the indexes.
	if existing != nil {
		entry.CreateIndex = existing.(*structs.DirEntry).CreateIndex
	} else {
		entry.CreateIndex = idx
	}
	entry.ModifyIndex = idx

	// Preserve the existing session unless told otherwise. The "existing"
	// session for a new entry is "no session".
	if !updateSession {
		if existing != nil {
			entry.Session = existing.(*structs.DirEntry).Session
		} else {
			entry.Session = ""
		}
	}

	// Store the kv pair in the state store and update the index.
	if err := tx.Insert("kvs", entry); err != nil {
		return fmt.Errorf("failed inserting kvs entry: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"kvs", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.kvsWatch.Notify(entry.Key, false) })
	return nil
}

// KVSGet is used to retrieve a key/value pair from the state store.
func (s *StateStore) KVSGet(key string) (uint64, *structs.DirEntry, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	return s.kvsGetTxn(tx, key)
}

// kvsGetTxn is the inner method that gets a KVS entry inside an existing
// transaction.
func (s *StateStore) kvsGetTxn(tx *memdb.Txn, key string) (uint64, *structs.DirEntry, error) {
	// Get the table index.
	idx := maxIndexTxn(tx, "kvs", "tombstones")

	// Retrieve the key.
	entry, err := tx.First("kvs", "id", key)
	if err != nil {
		return 0, nil, fmt.Errorf("failed kvs lookup: %s", err)
	}
	if entry != nil {
		return idx, entry.(*structs.DirEntry), nil
	}
	return idx, nil, nil
}

// KVSList is used to list out all keys under a given prefix. If the
// prefix is left empty, all keys in the KVS will be returned. The returned
// is the max index of the returned kvs entries or applicable tombstones, or
// else it's the full table indexes for kvs and tombstones.
func (s *StateStore) KVSList(prefix string) (uint64, structs.DirEntries, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	return s.kvsListTxn(tx, prefix)
}

// kvsListTxn is the inner method that gets a list of KVS entries matching a
// prefix.
func (s *StateStore) kvsListTxn(tx *memdb.Txn, prefix string) (uint64, structs.DirEntries, error) {
	// Get the table indexes.
	idx := maxIndexTxn(tx, "kvs", "tombstones")

	// Query the prefix and list the available keys
	entries, err := tx.Get("kvs", "id_prefix", prefix)
	if err != nil {
		return 0, nil, fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Gather all of the keys found in the store
	var ents structs.DirEntries
	var lindex uint64
	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		e := entry.(*structs.DirEntry)
		ents = append(ents, e)
		if e.ModifyIndex > lindex {
			lindex = e.ModifyIndex
		}
	}

	// Check for the highest index in the graveyard. If the prefix is empty
	// then just use the full table indexes since we are listing everything.
	if prefix != "" {
		gindex, err := s.kvsGraveyard.GetMaxIndexTxn(tx, prefix)
		if err != nil {
			return 0, nil, fmt.Errorf("failed graveyard lookup: %s", err)
		}
		if gindex > lindex {
			lindex = gindex
		}
	} else {
		lindex = idx
	}

	// Use the sub index if it was set and there are entries, otherwise use
	// the full table index from above.
	if lindex != 0 {
		idx = lindex
	}
	return idx, ents, nil
}

// KVSListKeys is used to query the KV store for keys matching the given prefix.
// An optional separator may be specified, which can be used to slice off a part
// of the response so that only a subset of the prefix is returned. In this
// mode, the keys which are omitted are still counted in the returned index.
func (s *StateStore) KVSListKeys(prefix, sep string) (uint64, []string, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table indexes.
	idx := maxIndexTxn(tx, "kvs", "tombstones")

	// Fetch keys using the specified prefix
	entries, err := tx.Get("kvs", "id_prefix", prefix)
	if err != nil {
		return 0, nil, fmt.Errorf("failed kvs lookup: %s", err)
	}

	prefixLen := len(prefix)
	sepLen := len(sep)

	var keys []string
	var lindex uint64
	var last string
	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		e := entry.(*structs.DirEntry)

		// Accumulate the high index
		if e.ModifyIndex > lindex {
			lindex = e.ModifyIndex
		}

		// Always accumulate if no separator provided
		if sepLen == 0 {
			keys = append(keys, e.Key)
			continue
		}

		// Parse and de-duplicate the returned keys based on the
		// key separator, if provided.
		after := e.Key[prefixLen:]
		sepIdx := strings.Index(after, sep)
		if sepIdx > -1 {
			key := e.Key[:prefixLen+sepIdx+sepLen]
			if key != last {
				keys = append(keys, key)
				last = key
			}
		} else {
			keys = append(keys, e.Key)
		}
	}

	// Check for the highest index in the graveyard. If the prefix is empty
	// then just use the full table indexes since we are listing everything.
	if prefix != "" {
		gindex, err := s.kvsGraveyard.GetMaxIndexTxn(tx, prefix)
		if err != nil {
			return 0, nil, fmt.Errorf("failed graveyard lookup: %s", err)
		}
		if gindex > lindex {
			lindex = gindex
		}
	} else {
		lindex = idx
	}

	// Use the sub index if it was set and there are entries, otherwise use
	// the full table index from above.
	if lindex != 0 {
		idx = lindex
	}
	return idx, keys, nil
}

// KVSDelete is used to perform a shallow delete on a single key in the
// the state store.
func (s *StateStore) KVSDelete(idx uint64, key string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Perform the actual delete
	if err := s.kvsDeleteTxn(tx, idx, key); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// kvsDeleteTxn is the inner method used to perform the actual deletion
// of a key/value pair within an existing transaction.
func (s *StateStore) kvsDeleteTxn(tx *memdb.Txn, idx uint64, key string) error {
	// Look up the entry in the state store.
	entry, err := tx.First("kvs", "id", key)
	if err != nil {
		return fmt.Errorf("failed kvs lookup: %s", err)
	}
	if entry == nil {
		return nil
	}

	// Create a tombstone.
	if err := s.kvsGraveyard.InsertTxn(tx, key, idx); err != nil {
		return fmt.Errorf("failed adding to graveyard: %s", err)
	}

	// Delete the entry and update the index.
	if err := tx.Delete("kvs", entry); err != nil {
		return fmt.Errorf("failed deleting kvs entry: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"kvs", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.kvsWatch.Notify(key, false) })
	return nil
}

// KVSDeleteCAS is used to try doing a KV delete operation with a given
// raft index. If the CAS index specified is not equal to the last
// observed index for the given key, then the call is a noop, otherwise
// a normal KV delete is invoked.
func (s *StateStore) KVSDeleteCAS(idx, cidx uint64, key string) (bool, error) {
	tx := s.db.Txn(true)
	defer tx.Abort()

	set, err := s.kvsDeleteCASTxn(tx, idx, cidx, key)
	if !set || err != nil {
		return false, err
	}

	tx.Commit()
	return true, nil
}

// kvsDeleteCASTxn is the inner method that does a CAS delete within an existing
// transaction.
func (s *StateStore) kvsDeleteCASTxn(tx *memdb.Txn, idx, cidx uint64, key string) (bool, error) {
	// Retrieve the existing kvs entry, if any exists.
	entry, err := tx.First("kvs", "id", key)
	if err != nil {
		return false, fmt.Errorf("failed kvs lookup: %s", err)
	}

	// If the existing index does not match the provided CAS
	// index arg, then we shouldn't update anything and can safely
	// return early here.
	e, ok := entry.(*structs.DirEntry)
	if !ok || e.ModifyIndex != cidx {
		return entry == nil, nil
	}

	// Call the actual deletion if the above passed.
	if err := s.kvsDeleteTxn(tx, idx, key); err != nil {
		return false, err
	}
	return true, nil
}

// KVSSetCAS is used to do a check-and-set operation on a KV entry. The
// ModifyIndex in the provided entry is used to determine if we should
// write the entry to the state store or bail. Returns a bool indicating
// if a write happened and any error.
func (s *StateStore) KVSSetCAS(idx uint64, entry *structs.DirEntry) (bool, error) {
	tx := s.db.Txn(true)
	defer tx.Abort()

	set, err := s.kvsSetCASTxn(tx, idx, entry)
	if !set || err != nil {
		return false, err
	}

	tx.Commit()
	return true, nil
}

// kvsSetCASTxn is the inner method used to do a CAS inside an existing
// transaction.
func (s *StateStore) kvsSetCASTxn(tx *memdb.Txn, idx uint64, entry *structs.DirEntry) (bool, error) {
	// Retrieve the existing entry.
	existing, err := tx.First("kvs", "id", entry.Key)
	if err != nil {
		return false, fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Check if the we should do the set. A ModifyIndex of 0 means that
	// we are doing a set-if-not-exists.
	if entry.ModifyIndex == 0 && existing != nil {
		return false, nil
	}
	if entry.ModifyIndex != 0 && existing == nil {
		return false, nil
	}
	e, ok := existing.(*structs.DirEntry)
	if ok && entry.ModifyIndex != 0 && entry.ModifyIndex != e.ModifyIndex {
		return false, nil
	}

	// If we made it this far, we should perform the set.
	if err := s.kvsSetTxn(tx, idx, entry, false); err != nil {
		return false, err
	}
	return true, nil
}

// KVSDeleteTree is used to do a recursive delete on a key prefix
// in the state store. If any keys are modified, the last index is
// set, otherwise this is a no-op.
func (s *StateStore) KVSDeleteTree(idx uint64, prefix string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	if err := s.kvsDeleteTreeTxn(tx, idx, prefix); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// kvsDeleteTreeTxn is the inner method that does a recursive delete inside an
// existing transaction.
func (s *StateStore) kvsDeleteTreeTxn(tx *memdb.Txn, idx uint64, prefix string) error {
	// Get an iterator over all of the keys with the given prefix.
	entries, err := tx.Get("kvs", "id_prefix", prefix)
	if err != nil {
		return fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Go over all of the keys and remove them. We call the delete
	// directly so that we only update the index once. We also add
	// tombstones as we go.
	var modified bool
	var objs []interface{}
	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		e := entry.(*structs.DirEntry)
		if err := s.kvsGraveyard.InsertTxn(tx, e.Key, idx); err != nil {
			return fmt.Errorf("failed adding to graveyard: %s", err)
		}
		objs = append(objs, entry)
		modified = true
	}

	// Do the actual deletes in a separate loop so we don't trash the
	// iterator as we go.
	for _, obj := range objs {
		if err := tx.Delete("kvs", obj); err != nil {
			return fmt.Errorf("failed deleting kvs entry: %s", err)
		}
	}

	// Update the index
	if modified {
		tx.Defer(func() { s.kvsWatch.Notify(prefix, true) })
		if err := tx.Insert("index", &IndexEntry{"kvs", idx}); err != nil {
			return fmt.Errorf("failed updating index: %s", err)
		}
	}
	return nil
}

// KVSLockDelay returns the expiration time for any lock delay associated with
// the given key.
func (s *StateStore) KVSLockDelay(key string) time.Time {
	return s.lockDelay.GetExpiration(key)
}

// KVSLock is similar to KVSSet but only performs the set if the lock can be
// acquired.
func (s *StateStore) KVSLock(idx uint64, entry *structs.DirEntry) (bool, error) {
	tx := s.db.Txn(true)
	defer tx.Abort()

	locked, err := s.kvsLockTxn(tx, idx, entry)
	if !locked || err != nil {
		return false, err
	}

	tx.Commit()
	return true, nil
}

// kvsLockTxn is the inner method that does a lock inside an existing
// transaction.
func (s *StateStore) kvsLockTxn(tx *memdb.Txn, idx uint64, entry *structs.DirEntry) (bool, error) {
	// Verify that a session is present.
	if entry.Session == "" {
		return false, fmt.Errorf("missing session")
	}

	// Verify that the session exists.
	sess, err := tx.First("sessions", "id", entry.Session)
	if err != nil {
		return false, fmt.Errorf("failed session lookup: %s", err)
	}
	if sess == nil {
		return false, fmt.Errorf("invalid session %#v", entry.Session)
	}

	// Retrieve the existing entry.
	existing, err := tx.First("kvs", "id", entry.Key)
	if err != nil {
		return false, fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Set up the entry, using the existing entry if present.
	if existing != nil {
		e := existing.(*structs.DirEntry)
		if e.Session == entry.Session {
			// We already hold this lock, good to go.
			entry.CreateIndex = e.CreateIndex
			entry.LockIndex = e.LockIndex
		} else if e.Session != "" {
			// Bail out, someone else holds this lock.
			return false, nil
		} else {
			// Set up a new lock with this session.
			entry.CreateIndex = e.CreateIndex
			entry.LockIndex = e.LockIndex + 1
		}
	} else {
		entry.CreateIndex = idx
		entry.LockIndex = 1
	}
	entry.ModifyIndex = idx

	// If we made it this far, we should perform the set.
	if err := s.kvsSetTxn(tx, idx, entry, true); err != nil {
		return false, err
	}
	return true, nil
}

// KVSUnlock is similar to KVSSet but only performs the set if the lock can be
// unlocked (the key must already exist and be locked).
func (s *StateStore) KVSUnlock(idx uint64, entry *structs.DirEntry) (bool, error) {
	tx := s.db.Txn(true)
	defer tx.Abort()

	unlocked, err := s.kvsUnlockTxn(tx, idx, entry)
	if !unlocked || err != nil {
		return false, err
	}

	tx.Commit()
	return true, nil
}

// kvsUnlockTxn is the inner method that does an unlock inside an existing
// transaction.
func (s *StateStore) kvsUnlockTxn(tx *memdb.Txn, idx uint64, entry *structs.DirEntry) (bool, error) {
	// Verify that a session is present.
	if entry.Session == "" {
		return false, fmt.Errorf("missing session")
	}

	// Retrieve the existing entry.
	existing, err := tx.First("kvs", "id", entry.Key)
	if err != nil {
		return false, fmt.Errorf("failed kvs lookup: %s", err)
	}

	// Bail if there's no existing key.
	if existing == nil {
		return false, nil
	}

	// Make sure the given session is the lock holder.
	e := existing.(*structs.DirEntry)
	if e.Session != entry.Session {
		return false, nil
	}

	// Clear the lock and update the entry.
	entry.Session = ""
	entry.LockIndex = e.LockIndex
	entry.CreateIndex = e.CreateIndex
	entry.ModifyIndex = idx

	// If we made it this far, we should perform the set.
	if err := s.kvsSetTxn(tx, idx, entry, true); err != nil {
		return false, err
	}
	return true, nil
}

// kvsCheckSessionTxn checks to see if the given session matches the current
// entry for a key.
func (s *StateStore) kvsCheckSessionTxn(tx *memdb.Txn, key string, session string) (*structs.DirEntry, error) {
	entry, err := tx.First("kvs", "id", key)
	if err != nil {
		return nil, fmt.Errorf("failed kvs lookup: %s", err)
	}
	if entry == nil {
		return nil, fmt.Errorf("failed to check session, key %q doesn't exist", key)
	}

	e := entry.(*structs.DirEntry)
	if e.Session != session {
		return nil, fmt.Errorf("failed session check for key %q, current session %q != %q", key, e.Session, session)
	}

	return e, nil
}

// kvsCheckIndexTxn checks to see if the given modify index matches the current
// entry for a key.
func (s *StateStore) kvsCheckIndexTxn(tx *memdb.Txn, key string, cidx uint64) (*structs.DirEntry, error) {
	entry, err := tx.First("kvs", "id", key)
	if err != nil {
		return nil, fmt.Errorf("failed kvs lookup: %s", err)
	}
	if entry == nil {
		return nil, fmt.Errorf("failed to check index, key %q doesn't exist", key)
	}

	e := entry.(*structs.DirEntry)
	if e.ModifyIndex != cidx {
		return nil, fmt.Errorf("failed index check for key %q, current modify index %d != %d", key, e.ModifyIndex, cidx)
	}

	return e, nil
}
