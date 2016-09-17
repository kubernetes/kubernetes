package state

import (
	"fmt"

	"github.com/hashicorp/go-memdb"
)

// Tombstone is the internal type used to track tombstones.
type Tombstone struct {
	Key   string
	Index uint64
}

// Graveyard manages a set of tombstones.
type Graveyard struct {
	// GC is when we create tombstones to track their time-to-live.
	// The GC is consumed upstream to manage clearing of tombstones.
	gc *TombstoneGC
}

// NewGraveyard returns a new graveyard.
func NewGraveyard(gc *TombstoneGC) *Graveyard {
	return &Graveyard{gc: gc}
}

// InsertTxn adds a new tombstone.
func (g *Graveyard) InsertTxn(tx *memdb.Txn, key string, idx uint64) error {
	// Insert the tombstone.
	stone := &Tombstone{Key: key, Index: idx}
	if err := tx.Insert("tombstones", stone); err != nil {
		return fmt.Errorf("failed inserting tombstone: %s", err)
	}

	if err := tx.Insert("index", &IndexEntry{"tombstones", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// If GC is configured, then we hint that this index requires reaping.
	if g.gc != nil {
		tx.Defer(func() { g.gc.Hint(idx) })
	}
	return nil
}

// GetMaxIndexTxn returns the highest index tombstone whose key matches the
// given context, using a prefix match.
func (g *Graveyard) GetMaxIndexTxn(tx *memdb.Txn, prefix string) (uint64, error) {
	stones, err := tx.Get("tombstones", "id_prefix", prefix)
	if err != nil {
		return 0, fmt.Errorf("failed querying tombstones: %s", err)
	}

	var lindex uint64
	for stone := stones.Next(); stone != nil; stone = stones.Next() {
		s := stone.(*Tombstone)
		if s.Index > lindex {
			lindex = s.Index
		}
	}
	return lindex, nil
}

// DumpTxn returns all the tombstones.
func (g *Graveyard) DumpTxn(tx *memdb.Txn) (memdb.ResultIterator, error) {
	iter, err := tx.Get("tombstones", "id")
	if err != nil {
		return nil, err
	}

	return iter, nil
}

// RestoreTxn is used when restoring from a snapshot. For general inserts, use
// InsertTxn.
func (g *Graveyard) RestoreTxn(tx *memdb.Txn, stone *Tombstone) error {
	if err := tx.Insert("tombstones", stone); err != nil {
		return fmt.Errorf("failed inserting tombstone: %s", err)
	}

	if err := indexUpdateMaxTxn(tx, stone.Index, "tombstones"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}
	return nil
}

// ReapTxn cleans out all tombstones whose index values are less than or equal
// to the given idx. This prevents unbounded storage growth of the tombstones.
func (g *Graveyard) ReapTxn(tx *memdb.Txn, idx uint64) error {
	// This does a full table scan since we currently can't index on a
	// numeric value. Since this is all in-memory and done infrequently
	// this pretty reasonable.
	stones, err := tx.Get("tombstones", "id")
	if err != nil {
		return fmt.Errorf("failed querying tombstones: %s", err)
	}

	// Find eligible tombstones.
	var objs []interface{}
	for stone := stones.Next(); stone != nil; stone = stones.Next() {
		if stone.(*Tombstone).Index <= idx {
			objs = append(objs, stone)
		}
	}

	// Delete the tombstones in a separate loop so we don't trash the
	// iterator.
	for _, obj := range objs {
		if err := tx.Delete("tombstones", obj); err != nil {
			return fmt.Errorf("failed deleting tombstone: %s", err)
		}
	}
	return nil
}
