package state

import (
	"reflect"
	"testing"
	"time"
)

func TestGraveyard_Lifecycle(t *testing.T) {
	g := NewGraveyard(nil)

	// Make a donor state store to steal its database, all prepared for
	// tombstones.
	s := testStateStore(t)

	// Create some tombstones.
	func() {
		tx := s.db.Txn(true)
		defer tx.Abort()

		if err := g.InsertTxn(tx, "foo/in/the/house", 2); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "foo/bar/baz", 5); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "foo/bar/zoo", 8); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "some/other/path", 9); err != nil {
			t.Fatalf("err: %s", err)
		}
		tx.Commit()
	}()

	// Check some prefixes.
	func() {
		tx := s.db.Txn(false)
		defer tx.Abort()

		if idx, err := g.GetMaxIndexTxn(tx, "foo"); idx != 8 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/in/the/house"); idx != 2 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/bar/baz"); idx != 5 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/bar/zoo"); idx != 8 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "some/other/path"); idx != 9 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, ""); idx != 9 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "nope"); idx != 0 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
	}()

	// Reap some tombstones.
	func() {
		tx := s.db.Txn(true)
		defer tx.Abort()

		if err := g.ReapTxn(tx, 6); err != nil {
			t.Fatalf("err: %s", err)
		}
		tx.Commit()
	}()

	// Check prefixes to see that the reap took effect at the right index.
	func() {
		tx := s.db.Txn(false)
		defer tx.Abort()

		if idx, err := g.GetMaxIndexTxn(tx, "foo"); idx != 8 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/in/the/house"); idx != 0 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/bar/baz"); idx != 0 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "foo/bar/zoo"); idx != 8 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "some/other/path"); idx != 9 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, ""); idx != 9 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
		if idx, err := g.GetMaxIndexTxn(tx, "nope"); idx != 0 || err != nil {
			t.Fatalf("bad: %d (%s)", idx, err)
		}
	}()
}

func TestGraveyard_GC_Trigger(t *testing.T) {
	// Set up a fast-expiring GC.
	ttl, granularity := 100*time.Millisecond, 20*time.Millisecond
	gc, err := NewTombstoneGC(ttl, granularity)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make a new graveyard and assign the GC.
	g := NewGraveyard(gc)
	gc.SetEnabled(true)

	// Make sure there's nothing already expiring.
	if gc.PendingExpiration() {
		t.Fatalf("should not have any expiring items")
	}

	// Create a tombstone but abort the transaction, this should not trigger
	// GC.
	s := testStateStore(t)
	func() {
		tx := s.db.Txn(true)
		defer tx.Abort()

		if err := g.InsertTxn(tx, "foo/in/the/house", 2); err != nil {
			t.Fatalf("err: %s", err)
		}
	}()

	// Make sure there's nothing already expiring.
	if gc.PendingExpiration() {
		t.Fatalf("should not have any expiring items")
	}

	// Now commit.
	func() {
		tx := s.db.Txn(true)
		defer tx.Abort()

		if err := g.InsertTxn(tx, "foo/in/the/house", 2); err != nil {
			t.Fatalf("err: %s", err)
		}
		tx.Commit()
	}()

	// Make sure the GC got hinted.
	if !gc.PendingExpiration() {
		t.Fatalf("should have a pending expiration")
	}

	// Make sure the index looks good.
	select {
	case idx := <-gc.ExpireCh():
		if idx != 2 {
			t.Fatalf("bad index: %d", idx)
		}
	case <-time.After(2 * ttl):
		t.Fatalf("should have gotten an expire notice")
	}
}

func TestGraveyard_Snapshot_Restore(t *testing.T) {
	g := NewGraveyard(nil)

	// Make a donor state store to steal its database, all prepared for
	// tombstones.
	s := testStateStore(t)

	// Create some tombstones.
	func() {
		tx := s.db.Txn(true)
		defer tx.Abort()

		if err := g.InsertTxn(tx, "foo/in/the/house", 2); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "foo/bar/baz", 5); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "foo/bar/zoo", 8); err != nil {
			t.Fatalf("err: %s", err)
		}
		if err := g.InsertTxn(tx, "some/other/path", 9); err != nil {
			t.Fatalf("err: %s", err)
		}
		tx.Commit()
	}()

	// Verify the index was set correctly.
	if idx := s.maxIndex("tombstones"); idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// Dump them as if we are doing a snapshot.
	dump := func() []*Tombstone {
		tx := s.db.Txn(false)
		defer tx.Abort()

		iter, err := g.DumpTxn(tx)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		var dump []*Tombstone
		for ti := iter.Next(); ti != nil; ti = iter.Next() {
			dump = append(dump, ti.(*Tombstone))
		}
		return dump
	}()

	// Verify the dump, which should be ordered by key.
	expected := []*Tombstone{
		&Tombstone{Key: "foo/bar/baz", Index: 5},
		&Tombstone{Key: "foo/bar/zoo", Index: 8},
		&Tombstone{Key: "foo/in/the/house", Index: 2},
		&Tombstone{Key: "some/other/path", Index: 9},
	}
	if !reflect.DeepEqual(dump, expected) {
		t.Fatalf("bad: %v", dump)
	}

	// Make another state store and restore from the dump.
	func() {
		s := testStateStore(t)
		func() {
			tx := s.db.Txn(true)
			defer tx.Abort()

			for _, stone := range dump {
				if err := g.RestoreTxn(tx, stone); err != nil {
					t.Fatalf("err: %s", err)
				}
			}
			tx.Commit()
		}()

		// Verify that the restore works.
		if idx := s.maxIndex("tombstones"); idx != 9 {
			t.Fatalf("bad index: %d", idx)
		}

		dump := func() []*Tombstone {
			tx := s.db.Txn(false)
			defer tx.Abort()

			iter, err := g.DumpTxn(tx)
			if err != nil {
				t.Fatalf("err: %s", err)
			}
			var dump []*Tombstone
			for ti := iter.Next(); ti != nil; ti = iter.Next() {
				dump = append(dump, ti.(*Tombstone))
			}
			return dump
		}()
		if !reflect.DeepEqual(dump, expected) {
			t.Fatalf("bad: %v", dump)
		}
	}()
}
