package tsm1_test

import (
	"os"
	"testing"

	"github.com/influxdb/influxdb/tsdb/engine/tsm1"
)

func TestTombstoner_Add(t *testing.T) {
	dir := MustTempDir()
	defer func() { os.RemoveAll(dir) }()

	f := MustTempFile(dir)
	ts := &tsm1.Tombstoner{Path: f.Name()}

	entries, err := ts.ReadAll()
	if err != nil {
		fatal(t, "ReadAll", err)
	}

	if got, exp := len(entries), 0; got != exp {
		t.Fatalf("length mismatch: got %v, exp %v", got, exp)
	}

	ts.Add([]string{"foo"})

	entries, err = ts.ReadAll()
	if err != nil {
		fatal(t, "ReadAll", err)
	}

	if got, exp := len(entries), 1; got != exp {
		t.Fatalf("length mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := entries[0], "foo"; got != exp {
		t.Fatalf("value mismatch: got %v, exp %v", got, exp)
	}

	// Use a new Tombstoner to verify values are persisted
	ts = &tsm1.Tombstoner{Path: f.Name()}
	entries, err = ts.ReadAll()
	if err != nil {
		fatal(t, "ReadAll", err)
	}

	if got, exp := len(entries), 1; got != exp {
		t.Fatalf("length mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := entries[0], "foo"; got != exp {
		t.Fatalf("value mismatch: got %v, exp %v", got, exp)
	}
}

func TestTombstoner_Delete(t *testing.T) {
	dir := MustTempDir()
	defer func() { os.RemoveAll(dir) }()

	f := MustTempFile(dir)
	ts := &tsm1.Tombstoner{Path: f.Name()}

	ts.Add([]string{"foo"})

	// Use a new Tombstoner to verify values are persisted
	ts = &tsm1.Tombstoner{Path: f.Name()}
	entries, err := ts.ReadAll()
	if err != nil {
		fatal(t, "ReadAll", err)
	}

	if got, exp := len(entries), 1; got != exp {
		t.Fatalf("length mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := entries[0], "foo"; got != exp {
		t.Fatalf("value mismatch: got %v, exp %v", got, exp)
	}

	if err := ts.Delete(); err != nil {
		fatal(t, "delete tombstone", err)
	}

	ts = &tsm1.Tombstoner{Path: f.Name()}
	entries, err = ts.ReadAll()
	if err != nil {
		fatal(t, "ReadAll", err)
	}

	if got, exp := len(entries), 0; got != exp {
		t.Fatalf("length mismatch: got %v, exp %v", got, exp)
	}

}
