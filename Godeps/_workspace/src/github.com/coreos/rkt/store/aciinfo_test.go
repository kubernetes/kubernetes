package store

import (
	"database/sql"
	"io/ioutil"
	"os"
	"testing"
)

func TestWriteACIInfo(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfo := &ACIInfo{
			BlobKey: "key01",
			AppName: "name01",
		}
		if err := WriteACIInfo(tx, aciinfo); err != nil {
			return err
		}
		// Insert it another time to check that is should be overwritten
		if err := WriteACIInfo(tx, aciinfo); err != nil {
			return err
		}
		return nil
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	aciinfos := []*ACIInfo{}
	ok := false
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfos, ok, err = GetACIInfosWithAppName(tx, "name01")
		return err
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !ok {
		t.Fatalf("expected some records but none found")
	}
	if len(aciinfos) != 1 {
		t.Fatalf("wrong number of records returned, wanted: 1, got: %d", len(aciinfos))
	}

	// Add another ACIInfo for the same app name
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfo := &ACIInfo{
			BlobKey: "key02",
			AppName: "name01",
		}
		if err := WriteACIInfo(tx, aciinfo); err != nil {
			return err
		}
		return nil
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfos, ok, err = GetACIInfosWithAppName(tx, "name01")
		return err
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !ok {
		t.Fatalf("expected some records but none found")
	}
	if len(aciinfos) != 2 {
		t.Fatalf("wrong number of records returned, wanted: 2, got: %d", len(aciinfos))
	}
}
