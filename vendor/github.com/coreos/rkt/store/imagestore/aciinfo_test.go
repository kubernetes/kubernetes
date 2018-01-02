// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package imagestore

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
			Name:    "name01",
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

	var aciinfos []*ACIInfo
	ok := false
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfos, ok, err = GetACIInfosWithName(tx, "name01")
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
			Name:    "name01",
		}
		if err := WriteACIInfo(tx, aciinfo); err != nil {
			return err
		}
		return nil
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = s.db.Do(func(tx *sql.Tx) error {
		aciinfos, ok, err = GetACIInfosWithName(tx, "name01")
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
