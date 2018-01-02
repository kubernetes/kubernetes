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

package db

import (
	"database/sql"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func queryValue(query string, tx *sql.Tx) (int, error) {
	var value int
	rows, err := tx.Query(query)
	if err != nil {
		return -1, err
	}
	defer rows.Close()

	if !rows.Next() {
		return -1, fmt.Errorf("no result of %q", query)
	}
	if err := rows.Scan(&value); err != nil {
		return -1, err
	}
	return value, nil
}

func insertValue(db *DB) error {
	return db.Do(func(tx *sql.Tx) error {
		// Get the current count.
		count, err := queryValue("SELECT count(*) FROM rkt_db_test", tx)
		if err != nil {
			return err
		}
		// Increase the count.
		_, err = tx.Exec(fmt.Sprintf("INSERT INTO rkt_db_test VALUES (%d)", count+1))
		return err
	})
}

func getMaxCount(db *DB, t *testing.T) int {
	var maxCount int
	var err error
	if err := db.Do(func(tx *sql.Tx) error {
		// Get the maximum count.
		maxCount, err = queryValue("SELECT max(counts) FROM rkt_db_test", tx)
		return err
	}); err != nil {
		t.Fatalf("Failed to get the maximum count: %v", err)
	}
	return maxCount
}

func createTable(db *DB, t *testing.T) {
	if err := db.Do(func(tx *sql.Tx) error {
		_, err := tx.Exec("CREATE TABLE rkt_db_test (counts int)")
		return err
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestDBRace(t *testing.T) {
	oldGoMaxProcs := runtime.GOMAXPROCS(runtime.NumCPU())
	defer runtime.GOMAXPROCS(oldGoMaxProcs)

	dir, err := ioutil.TempDir("", "rkt_db_test")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(dir)

	db, err := NewDB(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Create the table.
	createTable(db, t)

	// Insert values concurrently.
	ga := testutils.NewGoroutineAssistant(t)
	runs := 100
	ga.Add(runs)
	for i := 0; i < runs; i++ {
		go func() {
			if err := insertValue(db); err != nil {
				ga.Fatalf("Failed to insert value: %v", err)
			}
			ga.Done()
		}()
	}
	ga.Wait()

	// Check the final values.
	maxCount := getMaxCount(db, t)
	if maxCount != runs {
		t.Errorf("Expected: %v, saw: %v", runs, maxCount)
	}
}
