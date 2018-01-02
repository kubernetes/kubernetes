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
	"fmt"
)

const (
	// Incremental db version at the current code revision.
	dbVersion = 7
)

// Statement to run when creating a db. These are the statements to create the
// db at the latest db version (dbVersion) provided by this rkt version.
// If the db already exists migration statements should be executed
var dbCreateStmts = [...]string{
	// version table
	"CREATE TABLE IF NOT EXISTS version (version int);",
	fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

	// remote table. The primary key is "aciurl".
	"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
	"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

	// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
	"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64 DEFAULT 0, treestoresize int64 DEFAULT 0);",
	"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
	"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
}

// dbIsPopulated checks if the db is already populated (at any version) verifing if the "version" table exists
func dbIsPopulated(tx *sql.Tx) (bool, error) {
	rows, err := tx.Query("SELECT Name FROM __Table where Name == $1", "version")
	if err != nil {
		return false, err
	}
	count := 0
	for rows.Next() {
		count++
	}
	if err := rows.Err(); err != nil {
		return false, err
	}
	if count > 0 {
		return true, nil
	}
	return false, nil
}

// getDBVersion retrieves the current db version
func getDBVersion(tx *sql.Tx) (int, error) {
	var version int
	rows, err := tx.Query("SELECT version FROM version")
	if err != nil {
		return -1, err
	}
	found := false
	for rows.Next() {
		if err := rows.Scan(&version); err != nil {
			return -1, err
		}
		found = true
		break
	}
	if err := rows.Err(); err != nil {
		return -1, err
	}
	if !found {
		return -1, fmt.Errorf("db version table empty")
	}
	return version, nil
}

// updateDBVersion updates the db version
func updateDBVersion(tx *sql.Tx, version int) error {
	// ql doesn't have an INSERT OR UPDATE function so
	// it's faster to remove and reinsert the row
	_, err := tx.Exec("DELETE FROM version")
	if err != nil {
		return err
	}
	_, err = tx.Exec("INSERT INTO version VALUES ($1)", version)
	if err != nil {
		return err
	}
	return nil
}
