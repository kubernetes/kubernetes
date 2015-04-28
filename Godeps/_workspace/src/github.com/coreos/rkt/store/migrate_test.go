// Copyright 2015 CoreOS, Inc.
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

package store

import (
	"database/sql"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type testdb interface {
	version() int
	populate(db *DB) error
	load(db *DB) error
	compare(db testdb) bool
}

type DBV0 struct {
	aciinfos []*ACIInfoV0_1
	remotes  []*RemoteV0_1
}

func (d *DBV0) version() int {
	return 0
}

func (d *DBV0) populate(db *DB) error {
	// As DBV0 and DBV1 have the same schema use a common populate
	// function.
	return populateDBV0_1(db, d.version(), d.aciinfos, d.remotes)
}

// load populates the given struct with the data in db.
// the given struct d should be empty
func (d *DBV0) load(db *DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV0_1(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV0_1(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV0) compare(td testdb) bool {
	d2, ok := td.(*DBV0)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d2.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d2.remotes) {
		return false
	}
	return true
}

type DBV1 struct {
	aciinfos []*ACIInfoV0_1
	remotes  []*RemoteV0_1
}

func (d *DBV1) version() int {
	return 1
}
func (d *DBV1) populate(db *DB) error {
	return populateDBV0_1(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV1) load(db *DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV0_1(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV0_1(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV1) compare(td testdb) bool {
	d2, ok := td.(*DBV1)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d2.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d2.remotes) {
		return false
	}
	return true
}

// The ACIInfo struct for different db versions. The ending VX_Y represent the
// first and the last version where the format isn't changed
// The latest existing struct should to be updated when updating the db version
// without changing the struct format (ex. V0_1 to V0_2).
// A new struct and its relative function should be added if the format is changed.
// The same applies for all of the the other structs.
type ACIInfoV0_1 struct {
	BlobKey    string
	AppName    string
	ImportTime time.Time
	Latest     bool
}

func getAllACIInfosV0_1(tx *sql.Tx) ([]*ACIInfoV0_1, error) {
	aciinfos := []*ACIInfoV0_1{}
	rows, err := tx.Query("SELECT * from aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV0_1{}
		if err := rows.Scan(&aciinfo.BlobKey, &aciinfo.AppName, &aciinfo.ImportTime, &aciinfo.Latest); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

type RemoteV0_1 struct {
	ACIURL  string
	SigURL  string
	ETag    string
	BlobKey string
}

func getAllRemoteV0_1(tx *sql.Tx) ([]*RemoteV0_1, error) {
	remotes := []*RemoteV0_1{}
	rows, err := tx.Query("SELECT * from remote")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		remote := &RemoteV0_1{}
		if err := rows.Scan(&remote.ACIURL, &remote.SigURL, &remote.ETag, &remote.BlobKey); err != nil {
			return nil, err
		}
		remotes = append(remotes, remote)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return remotes, nil
}

func populateDBV0_1(db *DB, dbVersion int, aciInfos []*ACIInfoV0_1, remotes []*RemoteV0_1) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, appname string, importtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS appnameidx ON aciinfo (appname)",
	}

	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT into aciinfo values ($1, $2, $3, $4)", aciinfo.BlobKey, aciinfo.AppName, aciinfo.ImportTime, aciinfo.Latest)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

type migrateTest struct {
	predb  testdb
	postdb testdb
	// Needed to have the right DB type to load from
	curdb testdb
}

func testMigrate(tt migrateTest) error {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		return fmt.Errorf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)

	casDir := filepath.Join(dir, "cas")
	db, err := NewDB(filepath.Join(casDir, "db"))
	if err != nil {
		return err
	}
	if err = tt.predb.populate(db); err != nil {
		return err
	}

	fn := func(tx *sql.Tx) error {
		err := migrate(tx, tt.postdb.version())
		if err != nil {
			return err
		}
		return nil
	}
	if err = db.Do(fn); err != nil {
		return err
	}

	var curDBVersion int
	fn = func(tx *sql.Tx) error {
		var err error
		curDBVersion, err = getDBVersion(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err = db.Do(fn); err != nil {
		return err
	}
	if curDBVersion != tt.postdb.version() {
		return fmt.Errorf("wrong db version: got %#v, want %#v", curDBVersion, tt.postdb.version())
	}

	if err := tt.curdb.load(db); err != nil {
		return err
	}
	if !tt.curdb.compare(tt.postdb) {
		// TODO(sgotti) not very useful as these are pointers.
		// Use something like go-spew to write the full data?
		return fmt.Errorf("got %#v, want %#v", tt.curdb, tt.postdb)
	}
	return nil
}

func TestMigrate(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)

	now := time.Now()
	tests := []migrateTest{
		// Test migration from V0 to V1
		// Empty db
		{
			&DBV0{},
			&DBV1{},
			&DBV1{},
		},
		{
			&DBV0{
				[]*ACIInfoV0_1{
					{"sha512-aaaaaaaa", "example.com/app01", now, false},
					{"sha512-bbbbbbbb", "example.com/app02", now, true},
				},
				[]*RemoteV0_1{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", "sha512-aaaaaaaa"},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", "sha512-bbbbbbbb"},
				},
			},
			&DBV1{
				[]*ACIInfoV0_1{
					{"sha512-aaaaaaaa", "example.com/app01", now, false},
					{"sha512-bbbbbbbb", "example.com/app02", now, true},
				},
				[]*RemoteV0_1{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", "sha512-aaaaaaaa"},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", "sha512-bbbbbbbb"},
				},
			},
			&DBV1{},
		},
	}

	for i, tt := range tests {
		if err := testMigrate(tt); err != nil {
			t.Errorf("#%d: unexpected error: %v", i, err)
		}
	}
}

// compareSlices compare slices regardless of the slice elements order
func compareSlicesNoOrder(i1 interface{}, i2 interface{}) bool {
	s1 := interfaceToSlice(i1)
	s2 := interfaceToSlice(i2)

	if len(s1) != len(s2) {
		return false
	}

	seen := map[int]bool{}
	for _, v1 := range s1 {
		found := false
		for i2, v2 := range s2 {
			if _, ok := seen[i2]; ok {
				continue
			}
			if reflect.DeepEqual(v1, v2) {
				found = true
				seen[i2] = true
				continue
			}
		}
		if !found {
			return false
		}

	}
	return true
}

func interfaceToSlice(s interface{}) []interface{} {
	v := reflect.ValueOf(s)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		panic(fmt.Errorf("Expected slice or array, got %T", s))
	}
	l := v.Len()
	m := make([]interface{}, l)
	for i := 0; i < l; i++ {
		m[i] = v.Index(i).Interface()
	}
	return m
}
