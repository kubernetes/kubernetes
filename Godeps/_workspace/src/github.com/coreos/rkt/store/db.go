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
	"os"
	"path/filepath"

	"github.com/coreos/rkt/pkg/lock"

	_ "github.com/cznic/ql/driver"
)

const (
	DbFilename = "ql.db"
)

type DB struct {
	dbdir string
	lock  *lock.FileLock
	sqldb *sql.DB
}

func NewDB(dbdir string) (*DB, error) {
	if err := os.MkdirAll(dbdir, defaultPathPerm); err != nil {
		return nil, err
	}
	return &DB{dbdir: dbdir}, nil
}

func (db *DB) Open() error {
	// take a lock on db dir
	if db.lock != nil {
		panic("cas db lock already gained")
	}
	dl, err := lock.ExclusiveLock(db.dbdir, lock.Dir)
	if err != nil {
		return err
	}
	db.lock = dl

	sqldb, err := sql.Open("ql", filepath.Join(db.dbdir, DbFilename))
	if err != nil {
		dl.Close()
		return err
	}
	db.sqldb = sqldb
	return nil
}

func (db *DB) Close() error {
	if db.lock == nil {
		panic("cas db, Close called without lock")
	}
	if db.sqldb == nil {
		panic("cas db, Close called without an open sqldb")
	}
	if err := db.sqldb.Close(); err != nil {
		return fmt.Errorf("cas db close failed: %v", err)
	}
	db.sqldb = nil

	db.lock.Close()
	db.lock = nil
	return nil
}

func (db *DB) Begin() (*sql.Tx, error) {
	return db.sqldb.Begin()
}

type txfunc func(*sql.Tx) error

// Do Opens the db, executes DoTx and then Closes the DB
func (db *DB) Do(fns ...txfunc) error {
	err := db.Open()
	if err != nil {
		return err
	}
	defer db.Close()

	return db.DoTx(fns...)
}

// DoTx executes the provided txfuncs inside a unique transaction.
// If one of the functions returns an error the whole transaction is rolled back.
func (db *DB) DoTx(fns ...txfunc) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	for _, fn := range fns {
		if err := fn(tx); err != nil {
			tx.Rollback()
			return err
		}
	}
	tx.Commit()
	return nil
}
