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

package store

import (
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"sync"

	"github.com/coreos/rkt/pkg/lock"
	"github.com/hashicorp/errwrap"

	_ "github.com/cznic/ql/driver"
)

const (
	DbFilename = "ql.db"
)

// dbLock is used to guarantee both thread-safety and process-safety
// for db access.
type dbLock struct {
	fl *lock.FileLock
	// Although the FileLock already ensures thread safety, the Go runtime is unaware
	// of this, and so Mutex is necessary to satisfy the Go race detector.
	sync.Mutex
}

func newDBLock(dirPath string) (*dbLock, error) {
	l, err := lock.NewLock(dirPath, lock.Dir)
	if err != nil {
		return nil, err
	}
	return &dbLock{fl: l}, nil
}

func (dl *dbLock) lock() error {
	if err := dl.fl.ExclusiveLock(); err != nil {
		return err
	}
	dl.Lock()
	return nil
}

func (dl *dbLock) unlock() error {
	if err := dl.fl.Unlock(); err != nil {
		return err
	}
	dl.Unlock()
	return nil
}

type DB struct {
	dbdir string
	dl    *dbLock
	sqldb *sql.DB
}

func NewDB(dbdir string) (*DB, error) {
	if err := os.MkdirAll(dbdir, defaultPathPerm); err != nil {
		return nil, err
	}

	dl, err := newDBLock(dbdir)
	if err != nil {
		return nil, err
	}

	return &DB{dbdir: dbdir, dl: dl}, nil
}

func (db *DB) Open() error {
	if err := db.dl.lock(); err != nil {
		return err
	}

	sqldb, err := sql.Open("ql", filepath.Join(db.dbdir, DbFilename))
	if err != nil {
		unlockErr := db.dl.unlock()
		if unlockErr != nil {
			err = errwrap.Wrap(unlockErr, err)
		}
		return err
	}
	db.sqldb = sqldb

	return nil
}

func (db *DB) Close() error {
	if db.sqldb == nil {
		panic("cas db, Close called without an open sqldb")
	}

	if err := db.sqldb.Close(); err != nil {
		return errwrap.Wrap(errors.New("cas db close failed"), err)
	}
	db.sqldb = nil

	// Don't close the flock as it will be reused.
	return db.dl.unlock()
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
	return tx.Commit()
}
