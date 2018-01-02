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
	"os"
	"path/filepath"

	"github.com/coreos/rkt/pkg/lock"

	_ "github.com/cznic/ql/driver"
)

const (
	defaultPathPerm = os.FileMode(0770 | os.ModeSetgid)

	DbFilename = "ql.db"
)

type DB struct {
	dbdir string
}

func NewDB(dbdir string) (*DB, error) {
	if err := os.MkdirAll(dbdir, defaultPathPerm); err != nil {
		return nil, err
	}
	return &DB{dbdir: dbdir}, nil
}

type txfunc func(*sql.Tx) error

// Do Opens the db, executes DoTx and then Closes the DB
// To support a multiprocess and multigoroutine model on a single file access
// database like ql there's the need to exlusively lock, open, close, unlock the
// db for every transaction. For this reason every db transaction should be
// fast to not block other processes/goroutines.
func (db *DB) Do(fns ...txfunc) error {
	l, err := lock.ExclusiveLock(db.dbdir, lock.Dir)
	if err != nil {
		return err
	}
	defer l.Close()

	sqldb, err := sql.Open("ql", filepath.Join(db.dbdir, DbFilename))
	if err != nil {
		return err
	}
	defer sqldb.Close()

	tx, err := sqldb.Begin()
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
