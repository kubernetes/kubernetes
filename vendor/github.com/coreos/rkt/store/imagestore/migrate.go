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
	"time"

	"github.com/hashicorp/errwrap"
	"github.com/jonboulle/clockwork"
)

type migrateFunc func(*sql.Tx) error

var (
	// migrateTable is a map of migrate functions. The key is the db
	// version to migrate to.
	migrateTable = map[int]migrateFunc{
		1: migrateToV1,
		2: migrateToV2,
		3: migrateToV3,
		4: migrateToV4,
		5: migrateToV5,
		6: migrateToV6,
		7: migrateToV7,
	}

	clock = clockwork.NewRealClock()
)

func migrate(tx *sql.Tx, finalVersion int) error {
	if finalVersion > dbVersion {
		return fmt.Errorf("required migrate final version greater than the last supported db version")
	}
	version, err := getDBVersion(tx)
	if err != nil {
		return err
	}

	for v := version + 1; v <= finalVersion; v++ {
		migrate, ok := migrateTable[v]
		if !ok {
			return fmt.Errorf("missing migrate function for version %d", v)
		}

		if err := migrate(tx); err != nil {
			return errwrap.Wrap(fmt.Errorf("failed to migrate db to version %d", v), err)
		}

		if err := updateDBVersion(tx, v); err != nil {
			return errwrap.Wrap(fmt.Errorf("updateDBVersion() failed to update the db to version %d", v), err)
		}
	}
	return nil
}

func migrateToV1(tx *sql.Tx) error {
	return nil
}

func migrateToV2(tx *sql.Tx) error {
	_, err := tx.Exec("ALTER TABLE remote ADD cachemaxage int")
	if err != nil {
		return err
	}
	_, err = tx.Exec("ALTER TABLE remote ADD downloadedtime time")
	if err != nil {
		return err
	}
	// Set the default values for the new columns on the current rows
	_, err = tx.Exec("UPDATE remote cachemaxage = 0")
	if err != nil {
		return err
	}
	t := time.Time{}.UTC()
	_, err = tx.Exec("UPDATE remote downloadedtime = $1", t)
	if err != nil {
		return err
	}
	return nil
}

func migrateToV3(tx *sql.Tx) error {
	for _, t := range []string{
		"CREATE TABLE aciinfo_tmp (blobkey string, name string, importtime time, latest bool);",
		"INSERT INTO aciinfo_tmp (blobkey, name, importtime, latest) SELECT blobkey, appname, importtime, latest from aciinfo",
		"DROP TABLE aciinfo",
		"CREATE TABLE aciinfo (blobkey string, name string, importtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
		"INSERT INTO aciinfo SELECT * from aciinfo_tmp",
		"DROP TABLE aciinfo_tmp",
	} {
		_, err := tx.Exec(t)
		if err != nil {
			return err
		}
	}
	return nil
}

func migrateToV4(tx *sql.Tx) error {
	for _, t := range []string{
		"CREATE TABLE aciinfo_tmp (blobkey string, name string, importtime time, lastusedtime time, latest bool);",
		"INSERT INTO aciinfo_tmp (blobkey, name, importtime, latest) SELECT blobkey, name, importtime, latest from aciinfo",
		"DROP TABLE aciinfo",
		// We don't use now() as a DEFAULT for lastusedtime because it doesn't
		// return a UTC time, which is what we want. Instead, we UPDATE it
		// below.
		"CREATE TABLE aciinfo (blobkey string, name string, importtime time, lastusedtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
		"INSERT INTO aciinfo SELECT * from aciinfo_tmp",
		"DROP TABLE aciinfo_tmp",
	} {
		_, err := tx.Exec(t)
		if err != nil {
			return err
		}
	}
	t := clock.Now().UTC()
	_, err := tx.Exec("UPDATE aciinfo lastusedtime = $1", t)
	if err != nil {
		return err
	}
	return nil
}

func migrateToV5(tx *sql.Tx) error {
	for _, t := range []string{
		"CREATE TABLE aciinfo_tmp (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64);",
		"INSERT INTO aciinfo_tmp (blobkey, name, importtime, lastused, latest) SELECT blobkey, name, importtime, lastusedtime, latest from aciinfo",
		"DROP TABLE aciinfo",
		"CREATE TABLE aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64 DEFAULT 0, treestoresize int64 DEFAULT 0);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
		"INSERT INTO aciinfo SELECT * from aciinfo_tmp",
		"DROP TABLE aciinfo_tmp",
	} {
		_, err := tx.Exec(t)
		if err != nil {
			return err
		}
	}

	return nil
}

func migrateToV6(tx *sql.Tx) error {
	for _, t := range []string{
		"CREATE TABLE aciinfo_tmp (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64, insecureoptions int64 DEFAULT 0);",
		"INSERT INTO aciinfo_tmp (blobkey, name, importtime, lastused, latest, size, treestoresize) SELECT blobkey, name, importtime, lastused, latest, size, treestoresize from aciinfo",
		"DROP TABLE aciinfo",
		"CREATE TABLE aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64 DEFAULT 0, treestoresize int64 DEFAULT 0, insecureoptions int64 DEFAULT 0);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
		"INSERT INTO aciinfo SELECT * from aciinfo_tmp",
		"DROP TABLE aciinfo_tmp",
	} {
		_, err := tx.Exec(t)
		if err != nil {
			return err
		}
	}

	return nil
}

func migrateToV7(tx *sql.Tx) error {
	for _, t := range []string{
		"CREATE TABLE aciinfo_tmp (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64, insecureoptions int64 DEFAULT 0);",
		"INSERT INTO aciinfo_tmp (blobkey, name, importtime, lastused, latest, size, treestoresize) SELECT blobkey, name, importtime, lastused, latest, size, treestoresize from aciinfo",
		"DROP TABLE aciinfo",
		"CREATE TABLE aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64 DEFAULT 0, treestoresize int64 DEFAULT 0);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
		"INSERT INTO aciinfo SELECT blobkey, name, importtime, lastused, latest, size, treestoresize from aciinfo_tmp",
		"DROP TABLE aciinfo_tmp",
	} {
		_, err := tx.Exec(t)
		if err != nil {
			return err
		}
	}

	return nil
}
