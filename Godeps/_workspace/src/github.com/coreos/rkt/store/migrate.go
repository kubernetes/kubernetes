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
)

type migrateFunc func(*sql.Tx) error

var (
	// migrateTable is a map of migrate functions. The key is the db
	// version to migrate to.
	migrateTable = map[int]migrateFunc{
		1: migrateToV1,
	}
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
		f, ok := migrateTable[v]
		if !ok {
			return fmt.Errorf("missing migrate function for version %d", v)
		}
		err := f(tx)
		if err == nil {
			updateDBVersion(tx, v)
		}
		if err != nil {
			return fmt.Errorf("failed to migrate db to version %d: %v", v, err)
		}
	}
	return nil
}

func migrateToV1(tx *sql.Tx) error {
	return nil
}
