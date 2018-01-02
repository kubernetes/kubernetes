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

package backup

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"

	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/user"
)

// CreateBackup backs a directory up in a given directory. It basically
// copies this directory into a given backups directory. The backups
// directory has a simple structure - a directory inside named "0" is
// the most recent backup. A directory name for oldest backup is
// deduced from a given limit. For instance, for limit being 5 the
// name for the oldest backup would be "4". If a backups number
// exceeds the given limit then only newest ones are kept and the rest
// is removed.
func CreateBackup(dir, backupsDir string, limit int) error {
	tmpBackupDir := filepath.Join(backupsDir, "tmp")
	if err := os.MkdirAll(backupsDir, 0750); err != nil {
		return err
	}
	if err := fileutil.CopyTree(dir, tmpBackupDir, user.NewBlankUidRange()); err != nil {
		return err
	}
	defer os.RemoveAll(tmpBackupDir)
	// prune backups
	if err := pruneOldBackups(backupsDir, limit-1); err != nil {
		return err
	}
	if err := shiftBackups(backupsDir, limit-2); err != nil {
		return err
	}
	if err := os.Rename(tmpBackupDir, filepath.Join(backupsDir, "0")); err != nil {
		return err
	}
	return nil
}

// pruneOldBackups removes old backups, that is - directories with
// names greater or equal than given limit.
func pruneOldBackups(dir string, limit int) error {
	if list, err := ioutil.ReadDir(dir); err != nil {
		return err
	} else {
		for _, fi := range list {
			if num, err := strconv.Atoi(fi.Name()); err != nil {
				// directory name is not a number,
				// leave it alone
				continue
			} else if num < limit {
				// directory name is a number lower
				// than a limit, leave it alone
				continue
			}
			path := filepath.Join(dir, fi.Name())
			if err := os.RemoveAll(path); err != nil {
				return err
			}
		}
	}
	return nil
}

// shiftBackups renames all directories with names being numbers up to
// oldest to names with numbers greater by one.
func shiftBackups(dir string, oldest int) error {
	if oldest < 0 {
		return nil
	}
	for i := oldest; i >= 0; i-- {
		current := filepath.Join(dir, strconv.Itoa(i))
		inc := filepath.Join(dir, strconv.Itoa(i+1))
		if err := os.Rename(current, inc); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}
