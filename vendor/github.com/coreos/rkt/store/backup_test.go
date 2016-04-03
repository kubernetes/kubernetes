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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// TestBackup tests backup creation with limit 4
func TestBackup(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	dbDir := filepath.Join(dir, "db")
	backupsDir := filepath.Join(dir, "db-backups")
	limit := 4

	for i := 0; i < limit*2; i++ {
		if err := os.RemoveAll(dbDir); err != nil {
			t.Fatalf("error removing dbdir: %v", err)
		}
		if err := os.MkdirAll(dbDir, 0755); err != nil {
			t.Fatalf("error creating dbdir: %v", err)
		}
		if err := touch(filepath.Join(dbDir, strconv.Itoa(i))); err != nil {
			t.Fatalf("error creating a file: %v", err)
		}
		if err := createBackup(dbDir, backupsDir, limit); err != nil {
			t.Fatalf("error creating a backup: %v", err)
		}
		if err := checkBackups(backupsDir, limit, i); err != nil {
			t.Fatalf("error checking the backup: %v", err)
		}
	}
}

func touch(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return nil
}

func checkBackups(backupsDir string, limit, i int) error {
	expected := getExpectedTree(limit, i)
	return checkDirectory(backupsDir, expected)
}

// Expectations for limit L
//
// for iteration I (0, 1, 2, ...)
// at most D directories (where D = min(L,I + 1)
// dir 0 with file I
// dir 1 with file I - 1
// ...
// dir D - 1 with file I - L
func getExpectedTree(limit, iteration int) map[string]string {
	dirCount := iteration
	if iteration > limit {
		dirCount = limit
	}
	expected := make(map[string]string, dirCount)
	for j := 0; j <= dirCount; j++ {
		expected[strconv.Itoa(j)] = strconv.Itoa(iteration - j)
	}
	return expected
}

func checkDirectory(dir string, expected map[string]string) error {
	list, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, fi := range list {
		if !fi.IsDir() {
			return fmt.Errorf("%s is not a directory", fi.Name())
		}
		dirName := fi.Name()
		if _, err := strconv.Atoi(dirName); err != nil {
			return fmt.Errorf("Name is not a number: %v", err)
		}
		expectedFileName, ok := expected[dirName]
		if !ok {
			return fmt.Errorf("Unexpected name found: %s", dirName)
		}
		subList, err := ioutil.ReadDir(filepath.Join(dir, dirName))
		if err != nil {
			return fmt.Errorf("Failed to get a list of files in %s: %v",
				dirName, err)
		}
		filesCount := len(subList)
		if filesCount != 1 {
			return fmt.Errorf("Expected only 1 file in %s, got %d",
				dirName, filesCount)
		}
		gottenFileName := subList[0].Name()
		if gottenFileName != expectedFileName {
			return fmt.Errorf("Expected %s in %s, got %s",
				expectedFileName, dirName, gottenFileName)
		}
	}
	return nil
}
