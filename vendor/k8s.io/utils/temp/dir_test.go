/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package temp

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTempDir(t *testing.T) {
	dir, err := CreateTempDir("prefix")
	if err != nil {
		t.Fatal(err)
	}

	// Delete the directory no matter what.
	defer dir.Delete()

	// Make sure we have created the dir, with the proper name
	_, err = os.Stat(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(dir.Name), "prefix") {
		t.Fatalf(`Directory doesn't start with "prefix": %q`,
			dir.Name)
	}

	// Verify that the directory is empty
	entries, err := ioutil.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("Directory should be empty, has %d elements",
			len(entries))
	}

	// Create a couple of files
	_, err = dir.NewFile("ONE")
	if err != nil {
		t.Fatal(err)
	}
	_, err = dir.NewFile("TWO")
	if err != nil {
		t.Fatal(err)
	}
	// We can't create the same file twice
	_, err = dir.NewFile("TWO")
	if err == nil {
		t.Fatal("NewFile should fail to create the same file twice")
	}

	// We have created only two files
	entries, err = ioutil.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("ReadDir should have two elements, has %d elements",
			len(entries))
	}

	// Verify that deletion works
	err = dir.Delete()
	if err != nil {
		t.Fatal(err)
	}
	_, err = os.Stat(dir.Name)
	if err == nil {
		t.Fatal("Directory should be gone, still present.")
	}
}
