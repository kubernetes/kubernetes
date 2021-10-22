// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package api

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// Files in this package use a BSD-style license.
var sentinel = regexp.MustCompile(`(//|#) Copyright \d\d\d\d (Google LLC|The Go Authors)(\.)*( All rights reserved\.)*
(//|#) Use of this source code is governed by a BSD-style
(//|#) license that can be found in the LICENSE file.
`)

const prefix = "// Copyright"

// A few files have to be skipped.
var skip = map[string]bool{
	"tools.go": true, // This file requires another comment above the license.
	"internal/third_party/uritemplates/uritemplates.go":      true, // This file is licensed to an individual.
	"internal/third_party/uritemplates/uritemplates_test.go": true, // This file is licensed to an individual.
}

// This test validates that all go files in the repo start with an appropriate license.
func TestLicense(t *testing.T) {
	err := filepath.Walk(".", func(path string, fi os.FileInfo, err error) error {
		if skip[path] {
			return nil
		}

		if err != nil {
			return err
		}

		if filepath.Ext(path) != ".go" && filepath.Ext(path) != ".sh" {
			return nil
		}

		src, err := ioutil.ReadFile(path)
		if err != nil {
			return nil
		}

		// Verify that the license is matched.
		if !sentinel.Match(src) {
			t.Errorf("%v: license header not present", path)
			return nil
		}

		// Also check it is at the top of .go files (but not .sh files, because they must have a shebang first).
		if filepath.Ext(path) == ".go" && !bytes.HasPrefix(src, []byte(prefix)) {
			t.Errorf("%v: license header not at the top", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
