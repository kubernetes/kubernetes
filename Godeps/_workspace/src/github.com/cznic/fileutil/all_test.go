// Copyright (c) 2014 The fileutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fileutil

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTempFile(t *testing.T) {
	f, err := TempFile("", "abc", "mno.xyz")
	if err != nil {
		t.Fatal(err)
	}

	n := f.Name()
	t.Log(n)
	defer func() {
		f.Close()
		os.Remove(n)
	}()

	base := filepath.Base(n)
	if base == "abcmno.xyz" {
		t.Fatal(base)
	}

	if !strings.HasPrefix(base, "abc") {
		t.Fatal(base)
	}

	if !strings.HasSuffix(base, "mno.xyz") {
		t.Fatal(base)
	}
}
