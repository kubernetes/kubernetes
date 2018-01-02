// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

func TestGoogleClientHeader(t *testing.T) {
	const genVersion = "20170101"
	gv := strings.Replace(runtime.Version(), " ", "_", -1)
	got := GoogleClientHeader(genVersion, "gccl/xyz")
	want := fmt.Sprintf("gl-go/%s gccl/xyz gdcl/%s", gv, genVersion)
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	got = GoogleClientHeader(genVersion, "")
	want = fmt.Sprintf("gl-go/%s gdcl/%s", gv, genVersion)
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
