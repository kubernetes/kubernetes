// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"bytes"
	"io"
	"log"
	"os"

	"github.com/ulikunitz/xz"
)

func main() {
	const text = "The quick brown fox jumps over the lazy dog.\n"
	var buf bytes.Buffer
	// compress text
	w, err := xz.NewWriter(&buf)
	if err != nil {
		log.Fatalf("xz.NewWriter error %s", err)
	}
	if _, err := io.WriteString(w, text); err != nil {
		log.Fatalf("WriteString error %s", err)
	}
	if err := w.Close(); err != nil {
		log.Fatalf("w.Close error %s", err)
	}
	// decompress buffer and write output to stdout
	r, err := xz.NewReader(&buf)
	if err != nil {
		log.Fatalf("NewReader error %s", err)
	}
	if _, err = io.Copy(os.Stdout, r); err != nil {
		log.Fatalf("io.Copy error %s", err)
	}
}
