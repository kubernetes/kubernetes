// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ianaindex

import (
	"testing"
	"unicode"

	"golang.org/x/text/encoding"
)

func TestASCIIDecoder(t *testing.T) {
	repl := string(unicode.ReplacementChar)
	input := "Comment Candide fut élevé dans un beau château"
	want := "Comment Candide fut " + repl + repl + "lev" + repl + repl + " dans un beau ch" + repl + repl + "teau"
	got, err := asciiEnc.NewDecoder().String(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Fatalf("asciiEnc.NewDecoder().String() = %q, want %q", got, want)
	}
}

func TestASCIIEncoder(t *testing.T) {
	repl := string(encoding.ASCIISub)
	input := "Comment Candide fut élevé dans un beau château"
	want := "Comment Candide fut " + repl + "lev" + repl + " dans un beau ch" + repl + "teau"
	got, err := encoding.ReplaceUnsupported(asciiEnc.NewEncoder()).String(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != want {
		t.Fatalf("asciiEnc.NewEncoder().String() = %q, want %q", got, want)
	}
}
