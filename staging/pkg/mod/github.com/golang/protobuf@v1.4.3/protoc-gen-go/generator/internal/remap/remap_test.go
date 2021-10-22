// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package remap

import (
	"go/format"
	"testing"
)

func TestErrors(t *testing.T) {
	tests := []struct {
		in, out string
	}{
		{"", "x"},
		{"x", ""},
		{"var x int = 5\n", "var x = 5\n"},
		{"these are \"one\" thing", "those are 'another' thing"},
	}
	for _, test := range tests {
		m, err := Compute([]byte(test.in), []byte(test.out))
		if err != nil {
			t.Logf("Got expected error: %v", err)
			continue
		}
		t.Errorf("Compute(%q, %q): got %+v, wanted error", test.in, test.out, m)
	}
}

func TestMatching(t *testing.T) {
	// The input is a source text that will be rearranged by the formatter.
	const input = `package foo
var s int
func main(){}
`

	output, err := format.Source([]byte(input))
	if err != nil {
		t.Fatalf("Formatting failed: %v", err)
	}
	m, err := Compute([]byte(input), output)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Verify that the mapped locations have the same text.
	for key, val := range m {
		want := input[key.Pos:key.End]
		got := string(output[val.Pos:val.End])
		if got != want {
			t.Errorf("Token at %d:%d: got %q, want %q", key.Pos, key.End, got, want)
		}
	}
}
