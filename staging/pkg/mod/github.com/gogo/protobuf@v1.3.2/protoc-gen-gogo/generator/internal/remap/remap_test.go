// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2017 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
