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

/*
Package remap handles tracking the locations of Go tokens in a source text
across a rewrite by the Go formatter.
*/
package remap

import (
	"fmt"
	"go/scanner"
	"go/token"
)

// A Location represents a span of byte offsets in the source text.
type Location struct {
	Pos, End int // End is exclusive
}

// A Map represents a mapping between token locations in an input source text
// and locations in the correspnding output text.
type Map map[Location]Location

// Find reports whether the specified span is recorded by m, and if so returns
// the new location it was mapped to. If the input span was not found, the
// returned location is the same as the input.
func (m Map) Find(pos, end int) (Location, bool) {
	key := Location{
		Pos: pos,
		End: end,
	}
	if loc, ok := m[key]; ok {
		return loc, true
	}
	return key, false
}

func (m Map) add(opos, oend, npos, nend int) {
	m[Location{Pos: opos, End: oend}] = Location{Pos: npos, End: nend}
}

// Compute constructs a location mapping from input to output.  An error is
// reported if any of the tokens of output cannot be mapped.
func Compute(input, output []byte) (Map, error) {
	itok := tokenize(input)
	otok := tokenize(output)
	if len(itok) != len(otok) {
		return nil, fmt.Errorf("wrong number of tokens, %d ≠ %d", len(itok), len(otok))
	}
	m := make(Map)
	for i, ti := range itok {
		to := otok[i]
		if ti.Token != to.Token {
			return nil, fmt.Errorf("token %d type mismatch: %s ≠ %s", i+1, ti, to)
		}
		m.add(ti.pos, ti.end, to.pos, to.end)
	}
	return m, nil
}

// tokinfo records the span and type of a source token.
type tokinfo struct {
	pos, end int
	token.Token
}

func tokenize(src []byte) []tokinfo {
	fs := token.NewFileSet()
	var s scanner.Scanner
	s.Init(fs.AddFile("src", fs.Base(), len(src)), src, nil, scanner.ScanComments)
	var info []tokinfo
	for {
		pos, next, lit := s.Scan()
		switch next {
		case token.SEMICOLON:
			continue
		}
		info = append(info, tokinfo{
			pos:   int(pos - 1),
			end:   int(pos + token.Pos(len(lit)) - 1),
			Token: next,
		})
		if next == token.EOF {
			break
		}
	}
	return info
}
