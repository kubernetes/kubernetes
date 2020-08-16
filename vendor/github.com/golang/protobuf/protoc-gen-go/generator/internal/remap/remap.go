// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package remap handles tracking the locations of Go tokens in a source text
// across a rewrite by the Go formatter.
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
