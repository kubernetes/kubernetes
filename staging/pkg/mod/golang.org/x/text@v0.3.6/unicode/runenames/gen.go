// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"bytes"
	"log"
	"sort"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/gen/bitfield"
	"golang.org/x/text/internal/ucd"
)

var (
	// computed by computeDirectOffsets
	directOffsets = map[string]int{}
	directData    bytes.Buffer

	// computed by computeEntries
	entries    []entry
	singleData bytes.Buffer
	index      []uint16
)

type entry struct {
	start    rune `bitfield:"21,startRune"`
	numRunes int  `bitfield:"16"`
	end      rune
	index    int  `bitfield:"16"`
	base     int  `bitfield:"6"`
	direct   bool `bitfield:""`
	name     string
}

func main() {
	gen.Init()

	w := gen.NewCodeWriter()
	defer w.WriteVersionedGoFile("tables.go", "runenames")

	gen.WriteUnicodeVersion(w)

	computeDirectOffsets()
	computeEntries()

	if err := bitfield.Gen(w, entry{}, nil); err != nil {
		log.Fatal(err)
	}

	type entry uint64 // trick the generation code to use the entry type
	packed := []entry{}
	for _, e := range entries {
		e.numRunes = int(e.end - e.start + 1)
		v, err := bitfield.Pack(e, nil)
		if err != nil {
			log.Fatal(err)
		}
		packed = append(packed, entry(v))
	}

	index = append(index, uint16(singleData.Len()))

	w.WriteVar("entries", packed)
	w.WriteVar("index", index)
	w.WriteConst("directData", directData.String())
	w.WriteConst("singleData", singleData.String())
}

func computeDirectOffsets() {
	counts := map[string]int{}

	p := ucd.New(gen.OpenUCDFile("UnicodeData.txt"), ucd.KeepRanges)
	for p.Next() {
		start, end := p.Range(0)
		counts[getName(p)] += int(end-start) + 1
	}

	direct := []string{}
	for k, v := range counts {
		if v > 1 {
			direct = append(direct, k)
		}
	}
	sort.Strings(direct)

	for _, s := range direct {
		directOffsets[s] = directData.Len()
		directData.WriteString(s)
	}
}

func computeEntries() {
	p := ucd.New(gen.OpenUCDFile("UnicodeData.txt"), ucd.KeepRanges)
	for p.Next() {
		start, end := p.Range(0)

		last := entry{}
		if len(entries) > 0 {
			last = entries[len(entries)-1]
		}

		name := getName(p)
		if index, ok := directOffsets[name]; ok {
			if last.name == name && last.end+1 == start {
				entries[len(entries)-1].end = end
				continue
			}
			entries = append(entries, entry{
				start:  start,
				end:    end,
				index:  index,
				base:   len(name),
				direct: true,
				name:   name,
			})
			continue
		}

		if start != end {
			log.Fatalf("Expected start == end, found %x != %x", start, end)
		}

		offset := singleData.Len()
		base := offset >> 16
		index = append(index, uint16(offset))
		singleData.WriteString(name)

		if last.base == base && last.end+1 == start {
			entries[len(entries)-1].end = start
			continue
		}

		entries = append(entries, entry{
			start: start,
			end:   end,
			index: len(index) - 1,
			base:  base,
			name:  name,
		})
	}
}

func getName(p *ucd.Parser) string {
	s := p.String(ucd.Name)
	if s == "" {
		return ""
	}
	if s[0] == '<' {
		const first = ", First>"
		if i := strings.Index(s, first); i >= 0 {
			s = s[:i] + ">"
		}

	}
	return s
}
