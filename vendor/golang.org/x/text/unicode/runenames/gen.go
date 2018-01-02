// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"log"
	"strings"
	"unicode"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/ucd"
)

// snippet is a slice of data; data is the concatenation of all of the names.
type snippet struct {
	offset int
	length int
	s      string
}

func makeTable0EntryDirect(rOffset, rLength, dOffset, dLength int) uint64 {
	if rOffset >= 1<<bitsRuneOffset {
		log.Fatalf("makeTable0EntryDirect: rOffset %d is too large", rOffset)
	}
	if rLength >= 1<<bitsRuneLength {
		log.Fatalf("makeTable0EntryDirect: rLength %d is too large", rLength)
	}
	if dOffset >= 1<<bitsDataOffset {
		log.Fatalf("makeTable0EntryDirect: dOffset %d is too large", dOffset)
	}
	if dLength >= 1<<bitsRuneLength {
		log.Fatalf("makeTable0EntryDirect: dLength %d is too large", dLength)
	}
	return uint64(rOffset)<<shiftRuneOffset |
		uint64(rLength)<<shiftRuneLength |
		uint64(dOffset)<<shiftDataOffset |
		uint64(dLength)<<shiftDataLength |
		1 // Direct bit.
}

func makeTable0EntryIndirect(rOffset, rLength, dBase, t1Offset int) uint64 {
	if rOffset >= 1<<bitsRuneOffset {
		log.Fatalf("makeTable0EntryIndirect: rOffset %d is too large", rOffset)
	}
	if rLength >= 1<<bitsRuneLength {
		log.Fatalf("makeTable0EntryIndirect: rLength %d is too large", rLength)
	}
	if dBase >= 1<<bitsDataBase {
		log.Fatalf("makeTable0EntryIndirect: dBase %d is too large", dBase)
	}
	if t1Offset >= 1<<bitsTable1Offset {
		log.Fatalf("makeTable0EntryIndirect: t1Offset %d is too large", t1Offset)
	}
	return uint64(rOffset)<<shiftRuneOffset |
		uint64(rLength)<<shiftRuneLength |
		uint64(dBase)<<shiftDataBase |
		uint64(t1Offset)<<shiftTable1Offset |
		0 // Direct bit.
}

func makeTable1Entry(x int) uint16 {
	if x < 0 || 0xffff < x {
		log.Fatalf("makeTable1Entry: entry %d is out of range", x)
	}
	return uint16(x)
}

var (
	data     []byte
	snippets = make([]snippet, 1+unicode.MaxRune)
)

func main() {
	gen.Init()

	names, counts := parse()
	appendRepeatNames(names, counts)
	appendUniqueNames(names, counts)

	table0, table1 := makeTables()

	gen.Repackage("gen_bits.go", "bits.go", "runenames")

	w := gen.NewCodeWriter()
	w.WriteVar("table0", table0)
	w.WriteVar("table1", table1)
	w.WriteConst("data", string(data))
	w.WriteGoFile("tables.go", "runenames")
}

func parse() (names []string, counts map[string]int) {
	names = make([]string, 1+unicode.MaxRune)
	counts = map[string]int{}
	ucd.Parse(gen.OpenUCDFile("UnicodeData.txt"), func(p *ucd.Parser) {
		r, s := p.Rune(0), p.String(ucd.Name)
		if s == "" {
			return
		}
		if s[0] == '<' {
			const first = ", First>"
			if i := strings.Index(s, first); i >= 0 {
				s = s[:i] + ">"
			}
		}
		names[r] = s
		counts[s]++
	})
	return names, counts
}

func appendRepeatNames(names []string, counts map[string]int) {
	alreadySeen := map[string]snippet{}
	for r, s := range names {
		if s == "" || counts[s] == 1 {
			continue
		}
		if s[0] != '<' {
			log.Fatalf("Repeated name %q does not start with a '<'", s)
		}

		if z, ok := alreadySeen[s]; ok {
			snippets[r] = z
			continue
		}

		z := snippet{
			offset: len(data),
			length: len(s),
			s:      s,
		}
		data = append(data, s...)
		snippets[r] = z
		alreadySeen[s] = z
	}
}

func appendUniqueNames(names []string, counts map[string]int) {
	for r, s := range names {
		if s == "" || counts[s] != 1 {
			continue
		}
		if s[0] == '<' {
			log.Fatalf("Unique name %q starts with a '<'", s)
		}

		z := snippet{
			offset: len(data),
			length: len(s),
			s:      s,
		}
		data = append(data, s...)
		snippets[r] = z
	}
}

func makeTables() (table0 []uint64, table1 []uint16) {
	for i := 0; i < len(snippets); {
		zi := snippets[i]
		if zi == (snippet{}) {
			i++
			continue
		}

		// Look for repeat names. If we have one, we only need a table0 entry.
		j := i + 1
		for ; j < len(snippets) && zi == snippets[j]; j++ {
		}
		if j > i+1 {
			table0 = append(table0, makeTable0EntryDirect(i, j-i, zi.offset, zi.length))
			i = j
			continue
		}

		// Otherwise, we have a run of unique names. We need one table0 entry
		// and two or more table1 entries.
		base := zi.offset &^ (1<<dataBaseUnit - 1)
		t1Offset := len(table1) + 1
		table1 = append(table1, makeTable1Entry(zi.offset-base))
		table1 = append(table1, makeTable1Entry(zi.offset+zi.length-base))
		for ; j < len(snippets) && snippets[j] != (snippet{}); j++ {
			zj := snippets[j]
			if data[zj.offset] == '<' {
				break
			}
			table1 = append(table1, makeTable1Entry(zj.offset+zj.length-base))
		}
		table0 = append(table0, makeTable0EntryIndirect(i, j-i, base>>dataBaseUnit, t1Offset))
		i = j
	}
	return table0, table1
}
