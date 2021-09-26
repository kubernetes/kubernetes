// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/internal/gen"
)

type registry struct {
	XMLName  xml.Name `xml:"registry"`
	Updated  string   `xml:"updated"`
	Registry []struct {
		ID     string `xml:"id,attr"`
		Record []struct {
			Name string `xml:"name"`
			Xref []struct {
				Type string `xml:"type,attr"`
				Data string `xml:"data,attr"`
			} `xml:"xref"`
			Desc struct {
				Data string `xml:",innerxml"`
			} `xml:"description,"`
			MIB   string   `xml:"value"`
			Alias []string `xml:"alias"`
			MIME  string   `xml:"preferred_alias"`
		} `xml:"record"`
	} `xml:"registry"`
}

func main() {
	r := gen.OpenIANAFile("assignments/character-sets/character-sets.xml")
	reg := &registry{}
	if err := xml.NewDecoder(r).Decode(&reg); err != nil && err != io.EOF {
		log.Fatalf("Error decoding charset registry: %v", err)
	}
	if len(reg.Registry) == 0 || reg.Registry[0].ID != "character-sets-1" {
		log.Fatalf("Unexpected ID %s", reg.Registry[0].ID)
	}

	x := &indexInfo{}

	for _, rec := range reg.Registry[0].Record {
		mib := identifier.MIB(parseInt(rec.MIB))
		x.addEntry(mib, rec.Name)
		for _, a := range rec.Alias {
			a = strings.Split(a, " ")[0] // strip comments.
			x.addAlias(a, mib)
			// MIB name aliases are prefixed with a "cs" (character set) in the
			// registry to identify them as display names and to ensure that
			// the name starts with a lowercase letter in case it is used as
			// an identifier. We remove it to be left with a nice clean name.
			if strings.HasPrefix(a, "cs") {
				x.setName(2, a[2:])
			}
		}
		if rec.MIME != "" {
			x.addAlias(rec.MIME, mib)
			x.setName(1, rec.MIME)
		}
	}

	w := gen.NewCodeWriter()

	fmt.Fprintln(w, `import "golang.org/x/text/encoding/internal/identifier"`)

	writeIndex(w, x)

	w.WriteGoFile("tables.go", "ianaindex")
}

type alias struct {
	name string
	mib  identifier.MIB
}

type indexInfo struct {
	// compacted index from code to MIB
	codeToMIB []identifier.MIB
	alias     []alias
	names     [][3]string
}

func (ii *indexInfo) Len() int {
	return len(ii.codeToMIB)
}

func (ii *indexInfo) Less(a, b int) bool {
	return ii.codeToMIB[a] < ii.codeToMIB[b]
}

func (ii *indexInfo) Swap(a, b int) {
	ii.codeToMIB[a], ii.codeToMIB[b] = ii.codeToMIB[b], ii.codeToMIB[a]
	// Co-sort the names.
	ii.names[a], ii.names[b] = ii.names[b], ii.names[a]
}

func (ii *indexInfo) setName(i int, name string) {
	ii.names[len(ii.names)-1][i] = name
}

func (ii *indexInfo) addEntry(mib identifier.MIB, name string) {
	ii.names = append(ii.names, [3]string{name, name, name})
	ii.addAlias(name, mib)
	ii.codeToMIB = append(ii.codeToMIB, mib)
}

func (ii *indexInfo) addAlias(name string, mib identifier.MIB) {
	// Don't add duplicates for the same mib. Adding duplicate aliases for
	// different MIBs will cause the compiler to barf on an invalid map: great!.
	for i := len(ii.alias) - 1; i >= 0 && ii.alias[i].mib == mib; i-- {
		if ii.alias[i].name == name {
			return
		}
	}
	ii.alias = append(ii.alias, alias{name, mib})
	lower := strings.ToLower(name)
	if lower != name {
		ii.addAlias(lower, mib)
	}
}

const maxMIMENameLen = '0' - 1 // officially 40, but we leave some buffer.

func writeIndex(w *gen.CodeWriter, x *indexInfo) {
	sort.Stable(x)

	// Write constants.
	fmt.Fprintln(w, "const (")
	for i, m := range x.codeToMIB {
		if i == 0 {
			fmt.Fprintf(w, "enc%d = iota\n", m)
		} else {
			fmt.Fprintf(w, "enc%d\n", m)
		}
	}
	fmt.Fprintln(w, "numIANA")
	fmt.Fprintln(w, ")")

	w.WriteVar("ianaToMIB", x.codeToMIB)

	var ianaNames, mibNames []string
	for _, names := range x.names {
		n := names[0]
		if names[0] != names[1] {
			// MIME names are mostly identical to IANA names. We share the
			// tables by setting the first byte of the string to an index into
			// the string itself (< maxMIMENameLen) to the IANA name. The MIME
			// name immediately follows the index.
			x := len(names[1]) + 1
			if x > maxMIMENameLen {
				log.Fatalf("MIME name length (%d) > %d", x, maxMIMENameLen)
			}
			n = string(x) + names[1] + names[0]
		}
		ianaNames = append(ianaNames, n)
		mibNames = append(mibNames, names[2])
	}

	w.WriteVar("ianaNames", ianaNames)
	w.WriteVar("mibNames", mibNames)

	w.WriteComment(`
	TODO: Instead of using a map, we could use binary search strings doing
	on-the fly lower-casing per character. This allows to always avoid
	allocation and will be considerably more compact.`)
	fmt.Fprintln(w, "var ianaAliases = map[string]int{")
	for _, a := range x.alias {
		fmt.Fprintf(w, "%q: enc%d,\n", a.name, a.mib)
	}
	fmt.Fprintln(w, "}")
}

func parseInt(s string) int {
	x, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		log.Fatalf("Could not parse integer: %v", err)
	}
	return int(x)
}
