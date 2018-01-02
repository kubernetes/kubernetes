// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package display

// This file contains common lookup code that is shared between the various
// implementations of Namer and Dictionaries.

import (
	"fmt"
	"sort"
	"strings"

	"golang.org/x/text/language"
)

type namer interface {
	// name gets the string for the given index. It should walk the
	// inheritance chain if a value is not present in the base index.
	name(idx int) string
}

func nameLanguage(n namer, x interface{}) string {
	t, _ := language.All.Compose(x)
	for {
		i, _, _ := langTagSet.index(t.Raw())
		if s := n.name(i); s != "" {
			return s
		}
		if t = t.Parent(); t == language.Und {
			return ""
		}
	}
}

func nameScript(n namer, x interface{}) string {
	t, _ := language.DeprecatedScript.Compose(x)
	_, s, _ := t.Raw()
	return n.name(scriptIndex.index(s.String()))
}

func nameRegion(n namer, x interface{}) string {
	t, _ := language.DeprecatedRegion.Compose(x)
	_, _, r := t.Raw()
	return n.name(regionIndex.index(r.String()))
}

func nameTag(langN, scrN, regN namer, x interface{}) string {
	t, ok := x.(language.Tag)
	if !ok {
		return ""
	}
	const form = language.All &^ language.SuppressScript
	if c, err := form.Canonicalize(t); err == nil {
		t = c
	}
	_, sRaw, rRaw := t.Raw()
	i, scr, reg := langTagSet.index(t.Raw())
	for i != -1 {
		if str := langN.name(i); str != "" {
			if hasS, hasR := (scr != language.Script{}), (reg != language.Region{}); hasS || hasR {
				ss, sr := "", ""
				if hasS {
					ss = scrN.name(scriptIndex.index(scr.String()))
				}
				if hasR {
					sr = regN.name(regionIndex.index(reg.String()))
				}
				// TODO: use patterns in CLDR or at least confirm they are the
				// same for all languages.
				if ss != "" && sr != "" {
					return fmt.Sprintf("%s (%s, %s)", str, ss, sr)
				}
				if ss != "" || sr != "" {
					return fmt.Sprintf("%s (%s%s)", str, ss, sr)
				}
			}
			return str
		}
		scr, reg = sRaw, rRaw
		if t = t.Parent(); t == language.Und {
			return ""
		}
		i, _, _ = langTagSet.index(t.Raw())
	}
	return ""
}

// header contains the data and indexes for a single namer.
// data contains a series of strings concatenated into one. index contains the
// offsets for a string in data. For example, consider a header that defines
// strings for the languages de, el, en, fi, and nl:
//
// 		header{
// 			data: "GermanGreekEnglishDutch",
//  		index: []uint16{ 0, 6, 11, 18, 18, 23 },
// 		}
//
// For a language with index i, the string is defined by
// data[index[i]:index[i+1]]. So the number of elements in index is always one
// greater than the number of languages for which header defines a value.
// A string for a language may be empty, which means the name is undefined. In
// the above example, the name for fi (Finnish) is undefined.
type header struct {
	data  string
	index []uint16
}

// name looks up the name for a tag in the dictionary, given its index.
func (h *header) name(i int) string {
	if 0 <= i && i < len(h.index)-1 {
		return h.data[h.index[i]:h.index[i+1]]
	}
	return ""
}

// tagSet is used to find the index of a language in a set of tags.
type tagSet struct {
	single tagIndex
	long   []string
}

var (
	langTagSet = tagSet{
		single: langIndex,
		long:   langTagsLong,
	}

	// selfTagSet is used for indexing the language strings in their own
	// language.
	selfTagSet = tagSet{
		single: selfIndex,
		long:   selfTagsLong,
	}

	zzzz = language.MustParseScript("Zzzz")
	zz   = language.MustParseRegion("ZZ")
)

// index returns the index of the tag for the given base, script and region or
// its parent if the tag is not available. If the match is for a parent entry,
// the excess script and region are returned.
func (ts *tagSet) index(base language.Base, scr language.Script, reg language.Region) (int, language.Script, language.Region) {
	lang := base.String()
	index := -1
	if (scr != language.Script{} || reg != language.Region{}) {
		if scr == zzzz {
			scr = language.Script{}
		}
		if reg == zz {
			reg = language.Region{}
		}

		i := sort.SearchStrings(ts.long, lang)
		// All entries have either a script or a region and not both.
		scrStr, regStr := scr.String(), reg.String()
		for ; i < len(ts.long) && strings.HasPrefix(ts.long[i], lang); i++ {
			if s := ts.long[i][len(lang)+1:]; s == scrStr {
				scr = language.Script{}
				index = i + ts.single.len()
				break
			} else if s == regStr {
				reg = language.Region{}
				index = i + ts.single.len()
				break
			}
		}
	}
	if index == -1 {
		index = ts.single.index(lang)
	}
	return index, scr, reg
}

func (ts *tagSet) Tags() []language.Tag {
	tags := make([]language.Tag, 0, ts.single.len()+len(ts.long))
	ts.single.keys(func(s string) {
		tags = append(tags, language.Raw.MustParse(s))
	})
	for _, s := range ts.long {
		tags = append(tags, language.Raw.MustParse(s))
	}
	return tags
}

func supportedScripts() []language.Script {
	scr := make([]language.Script, 0, scriptIndex.len())
	scriptIndex.keys(func(s string) {
		scr = append(scr, language.MustParseScript(s))
	})
	return scr
}

func supportedRegions() []language.Region {
	reg := make([]language.Region, 0, regionIndex.len())
	regionIndex.keys(func(s string) {
		reg = append(reg, language.MustParseRegion(s))
	})
	return reg
}

// tagIndex holds a concatenated lists of subtags of length 2 to 4, one string
// for each length, which can be used in combination with binary search to get
// the index associated with a tag.
// For example, a tagIndex{
//   "arenesfrruzh",  // 6 2-byte tags.
//   "barwae",        // 2 3-byte tags.
//   "",
// }
// would mean that the 2-byte tag "fr" had an index of 3, and the 3-byte tag
// "wae" had an index of 7.
type tagIndex [3]string

func (t *tagIndex) index(s string) int {
	sz := len(s)
	if sz < 2 || 4 < sz {
		return -1
	}
	a := t[sz-2]
	index := sort.Search(len(a)/sz, func(i int) bool {
		p := i * sz
		return a[p:p+sz] >= s
	})
	p := index * sz
	if end := p + sz; end > len(a) || a[p:end] != s {
		return -1
	}
	// Add the number of tags for smaller sizes.
	for i := 0; i < sz-2; i++ {
		index += len(t[i]) / (i + 2)
	}
	return index
}

// len returns the number of tags that are contained in the tagIndex.
func (t *tagIndex) len() (n int) {
	for i, s := range t {
		n += len(s) / (i + 2)
	}
	return n
}

// keys calls f for each tag.
func (t *tagIndex) keys(f func(key string)) {
	for i, s := range *t {
		for ; s != ""; s = s[i+2:] {
			f(s[:i+2])
		}
	}
}
