// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run maketables.go -output tables.go

// Package display provides display names for languages, scripts and regions in
// a requested language.
//
// The data is based on CLDR's localeDisplayNames. It includes the names of the
// draft level "contributed" or "approved". The resulting tables are quite
// large. The display package is designed so that users can reduce the linked-in
// table sizes by cherry picking the languages one wishes to support. There is a
// Dictionary defined for a selected set of common languages for this purpose.
package display // import "golang.org/x/text/language/display"

import (
	"strings"

	"golang.org/x/text/language"
)

/*
TODO:
All fairly low priority at the moment:
  - Include alternative and variants as an option (using func options).
  - Option for returning the empty string for undefined values.
  - Support variants, currencies, time zones, option names and other data
    provided in CLDR.
  - Do various optimizations:
    - Reduce size of offset tables.
    - Consider compressing infrequently used languages and decompress on demand.
*/

// A Namer is used to get the name for a given value, such as a Tag, Language,
// Script or Region.
type Namer interface {
	// Name returns a display string for the given value. A Namer returns an
	// empty string for values it does not support. A Namer may support naming
	// an unspecified value. For example, when getting the name for a region for
	// a tag that does not have a defined Region, it may return the name for an
	// unknown region. It is up to the user to filter calls to Name for values
	// for which one does not want to have a name string.
	Name(x interface{}) string
}

var (
	// Supported lists the languages for which names are defined.
	Supported language.Coverage

	// The set of all possible values for which names are defined. Note that not
	// all Namer implementations will cover all the values of a given type.
	// A Namer will return the empty string for unsupported values.
	Values language.Coverage

	matcher language.Matcher
)

func init() {
	tags := make([]language.Tag, numSupported)
	s := supported
	for i := range tags {
		p := strings.IndexByte(s, '|')
		tags[i] = language.Raw.Make(s[:p])
		s = s[p+1:]
	}
	matcher = language.NewMatcher(tags)
	Supported = language.NewCoverage(tags)

	Values = language.NewCoverage(langTagSet.Tags, supportedScripts, supportedRegions)
}

// Languages returns a Namer for naming languages. It returns nil if there is no
// data for the given tag. The type passed to Name must be either language.Base
// or language.Tag. Note that the result may differ between passing a tag or its
// base language. For example, for English, passing "nl-BE" would return Flemish
// whereas passing "nl" returns "Dutch".
func Languages(t language.Tag) Namer {
	if _, index, conf := matcher.Match(t); conf != language.No {
		return languageNamer(index)
	}
	return nil
}

type languageNamer int

func (n languageNamer) name(i int) string {
	return lookup(langHeaders[:], int(n), i)
}

// Name implements the Namer interface for language names.
func (n languageNamer) Name(x interface{}) string {
	return nameLanguage(n, x)
}

// nonEmptyIndex walks up the parent chain until a non-empty header is found.
// It returns -1 if no index could be found.
func nonEmptyIndex(h []header, index int) int {
	for ; index != -1 && h[index].data == ""; index = int(parents[index]) {
	}
	return index
}

// Scripts returns a Namer for naming scripts. It returns nil if there is no
// data for the given tag. The type passed to Name must be either a
// language.Script or a language.Tag. It will not attempt to infer a script for
// tags with an unspecified script.
func Scripts(t language.Tag) Namer {
	if _, index, conf := matcher.Match(t); conf != language.No {
		if index = nonEmptyIndex(scriptHeaders[:], index); index != -1 {
			return scriptNamer(index)
		}
	}
	return nil
}

type scriptNamer int

func (n scriptNamer) name(i int) string {
	return lookup(scriptHeaders[:], int(n), i)
}

// Name implements the Namer interface for script names.
func (n scriptNamer) Name(x interface{}) string {
	return nameScript(n, x)
}

// Regions returns a Namer for naming regions. It returns nil if there is no
// data for the given tag. The type passed to Name must be either a
// language.Region or a language.Tag. It will not attempt to infer a region for
// tags with an unspecified region.
func Regions(t language.Tag) Namer {
	if _, index, conf := matcher.Match(t); conf != language.No {
		if index = nonEmptyIndex(regionHeaders[:], index); index != -1 {
			return regionNamer(index)
		}
	}
	return nil
}

type regionNamer int

func (n regionNamer) name(i int) string {
	return lookup(regionHeaders[:], int(n), i)
}

// Name implements the Namer interface for region names.
func (n regionNamer) Name(x interface{}) string {
	return nameRegion(n, x)
}

// Tags returns a Namer for giving a full description of a tag. The names of
// scripts and regions that are not already implied by the language name will
// in appended within parentheses. It returns nil if there is not data for the
// given tag. The type passed to Name must be a tag.
func Tags(t language.Tag) Namer {
	if _, index, conf := matcher.Match(t); conf != language.No {
		return tagNamer(index)
	}
	return nil
}

type tagNamer int

// Name implements the Namer interface for tag names.
func (n tagNamer) Name(x interface{}) string {
	return nameTag(languageNamer(n), scriptNamer(n), regionNamer(n), x)
}

// lookup finds the name for an entry in a global table, traversing the
// inheritance hierarchy if needed.
func lookup(table []header, dict, want int) string {
	for dict != -1 {
		if s := table[dict].name(want); s != "" {
			return s
		}
		dict = int(parents[dict])
	}
	return ""
}

// A Dictionary holds a collection of Namers for a single language. One can
// reduce the amount of data linked in to a binary by only referencing
// Dictionaries for the languages one needs to support instead of using the
// generic Namer factories.
type Dictionary struct {
	parent *Dictionary
	lang   header
	script header
	region header
}

// Tags returns a Namer for giving a full description of a tag. The names of
// scripts and regions that are not already implied by the language name will
// in appended within parentheses. It returns nil if there is not data for the
// given tag. The type passed to Name must be a tag.
func (d *Dictionary) Tags() Namer {
	return dictTags{d}
}

type dictTags struct {
	d *Dictionary
}

// Name implements the Namer interface for tag names.
func (n dictTags) Name(x interface{}) string {
	return nameTag(dictLanguages{n.d}, dictScripts{n.d}, dictRegions{n.d}, x)
}

// Languages returns a Namer for naming languages. It returns nil if there is no
// data for the given tag. The type passed to Name must be either language.Base
// or language.Tag. Note that the result may differ between passing a tag or its
// base language. For example, for English, passing "nl-BE" would return Flemish
// whereas passing "nl" returns "Dutch".
func (d *Dictionary) Languages() Namer {
	return dictLanguages{d}
}

type dictLanguages struct {
	d *Dictionary
}

func (n dictLanguages) name(i int) string {
	for d := n.d; d != nil; d = d.parent {
		if s := d.lang.name(i); s != "" {
			return s
		}
	}
	return ""
}

// Name implements the Namer interface for language names.
func (n dictLanguages) Name(x interface{}) string {
	return nameLanguage(n, x)
}

// Scripts returns a Namer for naming scripts. It returns nil if there is no
// data for the given tag. The type passed to Name must be either a
// language.Script or a language.Tag. It will not attempt to infer a script for
// tags with an unspecified script.
func (d *Dictionary) Scripts() Namer {
	return dictScripts{d}
}

type dictScripts struct {
	d *Dictionary
}

func (n dictScripts) name(i int) string {
	for d := n.d; d != nil; d = d.parent {
		if s := d.script.name(i); s != "" {
			return s
		}
	}
	return ""
}

// Name implements the Namer interface for script names.
func (n dictScripts) Name(x interface{}) string {
	return nameScript(n, x)
}

// Regions returns a Namer for naming regions. It returns nil if there is no
// data for the given tag. The type passed to Name must be either a
// language.Region or a language.Tag. It will not attempt to infer a region for
// tags with an unspecified region.
func (d *Dictionary) Regions() Namer {
	return dictRegions{d}
}

type dictRegions struct {
	d *Dictionary
}

func (n dictRegions) name(i int) string {
	for d := n.d; d != nil; d = d.parent {
		if s := d.region.name(i); s != "" {
			return s
		}
	}
	return ""
}

// Name implements the Namer interface for region names.
func (n dictRegions) Name(x interface{}) string {
	return nameRegion(n, x)
}

// A SelfNamer implements a Namer that returns the name of language in this same
// language. It provides a very compact mechanism to provide a comprehensive
// list of languages to users in their native language.
type SelfNamer struct {
	// Supported defines the values supported by this Namer.
	Supported language.Coverage
}

var (
	// Self is a shared instance of a SelfNamer.
	Self *SelfNamer = &self

	self = SelfNamer{language.NewCoverage(selfTagSet.Tags)}
)

// Name returns the name of a given language tag in the language identified by
// this tag. It supports both the language.Base and language.Tag types.
func (n SelfNamer) Name(x interface{}) string {
	t, _ := language.All.Compose(x)
	base, scr, reg := t.Raw()
	baseScript := language.Script{}
	if (scr == language.Script{} && reg != language.Region{}) {
		// For looking up in the self dictionary, we need to select the
		// maximized script. This is even the case if the script isn't
		// specified.
		s1, _ := t.Script()
		if baseScript = getScript(base); baseScript != s1 {
			scr = s1
		}
	}

	i, scr, reg := selfTagSet.index(base, scr, reg)
	if i == -1 {
		return ""
	}

	// Only return the display name if the script matches the expected script.
	if (scr != language.Script{}) {
		if (baseScript == language.Script{}) {
			baseScript = getScript(base)
		}
		if baseScript != scr {
			return ""
		}
	}

	return selfHeaders[0].name(i)
}

// getScript returns the maximized script for a base language.
func getScript(b language.Base) language.Script {
	tag, _ := language.Raw.Compose(b)
	scr, _ := tag.Script()
	return scr
}
