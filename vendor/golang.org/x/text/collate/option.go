// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"sort"

	"golang.org/x/text/internal/colltab"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
)

// newCollator creates a new collator with default options configured.
func newCollator(t colltab.Weighter) *Collator {
	// Initialize a collator with default options.
	c := &Collator{
		options: options{
			ignore: [colltab.NumLevels]bool{
				colltab.Quaternary: true,
				colltab.Identity:   true,
			},
			f: norm.NFD,
			t: t,
		},
	}

	// TODO: store vt in tags or remove.
	c.variableTop = t.Top()

	return c
}

// An Option is used to change the behavior of a Collator. Options override the
// settings passed through the locale identifier.
type Option struct {
	priority int
	f        func(o *options)
}

type prioritizedOptions []Option

func (p prioritizedOptions) Len() int {
	return len(p)
}

func (p prioritizedOptions) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p prioritizedOptions) Less(i, j int) bool {
	return p[i].priority < p[j].priority
}

type options struct {
	// ignore specifies which levels to ignore.
	ignore [colltab.NumLevels]bool

	// caseLevel is true if there is an additional level of case matching
	// between the secondary and tertiary levels.
	caseLevel bool

	// backwards specifies the order of sorting at the secondary level.
	// This option exists predominantly to support reverse sorting of accents in French.
	backwards bool

	// numeric specifies whether any sequence of decimal digits (category is Nd)
	// is sorted at a primary level with its numeric value.
	// For example, "A-21" < "A-123".
	// This option is set by wrapping the main Weighter with NewNumericWeighter.
	numeric bool

	// alternate specifies an alternative handling of variables.
	alternate alternateHandling

	// variableTop is the largest primary value that is considered to be
	// variable.
	variableTop uint32

	t colltab.Weighter

	f norm.Form
}

func (o *options) setOptions(opts []Option) {
	sort.Sort(prioritizedOptions(opts))
	for _, x := range opts {
		x.f(o)
	}
}

// OptionsFromTag extracts the BCP47 collation options from the tag and
// configures a collator accordingly. These options are set before any other
// option.
func OptionsFromTag(t language.Tag) Option {
	return Option{0, func(o *options) {
		o.setFromTag(t)
	}}
}

func (o *options) setFromTag(t language.Tag) {
	o.caseLevel = ldmlBool(t, o.caseLevel, "kc")
	o.backwards = ldmlBool(t, o.backwards, "kb")
	o.numeric = ldmlBool(t, o.numeric, "kn")

	// Extract settings from the BCP47 u extension.
	switch t.TypeForKey("ks") { // strength
	case "level1":
		o.ignore[colltab.Secondary] = true
		o.ignore[colltab.Tertiary] = true
	case "level2":
		o.ignore[colltab.Tertiary] = true
	case "level3", "":
		// The default.
	case "level4":
		o.ignore[colltab.Quaternary] = false
	case "identic":
		o.ignore[colltab.Quaternary] = false
		o.ignore[colltab.Identity] = false
	}

	switch t.TypeForKey("ka") {
	case "shifted":
		o.alternate = altShifted
	// The following two types are not official BCP47, but we support them to
	// give access to this otherwise hidden functionality. The name blanked is
	// derived from the LDML name blanked and posix reflects the main use of
	// the shift-trimmed option.
	case "blanked":
		o.alternate = altBlanked
	case "posix":
		o.alternate = altShiftTrimmed
	}

	// TODO: caseFirst ("kf"), reorder ("kr"), and maybe variableTop ("vt").

	// Not used:
	// - normalization ("kk", not necessary for this implementation)
	// - hiraganaQuatenary ("kh", obsolete)
}

func ldmlBool(t language.Tag, old bool, key string) bool {
	switch t.TypeForKey(key) {
	case "true":
		return true
	case "false":
		return false
	default:
		return old
	}
}

var (
	// IgnoreCase sets case-insensitive comparison.
	IgnoreCase Option = ignoreCase
	ignoreCase        = Option{3, ignoreCaseF}

	// IgnoreDiacritics causes diacritical marks to be ignored. ("o" == "ö").
	IgnoreDiacritics Option = ignoreDiacritics
	ignoreDiacritics        = Option{3, ignoreDiacriticsF}

	// IgnoreWidth causes full-width characters to match their half-width
	// equivalents.
	IgnoreWidth Option = ignoreWidth
	ignoreWidth        = Option{2, ignoreWidthF}

	// Loose sets the collator to ignore diacritics, case and weight.
	Loose Option = loose
	loose        = Option{4, looseF}

	// Force ordering if strings are equivalent but not equal.
	Force Option = force
	force        = Option{5, forceF}

	// Numeric specifies that numbers should sort numerically ("2" < "12").
	Numeric Option = numeric
	numeric        = Option{5, numericF}
)

func ignoreWidthF(o *options) {
	o.ignore[colltab.Tertiary] = true
	o.caseLevel = true
}

func ignoreDiacriticsF(o *options) {
	o.ignore[colltab.Secondary] = true
}

func ignoreCaseF(o *options) {
	o.ignore[colltab.Tertiary] = true
	o.caseLevel = false
}

func looseF(o *options) {
	ignoreWidthF(o)
	ignoreDiacriticsF(o)
	ignoreCaseF(o)
}

func forceF(o *options) {
	o.ignore[colltab.Identity] = false
}

func numericF(o *options) { o.numeric = true }

// Reorder overrides the pre-defined ordering of scripts and character sets.
func Reorder(s ...string) Option {
	// TODO: need fractional weights to implement this.
	panic("TODO: implement")
}

// TODO: consider making these public again. These options cannot be fully
// specified in BCP47, so an API interface seems warranted. Still a higher-level
// interface would be nice (e.g. a POSIX option for enabling altShiftTrimmed)

// alternateHandling identifies the various ways in which variables are handled.
// A rune with a primary weight lower than the variable top is considered a
// variable.
// See http://www.unicode.org/reports/tr10/#Variable_Weighting for details.
type alternateHandling int

const (
	// altNonIgnorable turns off special handling of variables.
	altNonIgnorable alternateHandling = iota

	// altBlanked sets variables and all subsequent primary ignorables to be
	// ignorable at all levels. This is identical to removing all variables
	// and subsequent primary ignorables from the input.
	altBlanked

	// altShifted sets variables to be ignorable for levels one through three and
	// adds a fourth level based on the values of the ignored levels.
	altShifted

	// altShiftTrimmed is a slight variant of altShifted that is used to
	// emulate POSIX.
	altShiftTrimmed
)
