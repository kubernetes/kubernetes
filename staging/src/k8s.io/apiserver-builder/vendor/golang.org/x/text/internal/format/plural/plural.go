// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package plural defines the grammatical plural feature.
//
// The definitions in this package are based on the plural rule handling defined
// in CLDR. See
// http://unicode.org/reports/tr35/tr35-numbers.html#Language_Plural_Rules for
// details.
package plural

import "golang.org/x/text/internal/format"

// Form defines a plural form. The meaning of plural forms, as well as which
// forms are supported, vary per language. Each language must at least support
// the form "other".
type Form byte

const (
	Other Form = iota
	Zero
	One
	Two
	Few
	Many
)

// Interface is implemented by values that have a plural feature.
type Interface interface {
	// PluralForm reports the plural form of a value, depending on the
	// language declared by the given state.
	PluralForm(s format.State) Form
}

// TODO
// - Select function
// - Definition for message package.
