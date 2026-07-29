// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package format contains types for defining language-specific formatting of
// values.
//
// This package is internal now, but will eventually be exposed after the API
// settles.
package format // import "golang.org/x/text/internal/format"

import (
	"fmt"

	"golang.org/x/text/language"
)

// State represents the printer state passed to custom formatters. It provides
// access to the fmt.State interface and the sentence and language-related
// context.
type State interface {
	fmt.State

	// Language reports the requested language in which to render a message.
	Language() language.Tag

	// TODO: consider this and removing rune from the Format method in the
	// Formatter interface.
	//
	// Verb returns the format variant to render, analogous to the types used
	// in fmt. Use 'v' for the default or only variant.
	// Verb() rune

	// TODO: more info:
	// - sentence context such as linguistic features passed by the translator.
}

// Formatter is analogous to fmt.Formatter.
type Formatter interface {
	Format(state State, verb rune)
}
