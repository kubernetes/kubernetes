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

	// TODO: more info:
	// - sentence context
	// - user preferences, like measurement systems
	// - options
}

// A Statement is a Var or an Expression.
type Statement interface {
	statement()
}

// A String a literal string format.
type String string

func (String) statement() {}

// TODO: Select, Var, Case, StatementSequence
