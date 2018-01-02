// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go

// Package htmlindex maps character set encoding names to Encodings as
// recommended by the W3C for use in HTML 5. See http://www.w3.org/TR/encoding.
package htmlindex

// TODO: perhaps have a "bare" version of the index (used by this package) that
// is not pre-loaded with all encodings. Global variables in encodings prevent
// the linker from being able to purge unneeded tables. This means that
// referencing all encodings, as this package does for the default index, links
// in all encodings unconditionally.
//
// This issue can be solved by either solving the linking issue (see
// https://github.com/golang/go/issues/6330) or refactoring the encoding tables
// (e.g. moving the tables to internal packages that do not use global
// variables).

// TODO: allow canonicalizing names

import (
	"errors"
	"strings"
	"sync"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/language"
)

var (
	errInvalidName = errors.New("htmlindex: invalid encoding name")
	errUnknown     = errors.New("htmlindex: unknown Encoding")
	errUnsupported = errors.New("htmlindex: this encoding is not supported")
)

var (
	matcherOnce sync.Once
	matcher     language.Matcher
)

// LanguageDefault returns the canonical name of the default encoding for a
// given language.
func LanguageDefault(tag language.Tag) string {
	matcherOnce.Do(func() {
		tags := []language.Tag{}
		for _, t := range strings.Split(locales, " ") {
			tags = append(tags, language.MustParse(t))
		}
		matcher = language.NewMatcher(tags)
	})
	_, i, _ := matcher.Match(tag)
	return canonical[localeMap[i]] // Default is Windows-1252.
}

// Get returns an Encoding for one of the names listed in
// http://www.w3.org/TR/encoding using the Default Index. Matching is case-
// insensitive.
func Get(name string) (encoding.Encoding, error) {
	x, ok := nameMap[strings.ToLower(strings.TrimSpace(name))]
	if !ok {
		return nil, errInvalidName
	}
	return encodings[x], nil
}

// Name reports the canonical name of the given Encoding. It will return
// an error if e is not associated with a supported encoding scheme.
func Name(e encoding.Encoding) (string, error) {
	id, ok := e.(identifier.Interface)
	if !ok {
		return "", errUnknown
	}
	mib, _ := id.ID()
	if mib == 0 {
		return "", errUnknown
	}
	v, ok := mibMap[mib]
	if !ok {
		return "", errUnsupported
	}
	return canonical[v], nil
}
