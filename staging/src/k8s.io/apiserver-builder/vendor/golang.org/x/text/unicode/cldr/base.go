// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cldr provides a parser for LDML and related XML formats.
// This package is inteded to be used by the table generation tools
// for the various internationalization-related packages.
// As the XML types are generated from the CLDR DTD, and as the CLDR standard
// is periodically amended, this package may change considerably over time.
// This mostly means that data may appear and disappear between versions.
// That is, old code should keep compiling for newer versions, but data
// may have moved or changed.
// CLDR version 22 is the first version supported by this package.
// Older versions may not work.
package cldr

import (
	"encoding/xml"
	"regexp"
	"strconv"
)

// Elem is implemented by every XML element.
type Elem interface {
	setEnclosing(Elem)
	setName(string)
	enclosing() Elem

	GetCommon() *Common
}

type hidden struct {
	CharData string `xml:",chardata"`
	Alias    *struct {
		Common
		Source string `xml:"source,attr"`
		Path   string `xml:"path,attr"`
	} `xml:"alias"`
	Def *struct {
		Common
		Choice string `xml:"choice,attr,omitempty"`
		Type   string `xml:"type,attr,omitempty"`
	} `xml:"default"`
}

// Common holds several of the most common attributes and sub elements
// of an XML element.
type Common struct {
	XMLName         xml.Name
	name            string
	enclElem        Elem
	Type            string `xml:"type,attr,omitempty"`
	Reference       string `xml:"reference,attr,omitempty"`
	Alt             string `xml:"alt,attr,omitempty"`
	ValidSubLocales string `xml:"validSubLocales,attr,omitempty"`
	Draft           string `xml:"draft,attr,omitempty"`
	hidden
}

// Default returns the default type to select from the enclosed list
// or "" if no default value is specified.
func (e *Common) Default() string {
	if e.Def == nil {
		return ""
	}
	if e.Def.Choice != "" {
		return e.Def.Choice
	} else if e.Def.Type != "" {
		// Type is still used by the default element in collation.
		return e.Def.Type
	}
	return ""
}

// GetCommon returns e. It is provided such that Common implements Elem.
func (e *Common) GetCommon() *Common {
	return e
}

// Data returns the character data accumulated for this element.
func (e *Common) Data() string {
	e.CharData = charRe.ReplaceAllStringFunc(e.CharData, replaceUnicode)
	return e.CharData
}

func (e *Common) setName(s string) {
	e.name = s
}

func (e *Common) enclosing() Elem {
	return e.enclElem
}

func (e *Common) setEnclosing(en Elem) {
	e.enclElem = en
}

// Escape characters that can be escaped without further escaping the string.
var charRe = regexp.MustCompile(`&#x[0-9a-fA-F]*;|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2}|\\[0-7]{3}|\\[abtnvfr]`)

// replaceUnicode converts hexadecimal Unicode codepoint notations to a one-rune string.
// It assumes the input string is correctly formatted.
func replaceUnicode(s string) string {
	if s[1] == '#' {
		r, _ := strconv.ParseInt(s[3:len(s)-1], 16, 32)
		return string(r)
	}
	r, _, _, _ := strconv.UnquoteChar(s, 0)
	return string(r)
}
