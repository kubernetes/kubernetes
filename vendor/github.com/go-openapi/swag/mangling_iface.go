// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import "github.com/go-openapi/swag/mangling"

// GoNamePrefixFunc sets an optional rule to prefix go names
// which do not start with a letter.
//
// GoNamePrefixFunc should not be written to while concurrently using the other mangling functions of this package.
//
// Deprecated: use [mangling.WithGoNamePrefixFunc] instead.
var GoNamePrefixFunc mangling.PrefixFunc

// swagNameMangler is a global instance of the name mangler specifically alloted
// to support deprecated functions.
var swagNameMangler = mangling.NewNameMangler(
	mangling.WithGoNamePrefixFuncPtr(&GoNamePrefixFunc),
)

// AddInitialisms adds additional initialisms to the default list (see [mangling.DefaultInitialisms]).
//
// AddInitialisms is not safe to be called concurrently.
//
// Deprecated: use [mangling.WithAdditionalInitialisms] instead.
func AddInitialisms(words ...string) {
	swagNameMangler.AddInitialisms(words...)
}

// Camelize a single word.
//
// Deprecated: use [mangling.NameMangler.Camelize] instead.
func Camelize(word string) string { return swagNameMangler.Camelize(word) }

// ToFileName lowercases and underscores a go type name.
//
// Deprecated: use [mangling.NameMangler.ToFileName] instead.
func ToFileName(name string) string { return swagNameMangler.ToFileName(name) }

// ToCommandName lowercases and underscores a go type name.
//
// Deprecated: use [mangling.NameMangler.ToCommandName] instead.
func ToCommandName(name string) string { return swagNameMangler.ToCommandName(name) }

// ToHumanNameLower represents a code name as a human series of words.
//
// Deprecated: use [mangling.NameMangler.ToHumanNameLower] instead.
func ToHumanNameLower(name string) string { return swagNameMangler.ToHumanNameLower(name) }

// ToHumanNameTitle represents a code name as a human series of words with the first letters titleized.
//
// Deprecated: use [mangling.NameMangler.ToHumanNameTitle] instead.
func ToHumanNameTitle(name string) string { return swagNameMangler.ToHumanNameTitle(name) }

// ToJSONName camel-cases a name which can be underscored or pascal-cased.
//
// Deprecated: use [mangling.NameMangler.ToJSONName] instead.
func ToJSONName(name string) string { return swagNameMangler.ToJSONName(name) }

// ToVarName camel-cases a name which can be underscored or pascal-cased.
//
// Deprecated: use [mangling.NameMangler.ToVarName] instead.
func ToVarName(name string) string { return swagNameMangler.ToVarName(name) }

// ToGoName translates a swagger name which can be underscored or camel cased to a name that golint likes.
//
// Deprecated: use [mangling.NameMangler.ToGoName] instead.
func ToGoName(name string) string { return swagNameMangler.ToGoName(name) }
