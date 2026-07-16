// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package common defines types and utilities common to expression parsing,
// checking, and interpretation
package common

import (
	"strings"
	"unicode"
)

// DocKind indicates the type of documentation element.
type DocKind int

const (
	// DocEnv represents environment variable documentation.
	DocEnv DocKind = iota + 1
	// DocFunction represents function documentation.
	DocFunction
	// DocOverload represents function overload documentation.
	DocOverload
	// DocVariable represents variable documentation.
	DocVariable
	// DocMacro represents macro documentation.
	DocMacro
	// DocExample represents example documentation.
	DocExample
)

// Doc holds the documentation details for a specific program element like
// a variable, function, macro, or example.
type Doc struct {
	// Kind specifies the type of documentation element (e.g., Function, Variable).
	Kind DocKind

	// Name is the identifier of the documented element (e.g., function name, variable name).
	Name string

	// Type is the data type associated with the element, primarily used for variables.
	Type string

	// Signature represents the function or overload signature.
	Signature string

	// Description holds the textual description of the element, potentially spanning multiple lines.
	Description string

	// Children holds nested documentation elements, such as overloads for a function
	// or examples for a function/macro.
	Children []*Doc
}

// MultilineDescription combines multiple lines into a newline separated string.
func MultilineDescription(lines ...string) string {
	return strings.Join(lines, "\n")
}

// ParseDescription takes a single string containing newline characters and splits
// it into a multiline description. All empty lines will be skipped.
//
// Returns an empty string if the input string is empty.
func ParseDescription(doc string) string {
	var lines []string
	if len(doc) != 0 {
		// Split the input string by newline characters.
		for _, line := range strings.Split(doc, "\n") {
			l := strings.TrimRightFunc(line, unicode.IsSpace)
			if len(l) == 0 {
				continue
			}
			lines = append(lines, l)
		}
	}
	// Return an empty slice if the input is empty.
	return MultilineDescription(lines...)
}

// ParseDescriptions splits a documentation string into multiple multi-line description
// sections, using blank lines as delimiters.
func ParseDescriptions(doc string) []string {
	var examples []string
	if len(doc) != 0 {
		lines := strings.Split(doc, "\n")
		lineStart := 0
		for i, l := range lines {
			// Trim trailing whitespace to identify effectively blank lines.
			l = strings.TrimRightFunc(l, unicode.IsSpace)
			// If a line is blank, it marks the end of the current section.
			if len(l) == 0 {
				// Start the next section after the blank line.
				ex := lines[lineStart:i]
				if len(ex) != 0 {
					examples = append(examples, MultilineDescription(ex...))
				}
				lineStart = i + 1
			}
		}
		// Append the last section if it wasn't terminated by a blank line.
		if lineStart < len(lines) {
			examples = append(examples, MultilineDescription(lines[lineStart:]...))
		}
	}
	return examples
}

// NewVariableDoc creates a new Doc struct specifically for documenting a variable.
func NewVariableDoc(name, celType, description string) *Doc {
	return &Doc{
		Kind:        DocVariable,
		Name:        name,
		Type:        celType,
		Description: ParseDescription(description),
	}
}

// NewFunctionDoc creates a new Doc struct for documenting a function.
func NewFunctionDoc(name, description string, overloads ...*Doc) *Doc {
	return &Doc{
		Kind:        DocFunction,
		Name:        name,
		Description: ParseDescription(description),
		Children:    overloads,
	}
}

// NewOverloadDoc creates a new Doc struct for a function example.
func NewOverloadDoc(id, signature string, examples ...*Doc) *Doc {
	return &Doc{
		Kind:      DocOverload,
		Name:      id,
		Signature: signature,
		Children:  examples,
	}
}

// NewMacroDoc creates a new Doc struct for documenting a macro.
func NewMacroDoc(name, description string, examples ...*Doc) *Doc {
	return &Doc{
		Kind:        DocMacro,
		Name:        name,
		Description: ParseDescription(description),
		Children:    examples,
	}
}

// NewExampleDoc creates a new Doc struct specifically for holding an example.
func NewExampleDoc(ex string) *Doc {
	return &Doc{
		Kind:        DocExample,
		Description: ex,
	}
}

// Documentor is an interface for types that can provide their own documentation.
type Documentor interface {
	// Documentation returns the documentation coded by the DocKind to assist
	// with text formatting.
	Documentation() *Doc
}
