// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
)

var (
	// ValidACName is a regular expression that defines a valid ACName
	ValidACName = regexp.MustCompile("^[a-z0-9]+([-][a-z0-9]+)*$")

	invalidACNameChars = regexp.MustCompile("[^a-z0-9-]")
	invalidACNameEdges = regexp.MustCompile("(^[-]+)|([-]+$)")

	ErrEmptyACName         = ACNameError("ACName cannot be empty")
	ErrInvalidEdgeInACName = ACNameError("ACName must start and end with only lower case " +
		"alphanumeric characters")
	ErrInvalidCharInACName = ACNameError("ACName must contain only lower case " +
		`alphanumeric characters plus "-"`)
)

// ACName (an App-Container Name) is a format used by keys in different formats
// of the App Container Standard. An ACName is restricted to numeric and lowercase
// characters accepted by the DNS RFC[1] plus "-"; all alphabetical characters must
// be lowercase only. Furthermore, the first and last character ("edges") must be
// alphanumeric, and an ACName cannot be empty. Programmatically, an ACName must
// conform to the regular expression ValidACName.
//
// [1] http://tools.ietf.org/html/rfc1123#page-13
type ACName string

func (n ACName) String() string {
	return string(n)
}

// Set sets the ACName to the given value, if it is valid; if not,
// an error is returned.
func (n *ACName) Set(s string) error {
	nn, err := NewACName(s)
	if err == nil {
		*n = *nn
	}
	return err
}

// Equals checks whether a given ACName is equal to this one.
func (n ACName) Equals(o ACName) bool {
	return strings.ToLower(string(n)) == strings.ToLower(string(o))
}

// Empty returns a boolean indicating whether this ACName is empty.
func (n ACName) Empty() bool {
	return n.String() == ""
}

// NewACName generates a new ACName from a string. If the given string is
// not a valid ACName, nil and an error are returned.
func NewACName(s string) (*ACName, error) {
	n := ACName(s)
	if err := n.assertValid(); err != nil {
		return nil, err
	}
	return &n, nil
}

// MustACName generates a new ACName from a string, If the given string is
// not a valid ACName, it panics.
func MustACName(s string) *ACName {
	n, err := NewACName(s)
	if err != nil {
		panic(err)
	}
	return n
}

func (n ACName) assertValid() error {
	s := string(n)
	if len(s) == 0 {
		return ErrEmptyACName
	}
	if invalidACNameChars.MatchString(s) {
		return ErrInvalidCharInACName
	}
	if invalidACNameEdges.MatchString(s) {
		return ErrInvalidEdgeInACName
	}
	return nil
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (n *ACName) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	nn, err := NewACName(s)
	if err != nil {
		return err
	}
	*n = *nn
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (n ACName) MarshalJSON() ([]byte, error) {
	if err := n.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(n.String())
}

// SanitizeACName replaces every invalid ACName character in s with a dash
// making it a legal ACName string. If the character is an upper case letter it
// replaces it with its lower case. It also removes illegal edge characters
// (hyphens).
//
// This is a helper function and its algorithm is not part of the spec. It
// should not be called without the user explicitly asking for a suggestion.
func SanitizeACName(s string) (string, error) {
	s = strings.ToLower(s)
	s = invalidACNameChars.ReplaceAllString(s, "-")
	s = invalidACNameEdges.ReplaceAllString(s, "")

	if s == "" {
		return "", errors.New("must contain at least one valid character")
	}

	return s, nil
}
