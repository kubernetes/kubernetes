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
	// ValidACIdentifier is a regular expression that defines a valid ACIdentifier
	ValidACIdentifier = regexp.MustCompile("^[a-z0-9]+([-._~/][a-z0-9]+)*$")

	invalidACIdentifierChars = regexp.MustCompile("[^a-z0-9-._~/]")
	invalidACIdentifierEdges = regexp.MustCompile("(^[-._~/]+)|([-._~/]+$)")

	ErrEmptyACIdentifier         = ACIdentifierError("ACIdentifier cannot be empty")
	ErrInvalidEdgeInACIdentifier = ACIdentifierError("ACIdentifier must start and end with only lower case " +
		"alphanumeric characters")
	ErrInvalidCharInACIdentifier = ACIdentifierError("ACIdentifier must contain only lower case " +
		`alphanumeric characters plus "-._~/"`)
)

// ACIdentifier (an App-Container Identifier) is a format used by keys in image names
// and image labels of the App Container Standard. An ACIdentifier is restricted to numeric
// and lowercase URI unreserved characters defined in URI RFC[1]; all alphabetical characters
// must be lowercase only. Furthermore, the first and last character ("edges") must be
// alphanumeric, and an ACIdentifier cannot be empty. Programmatically, an ACIdentifier must
// conform to the regular expression ValidACIdentifier.
//
// [1] http://tools.ietf.org/html/rfc3986#section-2.3
type ACIdentifier string

func (n ACIdentifier) String() string {
	return string(n)
}

// Set sets the ACIdentifier to the given value, if it is valid; if not,
// an error is returned.
func (n *ACIdentifier) Set(s string) error {
	nn, err := NewACIdentifier(s)
	if err == nil {
		*n = *nn
	}
	return err
}

// Equals checks whether a given ACIdentifier is equal to this one.
func (n ACIdentifier) Equals(o ACIdentifier) bool {
	return strings.ToLower(string(n)) == strings.ToLower(string(o))
}

// Empty returns a boolean indicating whether this ACIdentifier is empty.
func (n ACIdentifier) Empty() bool {
	return n.String() == ""
}

// NewACIdentifier generates a new ACIdentifier from a string. If the given string is
// not a valid ACIdentifier, nil and an error are returned.
func NewACIdentifier(s string) (*ACIdentifier, error) {
	n := ACIdentifier(s)
	if err := n.assertValid(); err != nil {
		return nil, err
	}
	return &n, nil
}

// MustACIdentifier generates a new ACIdentifier from a string, If the given string is
// not a valid ACIdentifier, it panics.
func MustACIdentifier(s string) *ACIdentifier {
	n, err := NewACIdentifier(s)
	if err != nil {
		panic(err)
	}
	return n
}

func (n ACIdentifier) assertValid() error {
	s := string(n)
	if len(s) == 0 {
		return ErrEmptyACIdentifier
	}
	if invalidACIdentifierChars.MatchString(s) {
		return ErrInvalidCharInACIdentifier
	}
	if invalidACIdentifierEdges.MatchString(s) {
		return ErrInvalidEdgeInACIdentifier
	}
	return nil
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (n *ACIdentifier) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	nn, err := NewACIdentifier(s)
	if err != nil {
		return err
	}
	*n = *nn
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (n ACIdentifier) MarshalJSON() ([]byte, error) {
	if err := n.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(n.String())
}

// SanitizeACIdentifier replaces every invalid ACIdentifier character in s with an underscore
// making it a legal ACIdentifier string. If the character is an upper case letter it
// replaces it with its lower case. It also removes illegal edge characters
// (hyphens, period, underscore, tilde and slash).
//
// This is a helper function and its algorithm is not part of the spec. It
// should not be called without the user explicitly asking for a suggestion.
func SanitizeACIdentifier(s string) (string, error) {
	s = strings.ToLower(s)
	s = invalidACIdentifierChars.ReplaceAllString(s, "_")
	s = invalidACIdentifierEdges.ReplaceAllString(s, "")

	if s == "" {
		return "", errors.New("must contain at least one valid character")
	}

	return s, nil
}
