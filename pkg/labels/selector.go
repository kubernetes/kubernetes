/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package labels

import (
	"fmt"
	"sort"
	"strings"
)

const (
	SelectorSep = ","
	NotEq       = "!="
	SingleEq    = "="
	DoubleEq    = "=="
	WildCard    = "*"
	SetPref     = "<<"
	SetSuff     = ">>"
	SetSep      = "$"
)

// Represents a selector.
type Selector interface {
	// Returns true if this selector matches the given set of labels.
	Matches(Labels) bool

	// Prints a human readable version of this selector.
	String() string
}

// Everything returns a selector that matches all labels.
func Everything() Selector {
	return andTerm{}
}

type hasTerm struct {
	label, value string
}

func (t *hasTerm) Matches(ls Labels) bool {
	return ls.Get(t.label) == t.value
}

func (t *hasTerm) String() string {
	return t.label + SingleEq + t.value
}

type notHasTerm struct {
	label, value string
}

func (t *notHasTerm) Matches(ls Labels) bool {
	return ls.Get(t.label) != t.value
}

func (t *notHasTerm) String() string {
	return t.label + NotEq + t.value
}

type labelExists struct {
	label string
}

// There is an ambiguity here. It is possible
// to have an empty string as a label's value.
// Label's Get method only returns a single string
// value and no boolean. Therefore, the case of
// a key not existing is generalized to it having
// an empty string value in addition to it not
// being present at all.
func (t *labelExists) Matches(ls Labels) bool {
	return ls.Get(t.label) != ""
}

func (t *labelExists) String() string {
	return t.label + SingleEq + WildCard
}

type labelIsIn struct {
	label  string
	values map[string]bool
}

func (t *labelIsIn) Matches(ls Labels) bool {
	return t.values[ls.Get(t.label)]
}

func (t *labelIsIn) String() string {
	v := make([]string, 0, len(t.values))
	for val := range t.values {
		v = append(v, val)
	}
	// Sort for determinism.
	sort.Strings(v)
	return t.label + SingleEq + SetPref + strings.Join(v, SetSep) + SetSuff
}

type labelIsNotIn struct {
	label  string
	values map[string]bool
}

func (t *labelIsNotIn) Matches(ls Labels) bool {
	return !t.values[ls.Get(t.label)]
}

func (t *labelIsNotIn) String() string {
	v := make([]string, 0, len(t.values))
	for val := range t.values {
		v = append(v, val)
	}
	// Sort for determinism.
	sort.Strings(v)
	return t.label + NotEq + SetPref + strings.Join(v, SetSep) + SetSuff
}

type andTerm []Selector

func (t andTerm) Matches(ls Labels) bool {
	for _, q := range t {
		if !q.Matches(ls) {
			return false
		}
	}
	return true
}

func (t andTerm) String() string {
	var terms []string
	for _, q := range t {
		terms = append(terms, q.String())
	}
	return strings.Join(terms, SelectorSep)
}

func try(selectorPiece, op string) (lhs, rhs string, ok bool) {
	pieces := strings.Split(selectorPiece, op)
	if len(pieces) == 2 {
		return pieces[0], pieces[1], true
	}
	return "", "", false
}

// Parses string of values into set
func parseValueSet(valuesSetEncoded, op string) map[string]bool {
	valueSet := make(map[string]bool)
	values := strings.Split(valuesSetEncoded, op)
	for _, value := range values {
		valueSet[value] = true
	}
	return valueSet
}

// Determines if string of values has
// empty string as one of its values
func hasSetEncodedEmptyValues(setEncoded string) bool {
	return setEncoded == SetPref+SetSuff ||
		strings.Contains(setEncoded, SetSep+SetSep) ||
		strings.HasPrefix(setEncoded, SetPref+SetSep) ||
		strings.HasSuffix(setEncoded, SetSep+SetSuff)
}

func parseLabelValue(lhs, rhs string, isEqual bool) (Selector, error) {
	if rhs == WildCard {
		return &labelExists{label: lhs}, nil
	} else if strings.HasPrefix(rhs, SetPref) && strings.HasSuffix(rhs, SetSuff) {
		if hasSetEncodedEmptyValues(rhs) {
			return nil, fmt.Errorf("invalid value set: '%s'; can't have empty string as value", rhs)
		}
		valueSet := parseValueSet(rhs[len(SetPref):len(rhs)-len(SetSuff)], SetSep)
		if isEqual {
			return &labelIsIn{label: lhs, values: valueSet}, nil
		} else {
			return &labelIsNotIn{label: lhs, values: valueSet}, nil
		}
	} else {
		if isEqual {
			return &hasTerm{label: lhs, value: rhs}, nil
		} else {
			return &notHasTerm{label: lhs, value: rhs}, nil
		}
	}
}

// Given a Set, return a Selector which will match exactly that Set.
func SelectorFromSet(ls Set) Selector {
	items := make([]Selector, 0, len(ls))
	for label, value := range ls {
		items = append(items, &hasTerm{label: label, value: value})
	}
	if len(items) == 1 {
		return items[0]
	}
	return andTerm(items)
}

// Takes a string repsenting a selector and returns an object suitable for matching, or an error.
// Allowed formats: (1) x!=foo
//                  (2) x=foo
//                  (3) x=*
//                  (4) x=<<foo>> or x!=<<foo>>
//                  (5) x=<<foo$bar>> or x!=<<foo$bar>> or any number of '$' separated values
func ParseSelector(selector string) (Selector, error) {
	parts := strings.Split(selector, SelectorSep)
	sort.StringSlice(parts).Sort()
	var items []Selector
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, NotEq); ok {
			if rhs == WildCard {
				return nil, fmt.Errorf("invalid selector: '%s'; can't check non-existence", selector)
			}
			sel, err := parseLabelValue(lhs, rhs, false)
			if err != nil {
				return nil, err
			}
			items = append(items, sel)
		} else if lhs, rhs, ok := try(part, DoubleEq); ok {
			sel, err := parseLabelValue(lhs, rhs, true)
			if err != nil {
				return nil, err
			}
			items = append(items, sel)
		} else if lhs, rhs, ok := try(part, SingleEq); ok {
			sel, err := parseLabelValue(lhs, rhs, true)
			if err != nil {
				return nil, err
			}
			items = append(items, sel)
		} else {
			return nil, fmt.Errorf("invalid selector: '%s'; can't understand '%s'", selector, part)
		}
	}
	if len(items) == 1 {
		return items[0], nil
	}
	return andTerm(items), nil
}
