/*
Copyright 2015 The Kubernetes Authors.

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

package fields

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/client-go/pkg/selection"
)

// Selector represents a field selector.
type Selector interface {
	// Matches returns true if this selector matches the given set of fields.
	Matches(Fields) bool

	// Empty returns true if this selector does not restrict the selection space.
	Empty() bool

	// RequiresExactMatch allows a caller to introspect whether a given selector
	// requires a single specific field to be set, and if so returns the value it
	// requires.
	RequiresExactMatch(field string) (value string, found bool)

	// Transform returns a new copy of the selector after TransformFunc has been
	// applied to the entire selector, or an error if fn returns an error.
	Transform(fn TransformFunc) (Selector, error)

	// Requirements converts this interface to Requirements to expose
	// more detailed selection information.
	Requirements() Requirements

	// String returns a human readable string that represents this selector.
	String() string
}

// Everything returns a selector that matches all fields.
func Everything() Selector {
	return andTerm{}
}

type hasTerm struct {
	field, value string
}

func (t *hasTerm) Matches(ls Fields) bool {
	return ls.Get(t.field) == t.value
}

func (t *hasTerm) Empty() bool {
	return false
}

func (t *hasTerm) RequiresExactMatch(field string) (value string, found bool) {
	if t.field == field {
		return t.value, true
	}
	return "", false
}

func (t *hasTerm) Transform(fn TransformFunc) (Selector, error) {
	field, value, err := fn(t.field, t.value)
	if err != nil {
		return nil, err
	}
	return &hasTerm{field, value}, nil
}

func (t *hasTerm) Requirements() Requirements {
	return []Requirement{{
		Field:    t.field,
		Operator: selection.Equals,
		Value:    t.value,
	}}
}

func (t *hasTerm) String() string {
	return fmt.Sprintf("%v=%v", t.field, t.value)
}

type notHasTerm struct {
	field, value string
}

func (t *notHasTerm) Matches(ls Fields) bool {
	return ls.Get(t.field) != t.value
}

func (t *notHasTerm) Empty() bool {
	return false
}

func (t *notHasTerm) RequiresExactMatch(field string) (value string, found bool) {
	return "", false
}

func (t *notHasTerm) Transform(fn TransformFunc) (Selector, error) {
	field, value, err := fn(t.field, t.value)
	if err != nil {
		return nil, err
	}
	return &notHasTerm{field, value}, nil
}

func (t *notHasTerm) Requirements() Requirements {
	return []Requirement{{
		Field:    t.field,
		Operator: selection.NotEquals,
		Value:    t.value,
	}}
}

func (t *notHasTerm) String() string {
	return fmt.Sprintf("%v!=%v", t.field, t.value)
}

type andTerm []Selector

func (t andTerm) Matches(ls Fields) bool {
	for _, q := range t {
		if !q.Matches(ls) {
			return false
		}
	}
	return true
}

func (t andTerm) Empty() bool {
	if t == nil {
		return true
	}
	if len([]Selector(t)) == 0 {
		return true
	}
	for i := range t {
		if !t[i].Empty() {
			return false
		}
	}
	return true
}

func (t andTerm) RequiresExactMatch(field string) (string, bool) {
	if t == nil || len([]Selector(t)) == 0 {
		return "", false
	}
	for i := range t {
		if value, found := t[i].RequiresExactMatch(field); found {
			return value, found
		}
	}
	return "", false
}

func (t andTerm) Transform(fn TransformFunc) (Selector, error) {
	next := make([]Selector, len([]Selector(t)))
	for i, s := range []Selector(t) {
		n, err := s.Transform(fn)
		if err != nil {
			return nil, err
		}
		next[i] = n
	}
	return andTerm(next), nil
}

func (t andTerm) Requirements() Requirements {
	reqs := make([]Requirement, 0, len(t))
	for _, s := range []Selector(t) {
		rs := s.Requirements()
		reqs = append(reqs, rs...)
	}
	return reqs
}

func (t andTerm) String() string {
	var terms []string
	for _, q := range t {
		terms = append(terms, q.String())
	}
	return strings.Join(terms, ",")
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil Set is considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil {
		return Everything()
	}
	items := make([]Selector, 0, len(ls))
	for field, value := range ls {
		items = append(items, &hasTerm{field: field, value: value})
	}
	if len(items) == 1 {
		return items[0]
	}
	return andTerm(items)
}

// ParseSelectorOrDie takes a string representing a selector and returns an
// object suitable for matching, or panic when an error occur.
func ParseSelectorOrDie(s string) Selector {
	selector, err := ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

// ParseSelector takes a string representing a selector and returns an
// object suitable for matching, or an error.
func ParseSelector(selector string) (Selector, error) {
	return parseSelector(selector,
		func(lhs, rhs string) (newLhs, newRhs string, err error) {
			return lhs, rhs, nil
		})
}

// Parses the selector and runs them through the given TransformFunc.
func ParseAndTransformSelector(selector string, fn TransformFunc) (Selector, error) {
	return parseSelector(selector, fn)
}

// Function to transform selectors.
type TransformFunc func(field, value string) (newField, newValue string, err error)

func try(selectorPiece, op string) (lhs, rhs string, ok bool) {
	pieces := strings.Split(selectorPiece, op)
	if len(pieces) == 2 {
		return pieces[0], pieces[1], true
	}
	return "", "", false
}

func parseSelector(selector string, fn TransformFunc) (Selector, error) {
	parts := strings.Split(selector, ",")
	sort.StringSlice(parts).Sort()
	var items []Selector
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, "!="); ok {
			items = append(items, &notHasTerm{field: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "=="); ok {
			items = append(items, &hasTerm{field: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "="); ok {
			items = append(items, &hasTerm{field: lhs, value: rhs})
		} else {
			return nil, fmt.Errorf("invalid selector: '%s'; can't understand '%s'", selector, part)
		}
	}
	if len(items) == 1 {
		return items[0].Transform(fn)
	}
	return andTerm(items).Transform(fn)
}

// OneTermEqualSelector returns an object that matches objects where one field/field equals one value.
// Cannot return an error.
func OneTermEqualSelector(k, v string) Selector {
	return &hasTerm{field: k, value: v}
}
