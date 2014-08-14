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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Selector represents a label selector.
type Selector interface {
	// Matches returns true if this selector matches the given set of labels.
	Matches(Labels) bool

	// Empty returns true if this selector does not restrict the selection space.
	Empty() bool

	// RequiresExactMatch allows a caller to introspect whether a given selector
	// requires a single specific label to be set, and if so returns the value it
	// requires.
	// TODO: expand this to be more general
	RequiresExactMatch(label string) (value string, found bool)

	// String returns a human readable string that represents this selector.
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

func (t *hasTerm) Empty() bool {
	return false
}

func (t *hasTerm) RequiresExactMatch(label string) (value string, found bool) {
	if t.label == label {
		return t.value, true
	}
	return "", false
}

func (t *hasTerm) String() string {
	return fmt.Sprintf("%v=%v", t.label, t.value)
}

type notHasTerm struct {
	label, value string
}

func (t *notHasTerm) Matches(ls Labels) bool {
	return ls.Get(t.label) != t.value
}

func (t *notHasTerm) Empty() bool {
	return false
}

func (t *notHasTerm) RequiresExactMatch(label string) (value string, found bool) {
	return "", false
}

func (t *notHasTerm) String() string {
	return fmt.Sprintf("%v!=%v", t.label, t.value)
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

func (t andTerm) RequiresExactMatch(label string) (string, bool) {
	if t == nil || len([]Selector(t)) == 0 {
		return "", false
	}
	for i := range t {
		if value, found := t[i].RequiresExactMatch(label); found {
			return value, found
		}
	}
	return "", false
}

func (t andTerm) String() string {
	var terms []string
	for _, q := range t {
		terms = append(terms, q.String())
	}
	return strings.Join(terms, ",")
}

// Operator represents a key's relationship
// to a set of values in a Requirement.
// TODO: Should also represent key's existence.
type Operator int

const (
	IN Operator = iota + 1
	NOT_IN
)

// LabelSelector only not named 'Selector' due
// to name conflict until Selector is deprecated.
type LabelSelector struct {
	Requirements []Requirement
}

type Requirement struct {
	key       string
	operator  Operator
	strValues util.StringSet
}

func (r *Requirement) Matches(ls Labels) bool {
	switch r.operator {
	case IN:
		return r.strValues.Has(ls.Get(r.key))
	case NOT_IN:
		return !r.strValues.Has(ls.Get(r.key))
	default:
		return false
	}
}

func (sg *LabelSelector) Matches(ls Labels) bool {
	for _, req := range sg.Requirements {
		if !req.Matches(ls) {
			return false
		}
	}
	return true
}

func try(selectorPiece, op string) (lhs, rhs string, ok bool) {
	pieces := strings.Split(selectorPiece, op)
	if len(pieces) == 2 {
		return pieces[0], pieces[1], true
	}
	return "", "", false
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil Set is considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil {
		return Everything()
	}
	items := make([]Selector, 0, len(ls))
	for label, value := range ls {
		items = append(items, &hasTerm{label: label, value: value})
	}
	if len(items) == 1 {
		return items[0]
	}
	return andTerm(items)
}

// ParseSelector takes a string representing a selector and returns an
// object suitable for matching, or an error.
func ParseSelector(selector string) (Selector, error) {
	parts := strings.Split(selector, ",")
	sort.StringSlice(parts).Sort()
	var items []Selector
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, "!="); ok {
			items = append(items, &notHasTerm{label: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "=="); ok {
			items = append(items, &hasTerm{label: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "="); ok {
			items = append(items, &hasTerm{label: lhs, value: rhs})
		} else {
			return nil, fmt.Errorf("invalid selector: '%s'; can't understand '%s'", selector, part)
		}
	}
	if len(items) == 1 {
		return items[0], nil
	}
	return andTerm(items), nil
}
