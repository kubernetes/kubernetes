/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/util/selectors"
	"k8s.io/kubernetes/pkg/util/sets"
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

	// String returns a human readable string that represents this selector.
	String() string

	// Add adds requirements to the Selector
	Add(r ...selectors.Requirement) Selector
}

// Everything returns a selector that matches all labels.
func Everything() Selector {
	return internalSelector{}
}

type nothingSelector struct{}

func (n nothingSelector) Matches(_ Fields) bool                                      { return false }
func (n nothingSelector) Empty() bool                                                { return false }
func (n nothingSelector) String() string                                             { return "<null>" }
func (n nothingSelector) Transform(fn TransformFunc) (Selector, error)               { return nothingSelector{}, nil }
func (n nothingSelector) Add(_ ...selectors.Requirement) Selector                    { return n }
func (n nothingSelector) RequiresExactMatch(field string) (value string, found bool) { return "", false }

// Nothing returns a selector that matches no labels
func Nothing() Selector {
	return nothingSelector{}
}

func NewSelector() Selector {
	return internalSelector(nil)
}

type internalSelector []selectors.Requirement

// NewRequirement is the constructor for a Requirement.
// TODO: determine if we need any additional validation over selectors.NewRequirement unique to fields
//
// The empty string is a valid value in the input values set.
func NewRequirement(key string, op selectors.Operator, vals sets.String) (*selectors.Requirement, error) {
	return selectors.NewRequirement(key, op, vals)
}

// Return true if the internalSelector doesn't restrict selection space
func (lsel internalSelector) Empty() bool {
	if lsel == nil {
		return true
	}
	return len(lsel) == 0
}

// Add adds requirements to the selector. It copies the current selector returning a new one
func (lsel internalSelector) Add(reqs ...selectors.Requirement) Selector {
	var sel internalSelector
	for ix := range lsel {
		sel = append(sel, lsel[ix])
	}
	for _, r := range reqs {
		sel = append(sel, r)
	}
	sort.Sort(selectors.ByKey(sel))
	return sel
}

// Matches for a internalSelector returns true if all
// its Requirements match the input Fields. If any
// Requirement does not match, false is returned.
func (lsel internalSelector) Matches(l Fields) bool {
	for ix := range lsel {
		if matches := lsel[ix].Matches(l); !matches {
			return false
		}
	}
	return true
}

// String returns a comma-separated string of all
// the internalSelector Requirements' human-readable strings.
func (lsel internalSelector) String() string {
	var reqs []string
	for ix := range lsel {
		reqs = append(reqs, lsel[ix].String())
	}
	return strings.Join(reqs, ",")
}

// RequiresExactMatch looks over all internal requirements to determine if the field
// matches on a requirement key that accepts only 1 valid value where operator is IN, ==, =
func (lsel internalSelector) RequiresExactMatch(field string) (value string, found bool) {
	for ix := range lsel {
		if lsel[ix].Key() == field && len(lsel[ix].Values()) == 1 {
			switch lsel[ix].Operator() {
			case selectors.DoubleEqualsOperator,
				selectors.EqualsOperator,
				selectors.InOperator:
				return lsel[ix].Values().List()[0], true
			}
		}
	}
	return "", false
}

func (lsel internalSelector) Transform(fn TransformFunc) (Selector, error) {
	result := NewSelector()
	var (
		err         error
		newKey      string
		newValue    string
		requirement *selectors.Requirement
	)

	for ix := range lsel {
		key := lsel[ix].Key()
		values := lsel[ix].Values()

		newValues := sets.NewString()
		if len(values) == 0 {
			newKey, _, err = fn(key, "")
			if err != nil {
				return nil, err
			}
		} else {
			for _, value := range values.List() {
				newKey, newValue, err = fn(key, value)
				if err != nil {
					return nil, err
				}
				newValues.Insert(newValue)
			}
		}
		requirement, err = NewRequirement(newKey, lsel[ix].Operator(), newValues)
		if err != nil {
			return nil, err
		}
		result = result.Add(*requirement)
	}
	return result, nil
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil and empty Sets are considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil {
		return internalSelector{}
	}
	var requirements internalSelector
	for label, value := range ls {
		if r, err := NewRequirement(label, selectors.EqualsOperator, sets.NewString(value)); err != nil {
			//TODO: double check errors when input comes from serialization?
			return internalSelector{}
		} else {
			requirements = append(requirements, *r)
		}
	}
	// sort to have deterministic string representation
	sort.Sort(selectors.ByKey(requirements))
	return internalSelector(requirements)
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

	result := NewSelector()
	for _, part := range parts {
		if part == "" {
			continue
		}
		var (
			requirement *selectors.Requirement
			err         error
		)
		if lhs, rhs, ok := try(part, "!="); ok {
			requirement, err = NewRequirement(lhs, selectors.NotEqualsOperator, sets.NewString(rhs))
		} else if lhs, rhs, ok := try(part, "=="); ok {
			requirement, err = NewRequirement(lhs, selectors.DoubleEqualsOperator, sets.NewString(rhs))
		} else if lhs, rhs, ok := try(part, "="); ok {
			requirement, err = NewRequirement(lhs, selectors.EqualsOperator, sets.NewString(rhs))
		} else {
			return nil, fmt.Errorf("invalid selector: '%s'; can't understand '%s'", selector, part)
		}

		if err != nil {
			return nil, err
		}

		result = result.Add(*requirement)
	}
	return result.Transform(fn)
}

// OneTermEqualSelector returns an object that matches objects where one field/field equals one value.
// Cannot return an error.
func OneTermEqualSelector(k, v string) Selector {
	result := NewSelector()
	requirement, err := NewRequirement(k, selectors.EqualsOperator, sets.NewString(v))
	if err != nil {
		panic(err)
	}
	result = result.Add(*requirement)
	return result
}
