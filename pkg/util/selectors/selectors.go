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

package selectors

import (
	"bytes"
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/sets"
)

// KeyValues allows you to present keys independently from their storage
type KeyValues interface {
	// Has returns whether the provided key exists.
	Has(key string) (exists bool)

	// Get returns the value for the provided key.
	Get(key string) (value string)
}

// Operator represents a key's relationship
// to a set of values in a Requirement.
type Operator string

const (
	DoesNotExistOperator Operator = "!"
	EqualsOperator       Operator = "="
	DoubleEqualsOperator Operator = "=="
	InOperator           Operator = "in"
	NotEqualsOperator    Operator = "!="
	NotInOperator        Operator = "notin"
	ExistsOperator       Operator = "exists"
)

// Requirement is a selector that contains values, a key
// and an operator that relates the key and values. The zero
// value of Requirement is invalid.
// Requirement implements both set based match and exact match
// Requirement is initialized via NewRequirement constructor for creating a valid Requirement.
type Requirement struct {
	key       string
	operator  Operator
	strValues sets.String
}

// Sort by key to obtain deterministic parser
type ByKey []Requirement

func (a ByKey) Len() int { return len(a) }

func (a ByKey) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func (a ByKey) Less(i, j int) bool { return a[i].key < a[j].key }

// NewRequirement is the constructor for a Requirement.
// If any of these rules is violated, an error is returned:
// (1) The operator can only be In, NotIn, Equals, DoubleEquals, NotEquals, Exists, or DoesNotExist.
// (2) If the operator is In or NotIn, the values set must be non-empty.
// (3) If the operator is Equals, DoubleEquals, or NotEquals, the values set must contain one value.
// (4) If the operator is Exists or DoesNotExist, the value set must be empty.
//
// The empty string is a valid value in the input values set.
func NewRequirement(key string, op Operator, vals sets.String) (*Requirement, error) {
	switch op {
	case InOperator, NotInOperator:
		if len(vals) == 0 {
			return nil, fmt.Errorf("for 'in', 'notin' operators, values set can't be empty")
		}
	case EqualsOperator, DoubleEqualsOperator, NotEqualsOperator:
		if len(vals) != 1 {
			return nil, fmt.Errorf("exact-match compatibility requires one single value")
		}
	case ExistsOperator, DoesNotExistOperator:
		if len(vals) != 0 {
			return nil, fmt.Errorf("values set must be empty for exists and does not exist")
		}
	default:
		return nil, fmt.Errorf("operator '%v' is not recognized", op)
	}
	return &Requirement{key: key, operator: op, strValues: vals}, nil
}

// Matches returns true if the Requirement matches the input KeyValues.
// There is a match in the following cases:
// (1) The operator is Exists and KeyValues has the Requirement's key.
// (2) The operator is In, KeyValues has the Requirement's key and KeyValues'
//     value for that key is in Requirement's value set.
// (3) The operator is NotIn, Labels has the Requirement's key and
//     KeyValues' value for that key is not in Requirement's value set.
// (4) The operator is DoesNotExist or NotIn and KeyValues does not have the
//     Requirement's key.
func (r *Requirement) Matches(kvs KeyValues) bool {
	switch r.operator {
	case InOperator, EqualsOperator, DoubleEqualsOperator:
		if !kvs.Has(r.key) {
			return false
		}
		return r.strValues.Has(kvs.Get(r.key))
	case NotInOperator, NotEqualsOperator:
		if !kvs.Has(r.key) {
			return true
		}
		return !r.strValues.Has(kvs.Get(r.key))
	case ExistsOperator:
		return kvs.Has(r.key)
	case DoesNotExistOperator:
		return !kvs.Has(r.key)
	default:
		return false
	}
}

func (r *Requirement) Key() string {
	return r.key
}
func (r *Requirement) Operator() Operator {
	return r.operator
}
func (r *Requirement) Values() sets.String {
	ret := sets.String{}
	for k := range r.strValues {
		ret.Insert(k)
	}
	return ret
}

// String returns a human-readable string that represents this
// Requirement. If called on an invalid Requirement, an error is
// returned. See NewRequirement for creating a valid Requirement.
func (r *Requirement) String() string {
	var buffer bytes.Buffer
	if r.operator == DoesNotExistOperator {
		buffer.WriteString("!")
	}
	buffer.WriteString(r.key)

	switch r.operator {
	case EqualsOperator:
		buffer.WriteString("=")
	case DoubleEqualsOperator:
		buffer.WriteString("==")
	case NotEqualsOperator:
		buffer.WriteString("!=")
	case InOperator:
		buffer.WriteString(" in ")
	case NotInOperator:
		buffer.WriteString(" notin ")
	case ExistsOperator, DoesNotExistOperator:
		return buffer.String()
	}

	switch r.operator {
	case InOperator, NotInOperator:
		buffer.WriteString("(")
	}
	if len(r.strValues) == 1 {
		buffer.WriteString(r.strValues.List()[0])
	} else { // only > 1 since == 0 prohibited by NewRequirement
		buffer.WriteString(strings.Join(r.strValues.List(), ","))
	}

	switch r.operator {
	case InOperator, NotInOperator:
		buffer.WriteString(")")
	}
	return buffer.String()
}
