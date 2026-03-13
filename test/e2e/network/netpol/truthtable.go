/*
Copyright 2020 The Kubernetes Authors.

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

package netpol

import (
	"strings"

	"k8s.io/kubernetes/test/e2e/framework"
)

// TruthTable takes in n items and maintains an n x n table of booleans for each ordered pair
type TruthTable struct {
	Froms  []string
	Tos    []string
	toSet  map[string]bool
	Values map[string]map[string]bool
}

// NewTruthTableFromItems creates a new truth table with items
func NewTruthTableFromItems(items []string, defaultValue *bool) *TruthTable {
	return NewTruthTable(items, items, defaultValue)
}

// NewTruthTable creates a new truth table with froms and tos
func NewTruthTable(froms []string, tos []string, defaultValue *bool) *TruthTable {
	values := map[string]map[string]bool{}
	for _, from := range froms {
		values[from] = map[string]bool{}
		for _, to := range tos {
			if defaultValue != nil {
				values[from][to] = *defaultValue
			}
		}
	}
	toSet := map[string]bool{}
	for _, to := range tos {
		toSet[to] = true
	}
	return &TruthTable{
		Froms:  froms,
		Tos:    tos,
		toSet:  toSet,
		Values: values,
	}
}

// IsComplete returns true if there's a value set for every single pair of items, otherwise it returns false.
func (tt *TruthTable) IsComplete() bool {
	for _, from := range tt.Froms {
		for _, to := range tt.Tos {
			if _, ok := tt.Values[from][to]; !ok {
				return false
			}
		}
	}
	return true
}

// Set sets the value for from->to
func (tt *TruthTable) Set(from string, to string, value bool) {
	dict, ok := tt.Values[from]
	if !ok {
		framework.Failf("from-key %s not found", from)
	}
	if _, ok := tt.toSet[to]; !ok {
		framework.Failf("to-key %s not allowed", to)
	}
	dict[to] = value
}

// SetAllFrom sets all values where from = 'from'
func (tt *TruthTable) SetAllFrom(from string, value bool) {
	dict, ok := tt.Values[from]
	if !ok {
		framework.Failf("from-key %s not found", from)
	}
	for _, to := range tt.Tos {
		dict[to] = value
	}
}

// SetAllTo sets all values where to = 'to'
func (tt *TruthTable) SetAllTo(to string, value bool) {
	if _, ok := tt.toSet[to]; !ok {
		framework.Failf("to-key %s not found", to)
	}
	for _, from := range tt.Froms {
		tt.Values[from][to] = value
	}
}

// Get gets the specified value
func (tt *TruthTable) Get(from string, to string) bool {
	dict, ok := tt.Values[from]
	if !ok {
		framework.Failf("from-key %s not found", from)
	}
	val, ok := dict[to]
	if !ok {
		framework.Failf("to-key %s not found in map (%+v)", to, dict)
	}
	return val
}

// Compare is used to check two truth tables for equality, returning its
// result in the form of a third truth table.  Both tables are expected to
// have identical items.
func (tt *TruthTable) Compare(other *TruthTable) *TruthTable {
	if len(tt.Froms) != len(other.Froms) || len(tt.Tos) != len(other.Tos) {
		framework.Failf("cannot compare tables of different dimensions")
	}
	for i, fr := range tt.Froms {
		if other.Froms[i] != fr {
			framework.Failf("cannot compare: from keys at index %d do not match (%s vs %s)", i, other.Froms[i], fr)
		}
	}
	for i, to := range tt.Tos {
		if other.Tos[i] != to {
			framework.Failf("cannot compare: to keys at index %d do not match (%s vs %s)", i, other.Tos[i], to)
		}
	}

	values := map[string]map[string]bool{}
	for from, dict := range tt.Values {
		values[from] = map[string]bool{}
		for to, val := range dict {
			values[from][to] = val == other.Values[from][to]
		}
	}
	return &TruthTable{
		Froms:  tt.Froms,
		Tos:    tt.Tos,
		toSet:  tt.toSet,
		Values: values,
	}
}

// PrettyPrint produces a nice visual representation.
func (tt *TruthTable) PrettyPrint(indent string) string {
	header := indent + strings.Join(append([]string{"-\t"}, tt.Tos...), "\t")
	lines := []string{header}
	for _, from := range tt.Froms {
		line := []string{from}
		for _, to := range tt.Tos {
			mark := "X"
			val, ok := tt.Values[from][to]
			if !ok {
				mark = "?"
			} else if val {
				mark = "."
			}
			line = append(line, mark+"\t")
		}
		lines = append(lines, indent+strings.Join(line, "\t"))
	}
	return strings.Join(lines, "\n")
}
