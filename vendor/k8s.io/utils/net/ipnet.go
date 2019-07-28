/*
Copyright 2016 The Kubernetes Authors.

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

package net

import (
	"net"
	"strings"
)

// IPNetSet maps string to net.IPNet.
type IPNetSet map[string]*net.IPNet

// ParseIPNets parses string slice to IPNetSet.
func ParseIPNets(specs ...string) (IPNetSet, error) {
	ipnetset := make(IPNetSet)
	for _, spec := range specs {
		spec = strings.TrimSpace(spec)
		_, ipnet, err := net.ParseCIDR(spec)
		if err != nil {
			return nil, err
		}
		k := ipnet.String() // In case of normalization
		ipnetset[k] = ipnet
	}
	return ipnetset, nil
}

// Insert adds items to the set.
func (s IPNetSet) Insert(items ...*net.IPNet) {
	for _, item := range items {
		s[item.String()] = item
	}
}

// Delete removes all items from the set.
func (s IPNetSet) Delete(items ...*net.IPNet) {
	for _, item := range items {
		delete(s, item.String())
	}
}

// Has returns true if and only if item is contained in the set.
func (s IPNetSet) Has(item *net.IPNet) bool {
	_, contained := s[item.String()]
	return contained
}

// HasAll returns true if and only if all items are contained in the set.
func (s IPNetSet) HasAll(items ...*net.IPNet) bool {
	for _, item := range items {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// Difference returns a set of objects that are not in s2
// For example:
// s1 = {a1, a2, a3}
// s2 = {a1, a2, a4, a5}
// s1.Difference(s2) = {a3}
// s2.Difference(s1) = {a4, a5}
func (s IPNetSet) Difference(s2 IPNetSet) IPNetSet {
	result := make(IPNetSet)
	for k, i := range s {
		_, found := s2[k]
		if found {
			continue
		}
		result[k] = i
	}
	return result
}

// StringSlice returns a []string with the String representation of each element in the set.
// Order is undefined.
func (s IPNetSet) StringSlice() []string {
	a := make([]string, 0, len(s))
	for k := range s {
		a = append(a, k)
	}
	return a
}

// IsSuperset returns true if and only if s1 is a superset of s2.
func (s IPNetSet) IsSuperset(s2 IPNetSet) bool {
	for k := range s2 {
		_, found := s[k]
		if !found {
			return false
		}
	}
	return true
}

// Equal returns true if and only if s1 is equal (as a set) to s2.
// Two sets are equal if their membership is identical.
// (In practice, this means same elements, order doesn't matter)
func (s IPNetSet) Equal(s2 IPNetSet) bool {
	return len(s) == len(s2) && s.IsSuperset(s2)
}

// Len returns the size of the set.
func (s IPNetSet) Len() int {
	return len(s)
}
