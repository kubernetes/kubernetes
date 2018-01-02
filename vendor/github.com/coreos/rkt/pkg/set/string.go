// Copyright 2015 The rkt Authors
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

// For how the uidshift and uidcount are generated please check:
// http://cgit.freedesktop.org/systemd/systemd/commit/?id=03cfe0d51499e86b1573d1

package set

type String map[string]struct{}

func NewString(items ...string) String {
	s := String{}
	s.Insert(items...)
	return s
}

// Insert adds items to the set.
func (s String) Insert(items ...string) {
	for _, item := range items {
		s[item] = struct{}{}
	}
}

// Has returns true if and only if item is contained in the set.
func (s String) Has(item string) bool {
	_, contained := s[item]
	return contained
}

// HasAll returns true if and only if all items are contained in the set.
func (s String) HasAll(items ...string) bool {
	for _, item := range items {
		if !s.Has(item) {
			return false
		}
	}
	return true
}

// Delete removes all items from the set.
func (s String) Delete(items ...string) {
	for _, item := range items {
		delete(s, item)
	}
}

// ConditionalHas returns true if and only if there is any item 'source'
// in the set that satisfies the conditionFunc wrt 'item'.
func (s String) ConditionalHas(conditionFunc func(source, item string) bool, item string) bool {
	for source := range s {
		if conditionFunc(source, item) {
			return true
		}
	}
	return false
}
