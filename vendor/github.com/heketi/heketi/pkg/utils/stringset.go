//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package utils

import (
	"sort"
)

type StringSet struct {
	Set sort.StringSlice
}

// Create a string set.
//
// A string set is a list where each element appears only once
func NewStringSet() *StringSet {
	return &StringSet{
		Set: make(sort.StringSlice, 0),
	}
}

// Add a string to the string set
func (s *StringSet) Add(v string) {
	if !SortedStringHas(s.Set, v) {
		s.Set = append(s.Set, v)
		s.Set.Sort()
	}
}

// Return string list
func (s *StringSet) Strings() []string {
	return s.Set
}

func (s *StringSet) Len() int {
	return len(s.Set)
}
