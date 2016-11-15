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

// Check if a sorted string list has a string
func SortedStringHas(s sort.StringSlice, x string) bool {
	index := s.Search(x)
	if index == len(s) {
		return false
	}
	return s[s.Search(x)] == x
}

// Delete a string from a sorted string list
func SortedStringsDelete(s sort.StringSlice, x string) sort.StringSlice {
	index := s.Search(x)
	if len(s) != index && s[index] == x {
		s = append(s[:index], s[index+1:]...)
	}

	return s
}
