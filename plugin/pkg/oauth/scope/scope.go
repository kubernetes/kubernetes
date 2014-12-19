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

package scope

import (
	"sort"
	"strings"
)

// Add takes two sets of scopes, and returns a combined sorted set of scopes
func Add(has []string, new []string) []string {
	sorted := sortAndCopy(has)
	for _, s := range new {
		i := sort.SearchStrings(sorted, s)
		if i == len(sorted) {
			sorted = append(sorted, s)
		} else if sorted[i] != s {
			sorted = append(sorted, "")
			copy(sorted[i+1:], sorted[i:])
			sorted[i] = s
		}
	}
	return sorted
}

// Split turns a string containing space-delimited scopes into a slice (which may be of length 0)
func Split(scope string) []string {
	scope = strings.TrimSpace(scope)
	if scope == "" {
		return []string{}
	}
	return strings.Split(scope, " ")
}

// Join returns a space-delimited string of scopes
func Join(scopes []string) string {
	return strings.Join(scopes, " ")
}

// Covers returns true if the requested scopes are a subset of the granted scopes
func Covers(granted, requested []string) bool {
	granted, requested = sortAndCopy(granted), sortAndCopy(requested)
NextRequested:
	for i := range requested {
		for j := range granted {
			if granted[j] == requested[i] {
				continue NextRequested
			}
		}
		return false
	}
	return true
}

func sortAndCopy(arr []string) []string {
	newArr := make([]string, len(arr))
	copy(newArr, arr)
	sort.Sort(sort.StringSlice(newArr))
	return newArr
}
