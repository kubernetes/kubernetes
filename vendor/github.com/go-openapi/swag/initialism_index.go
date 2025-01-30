// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"sort"
	"strings"
	"sync"
)

var (
	// commonInitialisms are common acronyms that are kept as whole uppercased words.
	commonInitialisms *indexOfInitialisms

	// initialisms is a slice of sorted initialisms
	initialisms []string

	// a copy of initialisms pre-baked as []rune
	initialismsRunes      [][]rune
	initialismsUpperCased [][]rune

	isInitialism func(string) bool

	maxAllocMatches int
)

func init() {
	// Taken from https://github.com/golang/lint/blob/3390df4df2787994aea98de825b964ac7944b817/lint.go#L732-L769
	configuredInitialisms := map[string]bool{
		"ACL":   true,
		"API":   true,
		"ASCII": true,
		"CPU":   true,
		"CSS":   true,
		"DNS":   true,
		"EOF":   true,
		"GUID":  true,
		"HTML":  true,
		"HTTPS": true,
		"HTTP":  true,
		"ID":    true,
		"IP":    true,
		"IPv4":  true,
		"IPv6":  true,
		"JSON":  true,
		"LHS":   true,
		"OAI":   true,
		"QPS":   true,
		"RAM":   true,
		"RHS":   true,
		"RPC":   true,
		"SLA":   true,
		"SMTP":  true,
		"SQL":   true,
		"SSH":   true,
		"TCP":   true,
		"TLS":   true,
		"TTL":   true,
		"UDP":   true,
		"UI":    true,
		"UID":   true,
		"UUID":  true,
		"URI":   true,
		"URL":   true,
		"UTF8":  true,
		"VM":    true,
		"XML":   true,
		"XMPP":  true,
		"XSRF":  true,
		"XSS":   true,
	}

	// a thread-safe index of initialisms
	commonInitialisms = newIndexOfInitialisms().load(configuredInitialisms)
	initialisms = commonInitialisms.sorted()
	initialismsRunes = asRunes(initialisms)
	initialismsUpperCased = asUpperCased(initialisms)
	maxAllocMatches = maxAllocHeuristic(initialismsRunes)

	// a test function
	isInitialism = commonInitialisms.isInitialism
}

func asRunes(in []string) [][]rune {
	out := make([][]rune, len(in))
	for i, initialism := range in {
		out[i] = []rune(initialism)
	}

	return out
}

func asUpperCased(in []string) [][]rune {
	out := make([][]rune, len(in))

	for i, initialism := range in {
		out[i] = []rune(upper(trim(initialism)))
	}

	return out
}

func maxAllocHeuristic(in [][]rune) int {
	heuristic := make(map[rune]int)
	for _, initialism := range in {
		heuristic[initialism[0]]++
	}

	var maxAlloc int
	for _, val := range heuristic {
		if val > maxAlloc {
			maxAlloc = val
		}
	}

	return maxAlloc
}

// AddInitialisms add additional initialisms
func AddInitialisms(words ...string) {
	for _, word := range words {
		// commonInitialisms[upper(word)] = true
		commonInitialisms.add(upper(word))
	}
	// sort again
	initialisms = commonInitialisms.sorted()
	initialismsRunes = asRunes(initialisms)
	initialismsUpperCased = asUpperCased(initialisms)
}

// indexOfInitialisms is a thread-safe implementation of the sorted index of initialisms.
// Since go1.9, this may be implemented with sync.Map.
type indexOfInitialisms struct {
	sortMutex *sync.Mutex
	index     *sync.Map
}

func newIndexOfInitialisms() *indexOfInitialisms {
	return &indexOfInitialisms{
		sortMutex: new(sync.Mutex),
		index:     new(sync.Map),
	}
}

func (m *indexOfInitialisms) load(initial map[string]bool) *indexOfInitialisms {
	m.sortMutex.Lock()
	defer m.sortMutex.Unlock()
	for k, v := range initial {
		m.index.Store(k, v)
	}
	return m
}

func (m *indexOfInitialisms) isInitialism(key string) bool {
	_, ok := m.index.Load(key)
	return ok
}

func (m *indexOfInitialisms) add(key string) *indexOfInitialisms {
	m.index.Store(key, true)
	return m
}

func (m *indexOfInitialisms) sorted() (result []string) {
	m.sortMutex.Lock()
	defer m.sortMutex.Unlock()
	m.index.Range(func(key, _ interface{}) bool {
		k := key.(string)
		result = append(result, k)
		return true
	})
	sort.Sort(sort.Reverse(byInitialism(result)))
	return
}

type byInitialism []string

func (s byInitialism) Len() int {
	return len(s)
}
func (s byInitialism) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s byInitialism) Less(i, j int) bool {
	if len(s[i]) != len(s[j]) {
		return len(s[i]) < len(s[j])
	}

	return strings.Compare(s[i], s[j]) > 0
}
