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

package util

// A simple trie implementation with Add an HasPrefix methods only.
type Trie struct {
	children map[byte]*Trie
	wordTail bool
	word     string
}

// CreateTrie creates a Trie and add all strings in the provided list to it.
func CreateTrie(list []string) Trie {
	ret := Trie{
		children: make(map[byte]*Trie),
		wordTail: false,
	}
	for _, v := range list {
		ret.Add(v)
	}
	return ret
}

// Add adds a word to this trie
func (t *Trie) Add(v string) {
	root := t
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			child = &Trie{
				children: make(map[byte]*Trie),
				wordTail: false,
			}
			root.children[b] = child
		}
		root = child
	}
	root.wordTail = true
	root.word = v
}

// HasPrefix returns true of v has any of the prefixes stored in this trie.
func (t *Trie) HasPrefix(v string) bool {
	_, has := t.GetPrefix(v)
	return has
}

// GetPrefix is like HasPrefix but return the prefix in case of match or empty string otherwise.
func (t *Trie) GetPrefix(v string) (string, bool) {
	root := t
	if root.wordTail {
		return root.word, true
	}
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			return "", false
		}
		if child.wordTail {
			return child.word, true
		}
		root = child
	}
	return "", false
}
