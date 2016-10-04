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
}

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
}

func (t *Trie) HasPrefix(v string) bool {
	root := t
	if root.wordTail {
		return true
	}
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			return false
		}
		if child.wordTail {
			return true
		}
		root = child
	}
	return false
}


