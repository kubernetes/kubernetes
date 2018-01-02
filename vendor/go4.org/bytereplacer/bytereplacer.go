/*
Copyright 2015 The Camlistore Authors

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

// Package bytereplacer provides a utility for replacing parts of byte slices.
package bytereplacer // import "go4.org/bytereplacer"

import "bytes"

// Replacer replaces a list of strings with replacements.
// It is safe for concurrent use by multiple goroutines.
type Replacer struct {
	r replacer
}

// replacer is the interface that a replacement algorithm needs to implement.
type replacer interface {
	// Replace performs all replacements, in-place if possible.
	Replace(s []byte) []byte
}

// New returns a new Replacer from a list of old, new string pairs.
// Replacements are performed in order, without overlapping matches.
func New(oldnew ...string) *Replacer {
	if len(oldnew)%2 == 1 {
		panic("bytes.NewReplacer: odd argument count")
	}

	allNewBytes := true
	for i := 0; i < len(oldnew); i += 2 {
		if len(oldnew[i]) != 1 {
			return &Replacer{r: makeGenericReplacer(oldnew)}
		}
		if len(oldnew[i+1]) != 1 {
			allNewBytes = false
		}
	}

	if allNewBytes {
		r := byteReplacer{}
		for i := range r {
			r[i] = byte(i)
		}
		// The first occurrence of old->new map takes precedence
		// over the others with the same old string.
		for i := len(oldnew) - 2; i >= 0; i -= 2 {
			o := oldnew[i][0]
			n := oldnew[i+1][0]
			r[o] = n
		}
		return &Replacer{r: &r}
	}

	return &Replacer{r: makeGenericReplacer(oldnew)}
}

// Replace performs all replacements in-place on s. If the capacity
// of s is not sufficient, a new slice is allocated, otherwise Replace
// returns s.
func (r *Replacer) Replace(s []byte) []byte {
	return r.r.Replace(s)
}

type trieNode struct {
	value    []byte
	priority int
	prefix   []byte
	next     *trieNode
	table    []*trieNode
}

func (t *trieNode) add(key, val []byte, priority int, r *genericReplacer) {
	if len(key) == 0 {
		if t.priority == 0 {
			t.value = val
			t.priority = priority
		}
		return
	}

	if len(t.prefix) > 0 {
		// Need to split the prefix among multiple nodes.
		var n int // length of the longest common prefix
		for ; n < len(t.prefix) && n < len(key); n++ {
			if t.prefix[n] != key[n] {
				break
			}
		}
		if n == len(t.prefix) {
			t.next.add(key[n:], val, priority, r)
		} else if n == 0 {
			// First byte differs, start a new lookup table here. Looking up
			// what is currently t.prefix[0] will lead to prefixNode, and
			// looking up key[0] will lead to keyNode.
			var prefixNode *trieNode
			if len(t.prefix) == 1 {
				prefixNode = t.next
			} else {
				prefixNode = &trieNode{
					prefix: t.prefix[1:],
					next:   t.next,
				}
			}
			keyNode := new(trieNode)
			t.table = make([]*trieNode, r.tableSize)
			t.table[r.mapping[t.prefix[0]]] = prefixNode
			t.table[r.mapping[key[0]]] = keyNode
			t.prefix = nil
			t.next = nil
			keyNode.add(key[1:], val, priority, r)
		} else {
			// Insert new node after the common section of the prefix.
			next := &trieNode{
				prefix: t.prefix[n:],
				next:   t.next,
			}
			t.prefix = t.prefix[:n]
			t.next = next
			next.add(key[n:], val, priority, r)
		}
	} else if t.table != nil {
		// Insert into existing table.
		m := r.mapping[key[0]]
		if t.table[m] == nil {
			t.table[m] = new(trieNode)
		}
		t.table[m].add(key[1:], val, priority, r)
	} else {
		t.prefix = key
		t.next = new(trieNode)
		t.next.add(nil, val, priority, r)
	}
}

func (r *genericReplacer) lookup(s []byte, ignoreRoot bool) (val []byte, keylen int, found bool) {
	// Iterate down the trie to the end, and grab the value and keylen with
	// the highest priority.
	bestPriority := 0
	node := &r.root
	n := 0
	for node != nil {
		if node.priority > bestPriority && !(ignoreRoot && node == &r.root) {
			bestPriority = node.priority
			val = node.value
			keylen = n
			found = true
		}

		if len(s) == 0 {
			break
		}
		if node.table != nil {
			index := r.mapping[s[0]]
			if int(index) == r.tableSize {
				break
			}
			node = node.table[index]
			s = s[1:]
			n++
		} else if len(node.prefix) > 0 && bytes.HasPrefix(s, node.prefix) {
			n += len(node.prefix)
			s = s[len(node.prefix):]
			node = node.next
		} else {
			break
		}
	}
	return
}

// genericReplacer is the fully generic algorithm.
// It's used as a fallback when nothing faster can be used.
type genericReplacer struct {
	root trieNode
	// tableSize is the size of a trie node's lookup table. It is the number
	// of unique key bytes.
	tableSize int
	// mapping maps from key bytes to a dense index for trieNode.table.
	mapping [256]byte
}

func makeGenericReplacer(oldnew []string) *genericReplacer {
	r := new(genericReplacer)
	// Find each byte used, then assign them each an index.
	for i := 0; i < len(oldnew); i += 2 {
		key := oldnew[i]
		for j := 0; j < len(key); j++ {
			r.mapping[key[j]] = 1
		}
	}

	for _, b := range r.mapping {
		r.tableSize += int(b)
	}

	var index byte
	for i, b := range r.mapping {
		if b == 0 {
			r.mapping[i] = byte(r.tableSize)
		} else {
			r.mapping[i] = index
			index++
		}
	}
	// Ensure root node uses a lookup table (for performance).
	r.root.table = make([]*trieNode, r.tableSize)

	for i := 0; i < len(oldnew); i += 2 {
		r.root.add([]byte(oldnew[i]), []byte(oldnew[i+1]), len(oldnew)-i, r)
	}
	return r
}

func (r *genericReplacer) Replace(s []byte) []byte {
	var last int
	var prevMatchEmpty bool
	dst := s[:0]
	grown := false
	for i := 0; i <= len(s); {
		// Fast path: s[i] is not a prefix of any pattern.
		if i != len(s) && r.root.priority == 0 {
			index := int(r.mapping[s[i]])
			if index == r.tableSize || r.root.table[index] == nil {
				i++
				continue
			}
		}

		// Ignore the empty match iff the previous loop found the empty match.
		val, keylen, match := r.lookup(s[i:], prevMatchEmpty)
		prevMatchEmpty = match && keylen == 0
		if match {
			dst = append(dst, s[last:i]...)
			if diff := len(val) - keylen; grown || diff < 0 {
				dst = append(dst, val...)
				i += keylen
			} else if diff <= cap(s)-len(s) {
				// The replacement is larger than the original, but can still fit in the original buffer.
				copy(s[i+len(val):cap(dst)], s[i+keylen:])
				dst = append(dst, val...)
				s = s[:len(s)+diff]
				i += len(val)
			} else {
				// The output will grow larger than the original buffer.  Allocate a new one.
				grown = true
				newDst := make([]byte, len(dst), cap(dst)+diff)
				copy(newDst, dst)
				dst = newDst

				dst = append(dst, val...)
				i += keylen
			}
			last = i
			continue
		}
		i++
	}
	if last != len(s) {
		dst = append(dst, s[last:]...)
	}
	return dst
}

// byteReplacer is the implementation that's used when all the "old"
// and "new" values are single ASCII bytes.
// The array contains replacement bytes indexed by old byte.
type byteReplacer [256]byte

func (r *byteReplacer) Replace(s []byte) []byte {
	for i, b := range s {
		s[i] = r[b]
	}
	return s
}
