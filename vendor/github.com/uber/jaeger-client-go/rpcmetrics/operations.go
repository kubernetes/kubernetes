// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package rpcmetrics

import "sync"

// normalizedOperations is a cache for operationName -> safeName mappings.
type normalizedOperations struct {
	names       map[string]string
	maxSize     int
	defaultName string
	normalizer  NameNormalizer
	mux         sync.RWMutex
}

func newNormalizedOperations(maxSize int, normalizer NameNormalizer) *normalizedOperations {
	return &normalizedOperations{
		maxSize:    maxSize,
		normalizer: normalizer,
		names:      make(map[string]string, maxSize),
	}
}

// normalize looks up the name in the cache, if not found it uses normalizer
// to convert the name to a safe name. If called with more than maxSize unique
// names it returns "" for all other names beyond those already cached.
func (n *normalizedOperations) normalize(name string) string {
	n.mux.RLock()
	norm, ok := n.names[name]
	l := len(n.names)
	n.mux.RUnlock()
	if ok {
		return norm
	}
	if l >= n.maxSize {
		return ""
	}
	return n.normalizeWithLock(name)
}

func (n *normalizedOperations) normalizeWithLock(name string) string {
	norm := n.normalizer.Normalize(name)
	n.mux.Lock()
	defer n.mux.Unlock()
	// cache may have grown while we were not holding the lock
	if len(n.names) >= n.maxSize {
		return ""
	}
	n.names[name] = norm
	return norm
}
