// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

// testWeighter is a simple Weighter that returns weights from a user-defined map.
type testWeighter map[string][]Elem

func (t testWeighter) Start(int, []byte) int       { return 0 }
func (t testWeighter) StartString(int, string) int { return 0 }
func (t testWeighter) Domain() []string            { return nil }
func (t testWeighter) Top() uint32                 { return 0 }

// maxContractBytes is the maximum length of any key in the map.
const maxContractBytes = 10

func (t testWeighter) AppendNext(buf []Elem, s []byte) ([]Elem, int) {
	n := len(s)
	if n > maxContractBytes {
		n = maxContractBytes
	}
	for i := n; i > 0; i-- {
		if e, ok := t[string(s[:i])]; ok {
			return append(buf, e...), i
		}
	}
	panic("incomplete testWeighter: could not find " + string(s))
}

func (t testWeighter) AppendNextString(buf []Elem, s string) ([]Elem, int) {
	n := len(s)
	if n > maxContractBytes {
		n = maxContractBytes
	}
	for i := n; i > 0; i-- {
		if e, ok := t[s[:i]]; ok {
			return append(buf, e...), i
		}
	}
	panic("incomplete testWeighter: could not find " + s)
}
