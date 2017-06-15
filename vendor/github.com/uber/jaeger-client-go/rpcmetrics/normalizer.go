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

// NameNormalizer is used to convert the operation names to strings
// that can be safely used as tags in the metrics.
type NameNormalizer interface {
	Normalize(name string) string
}

// DefaultNameNormalizer converts operation names so that they contain only characters
// from the safe charset [a-zA-Z0-9-./_]. All other characters are replaced with '-'.
var DefaultNameNormalizer = &SimpleNameNormalizer{
	SafeSets: []SafeCharacterSet{
		&Range{From: 'a', To: 'z'},
		&Range{From: 'A', To: 'Z'},
		&Range{From: '0', To: '9'},
		&Char{'-'},
		&Char{'_'},
		&Char{'/'},
		&Char{'.'},
	},
	Replacement: '-',
}

// SimpleNameNormalizer uses a set of safe character sets.
type SimpleNameNormalizer struct {
	SafeSets    []SafeCharacterSet
	Replacement byte
}

// SafeCharacterSet determines if the given character is "safe"
type SafeCharacterSet interface {
	IsSafe(c byte) bool
}

// Range implements SafeCharacterSet
type Range struct {
	From, To byte
}

// IsSafe implements SafeCharacterSet
func (r *Range) IsSafe(c byte) bool {
	return c >= r.From && c <= r.To
}

// Char implements SafeCharacterSet
type Char struct {
	Val byte
}

// IsSafe implements SafeCharacterSet
func (ch *Char) IsSafe(c byte) bool {
	return c == ch.Val
}

// Normalize checks each character in the string against SafeSets,
// and if it's not safe substitutes it with Replacement.
func (n *SimpleNameNormalizer) Normalize(name string) string {
	var retMe []byte
	nameBytes := []byte(name)
	for i, b := range nameBytes {
		if n.safeByte(b) {
			if retMe != nil {
				retMe[i] = b
			}
		} else {
			if retMe == nil {
				retMe = make([]byte, len(nameBytes))
				copy(retMe[0:i], nameBytes[0:i])
			}
			retMe[i] = n.Replacement
		}
	}
	if retMe == nil {
		return name
	}
	return string(retMe)
}

// safeByte checks if b against all safe charsets.
func (n *SimpleNameNormalizer) safeByte(b byte) bool {
	for i := range n.SafeSets {
		if n.SafeSets[i].IsSafe(b) {
			return true
		}
	}
	return false
}
