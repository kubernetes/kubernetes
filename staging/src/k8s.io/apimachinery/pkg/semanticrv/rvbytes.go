/*
Copyright 2022 The Kubernetes Authors.

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

package semanticrv

import (
	"bytes"
	"strings"
)

// ResourceVersion, except for two special values, identifies a closed set of
// write transactions on an API service and thus is associated with
// the state that those transactions collectively produce.
// Here "closed" means that for every included write X,
// every write W that happens-before X is also included.
//
// This definition contemplates that write transactions might not be
// totally ordered.
// This does not preclude the notifications from a WATCH from having
// the properties that:
// (1) a notification with RV1 preceding a notification with RV2 implies that
//     RV2 is not a subset of RV1; and
// (2) in a WATCH that starts from a specific RV0, every notification carries
//     a ResourceVersion that is a superset of RV0.
// These properties are enough to allow a reflector to maintain a single ResourceVersion
// that tracks the accumulation of what the reflector has seen.
//
// There are two special values of ResourceVersion,
// which do not identify a set of transactions but are used
// by a client in a position that _may_ specify a set of transactions; see
// https://kubernetes.io/docs/reference/using-api/api-concepts/#the-resourceversion-parameter
// for the two special values and their meaning.
type ResourceVersion struct {
	// The concrete representation used here is slice of byte.
	// Ordering among non-special values is first by slice length and second,
	// among equal length slices, by lexicographic comparison.
	// This has the property that values produced by `fmt.Sprintf("%d", .)` applied
	// to positive integers are ordered the same as those integers.
	rep []byte
}

// UndefinedAsString is the string representation of "undefined"
const UndefinedAsString = ""

// AnyAsString is the string representation of "any"
const AnyAsString = "0"

// Parse parses the given string into a ResourceVersion, returning an error
// if the input is not a valid string representation of a ResourceVersion.
func Parse(str string) (ResourceVersion, error) {
	return ResourceVersion{rep: []byte(str)}, nil
}

// IsUndefined tells whether the receiver is the special value that means
// "unspecified" or "undefined".
// This is true for the zero value of this type.
func (rv ResourceVersion) IsUndefined() bool {
	return len(rv.rep) == 0
}

// IsAny tells whether the receiver is the special value that means any
// closed set of write transactions is acceptable.
func (rv ResourceVersion) IsAny() bool {
	return string(rv.rep) == AnyAsString
}

// IsSpecial tells whether the receiver is either of the special values.
func (rv ResourceVersion) IsSpecial() bool {
	return rv.IsAny() || rv.IsUndefined()
}

// String returns the string representation of the receiver.
// For every string S: if `R, err := Parse(S); err == nil`
// then `S == R.String()`.
// For every pair of ResourceVersions X and Y,
// `X.Compare(Y).IsEqual() == (X.String() == Y.String())`.
func (rv ResourceVersion) String() string {
	return string(rv.rep)
}

// StringForFilename returns a string representation of the receiver
// that is safe to put in a filename in Linux, MacOS, and Windows.
// The returned string will not include a directory delimiter.
func (rv ResourceVersion) StringForFilename() string {
	if rv.IsUndefined() {
		return "%NDEF"
	}
	var bld strings.Builder
	for _, b := range rv.rep {
		if b >= '0' && b <= '9' ||
			b >= 'A' && b <= 'Z' {
			bld.WriteByte(b)
			continue
		}
		switch b {
		case '!', '@', '#', '$', '^', '&', '*', '(', ')',
			'-', '_', '=', '+', '.', ',':
			bld.WriteByte(b)
			continue
		}
		bld.WriteRune('%')
		bld.WriteByte(hexDigits[b/16])
		bld.WriteByte(hexDigits[b%16])
	}
	return bld.String()
}

const hexDigits = "0123456789ABCDEF"

// Compare compares the write transaction sets identified by two ResourceVersions.
// If the transaction set of RV X is a subset of the transaction set of RV Y then X.Compare(Y).LE.
// Naturally, this is reflexive ( X.Compare(X).IsEqual() )
// and anti-symmetric ( X.Compare(Y) == Y.Compare(X).Reverse() ).
// Also naturally, these chain: X.Compare(Y).And(Y.Compare(Z)).Implies(X.Compare(Z)).
// The special values are equal to themselves and incomparable with all other values.
func (rv ResourceVersion) Compare(other ResourceVersion) Comparison {
	if rv.IsUndefined() {
		otherIsUndef := other.IsUndefined()
		return Comparison{LE: otherIsUndef, GE: otherIsUndef}
	}
	if rv.IsAny() {
		otherIsAny := other.IsAny()
		return Comparison{LE: otherIsAny, GE: otherIsAny}
	}
	if other.IsSpecial() {
		return Incomparable()
	}
	l1 := len(rv.rep)
	l2 := len(other.rep)
	if l1 < l2 {
		return Less()
	}
	if l1 > l2 {
		return Greater()
	}
	c := bytes.Compare(rv.rep, other.rep)
	return Comparison{LE: c <= 0, GE: c >= 0}
}

// Union returns the ResourceVersion that identifies the union of the two
// input transaction sets, with the two special values handled as follows.
// For every ResourceVersion X: X.Union(undefined) = undfined.Union(X) = X.
// For every ResourceVersion X other than undefined: X.Union(any) = any.Union(X) = X.
func (rv ResourceVersion) Union(other ResourceVersion) ResourceVersion {
	if rv.IsUndefined() {
		return other
	}
	if other.IsSpecial() {
		return rv
	}
	cmp := rv.Compare(other)
	if cmp.GE {
		return rv
	}
	return other
}
