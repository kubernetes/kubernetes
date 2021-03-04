// Copyright 2013 The Prometheus Authors
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

package model

import (
	"fmt"
	"strconv"
)

// Fingerprint provides a hash-capable representation of a Metric.
// For our purposes, FNV-1A 64-bit is used.
type Fingerprint uint64

// FingerprintFromString transforms a string representation into a Fingerprint.
func FingerprintFromString(s string) (Fingerprint, error) {
	num, err := strconv.ParseUint(s, 16, 64)
	return Fingerprint(num), err
}

// ParseFingerprint parses the input string into a fingerprint.
func ParseFingerprint(s string) (Fingerprint, error) {
	num, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return 0, err
	}
	return Fingerprint(num), nil
}

func (f Fingerprint) String() string {
	return fmt.Sprintf("%016x", uint64(f))
}

// Fingerprints represents a collection of Fingerprint subject to a given
// natural sorting scheme. It implements sort.Interface.
type Fingerprints []Fingerprint

// Len implements sort.Interface.
func (f Fingerprints) Len() int {
	return len(f)
}

// Less implements sort.Interface.
func (f Fingerprints) Less(i, j int) bool {
	return f[i] < f[j]
}

// Swap implements sort.Interface.
func (f Fingerprints) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

// FingerprintSet is a set of Fingerprints.
type FingerprintSet map[Fingerprint]struct{}

// Equal returns true if both sets contain the same elements (and not more).
func (s FingerprintSet) Equal(o FingerprintSet) bool {
	if len(s) != len(o) {
		return false
	}

	for k := range s {
		if _, ok := o[k]; !ok {
			return false
		}
	}

	return true
}

// Intersection returns the elements contained in both sets.
func (s FingerprintSet) Intersection(o FingerprintSet) FingerprintSet {
	myLength, otherLength := len(s), len(o)
	if myLength == 0 || otherLength == 0 {
		return FingerprintSet{}
	}

	subSet := s
	superSet := o

	if otherLength < myLength {
		subSet = o
		superSet = s
	}

	out := FingerprintSet{}

	for k := range subSet {
		if _, ok := superSet[k]; ok {
			out[k] = struct{}{}
		}
	}

	return out
}
