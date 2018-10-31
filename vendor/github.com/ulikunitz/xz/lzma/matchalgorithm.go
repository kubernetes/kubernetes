// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import "errors"

// MatchAlgorithm identifies an algorithm to find matches in the
// dictionary.
type MatchAlgorithm byte

// Supported matcher algorithms.
const (
	HashTable4 MatchAlgorithm = iota
	BinaryTree
)

// maStrings are used by the String method.
var maStrings = map[MatchAlgorithm]string{
	HashTable4: "HashTable4",
	BinaryTree: "BinaryTree",
}

// String returns a string representation of the Matcher.
func (a MatchAlgorithm) String() string {
	if s, ok := maStrings[a]; ok {
		return s
	}
	return "unknown"
}

var errUnsupportedMatchAlgorithm = errors.New(
	"lzma: unsupported match algorithm value")

// verify checks whether the matcher value is supported.
func (a MatchAlgorithm) verify() error {
	if _, ok := maStrings[a]; !ok {
		return errUnsupportedMatchAlgorithm
	}
	return nil
}

func (a MatchAlgorithm) new(dictCap int) (m matcher, err error) {
	switch a {
	case HashTable4:
		return newHashTable(dictCap, 4)
	case BinaryTree:
		return newBinTree(dictCap)
	}
	return nil, errUnsupportedMatchAlgorithm
}
