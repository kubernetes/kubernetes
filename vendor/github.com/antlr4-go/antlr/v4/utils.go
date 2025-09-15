// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"bytes"
	"errors"
	"fmt"
	"math/bits"
	"os"
	"strconv"
	"strings"
	"syscall"
)

func intMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func intMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// A simple integer stack

type IntStack []int

var ErrEmptyStack = errors.New("stack is empty")

func (s *IntStack) Pop() (int, error) {
	l := len(*s) - 1
	if l < 0 {
		return 0, ErrEmptyStack
	}
	v := (*s)[l]
	*s = (*s)[0:l]
	return v, nil
}

func (s *IntStack) Push(e int) {
	*s = append(*s, e)
}

const bitsPerWord = 64

func indexForBit(bit int) int {
	return bit / bitsPerWord
}

//goland:noinspection GoUnusedExportedFunction,GoUnusedFunction
func wordForBit(data []uint64, bit int) uint64 {
	idx := indexForBit(bit)
	if idx >= len(data) {
		return 0
	}
	return data[idx]
}

func maskForBit(bit int) uint64 {
	return uint64(1) << (bit % bitsPerWord)
}

func wordsNeeded(bit int) int {
	return indexForBit(bit) + 1
}

type BitSet struct {
	data []uint64
}

// NewBitSet creates a new bitwise set
// TODO: See if we can replace with the standard library's BitSet
func NewBitSet() *BitSet {
	return &BitSet{}
}

func (b *BitSet) add(value int) {
	idx := indexForBit(value)
	if idx >= len(b.data) {
		size := wordsNeeded(value)
		data := make([]uint64, size)
		copy(data, b.data)
		b.data = data
	}
	b.data[idx] |= maskForBit(value)
}

func (b *BitSet) clear(index int) {
	idx := indexForBit(index)
	if idx >= len(b.data) {
		return
	}
	b.data[idx] &= ^maskForBit(index)
}

func (b *BitSet) or(set *BitSet) {
	// Get min size necessary to represent the bits in both sets.
	bLen := b.minLen()
	setLen := set.minLen()
	maxLen := intMax(bLen, setLen)
	if maxLen > len(b.data) {
		// Increase the size of len(b.data) to represent the bits in both sets.
		data := make([]uint64, maxLen)
		copy(data, b.data)
		b.data = data
	}
	// len(b.data) is at least setLen.
	for i := 0; i < setLen; i++ {
		b.data[i] |= set.data[i]
	}
}

func (b *BitSet) remove(value int) {
	b.clear(value)
}

func (b *BitSet) contains(value int) bool {
	idx := indexForBit(value)
	if idx >= len(b.data) {
		return false
	}
	return (b.data[idx] & maskForBit(value)) != 0
}

func (b *BitSet) minValue() int {
	for i, v := range b.data {
		if v == 0 {
			continue
		}
		return i*bitsPerWord + bits.TrailingZeros64(v)
	}
	return 2147483647
}

func (b *BitSet) equals(other interface{}) bool {
	otherBitSet, ok := other.(*BitSet)
	if !ok {
		return false
	}

	if b == otherBitSet {
		return true
	}

	// We only compare set bits, so we cannot rely on the two slices having the same size. Its
	// possible for two BitSets to have different slice lengths but the same set bits. So we only
	// compare the relevant words and ignore the trailing zeros.
	bLen := b.minLen()
	otherLen := otherBitSet.minLen()

	if bLen != otherLen {
		return false
	}

	for i := 0; i < bLen; i++ {
		if b.data[i] != otherBitSet.data[i] {
			return false
		}
	}

	return true
}

func (b *BitSet) minLen() int {
	for i := len(b.data); i > 0; i-- {
		if b.data[i-1] != 0 {
			return i
		}
	}
	return 0
}

func (b *BitSet) length() int {
	cnt := 0
	for _, val := range b.data {
		cnt += bits.OnesCount64(val)
	}
	return cnt
}

func (b *BitSet) String() string {
	vals := make([]string, 0, b.length())

	for i, v := range b.data {
		for v != 0 {
			n := bits.TrailingZeros64(v)
			vals = append(vals, strconv.Itoa(i*bitsPerWord+n))
			v &= ^(uint64(1) << n)
		}
	}

	return "{" + strings.Join(vals, ", ") + "}"
}

type AltDict struct {
	data map[string]interface{}
}

func NewAltDict() *AltDict {
	d := new(AltDict)
	d.data = make(map[string]interface{})
	return d
}

func (a *AltDict) Get(key string) interface{} {
	key = "k-" + key
	return a.data[key]
}

func (a *AltDict) put(key string, value interface{}) {
	key = "k-" + key
	a.data[key] = value
}

func (a *AltDict) values() []interface{} {
	vs := make([]interface{}, len(a.data))
	i := 0
	for _, v := range a.data {
		vs[i] = v
		i++
	}
	return vs
}

func EscapeWhitespace(s string, escapeSpaces bool) string {

	s = strings.Replace(s, "\t", "\\t", -1)
	s = strings.Replace(s, "\n", "\\n", -1)
	s = strings.Replace(s, "\r", "\\r", -1)
	if escapeSpaces {
		s = strings.Replace(s, " ", "\u00B7", -1)
	}
	return s
}

//goland:noinspection GoUnusedExportedFunction
func TerminalNodeToStringArray(sa []TerminalNode) []string {
	st := make([]string, len(sa))

	for i, s := range sa {
		st[i] = fmt.Sprintf("%v", s)
	}

	return st
}

//goland:noinspection GoUnusedExportedFunction
func PrintArrayJavaStyle(sa []string) string {
	var buffer bytes.Buffer

	buffer.WriteString("[")

	for i, s := range sa {
		buffer.WriteString(s)
		if i != len(sa)-1 {
			buffer.WriteString(", ")
		}
	}

	buffer.WriteString("]")

	return buffer.String()
}

// murmur hash
func murmurInit(seed int) int {
	return seed
}

func murmurUpdate(h int, value int) int {
	const c1 uint32 = 0xCC9E2D51
	const c2 uint32 = 0x1B873593
	const r1 uint32 = 15
	const r2 uint32 = 13
	const m uint32 = 5
	const n uint32 = 0xE6546B64

	k := uint32(value)
	k *= c1
	k = (k << r1) | (k >> (32 - r1))
	k *= c2

	hash := uint32(h) ^ k
	hash = (hash << r2) | (hash >> (32 - r2))
	hash = hash*m + n
	return int(hash)
}

func murmurFinish(h int, numberOfWords int) int {
	var hash = uint32(h)
	hash ^= uint32(numberOfWords) << 2
	hash ^= hash >> 16
	hash *= 0x85ebca6b
	hash ^= hash >> 13
	hash *= 0xc2b2ae35
	hash ^= hash >> 16

	return int(hash)
}

func isDirectory(dir string) (bool, error) {
	fileInfo, err := os.Stat(dir)
	if err != nil {
		switch {
		case errors.Is(err, syscall.ENOENT):
			// The given directory does not exist, so we will try to create it
			//
			err = os.MkdirAll(dir, 0755)
			if err != nil {
				return false, err
			}

			return true, nil
		case err != nil:
			return false, err
		default:
		}
	}
	return fileInfo.IsDir(), err
}
