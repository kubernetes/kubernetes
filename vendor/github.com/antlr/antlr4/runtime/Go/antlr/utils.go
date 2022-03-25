// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
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

var ErrEmptyStack = errors.New("Stack is empty")

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

func standardEqualsFunction(a interface{}, b interface{}) bool {

	ac, oka := a.(comparable)
	bc, okb := b.(comparable)

	if !oka || !okb {
		panic("Not Comparable")
	}

	return ac.equals(bc)
}

func standardHashFunction(a interface{}) int {
	if h, ok := a.(hasher); ok {
		return h.hash()
	}

	panic("Not Hasher")
}

type hasher interface {
	hash() int
}

type BitSet struct {
	data map[int]bool
}

func NewBitSet() *BitSet {
	b := new(BitSet)
	b.data = make(map[int]bool)
	return b
}

func (b *BitSet) add(value int) {
	b.data[value] = true
}

func (b *BitSet) clear(index int) {
	delete(b.data, index)
}

func (b *BitSet) or(set *BitSet) {
	for k := range set.data {
		b.add(k)
	}
}

func (b *BitSet) remove(value int) {
	delete(b.data, value)
}

func (b *BitSet) contains(value int) bool {
	return b.data[value]
}

func (b *BitSet) values() []int {
	ks := make([]int, len(b.data))
	i := 0
	for k := range b.data {
		ks[i] = k
		i++
	}
	sort.Ints(ks)
	return ks
}

func (b *BitSet) minValue() int {
	min := 2147483647

	for k := range b.data {
		if k < min {
			min = k
		}
	}

	return min
}

func (b *BitSet) equals(other interface{}) bool {
	otherBitSet, ok := other.(*BitSet)
	if !ok {
		return false
	}

	if len(b.data) != len(otherBitSet.data) {
		return false
	}

	for k, v := range b.data {
		if otherBitSet.data[k] != v {
			return false
		}
	}

	return true
}

func (b *BitSet) length() int {
	return len(b.data)
}

func (b *BitSet) String() string {
	vals := b.values()
	valsS := make([]string, len(vals))

	for i, val := range vals {
		valsS[i] = strconv.Itoa(val)
	}
	return "{" + strings.Join(valsS, ", ") + "}"
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

type DoubleDict struct {
	data map[int]map[int]interface{}
}

func NewDoubleDict() *DoubleDict {
	dd := new(DoubleDict)
	dd.data = make(map[int]map[int]interface{})
	return dd
}

func (d *DoubleDict) Get(a, b int) interface{} {
	data := d.data[a]

	if data == nil {
		return nil
	}

	return data[b]
}

func (d *DoubleDict) set(a, b int, o interface{}) {
	data := d.data[a]

	if data == nil {
		data = make(map[int]interface{})
		d.data[a] = data
	}

	data[b] = o
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

func TerminalNodeToStringArray(sa []TerminalNode) []string {
	st := make([]string, len(sa))

	for i, s := range sa {
		st[i] = fmt.Sprintf("%v", s)
	}

	return st
}

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
