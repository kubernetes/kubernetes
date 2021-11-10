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

type Set struct {
	data             map[int][]interface{}
	hashcodeFunction func(interface{}) int
	equalsFunction   func(interface{}, interface{}) bool
}

func NewSet(
	hashcodeFunction func(interface{}) int,
	equalsFunction func(interface{}, interface{}) bool) *Set {

	s := new(Set)

	s.data = make(map[int][]interface{})

	if hashcodeFunction != nil {
		s.hashcodeFunction = hashcodeFunction
	} else {
		s.hashcodeFunction = standardHashFunction
	}

	if equalsFunction == nil {
		s.equalsFunction = standardEqualsFunction
	} else {
		s.equalsFunction = equalsFunction
	}

	return s
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

func (s *Set) length() int {
	return len(s.data)
}

func (s *Set) add(value interface{}) interface{} {

	key := s.hashcodeFunction(value)

	values := s.data[key]

	if s.data[key] != nil {
		for i := 0; i < len(values); i++ {
			if s.equalsFunction(value, values[i]) {
				return values[i]
			}
		}

		s.data[key] = append(s.data[key], value)
		return value
	}

	v := make([]interface{}, 1, 10)
	v[0] = value
	s.data[key] = v

	return value
}

func (s *Set) contains(value interface{}) bool {

	key := s.hashcodeFunction(value)

	values := s.data[key]

	if s.data[key] != nil {
		for i := 0; i < len(values); i++ {
			if s.equalsFunction(value, values[i]) {
				return true
			}
		}
	}
	return false
}

func (s *Set) values() []interface{} {
	var l []interface{}

	for _, v := range s.data {
		l = append(l, v...)
	}

	return l
}

func (s *Set) String() string {
	r := ""

	for _, av := range s.data {
		for _, v := range av {
			r += fmt.Sprint(v)
		}
	}

	return r
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

// The following routines were lifted from bits.rotate* available in Go 1.9.

const uintSize = 32 << (^uint(0) >> 32 & 1) // 32 or 64

// rotateLeft returns the value of x rotated left by (k mod UintSize) bits.
// To rotate x right by k bits, call RotateLeft(x, -k).
func rotateLeft(x uint, k int) uint {
	if uintSize == 32 {
		return uint(rotateLeft32(uint32(x), k))
	}
	return uint(rotateLeft64(uint64(x), k))
}

// rotateLeft32 returns the value of x rotated left by (k mod 32) bits.
func rotateLeft32(x uint32, k int) uint32 {
	const n = 32
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}

// rotateLeft64 returns the value of x rotated left by (k mod 64) bits.
func rotateLeft64(x uint64, k int) uint64 {
	const n = 64
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}


// murmur hash
const (
	c1_32 uint = 0xCC9E2D51
	c2_32 uint = 0x1B873593
	n1_32 uint = 0xE6546B64
)

func murmurInit(seed int) int {
	return seed
}

func murmurUpdate(h1 int, k1 int) int {
	var k1u uint
	k1u = uint(k1) * c1_32
	k1u = rotateLeft(k1u, 15)
	k1u *= c2_32

	var h1u = uint(h1) ^ k1u
	k1u = rotateLeft(k1u, 13)
	h1u = h1u*5 + 0xe6546b64
	return int(h1u)
}

func murmurFinish(h1 int, numberOfWords int) int {
	var h1u uint = uint(h1)
	h1u ^= uint(numberOfWords * 4)
	h1u ^= h1u >> 16
	h1u *= uint(0x85ebca6b)
	h1u ^= h1u >> 13
	h1u *= 0xc2b2ae35
	h1u ^= h1u >> 16

	return int(h1u)
}
