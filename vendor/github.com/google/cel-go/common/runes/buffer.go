// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package runes provides interfaces and utilities for working with runes.
package runes

import (
	"strings"
	"unicode/utf8"
)

// Buffer is an interface for accessing a contiguous array of code points.
type Buffer interface {
	Get(i int) rune
	Slice(i, j int) string
	Len() int
}

type emptyBuffer struct{}

func (e *emptyBuffer) Get(i int) rune {
	panic("slice index out of bounds")
}

func (e *emptyBuffer) Slice(i, j int) string {
	if i != 0 || i != j {
		panic("slice index out of bounds")
	}
	return ""
}

func (e *emptyBuffer) Len() int {
	return 0
}

var _ Buffer = &emptyBuffer{}

// asciiBuffer is an implementation for an array of code points that contain code points only from
// the ASCII character set.
type asciiBuffer struct {
	arr []byte
}

func (a *asciiBuffer) Get(i int) rune {
	return rune(uint32(a.arr[i]))
}

func (a *asciiBuffer) Slice(i, j int) string {
	return string(a.arr[i:j])
}

func (a *asciiBuffer) Len() int {
	return len(a.arr)
}

var _ Buffer = &asciiBuffer{}

// basicBuffer is an implementation for an array of code points that contain code points from both
// the Latin-1 character set and Basic Multilingual Plane.
type basicBuffer struct {
	arr []uint16
}

func (b *basicBuffer) Get(i int) rune {
	return rune(uint32(b.arr[i]))
}

func (b *basicBuffer) Slice(i, j int) string {
	var str strings.Builder
	str.Grow((j - i) * 3) // Worst case encoding size for 0xffff is 3.
	for ; i < j; i++ {
		str.WriteRune(rune(uint32(b.arr[i])))
	}
	return str.String()
}

func (b *basicBuffer) Len() int {
	return len(b.arr)
}

var _ Buffer = &basicBuffer{}

// supplementalBuffer is an implementation for an array of code points that contain code points from
// the Latin-1 character set, Basic Multilingual Plane, or the Supplemental Multilingual Plane.
type supplementalBuffer struct {
	arr []rune
}

func (s *supplementalBuffer) Get(i int) rune {
	return rune(uint32(s.arr[i]))
}

func (s *supplementalBuffer) Slice(i, j int) string {
	return string(s.arr[i:j])
}

func (s *supplementalBuffer) Len() int {
	return len(s.arr)
}

var _ Buffer = &supplementalBuffer{}

var nilBuffer = &emptyBuffer{}

// NewBuffer returns an efficient implementation of Buffer for the given text based on the ranges of
// the encoded code points contained within.
//
// Code points are represented as an array of byte, uint16, or rune. This approach ensures that
// each index represents a code point by itself without needing to use an array of rune. At first
// we assume all code points are less than or equal to '\u007f'. If this holds true, the
// underlying storage is a byte array containing only ASCII characters. If we encountered a code
// point above this range but less than or equal to '\uffff' we allocate a uint16 array, copy the
// elements of previous byte array to the uint16 array, and continue. If this holds true, the
// underlying storage is a uint16 array containing only Unicode characters in the Basic Multilingual
// Plane. If we encounter a code point above '\uffff' we allocate an rune array, copy the previous
// elements of the byte or uint16 array, and continue. The underlying storage is an rune array
// containing any Unicode character.
func NewBuffer(data string) Buffer {
	if len(data) == 0 {
		return nilBuffer
	}
	var (
		idx   = 0
		buf8  = make([]byte, 0, len(data))
		buf16 []uint16
		buf32 []rune
	)
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		if r < utf8.RuneSelf {
			buf8 = append(buf8, byte(r))
			continue
		}
		if r <= 0xffff {
			buf16 = make([]uint16, len(buf8), len(data))
			for i, v := range buf8 {
				buf16[i] = uint16(v)
			}
			buf8 = nil
			buf16 = append(buf16, uint16(r))
			goto copy16
		}
		buf32 = make([]rune, len(buf8), len(data))
		for i, v := range buf8 {
			buf32[i] = rune(uint32(v))
		}
		buf8 = nil
		buf32 = append(buf32, r)
		goto copy32
	}
	return &asciiBuffer{
		arr: buf8,
	}
copy16:
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		if r <= 0xffff {
			buf16 = append(buf16, uint16(r))
			continue
		}
		buf32 = make([]rune, len(buf16), len(data))
		for i, v := range buf16 {
			buf32[i] = rune(uint32(v))
		}
		buf16 = nil
		buf32 = append(buf32, r)
		goto copy32
	}
	return &basicBuffer{
		arr: buf16,
	}
copy32:
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		buf32 = append(buf32, r)
	}
	return &supplementalBuffer{
		arr: buf32,
	}
}
