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
	"fmt"
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

// SizeLimitError indicates that the input exceeded the configured code point limit.
type SizeLimitError struct {
	Size  int
	Limit int
}

func (e *SizeLimitError) Error() string {
	return fmt.Sprintf("expression code point size exceeds limit: size: %d, limit %d", e.Size, e.Limit)
}

// NewBuffer returns an efficient implementation of Buffer for the given text based on the ranges of
// the encoded code points contained within.
func NewBuffer(data string) Buffer {
	buf, _, _ := newBufferWithLimit(data, false, -1)
	return buf
}

// NewBufferAndLineOffsets returns an efficient implementation of Buffer for the given text based on
// the ranges of the encoded code points contained within, as well as returning the line offsets.
func NewBufferAndLineOffsets(data string) (Buffer, []int32) {
	buf, offs, _ := newBufferWithLimit(data, true, -1)
	return buf, offs
}

// NewBufferAndLineOffsetsWithLimit returns an efficient implementation of Buffer for the given text
// and enforces a code point limit while constructing the buffer.
func NewBufferAndLineOffsetsWithLimit(data string, limit int) (Buffer, []int32, error) {
	if limit < 0 || len(data) <= limit {
		return newBufferWithLimit(data, true, -1)
	}
	return newBufferWithLimit(data, true, limit)
}

func countRemainingCodePoints(data string, idx int, count int) int {
	for idx < len(data) {
		_, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		count++
	}
	return count
}

func newBufferWithLimit(data string, lines bool, limit int) (Buffer, []int32, error) {
	if len(data) == 0 {
		return nilBuffer, []int32{0}, nil
	}
	if limit >= 0 && len(data) > limit {
		size := countRemainingCodePoints(data, 0, 0)
		if size > limit {
			return nil, nil, &SizeLimitError{
				Size:  size,
				Limit: limit,
			}
		}
	}

	// The resulting buffers store one element per code point, so the worst case
	// element count never exceeds len(data).
	var (
		idx         = 0
		off   int32 = 0
		buf8        = make([]byte, 0, len(data))
		buf16 []uint16
		buf32 []rune
		offs  []int32
	)
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		if lines && r == '\n' {
			offs = append(offs, off+1)
		}
		if r < utf8.RuneSelf {
			buf8 = append(buf8, byte(r))
			off++
			continue
		}
		if r <= 0xffff {
			buf16 = make([]uint16, len(buf8), len(data))
			for i, v := range buf8 {
				buf16[i] = uint16(v)
			}
			buf8 = nil
			buf16 = append(buf16, uint16(r))
			off++
			goto copy16
		}
		buf32 = make([]rune, len(buf8), len(data))
		for i, v := range buf8 {
			buf32[i] = rune(uint32(v))
		}
		buf8 = nil
		buf32 = append(buf32, r)
		off++
		goto copy32
	}
	if lines {
		offs = append(offs, off+1)
	}
	return &asciiBuffer{
		arr: buf8,
	}, offs, nil

copy16:
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		if lines && r == '\n' {
			offs = append(offs, off+1)
		}
		if r <= 0xffff {
			buf16 = append(buf16, uint16(r))
			off++
			continue
		}
		buf32 = make([]rune, len(buf16), len(data))
		for i, v := range buf16 {
			buf32[i] = rune(uint32(v))
		}
		buf16 = nil
		buf32 = append(buf32, r)
		off++
		goto copy32
	}
	if lines {
		offs = append(offs, off+1)
	}
	return &basicBuffer{
		arr: buf16,
	}, offs, nil

copy32:
	for idx < len(data) {
		r, s := utf8.DecodeRuneInString(data[idx:])
		idx += s
		if lines && r == '\n' {
			offs = append(offs, off+1)
		}
		buf32 = append(buf32, r)
		off++
	}
	if lines {
		offs = append(offs, off+1)
	}
	return &supplementalBuffer{
		arr: buf32,
	}, offs, nil
}
