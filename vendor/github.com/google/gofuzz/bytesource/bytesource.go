/*
Copyright 2014 Google Inc. All rights reserved.

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

// Package bytesource provides a rand.Source64 that is determined by a slice of bytes.
package bytesource

import (
	"bytes"
	"encoding/binary"
	"io"
	"math/rand"
)

// ByteSource implements rand.Source64 determined by a slice of bytes. The random numbers are
// generated from each 8 bytes in the slice, until the last bytes are consumed, from which a
// fallback pseudo random source is created in case more random numbers are required.
// It also exposes a `bytes.Reader` API, which lets callers consume the bytes directly.
type ByteSource struct {
	*bytes.Reader
	fallback rand.Source
}

// New returns a new ByteSource from a given slice of bytes.
func New(input []byte) *ByteSource {
	s := &ByteSource{
		Reader:   bytes.NewReader(input),
		fallback: rand.NewSource(0),
	}
	if len(input) > 0 {
		s.fallback = rand.NewSource(int64(s.consumeUint64()))
	}
	return s
}

func (s *ByteSource) Uint64() uint64 {
	// Return from input if it was not exhausted.
	if s.Len() > 0 {
		return s.consumeUint64()
	}

	// Input was exhausted, return random number from fallback (in this case fallback should not be
	// nil). Try first having a Uint64 output (Should work in current rand implementation),
	// otherwise return a conversion of Int63.
	if s64, ok := s.fallback.(rand.Source64); ok {
		return s64.Uint64()
	}
	return uint64(s.fallback.Int63())
}

func (s *ByteSource) Int63() int64 {
	return int64(s.Uint64() >> 1)
}

func (s *ByteSource) Seed(seed int64) {
	s.fallback = rand.NewSource(seed)
	s.Reader = bytes.NewReader(nil)
}

// consumeUint64 reads 8 bytes from the input and convert them to a uint64. It assumes that the the
// bytes reader is not empty.
func (s *ByteSource) consumeUint64() uint64 {
	var bytes [8]byte
	_, err := s.Read(bytes[:])
	if err != nil && err != io.EOF {
		panic("failed reading source") // Should not happen.
	}
	return binary.BigEndian.Uint64(bytes[:])
}
