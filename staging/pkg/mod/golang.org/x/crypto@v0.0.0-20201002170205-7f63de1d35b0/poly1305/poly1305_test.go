// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poly1305

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"testing"
	"unsafe"
)

var stressFlag = flag.Bool("stress", false, "run slow stress tests")

type test struct {
	in    string
	key   string
	tag   string
	state string
}

func (t *test) Input() []byte {
	in, err := hex.DecodeString(t.in)
	if err != nil {
		panic(err)
	}
	return in
}

func (t *test) Key() [32]byte {
	buf, err := hex.DecodeString(t.key)
	if err != nil {
		panic(err)
	}
	var key [32]byte
	copy(key[:], buf[:32])
	return key
}

func (t *test) Tag() [16]byte {
	buf, err := hex.DecodeString(t.tag)
	if err != nil {
		panic(err)
	}
	var tag [16]byte
	copy(tag[:], buf[:16])
	return tag
}

func (t *test) InitialState() [3]uint64 {
	// state is hex encoded in big-endian byte order
	if t.state == "" {
		return [3]uint64{0, 0, 0}
	}
	buf, err := hex.DecodeString(t.state)
	if err != nil {
		panic(err)
	}
	if len(buf) != 3*8 {
		panic("incorrect state length")
	}
	return [3]uint64{
		binary.BigEndian.Uint64(buf[16:24]),
		binary.BigEndian.Uint64(buf[8:16]),
		binary.BigEndian.Uint64(buf[0:8]),
	}
}

func testSum(t *testing.T, unaligned bool, sumImpl func(tag *[TagSize]byte, msg []byte, key *[32]byte)) {
	var tag [16]byte
	for i, v := range testData {
		// cannot set initial state before calling sum, so skip those tests
		if v.InitialState() != [3]uint64{0, 0, 0} {
			continue
		}

		in := v.Input()
		if unaligned {
			in = unalignBytes(in)
		}
		key := v.Key()
		sumImpl(&tag, in, &key)
		if tag != v.Tag() {
			t.Errorf("%d: expected %x, got %x", i, v.Tag(), tag[:])
		}
		if !Verify(&tag, in, &key) {
			t.Errorf("%d: tag didn't verify", i)
		}
		// If the key is zero, the tag will always be zero, independent of the input.
		if len(in) > 0 && key != [32]byte{} {
			in[0] ^= 0xff
			if Verify(&tag, in, &key) {
				t.Errorf("%d: tag verified after altering the input", i)
			}
			in[0] ^= 0xff
		}
		// If the input is empty, the tag only depends on the second half of the key.
		if len(in) > 0 {
			key[0] ^= 0xff
			if Verify(&tag, in, &key) {
				t.Errorf("%d: tag verified after altering the key", i)
			}
			key[0] ^= 0xff
		}
		tag[0] ^= 0xff
		if Verify(&tag, in, &key) {
			t.Errorf("%d: tag verified after altering the tag", i)
		}
		tag[0] ^= 0xff
	}
}

func TestBurnin(t *testing.T) {
	// This test can be used to sanity-check significant changes. It can
	// take about many minutes to run, even on fast machines. It's disabled
	// by default.
	if !*stressFlag {
		t.Skip("skipping without -stress")
	}

	var key [32]byte
	var input [25]byte
	var output [16]byte

	for i := range key {
		key[i] = 1
	}
	for i := range input {
		input[i] = 2
	}

	for i := uint64(0); i < 1e10; i++ {
		Sum(&output, input[:], &key)
		copy(key[0:], output[:])
		copy(key[16:], output[:])
		copy(input[:], output[:])
		copy(input[16:], output[:])
	}

	const expected = "5e3b866aea0b636d240c83c428f84bfa"
	if got := hex.EncodeToString(output[:]); got != expected {
		t.Errorf("expected %s, got %s", expected, got)
	}
}

func TestSum(t *testing.T)                 { testSum(t, false, Sum) }
func TestSumUnaligned(t *testing.T)        { testSum(t, true, Sum) }
func TestSumGeneric(t *testing.T)          { testSum(t, false, sumGeneric) }
func TestSumGenericUnaligned(t *testing.T) { testSum(t, true, sumGeneric) }

func TestWriteGeneric(t *testing.T)          { testWriteGeneric(t, false) }
func TestWriteGenericUnaligned(t *testing.T) { testWriteGeneric(t, true) }
func TestWrite(t *testing.T)                 { testWrite(t, false) }
func TestWriteUnaligned(t *testing.T)        { testWrite(t, true) }

func testWriteGeneric(t *testing.T, unaligned bool) {
	for i, v := range testData {
		key := v.Key()
		input := v.Input()
		var out [16]byte

		if unaligned {
			input = unalignBytes(input)
		}
		h := newMACGeneric(&key)
		if s := v.InitialState(); s != [3]uint64{0, 0, 0} {
			h.macState.h = s
		}
		n, err := h.Write(input[:len(input)/3])
		if err != nil || n != len(input[:len(input)/3]) {
			t.Errorf("#%d: unexpected Write results: n = %d, err = %v", i, n, err)
		}
		n, err = h.Write(input[len(input)/3:])
		if err != nil || n != len(input[len(input)/3:]) {
			t.Errorf("#%d: unexpected Write results: n = %d, err = %v", i, n, err)
		}
		h.Sum(&out)
		if tag := v.Tag(); out != tag {
			t.Errorf("%d: expected %x, got %x", i, tag[:], out[:])
		}
	}
}

func testWrite(t *testing.T, unaligned bool) {
	for i, v := range testData {
		key := v.Key()
		input := v.Input()
		var out [16]byte

		if unaligned {
			input = unalignBytes(input)
		}
		h := New(&key)
		if s := v.InitialState(); s != [3]uint64{0, 0, 0} {
			h.macState.h = s
		}
		n, err := h.Write(input[:len(input)/3])
		if err != nil || n != len(input[:len(input)/3]) {
			t.Errorf("#%d: unexpected Write results: n = %d, err = %v", i, n, err)
		}
		n, err = h.Write(input[len(input)/3:])
		if err != nil || n != len(input[len(input)/3:]) {
			t.Errorf("#%d: unexpected Write results: n = %d, err = %v", i, n, err)
		}
		h.Sum(out[:0])
		tag := v.Tag()
		if out != tag {
			t.Errorf("%d: expected %x, got %x", i, tag[:], out[:])
		}
		if !h.Verify(tag[:]) {
			t.Errorf("%d: Verify failed", i)
		}
		tag[0] ^= 0xff
		if h.Verify(tag[:]) {
			t.Errorf("%d: Verify succeeded after modifying the tag", i)
		}
	}
}

func benchmarkSum(b *testing.B, size int, unaligned bool) {
	var out [16]byte
	var key [32]byte
	in := make([]byte, size)
	if unaligned {
		in = unalignBytes(in)
	}
	rand.Read(in)
	b.SetBytes(int64(len(in)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sum(&out, in, &key)
	}
}

func benchmarkWrite(b *testing.B, size int, unaligned bool) {
	var key [32]byte
	h := New(&key)
	in := make([]byte, size)
	if unaligned {
		in = unalignBytes(in)
	}
	rand.Read(in)
	b.SetBytes(int64(len(in)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Write(in)
	}
}

func Benchmark64(b *testing.B)          { benchmarkSum(b, 64, false) }
func Benchmark1K(b *testing.B)          { benchmarkSum(b, 1024, false) }
func Benchmark2M(b *testing.B)          { benchmarkSum(b, 2*1024*1024, false) }
func Benchmark64Unaligned(b *testing.B) { benchmarkSum(b, 64, true) }
func Benchmark1KUnaligned(b *testing.B) { benchmarkSum(b, 1024, true) }
func Benchmark2MUnaligned(b *testing.B) { benchmarkSum(b, 2*1024*1024, true) }

func BenchmarkWrite64(b *testing.B)          { benchmarkWrite(b, 64, false) }
func BenchmarkWrite1K(b *testing.B)          { benchmarkWrite(b, 1024, false) }
func BenchmarkWrite2M(b *testing.B)          { benchmarkWrite(b, 2*1024*1024, false) }
func BenchmarkWrite64Unaligned(b *testing.B) { benchmarkWrite(b, 64, true) }
func BenchmarkWrite1KUnaligned(b *testing.B) { benchmarkWrite(b, 1024, true) }
func BenchmarkWrite2MUnaligned(b *testing.B) { benchmarkWrite(b, 2*1024*1024, true) }

func unalignBytes(in []byte) []byte {
	out := make([]byte, len(in)+1)
	if uintptr(unsafe.Pointer(&out[0]))&(unsafe.Alignof(uint32(0))-1) == 0 {
		out = out[1:]
	} else {
		out = out[:len(in)]
	}
	copy(out, in)
	return out
}
