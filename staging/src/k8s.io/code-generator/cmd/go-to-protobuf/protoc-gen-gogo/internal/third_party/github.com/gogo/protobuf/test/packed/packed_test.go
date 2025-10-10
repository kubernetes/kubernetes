// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package packed

import (
	"bytes"
	"fmt"
	math_rand "math/rand"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/gogo/protobuf/proto"
)

func BenchmarkVarintIssue436withCount(b *testing.B) {
	var arraySizes = []struct {
		name  string
		value int64
	}{
		{"2^0_ints", 1 << 0},
		{"2^1_ints", 1 << 1},
		{"2^2_ints", 1 << 2},
		{"2^3_ints", 1 << 3},
		{"2^4_ints", 1 << 4},
		{"2^8_ints", 1 << 8},
		{"2^16_ints", 1 << 16},
		{"2^20_ints", 1 << 20},
		{"2^24_ints", 1 << 24},
	}

	var varintSizes = []struct {
		name  string
		value int64
	}{
		{"max_int 2^7-4", 1<<7 - 4},
		{"max_int 2^15-4", 1<<15 - 4},
		{"max_int 2^31-4", 1<<31 - 4},
		{"max_int 2^63-4", 1<<63 - 4},
	}

	for _, arraySize := range arraySizes {
		for _, varintSize := range varintSizes {
			seed := time.Now().UnixNano()
			rng := math_rand.New(math_rand.NewSource(seed))
			buf := make([]int64, arraySize.value)
			for j := range buf {
				buf[j] = rng.Int63n(varintSize.value)
			}

			b.Run(arraySize.name+", "+varintSize.name, func(b *testing.B) {
				msg := &NinRepNative{
					Field8: buf,
				}

				data, err := proto.Marshal(msg)
				if err != nil {
					b.Fatal(err)
				}

				normalmsg := &NinRepNative{}

				for i := 0; i < b.N; i++ {
					err = proto.Unmarshal(data, normalmsg)
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestVarintIssue436(t *testing.T) {
	n := 1 << 22 // Makes for 32 MiB

	m := &runtime.MemStats{}

	msgNormal := &NinRepNative{
		Field8: make([]int64, n),
	}
	dataNormal, err := proto.Marshal(msgNormal)
	if err != nil {
		t.Fatal(err)
	}

	normalmsg := &NinRepNative{}
	runtime.ReadMemStats(m)
	beforeNormal := m.TotalAlloc
	err = proto.Unmarshal(dataNormal, normalmsg)
	runtime.ReadMemStats(m)
	afterNormal := m.TotalAlloc
	if err != nil {
		t.Fatal(err)
	}

	msgPacked := &NinRepPackedNative{
		Field8: make([]int64, n),
	}
	dataPacked, err := proto.Marshal(msgPacked)
	if err != nil {
		t.Fatal(err)
	}

	packedmsg := &NinRepPackedNative{}
	runtime.ReadMemStats(m)
	beforePacked := m.TotalAlloc
	err = proto.Unmarshal(dataPacked, packedmsg)
	runtime.ReadMemStats(m)
	afterPacked := m.TotalAlloc
	if err != nil {
		t.Fatal(err)
	}

	totalNormal := afterNormal - beforeNormal
	totalPacked := afterPacked - beforePacked
	usedRatio := float64(totalPacked) / float64(totalNormal)
	if usedRatio > 0.5 {
		t.Fatalf("unmarshaling packed msg allocated too much memory:\nnormal:\t\t%d bytes\npacked:\t\t%d bytes\nused ratio:\t%.2f%%", totalNormal, totalPacked, usedRatio*100)
	}
}

func TestIssue436(t *testing.T) {
	n := 1 << 22 // Makes for 32 MiB

	m := &runtime.MemStats{}

	msgNormal := &NinRepNative{
		Field1: make([]float64, n),
	}
	dataNormal, err := proto.Marshal(msgNormal)
	if err != nil {
		t.Fatal(err)
	}

	normalmsg := &NinRepNative{}
	runtime.ReadMemStats(m)
	beforeNormal := m.TotalAlloc
	err = proto.Unmarshal(dataNormal, normalmsg)
	runtime.ReadMemStats(m)
	afterNormal := m.TotalAlloc
	if err != nil {
		t.Fatal(err)
	}

	msgPacked := &NinRepPackedNative{
		Field1: make([]float64, n),
	}
	dataPacked, err := proto.Marshal(msgPacked)
	if err != nil {
		t.Fatal(err)
	}

	packedmsg := &NinRepPackedNative{}
	runtime.ReadMemStats(m)
	beforePacked := m.TotalAlloc
	err = proto.Unmarshal(dataPacked, packedmsg)
	runtime.ReadMemStats(m)
	afterPacked := m.TotalAlloc
	if err != nil {
		t.Fatal(err)
	}

	totalNormal := afterNormal - beforeNormal
	totalPacked := afterPacked - beforePacked
	usedRatio := float64(totalPacked) / float64(totalNormal)
	if usedRatio > 0.5 {
		t.Fatalf("unmarshaling packed msg allocated too much memory:\nnormal:\t\t%d bytes\npacked:\t\t%d bytes\nused ratio:\t%.2f%%", totalNormal, totalPacked, usedRatio*100)
	}
}

/*
https://github.com/gogo/protobuf/issues/503
preallocation was estimating length over all data vs. just that for the given field.
*/
func TestTestPackedPreallocation(t *testing.T) {
	n1 := 900
	n2 := 700

	msgPacked := &NinRepPackedNative{
		Field3: make([]int32, n1),
		Field4: make([]int64, n2),
	}

	dataPacked, err := proto.Marshal(msgPacked)
	if err != nil {
		t.Fatal(err)
	}

	packedmsg := &NinRepPackedNative{}
	err = proto.Unmarshal(dataPacked, packedmsg)
	if err != nil {
		t.Fatal(err)
	}

	if v := len(packedmsg.Field3); v != n1 {
		t.Errorf("Field3 incorrect len: %v != %v", v, n1)
	}

	if v := len(packedmsg.Field4); v != n2 {
		t.Errorf("Field4 incorrect len: %v != %v", v, n2)
	}

	if v := cap(packedmsg.Field3); v != n1 {
		t.Errorf("Field3 incorrect cap: %v != %v", v, n1)
	}

	if v := cap(packedmsg.Field4); v != n2 {
		t.Errorf("Field4 incorrect cap: %v != %v", v, n2)
	}
}

/*
https://github.com/gogo/protobuf/issues/detail?id=21
https://developers.google.com/protocol-buffers/docs/proto#options
In 2.3.0 and later, this change is safe, as parsers for packable fields will always accept both formats,
*/
func TestSafeIssue21(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	msg1 := NewPopulatedNinRepNative(popr, true)
	data1, err := proto.Marshal(msg1)
	if err != nil {
		t.Fatal(err)
	}
	packedmsg := &NinRepPackedNative{}
	err = proto.Unmarshal(data1, packedmsg)
	if err != nil {
		t.Fatal(err)
	}
	if len(packedmsg.XXX_unrecognized) != 0 {
		t.Fatalf("packed msg unmarshaled unrecognized fields, even though there aren't any")
	}
	if err := VerboseEqual(msg1, packedmsg); err != nil {
		t.Fatalf("%v", err)
	}
}

func TestUnsafeIssue21(t *testing.T) {
	var bigendian uint32 = 0x01020304
	if *(*byte)(unsafe.Pointer(&bigendian)) == 1 {
		t.Skip("unsafe does not work on big endian architectures")
	}
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	msg1 := NewPopulatedNinRepNativeUnsafe(popr, true)
	data1, err := proto.Marshal(msg1)
	if err != nil {
		t.Fatal(err)
	}
	packedmsg := &NinRepPackedNativeUnsafe{}
	err = proto.Unmarshal(data1, packedmsg)
	if err != nil {
		t.Fatal(err)
	}
	if len(packedmsg.XXX_unrecognized) != 0 {
		t.Fatalf("packed msg unmarshaled unrecognized fields, even though there aren't any")
	}
	if err := VerboseEqualUnsafe(msg1, packedmsg); err != nil {
		t.Fatalf("%v", err)
	}
}

func VerboseEqual(this *NinRepNative, that *NinRepPackedNative) error {
	if that == nil {
		if this == nil {
			return nil
		}
		return fmt.Errorf("that == nil && this != nil")
	} else if this == nil {
		return fmt.Errorf("that != nil && this == nil")
	}

	if len(this.Field1) != len(that.Field1) {
		return fmt.Errorf("Field1 this(%v) Not Equal that(%v)", len(this.Field1), len(that.Field1))
	}
	for i := range this.Field1 {
		if this.Field1[i] != that.Field1[i] {
			return fmt.Errorf("Field1 this[%v](%v) Not Equal that[%v](%v)", i, this.Field1[i], i, that.Field1[i])
		}
	}
	if len(this.Field2) != len(that.Field2) {
		return fmt.Errorf("Field2 this(%v) Not Equal that(%v)", len(this.Field2), len(that.Field2))
	}
	for i := range this.Field2 {
		if this.Field2[i] != that.Field2[i] {
			return fmt.Errorf("Field2 this[%v](%v) Not Equal that[%v](%v)", i, this.Field2[i], i, that.Field2[i])
		}
	}
	if len(this.Field3) != len(that.Field3) {
		return fmt.Errorf("Field3 this(%v) Not Equal that(%v)", len(this.Field3), len(that.Field3))
	}
	for i := range this.Field3 {
		if this.Field3[i] != that.Field3[i] {
			return fmt.Errorf("Field3 this[%v](%v) Not Equal that[%v](%v)", i, this.Field3[i], i, that.Field3[i])
		}
	}
	if len(this.Field4) != len(that.Field4) {
		return fmt.Errorf("Field4 this(%v) Not Equal that(%v)", len(this.Field4), len(that.Field4))
	}
	for i := range this.Field4 {
		if this.Field4[i] != that.Field4[i] {
			return fmt.Errorf("Field4 this[%v](%v) Not Equal that[%v](%v)", i, this.Field4[i], i, that.Field4[i])
		}
	}
	if len(this.Field5) != len(that.Field5) {
		return fmt.Errorf("Field5 this(%v) Not Equal that(%v)", len(this.Field5), len(that.Field5))
	}
	for i := range this.Field5 {
		if this.Field5[i] != that.Field5[i] {
			return fmt.Errorf("Field5 this[%v](%v) Not Equal that[%v](%v)", i, this.Field5[i], i, that.Field5[i])
		}
	}
	if len(this.Field6) != len(that.Field6) {
		return fmt.Errorf("Field6 this(%v) Not Equal that(%v)", len(this.Field6), len(that.Field6))
	}
	for i := range this.Field6 {
		if this.Field6[i] != that.Field6[i] {
			return fmt.Errorf("Field6 this[%v](%v) Not Equal that[%v](%v)", i, this.Field6[i], i, that.Field6[i])
		}
	}
	if len(this.Field7) != len(that.Field7) {
		return fmt.Errorf("Field7 this(%v) Not Equal that(%v)", len(this.Field7), len(that.Field7))
	}
	for i := range this.Field7 {
		if this.Field7[i] != that.Field7[i] {
			return fmt.Errorf("Field7 this[%v](%v) Not Equal that[%v](%v)", i, this.Field7[i], i, that.Field7[i])
		}
	}
	if len(this.Field8) != len(that.Field8) {
		return fmt.Errorf("Field8 this(%v) Not Equal that(%v)", len(this.Field8), len(that.Field8))
	}
	for i := range this.Field8 {
		if this.Field8[i] != that.Field8[i] {
			return fmt.Errorf("Field8 this[%v](%v) Not Equal that[%v](%v)", i, this.Field8[i], i, that.Field8[i])
		}
	}
	if len(this.Field9) != len(that.Field9) {
		return fmt.Errorf("Field9 this(%v) Not Equal that(%v)", len(this.Field9), len(that.Field9))
	}
	for i := range this.Field9 {
		if this.Field9[i] != that.Field9[i] {
			return fmt.Errorf("Field9 this[%v](%v) Not Equal that[%v](%v)", i, this.Field9[i], i, that.Field9[i])
		}
	}
	if len(this.Field10) != len(that.Field10) {
		return fmt.Errorf("Field10 this(%v) Not Equal that(%v)", len(this.Field10), len(that.Field10))
	}
	for i := range this.Field10 {
		if this.Field10[i] != that.Field10[i] {
			return fmt.Errorf("Field10 this[%v](%v) Not Equal that[%v](%v)", i, this.Field10[i], i, that.Field10[i])
		}
	}
	if len(this.Field11) != len(that.Field11) {
		return fmt.Errorf("Field11 this(%v) Not Equal that(%v)", len(this.Field11), len(that.Field11))
	}
	for i := range this.Field11 {
		if this.Field11[i] != that.Field11[i] {
			return fmt.Errorf("Field11 this[%v](%v) Not Equal that[%v](%v)", i, this.Field11[i], i, that.Field11[i])
		}
	}
	if len(this.Field12) != len(that.Field12) {
		return fmt.Errorf("Field12 this(%v) Not Equal that(%v)", len(this.Field12), len(that.Field12))
	}
	for i := range this.Field12 {
		if this.Field12[i] != that.Field12[i] {
			return fmt.Errorf("Field12 this[%v](%v) Not Equal that[%v](%v)", i, this.Field12[i], i, that.Field12[i])
		}
	}
	if len(this.Field13) != len(that.Field13) {
		return fmt.Errorf("Field13 this(%v) Not Equal that(%v)", len(this.Field13), len(that.Field13))
	}
	for i := range this.Field13 {
		if this.Field13[i] != that.Field13[i] {
			return fmt.Errorf("Field13 this[%v](%v) Not Equal that[%v](%v)", i, this.Field13[i], i, that.Field13[i])
		}
	}
	if !bytes.Equal(this.XXX_unrecognized, that.XXX_unrecognized) {
		return fmt.Errorf("XXX_unrecognized this(%v) Not Equal that(%v)", this.XXX_unrecognized, that.XXX_unrecognized)
	}
	return nil
}

func VerboseEqualUnsafe(this *NinRepNativeUnsafe, that *NinRepPackedNativeUnsafe) error {
	if that == nil {
		if this == nil {
			return nil
		}
		return fmt.Errorf("that == nil && this != nil")
	} else if this == nil {
		return fmt.Errorf("that != nil && this == nil")
	}

	if len(this.Field1) != len(that.Field1) {
		return fmt.Errorf("Field1 this(%v) Not Equal that(%v)", len(this.Field1), len(that.Field1))
	}
	for i := range this.Field1 {
		if this.Field1[i] != that.Field1[i] {
			return fmt.Errorf("Field1 this[%v](%v) Not Equal that[%v](%v)", i, this.Field1[i], i, that.Field1[i])
		}
	}
	if len(this.Field2) != len(that.Field2) {
		return fmt.Errorf("Field2 this(%v) Not Equal that(%v)", len(this.Field2), len(that.Field2))
	}
	for i := range this.Field2 {
		if this.Field2[i] != that.Field2[i] {
			return fmt.Errorf("Field2 this[%v](%v) Not Equal that[%v](%v)", i, this.Field2[i], i, that.Field2[i])
		}
	}
	if len(this.Field3) != len(that.Field3) {
		return fmt.Errorf("Field3 this(%v) Not Equal that(%v)", len(this.Field3), len(that.Field3))
	}
	for i := range this.Field3 {
		if this.Field3[i] != that.Field3[i] {
			return fmt.Errorf("Field3 this[%v](%v) Not Equal that[%v](%v)", i, this.Field3[i], i, that.Field3[i])
		}
	}
	if len(this.Field4) != len(that.Field4) {
		return fmt.Errorf("Field4 this(%v) Not Equal that(%v)", len(this.Field4), len(that.Field4))
	}
	for i := range this.Field4 {
		if this.Field4[i] != that.Field4[i] {
			return fmt.Errorf("Field4 this[%v](%v) Not Equal that[%v](%v)", i, this.Field4[i], i, that.Field4[i])
		}
	}
	if len(this.Field5) != len(that.Field5) {
		return fmt.Errorf("Field5 this(%v) Not Equal that(%v)", len(this.Field5), len(that.Field5))
	}
	for i := range this.Field5 {
		if this.Field5[i] != that.Field5[i] {
			return fmt.Errorf("Field5 this[%v](%v) Not Equal that[%v](%v)", i, this.Field5[i], i, that.Field5[i])
		}
	}
	if len(this.Field6) != len(that.Field6) {
		return fmt.Errorf("Field6 this(%v) Not Equal that(%v)", len(this.Field6), len(that.Field6))
	}
	for i := range this.Field6 {
		if this.Field6[i] != that.Field6[i] {
			return fmt.Errorf("Field6 this[%v](%v) Not Equal that[%v](%v)", i, this.Field6[i], i, that.Field6[i])
		}
	}
	if len(this.Field7) != len(that.Field7) {
		return fmt.Errorf("Field7 this(%v) Not Equal that(%v)", len(this.Field7), len(that.Field7))
	}
	for i := range this.Field7 {
		if this.Field7[i] != that.Field7[i] {
			return fmt.Errorf("Field7 this[%v](%v) Not Equal that[%v](%v)", i, this.Field7[i], i, that.Field7[i])
		}
	}
	if len(this.Field8) != len(that.Field8) {
		return fmt.Errorf("Field8 this(%v) Not Equal that(%v)", len(this.Field8), len(that.Field8))
	}
	for i := range this.Field8 {
		if this.Field8[i] != that.Field8[i] {
			return fmt.Errorf("Field8 this[%v](%v) Not Equal that[%v](%v)", i, this.Field8[i], i, that.Field8[i])
		}
	}
	if len(this.Field9) != len(that.Field9) {
		return fmt.Errorf("Field9 this(%v) Not Equal that(%v)", len(this.Field9), len(that.Field9))
	}
	for i := range this.Field9 {
		if this.Field9[i] != that.Field9[i] {
			return fmt.Errorf("Field9 this[%v](%v) Not Equal that[%v](%v)", i, this.Field9[i], i, that.Field9[i])
		}
	}
	if len(this.Field10) != len(that.Field10) {
		return fmt.Errorf("Field10 this(%v) Not Equal that(%v)", len(this.Field10), len(that.Field10))
	}
	for i := range this.Field10 {
		if this.Field10[i] != that.Field10[i] {
			return fmt.Errorf("Field10 this[%v](%v) Not Equal that[%v](%v)", i, this.Field10[i], i, that.Field10[i])
		}
	}
	if len(this.Field11) != len(that.Field11) {
		return fmt.Errorf("Field11 this(%v) Not Equal that(%v)", len(this.Field11), len(that.Field11))
	}
	for i := range this.Field11 {
		if this.Field11[i] != that.Field11[i] {
			return fmt.Errorf("Field11 this[%v](%v) Not Equal that[%v](%v)", i, this.Field11[i], i, that.Field11[i])
		}
	}
	if len(this.Field12) != len(that.Field12) {
		return fmt.Errorf("Field12 this(%v) Not Equal that(%v)", len(this.Field12), len(that.Field12))
	}
	for i := range this.Field12 {
		if this.Field12[i] != that.Field12[i] {
			return fmt.Errorf("Field12 this[%v](%v) Not Equal that[%v](%v)", i, this.Field12[i], i, that.Field12[i])
		}
	}
	if len(this.Field13) != len(that.Field13) {
		return fmt.Errorf("Field13 this(%v) Not Equal that(%v)", len(this.Field13), len(that.Field13))
	}
	for i := range this.Field13 {
		if this.Field13[i] != that.Field13[i] {
			return fmt.Errorf("Field13 this[%v](%v) Not Equal that[%v](%v)", i, this.Field13[i], i, that.Field13[i])
		}
	}
	if !bytes.Equal(this.XXX_unrecognized, that.XXX_unrecognized) {
		return fmt.Errorf("XXX_unrecognized this(%v) Not Equal that(%v)", this.XXX_unrecognized, that.XXX_unrecognized)
	}
	return nil
}
