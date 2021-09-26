// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2018, The GoGo Authors. All rights reserved.
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

package deterministic

import (
	"bytes"
	"github.com/gogo/protobuf/proto"
	"testing"
)

func getTestMap() map[string]string {
	return map[string]string{
		"a": "1",
		"b": "2",
		"c": "3",
		"d": "4",
		"e": "5",
		"f": "6",
		"g": "7",
		"h": "8",
		"i": "9",
		"j": "10",
		"k": "11",
		"l": "12",
		"m": "13",
		"n": "14",
	}

}

func TestOrderedMap(t *testing.T) {
	var b proto.Buffer
	m := getTestMap()
	in := &OrderedMap{
		StringMap: m,
	}
	if err := b.Marshal(in); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	data1 := b.Bytes()
	out := &OrderedMap{}
	if err := proto.Unmarshal(data1, out); err != nil {
		t.Fatal(err)
	}
	if err := in.VerboseEqual(out); err != nil {
		t.Fatal(err)
	}
	data2, err := proto.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Compare(data1, data2) != 0 {
		t.Fatal("byte arrays are not the same\n", data1, "\n", data2)
	}
}

func TestUnorderedMap(t *testing.T) {
	m := getTestMap()
	in := &UnorderedMap{
		StringMap: m,
	}
	var b proto.Buffer
	b.SetDeterministic(true)
	if err := b.Marshal(in); err == nil {
		t.Fatalf("Expected Marshal to return error rejecting deterministic flag")
	}
}

func TestMapNoMarshaler(t *testing.T) {
	m := getTestMap()
	in := &MapNoMarshaler{
		StringMap: m,
	}

	var b1 proto.Buffer
	b1.SetDeterministic(true)
	if err := b1.Marshal(in); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	data1 := b1.Bytes()

	out := &MapNoMarshaler{}
	err := proto.Unmarshal(data1, out)
	if err != nil {
		t.Fatal(err)
	}
	if err := in.VerboseEqual(out); err != nil {
		t.Fatal(err)
	}

	var b2 proto.Buffer
	b2.SetDeterministic(true)
	if err := b2.Marshal(in); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	data2 := b2.Bytes()

	if bytes.Compare(data1, data2) != 0 {
		t.Fatal("byte arrays are not the same:\n", data1, "\n", data2)
	}
}

func TestOrderedNestedMap(t *testing.T) {
	var b proto.Buffer
	in := &NestedOrderedMap{
		StringMap: getTestMap(),
		NestedMap: &NestedMap1{
			NestedStringMap: getTestMap(),
		},
	}
	if err := b.Marshal(in); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	data1 := b.Bytes()
	out := &NestedOrderedMap{}
	if err := proto.Unmarshal(data1, out); err != nil {
		t.Fatal(err)
	}
	if err := in.VerboseEqual(out); err != nil {
		t.Fatal(err)
	}
	data2, err := proto.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Compare(data1, data2) != 0 {
		t.Fatal("byte arrays are not the same\n", data1, "\n", data2)
	}
}

func TestUnorderedNestedMap(t *testing.T) {
	in := &NestedUnorderedMap{
		StringMap: getTestMap(),
		NestedMap: &NestedMap2{
			NestedStringMap: getTestMap(),
		},
	}
	var b proto.Buffer
	b.SetDeterministic(true)
	if err := b.Marshal(in); err == nil {
		t.Fatalf("Expected Marshal to return error rejecting deterministic flag")
	}
}

func TestOrderedNestedStructMap(t *testing.T) {
	var b proto.Buffer
	m := getTestMap()
	in := &NestedMap1{
		NestedStringMap: m,
	}
	if err := b.Marshal(in); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	data1 := b.Bytes()
	out := &NestedMap1{}
	if err := proto.Unmarshal(data1, out); err != nil {
		t.Fatal(err)
	}
	if err := in.VerboseEqual(out); err != nil {
		t.Fatal(err)
	}
	data2, err := proto.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Compare(data1, data2) != 0 {
		t.Fatal("byte arrays are not the same\n", data1, "\n", data2)
	}
}

func TestUnorderedNestedStructMap(t *testing.T) {
	m := getTestMap()
	in := &NestedMap2{
		NestedStringMap: m,
	}
	var b proto.Buffer
	b.SetDeterministic(true)
	if err := b.Marshal(in); err == nil {
		t.Fatalf("Expected Marshal to return error rejecting deterministic flag")
	}
}
