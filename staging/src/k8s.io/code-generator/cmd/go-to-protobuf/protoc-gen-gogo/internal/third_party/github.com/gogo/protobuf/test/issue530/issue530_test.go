// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2019, The GoGo Authors. All rights reserved.
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

package issue530

import (
	"strings"
	"testing"
)

// NN : NonNullable
// N  : Nullable
// R  : Repeated
func TestStringNNMessageNNString(t *testing.T) {
	exp := "MessageWith&Ampersand"
	m := &Foo5{
		Bar1: Bar1{Str: exp},
	}
	check(t, "Bar1", m.String(), exp)
}

func TestStringNMessageNNString(t *testing.T) {
	exp := "MessageWith&Ampersand"
	m := &Foo5{
		Bar2: &Bar1{Str: exp},
	}
	check(t, "Bar2", m.String(), exp)
}

func TestStringNNMessageNString(t *testing.T) {
	exp := "MessageWith&Ampersand"
	m := &Foo5{
		Bar3: Bar2{Str: &exp},
	}
	check(t, "Bar3", m.String(), exp)
}

func TestStringNMessageNString(t *testing.T) {
	exp := "MessageWith&Ampersand"
	m := &Foo5{
		Bar4: &Bar2{Str: &exp},
	}
	check(t, "Bar4", m.String(), exp)
}

func TestStringRNNMessageNNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Bars1: []Bar1{{Str: exp1}, {Str: exp2}},
	}
	check(t, "Bars1", m.String(), exp1, exp2)
}

func TestStringRNMessageNNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Bars2: []*Bar1{{Str: exp1}, nil, {Str: exp2}},
	}
	check(t, "Bars2", m.String(), exp1, exp2)
}

func TestStringRNNMessageNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Bars3: []Bar2{{Str: &exp1}, {Str: &exp2}},
	}
	check(t, "Bars3", m.String(), exp1, exp2)
}

func TestStringRNMessageNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Bars4: []*Bar2{{Str: &exp1}, {Str: &exp2}},
	}
	check(t, "Bars4", m.String(), exp1, exp2)
}

func TestStringDeepRNNMessageRNNMessageNNStringAndNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Barrs1: []Bar3{
			{
				Bars4: []Bar4{
					{
						Str: exp1,
					},
				},
				Bars2: []Bar2{
					{
						Str: &exp2,
					},
				},
			},
		},
	}
	check(t, "Barrs1", m.String(), exp1, exp2)
}

func TestStringDeepRNNMessageRNMessageNNStringAndNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Barrs2: []Bar5{
			{
				Bars2: []*Bar2{
					{
						Str: &exp1,
					},
				},
				Bars1: []*Bar1{
					{
						Str: exp2,
					},
				},
			},
		},
	}
	check(t, "Barrs2", m.String(), exp1, exp2)
}

func TestStringMapNMessageRNNMessageNNStringAndNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Barmap1: map[string]*Bar3{
			"one": {
				Bars4: []Bar4{
					{
						Str: exp1,
					},
				},
				Bars2: []Bar2{
					{
						Str: &exp2,
					},
				},
			},
		},
	}
	check(t, "Barmap1", m.String(), exp1, exp2)
}

func TestStringMapNMessageRNMessageNNStringAndNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Foo5{
		Barmap2: map[string]*Bar5{
			"one": {
				Bars2: []*Bar2{
					{
						Str: &exp1,
					},
				},
				Bars1: []*Bar1{
					{
						Str: exp2,
					},
				},
			},
		},
	}
	check(t, "Barmap2", m.String(), exp1, exp2)
}

func TestStringRNNMessageNNStringRNMessageNStringNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	exp3 := "MessageWith&Ampersand3"
	m := &Bar7{
		Bars71: []Bar7{
			{
				Bars72: []*Bar7{
					{
						Str2: &exp3,
					},
				},
				Str2: &exp2,
			},
		},
		Str1: exp1,
	}
	check(t, "Bar7", m.String(), exp1, exp2, exp3)
}

func TestStringRNNMessageWithNoStringerNNString(t *testing.T) {
	exp1 := "MessageWith&Ampersand1"
	exp2 := "MessageWith&Ampersand2"
	m := &Bar8{
		Bars1: []Bar9{{Str: exp1}, {Str: exp2}},
	}
	check(t, "Bars1", m.String(), exp1, exp2)
}

func check(t *testing.T, field, result string, expects ...string) {
	// t.Logf("result: %s \n", result)
	for _, expect := range expects {
		if !strings.Contains(result, expect) {
			t.Fatalf("Expected %s to contain: %s, but got: %s\n", field, expect, result)
		}
	}
}
