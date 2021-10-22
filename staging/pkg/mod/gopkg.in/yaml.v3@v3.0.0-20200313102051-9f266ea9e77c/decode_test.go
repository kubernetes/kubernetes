//
// Copyright (c) 2011-2019 Canonical Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package yaml_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"time"

	. "gopkg.in/check.v1"
	"gopkg.in/yaml.v3"
)

var unmarshalIntTest = 123

var unmarshalTests = []struct {
	data  string
	value interface{}
}{
	{
		"",
		(*struct{})(nil),
	},
	{
		"{}", &struct{}{},
	}, {
		"v: hi",
		map[string]string{"v": "hi"},
	}, {
		"v: hi", map[string]interface{}{"v": "hi"},
	}, {
		"v: true",
		map[string]string{"v": "true"},
	}, {
		"v: true",
		map[string]interface{}{"v": true},
	}, {
		"v: 10",
		map[string]interface{}{"v": 10},
	}, {
		"v: 0b10",
		map[string]interface{}{"v": 2},
	}, {
		"v: 0xA",
		map[string]interface{}{"v": 10},
	}, {
		"v: 4294967296",
		map[string]int64{"v": 4294967296},
	}, {
		"v: 0.1",
		map[string]interface{}{"v": 0.1},
	}, {
		"v: .1",
		map[string]interface{}{"v": 0.1},
	}, {
		"v: .Inf",
		map[string]interface{}{"v": math.Inf(+1)},
	}, {
		"v: -.Inf",
		map[string]interface{}{"v": math.Inf(-1)},
	}, {
		"v: -10",
		map[string]interface{}{"v": -10},
	}, {
		"v: -.1",
		map[string]interface{}{"v": -0.1},
	},

	// Simple values.
	{
		"123",
		&unmarshalIntTest,
	},

	// Floats from spec
	{
		"canonical: 6.8523e+5",
		map[string]interface{}{"canonical": 6.8523e+5},
	}, {
		"expo: 685.230_15e+03",
		map[string]interface{}{"expo": 685.23015e+03},
	}, {
		"fixed: 685_230.15",
		map[string]interface{}{"fixed": 685230.15},
	}, {
		"neginf: -.inf",
		map[string]interface{}{"neginf": math.Inf(-1)},
	}, {
		"fixed: 685_230.15",
		map[string]float64{"fixed": 685230.15},
	},
	//{"sexa: 190:20:30.15", map[string]interface{}{"sexa": 0}}, // Unsupported
	//{"notanum: .NaN", map[string]interface{}{"notanum": math.NaN()}}, // Equality of NaN fails.

	// Bools are per 1.2 spec.
	{
		"canonical: true",
		map[string]interface{}{"canonical": true},
	}, {
		"canonical: false",
		map[string]interface{}{"canonical": false},
	}, {
		"bool: True",
		map[string]interface{}{"bool": true},
	}, {
		"bool: False",
		map[string]interface{}{"bool": false},
	}, {
		"bool: TRUE",
		map[string]interface{}{"bool": true},
	}, {
		"bool: FALSE",
		map[string]interface{}{"bool": false},
	},
	// For backwards compatibility with 1.1, decoding old strings into typed values still works.
	{
		"option: on",
		map[string]bool{"option": true},
	}, {
		"option: y",
		map[string]bool{"option": true},
	}, {
		"option: Off",
		map[string]bool{"option": false},
	}, {
		"option: No",
		map[string]bool{"option": false},
	}, {
		"option: other",
		map[string]bool{},
	},
	// Ints from spec
	{
		"canonical: 685230",
		map[string]interface{}{"canonical": 685230},
	}, {
		"decimal: +685_230",
		map[string]interface{}{"decimal": 685230},
	}, {
		"octal: 02472256",
		map[string]interface{}{"octal": 685230},
	}, {
		"octal: -02472256",
		map[string]interface{}{"octal": -685230},
	}, {
		"octal: 0o2472256",
		map[string]interface{}{"octal": 685230},
	}, {
		"octal: -0o2472256",
		map[string]interface{}{"octal": -685230},
	}, {
		"hexa: 0x_0A_74_AE",
		map[string]interface{}{"hexa": 685230},
	}, {
		"bin: 0b1010_0111_0100_1010_1110",
		map[string]interface{}{"bin": 685230},
	}, {
		"bin: -0b101010",
		map[string]interface{}{"bin": -42},
	}, {
		"bin: -0b1000000000000000000000000000000000000000000000000000000000000000",
		map[string]interface{}{"bin": -9223372036854775808},
	}, {
		"decimal: +685_230",
		map[string]int{"decimal": 685230},
	},

	//{"sexa: 190:20:30", map[string]interface{}{"sexa": 0}}, // Unsupported

	// Nulls from spec
	{
		"empty:",
		map[string]interface{}{"empty": nil},
	}, {
		"canonical: ~",
		map[string]interface{}{"canonical": nil},
	}, {
		"english: null",
		map[string]interface{}{"english": nil},
	}, {
		"~: null key",
		map[interface{}]string{nil: "null key"},
	}, {
		"empty:",
		map[string]*bool{"empty": nil},
	},

	// Flow sequence
	{
		"seq: [A,B]",
		map[string]interface{}{"seq": []interface{}{"A", "B"}},
	}, {
		"seq: [A,B,C,]",
		map[string][]string{"seq": []string{"A", "B", "C"}},
	}, {
		"seq: [A,1,C]",
		map[string][]string{"seq": []string{"A", "1", "C"}},
	}, {
		"seq: [A,1,C]",
		map[string][]int{"seq": []int{1}},
	}, {
		"seq: [A,1,C]",
		map[string]interface{}{"seq": []interface{}{"A", 1, "C"}},
	},
	// Block sequence
	{
		"seq:\n - A\n - B",
		map[string]interface{}{"seq": []interface{}{"A", "B"}},
	}, {
		"seq:\n - A\n - B\n - C",
		map[string][]string{"seq": []string{"A", "B", "C"}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string][]string{"seq": []string{"A", "1", "C"}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string][]int{"seq": []int{1}},
	}, {
		"seq:\n - A\n - 1\n - C",
		map[string]interface{}{"seq": []interface{}{"A", 1, "C"}},
	},

	// Literal block scalar
	{
		"scalar: | # Comment\n\n literal\n\n \ttext\n\n",
		map[string]string{"scalar": "\nliteral\n\n\ttext\n"},
	},

	// Folded block scalar
	{
		"scalar: > # Comment\n\n folded\n line\n \n next\n line\n  * one\n  * two\n\n last\n line\n\n",
		map[string]string{"scalar": "\nfolded line\nnext line\n * one\n * two\n\nlast line\n"},
	},

	// Map inside interface with no type hints.
	{
		"a: {b: c}",
		map[interface{}]interface{}{"a": map[string]interface{}{"b": "c"}},
	},
	// Non-string map inside interface with no type hints.
	{
		"a: {b: c, 1: d}",
		map[interface{}]interface{}{"a": map[interface{}]interface{}{"b": "c", 1: "d"}},
	},

	// Structs and type conversions.
	{
		"hello: world",
		&struct{ Hello string }{"world"},
	}, {
		"a: {b: c}",
		&struct{ A struct{ B string } }{struct{ B string }{"c"}},
	}, {
		"a: {b: c}",
		&struct{ A *struct{ B string } }{&struct{ B string }{"c"}},
	}, {
		"a: 'null'",
		&struct{ A *unmarshalerType }{&unmarshalerType{"null"}},
	}, {
		"a: {b: c}",
		&struct{ A map[string]string }{map[string]string{"b": "c"}},
	}, {
		"a: {b: c}",
		&struct{ A *map[string]string }{&map[string]string{"b": "c"}},
	}, {
		"a:",
		&struct{ A map[string]string }{},
	}, {
		"a: 1",
		&struct{ A int }{1},
	}, {
		"a: 1",
		&struct{ A float64 }{1},
	}, {
		"a: 1.0",
		&struct{ A int }{1},
	}, {
		"a: 1.0",
		&struct{ A uint }{1},
	}, {
		"a: [1, 2]",
		&struct{ A []int }{[]int{1, 2}},
	}, {
		"a: [1, 2]",
		&struct{ A [2]int }{[2]int{1, 2}},
	}, {
		"a: 1",
		&struct{ B int }{0},
	}, {
		"a: 1",
		&struct {
			B int "a"
		}{1},
	}, {
		// Some limited backwards compatibility with the 1.1 spec.
		"a: YES",
		&struct{ A bool }{true},
	},

	// Some cross type conversions
	{
		"v: 42",
		map[string]uint{"v": 42},
	}, {
		"v: -42",
		map[string]uint{},
	}, {
		"v: 4294967296",
		map[string]uint64{"v": 4294967296},
	}, {
		"v: -4294967296",
		map[string]uint64{},
	},

	// int
	{
		"int_max: 2147483647",
		map[string]int{"int_max": math.MaxInt32},
	},
	{
		"int_min: -2147483648",
		map[string]int{"int_min": math.MinInt32},
	},
	{
		"int_overflow: 9223372036854775808", // math.MaxInt64 + 1
		map[string]int{},
	},

	// int64
	{
		"int64_max: 9223372036854775807",
		map[string]int64{"int64_max": math.MaxInt64},
	},
	{
		"int64_max_base2: 0b111111111111111111111111111111111111111111111111111111111111111",
		map[string]int64{"int64_max_base2": math.MaxInt64},
	},
	{
		"int64_min: -9223372036854775808",
		map[string]int64{"int64_min": math.MinInt64},
	},
	{
		"int64_neg_base2: -0b111111111111111111111111111111111111111111111111111111111111111",
		map[string]int64{"int64_neg_base2": -math.MaxInt64},
	},
	{
		"int64_overflow: 9223372036854775808", // math.MaxInt64 + 1
		map[string]int64{},
	},

	// uint
	{
		"uint_min: 0",
		map[string]uint{"uint_min": 0},
	},
	{
		"uint_max: 4294967295",
		map[string]uint{"uint_max": math.MaxUint32},
	},
	{
		"uint_underflow: -1",
		map[string]uint{},
	},

	// uint64
	{
		"uint64_min: 0",
		map[string]uint{"uint64_min": 0},
	},
	{
		"uint64_max: 18446744073709551615",
		map[string]uint64{"uint64_max": math.MaxUint64},
	},
	{
		"uint64_max_base2: 0b1111111111111111111111111111111111111111111111111111111111111111",
		map[string]uint64{"uint64_max_base2": math.MaxUint64},
	},
	{
		"uint64_maxint64: 9223372036854775807",
		map[string]uint64{"uint64_maxint64": math.MaxInt64},
	},
	{
		"uint64_underflow: -1",
		map[string]uint64{},
	},

	// float32
	{
		"float32_max: 3.40282346638528859811704183484516925440e+38",
		map[string]float32{"float32_max": math.MaxFloat32},
	},
	{
		"float32_nonzero: 1.401298464324817070923729583289916131280e-45",
		map[string]float32{"float32_nonzero": math.SmallestNonzeroFloat32},
	},
	{
		"float32_maxuint64: 18446744073709551615",
		map[string]float32{"float32_maxuint64": float32(math.MaxUint64)},
	},
	{
		"float32_maxuint64+1: 18446744073709551616",
		map[string]float32{"float32_maxuint64+1": float32(math.MaxUint64 + 1)},
	},

	// float64
	{
		"float64_max: 1.797693134862315708145274237317043567981e+308",
		map[string]float64{"float64_max": math.MaxFloat64},
	},
	{
		"float64_nonzero: 4.940656458412465441765687928682213723651e-324",
		map[string]float64{"float64_nonzero": math.SmallestNonzeroFloat64},
	},
	{
		"float64_maxuint64: 18446744073709551615",
		map[string]float64{"float64_maxuint64": float64(math.MaxUint64)},
	},
	{
		"float64_maxuint64+1: 18446744073709551616",
		map[string]float64{"float64_maxuint64+1": float64(math.MaxUint64 + 1)},
	},

	// Overflow cases.
	{
		"v: 4294967297",
		map[string]int32{},
	}, {
		"v: 128",
		map[string]int8{},
	},

	// Quoted values.
	{
		"'1': '\"2\"'",
		map[interface{}]interface{}{"1": "\"2\""},
	}, {
		"v:\n- A\n- 'B\n\n  C'\n",
		map[string][]string{"v": []string{"A", "B\nC"}},
	},

	// Explicit tags.
	{
		"v: !!float '1.1'",
		map[string]interface{}{"v": 1.1},
	}, {
		"v: !!float 0",
		map[string]interface{}{"v": float64(0)},
	}, {
		"v: !!float -1",
		map[string]interface{}{"v": float64(-1)},
	}, {
		"v: !!null ''",
		map[string]interface{}{"v": nil},
	}, {
		"%TAG !y! tag:yaml.org,2002:\n---\nv: !y!int '1'",
		map[string]interface{}{"v": 1},
	},

	// Non-specific tag (Issue #75)
	{
		"v: ! test",
		map[string]interface{}{"v": "test"},
	},

	// Anchors and aliases.
	{
		"a: &x 1\nb: &y 2\nc: *x\nd: *y\n",
		&struct{ A, B, C, D int }{1, 2, 1, 2},
	}, {
		"a: &a {c: 1}\nb: *a",
		&struct {
			A, B struct {
				C int
			}
		}{struct{ C int }{1}, struct{ C int }{1}},
	}, {
		"a: &a [1, 2]\nb: *a",
		&struct{ B []int }{[]int{1, 2}},
	},

	// Bug #1133337
	{
		"foo: ''",
		map[string]*string{"foo": new(string)},
	}, {
		"foo: null",
		map[string]*string{"foo": nil},
	}, {
		"foo: null",
		map[string]string{},
	}, {
		"foo: null",
		map[string]interface{}{"foo": nil},
	},

	// Support for ~
	{
		"foo: ~",
		map[string]*string{"foo": nil},
	}, {
		"foo: ~",
		map[string]string{},
	}, {
		"foo: ~",
		map[string]interface{}{"foo": nil},
	},

	// Ignored field
	{
		"a: 1\nb: 2\n",
		&struct {
			A int
			B int "-"
		}{1, 0},
	},

	// Bug #1191981
	{
		"" +
			"%YAML 1.1\n" +
			"--- !!str\n" +
			`"Generic line break (no glyph)\n\` + "\n" +
			` Generic line break (glyphed)\n\` + "\n" +
			` Line separator\u2028\` + "\n" +
			` Paragraph separator\u2029"` + "\n",
		"" +
			"Generic line break (no glyph)\n" +
			"Generic line break (glyphed)\n" +
			"Line separator\u2028Paragraph separator\u2029",
	},

	// Struct inlining
	{
		"a: 1\nb: 2\nc: 3\n",
		&struct {
			A int
			C inlineB `yaml:",inline"`
		}{1, inlineB{2, inlineC{3}}},
	},

	// Struct inlining as a pointer.
	{
		"a: 1\nb: 2\nc: 3\n",
		&struct {
			A int
			C *inlineB `yaml:",inline"`
		}{1, &inlineB{2, inlineC{3}}},
	}, {
		"a: 1\n",
		&struct {
			A int
			C *inlineB `yaml:",inline"`
		}{1, nil},
	}, {
		"a: 1\nc: 3\nd: 4\n",
		&struct {
			A int
			C *inlineD `yaml:",inline"`
		}{1, &inlineD{&inlineC{3}, 4}},
	},

	// Map inlining
	{
		"a: 1\nb: 2\nc: 3\n",
		&struct {
			A int
			C map[string]int `yaml:",inline"`
		}{1, map[string]int{"b": 2, "c": 3}},
	},

	// bug 1243827
	{
		"a: -b_c",
		map[string]interface{}{"a": "-b_c"},
	},
	{
		"a: +b_c",
		map[string]interface{}{"a": "+b_c"},
	},
	{
		"a: 50cent_of_dollar",
		map[string]interface{}{"a": "50cent_of_dollar"},
	},

	// issue #295 (allow scalars with colons in flow mappings and sequences)
	{
		"a: {b: https://github.com/go-yaml/yaml}",
		map[string]interface{}{"a": map[string]interface{}{
			"b": "https://github.com/go-yaml/yaml",
		}},
	},
	{
		"a: [https://github.com/go-yaml/yaml]",
		map[string]interface{}{"a": []interface{}{"https://github.com/go-yaml/yaml"}},
	},

	// Duration
	{
		"a: 3s",
		map[string]time.Duration{"a": 3 * time.Second},
	},

	// Issue #24.
	{
		"a: <foo>",
		map[string]string{"a": "<foo>"},
	},

	// Base 60 floats are obsolete and unsupported.
	{
		"a: 1:1\n",
		map[string]string{"a": "1:1"},
	},

	// Binary data.
	{
		"a: !!binary gIGC\n",
		map[string]string{"a": "\x80\x81\x82"},
	}, {
		"a: !!binary |\n  " + strings.Repeat("kJCQ", 17) + "kJ\n  CQ\n",
		map[string]string{"a": strings.Repeat("\x90", 54)},
	}, {
		"a: !!binary |\n  " + strings.Repeat("A", 70) + "\n  ==\n",
		map[string]string{"a": strings.Repeat("\x00", 52)},
	},

	// Issue #39.
	{
		"a:\n b:\n  c: d\n",
		map[string]struct{ B interface{} }{"a": {map[string]interface{}{"c": "d"}}},
	},

	// Custom map type.
	{
		"a: {b: c}",
		M{"a": M{"b": "c"}},
	},

	// Support encoding.TextUnmarshaler.
	{
		"a: 1.2.3.4\n",
		map[string]textUnmarshaler{"a": textUnmarshaler{S: "1.2.3.4"}},
	},
	{
		"a: 2015-02-24T18:19:39Z\n",
		map[string]textUnmarshaler{"a": textUnmarshaler{"2015-02-24T18:19:39Z"}},
	},

	// Timestamps
	{
		// Date only.
		"a: 2015-01-01\n",
		map[string]time.Time{"a": time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)},
	},
	{
		// RFC3339
		"a: 2015-02-24T18:19:39.12Z\n",
		map[string]time.Time{"a": time.Date(2015, 2, 24, 18, 19, 39, .12e9, time.UTC)},
	},
	{
		// RFC3339 with short dates.
		"a: 2015-2-3T3:4:5Z",
		map[string]time.Time{"a": time.Date(2015, 2, 3, 3, 4, 5, 0, time.UTC)},
	},
	{
		// ISO8601 lower case t
		"a: 2015-02-24t18:19:39Z\n",
		map[string]time.Time{"a": time.Date(2015, 2, 24, 18, 19, 39, 0, time.UTC)},
	},
	{
		// space separate, no time zone
		"a: 2015-02-24 18:19:39\n",
		map[string]time.Time{"a": time.Date(2015, 2, 24, 18, 19, 39, 0, time.UTC)},
	},
	// Some cases not currently handled. Uncomment these when
	// the code is fixed.
	//	{
	//		// space separated with time zone
	//		"a: 2001-12-14 21:59:43.10 -5",
	//		map[string]interface{}{"a": time.Date(2001, 12, 14, 21, 59, 43, .1e9, time.UTC)},
	//	},
	//	{
	//		// arbitrary whitespace between fields
	//		"a: 2001-12-14 \t\t \t21:59:43.10 \t Z",
	//		map[string]interface{}{"a": time.Date(2001, 12, 14, 21, 59, 43, .1e9, time.UTC)},
	//	},
	{
		// explicit string tag
		"a: !!str 2015-01-01",
		map[string]interface{}{"a": "2015-01-01"},
	},
	{
		// explicit timestamp tag on quoted string
		"a: !!timestamp \"2015-01-01\"",
		map[string]time.Time{"a": time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)},
	},
	{
		// explicit timestamp tag on unquoted string
		"a: !!timestamp 2015-01-01",
		map[string]time.Time{"a": time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)},
	},
	{
		// quoted string that's a valid timestamp
		"a: \"2015-01-01\"",
		map[string]interface{}{"a": "2015-01-01"},
	},
	{
		// explicit timestamp tag into interface.
		"a: !!timestamp \"2015-01-01\"",
		map[string]interface{}{"a": time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)},
	},
	{
		// implicit timestamp tag into interface.
		"a: 2015-01-01",
		map[string]interface{}{"a": time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)},
	},

	// Encode empty lists as zero-length slices.
	{
		"a: []",
		&struct{ A []int }{[]int{}},
	},

	// UTF-16-LE
	{
		"\xff\xfe\xf1\x00o\x00\xf1\x00o\x00:\x00 \x00v\x00e\x00r\x00y\x00 \x00y\x00e\x00s\x00\n\x00",
		M{"ñoño": "very yes"},
	},
	// UTF-16-LE with surrogate.
	{
		"\xff\xfe\xf1\x00o\x00\xf1\x00o\x00:\x00 \x00v\x00e\x00r\x00y\x00 \x00y\x00e\x00s\x00 \x00=\xd8\xd4\xdf\n\x00",
		M{"ñoño": "very yes 🟔"},
	},

	// UTF-16-BE
	{
		"\xfe\xff\x00\xf1\x00o\x00\xf1\x00o\x00:\x00 \x00v\x00e\x00r\x00y\x00 \x00y\x00e\x00s\x00\n",
		M{"ñoño": "very yes"},
	},
	// UTF-16-BE with surrogate.
	{
		"\xfe\xff\x00\xf1\x00o\x00\xf1\x00o\x00:\x00 \x00v\x00e\x00r\x00y\x00 \x00y\x00e\x00s\x00 \xd8=\xdf\xd4\x00\n",
		M{"ñoño": "very yes 🟔"},
	},

	// This *is* in fact a float number, per the spec. #171 was a mistake.
	{
		"a: 123456e1\n",
		M{"a": 123456e1},
	}, {
		"a: 123456E1\n",
		M{"a": 123456E1},
	},
	// yaml-test-suite 3GZX: Spec Example 7.1. Alias Nodes
	{
		"First occurrence: &anchor Foo\nSecond occurrence: *anchor\nOverride anchor: &anchor Bar\nReuse anchor: *anchor\n",
		map[string]interface{}{
			"First occurrence":  "Foo",
			"Second occurrence": "Foo",
			"Override anchor":   "Bar",
			"Reuse anchor":      "Bar",
		},
	},
	// Single document with garbage following it.
	{
		"---\nhello\n...\n}not yaml",
		"hello",
	},

	// Comment scan exhausting the input buffer (issue #469).
	{
		"true\n#" + strings.Repeat(" ", 512*3),
		"true",
	}, {
		"true #" + strings.Repeat(" ", 512*3),
		"true",
	},

	// CRLF
	{
		"a: b\r\nc:\r\n- d\r\n- e\r\n",
		map[string]interface{}{
			"a": "b",
			"c": []interface{}{"d", "e"},
		},
	},

}

type M map[string]interface{}

type inlineB struct {
	B       int
	inlineC `yaml:",inline"`
}

type inlineC struct {
	C int
}

type inlineD struct {
	C *inlineC `yaml:",inline"`
	D int
}

func (s *S) TestUnmarshal(c *C) {
	for i, item := range unmarshalTests {
		c.Logf("test %d: %q", i, item.data)
		t := reflect.ValueOf(item.value).Type()
		value := reflect.New(t)
		err := yaml.Unmarshal([]byte(item.data), value.Interface())
		if _, ok := err.(*yaml.TypeError); !ok {
			c.Assert(err, IsNil)
		}
		c.Assert(value.Elem().Interface(), DeepEquals, item.value, Commentf("error: %v", err))
	}
}

func (s *S) TestUnmarshalFullTimestamp(c *C) {
	// Full timestamp in same format as encoded. This is confirmed to be
	// properly decoded by Python as a timestamp as well.
	var str = "2015-02-24T18:19:39.123456789-03:00"
	var t interface{}
	err := yaml.Unmarshal([]byte(str), &t)
	c.Assert(err, IsNil)
	c.Assert(t, Equals, time.Date(2015, 2, 24, 18, 19, 39, 123456789, t.(time.Time).Location()))
	c.Assert(t.(time.Time).In(time.UTC), Equals, time.Date(2015, 2, 24, 21, 19, 39, 123456789, time.UTC))
}

func (s *S) TestDecoderSingleDocument(c *C) {
	// Test that Decoder.Decode works as expected on
	// all the unmarshal tests.
	for i, item := range unmarshalTests {
		c.Logf("test %d: %q", i, item.data)
		if item.data == "" {
			// Behaviour differs when there's no YAML.
			continue
		}
		t := reflect.ValueOf(item.value).Type()
		value := reflect.New(t)
		err := yaml.NewDecoder(strings.NewReader(item.data)).Decode(value.Interface())
		if _, ok := err.(*yaml.TypeError); !ok {
			c.Assert(err, IsNil)
		}
		c.Assert(value.Elem().Interface(), DeepEquals, item.value)
	}
}

var decoderTests = []struct {
	data   string
	values []interface{}
}{{
	"",
	nil,
}, {
	"a: b",
	[]interface{}{
		map[string]interface{}{"a": "b"},
	},
}, {
	"---\na: b\n...\n",
	[]interface{}{
		map[string]interface{}{"a": "b"},
	},
}, {
	"---\n'hello'\n...\n---\ngoodbye\n...\n",
	[]interface{}{
		"hello",
		"goodbye",
	},
}}

func (s *S) TestDecoder(c *C) {
	for i, item := range decoderTests {
		c.Logf("test %d: %q", i, item.data)
		var values []interface{}
		dec := yaml.NewDecoder(strings.NewReader(item.data))
		for {
			var value interface{}
			err := dec.Decode(&value)
			if err == io.EOF {
				break
			}
			c.Assert(err, IsNil)
			values = append(values, value)
		}
		c.Assert(values, DeepEquals, item.values)
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) {
	return 0, errors.New("some read error")
}

func (s *S) TestDecoderReadError(c *C) {
	err := yaml.NewDecoder(errReader{}).Decode(&struct{}{})
	c.Assert(err, ErrorMatches, `yaml: input error: some read error`)
}

func (s *S) TestUnmarshalNaN(c *C) {
	value := map[string]interface{}{}
	err := yaml.Unmarshal([]byte("notanum: .NaN"), &value)
	c.Assert(err, IsNil)
	c.Assert(math.IsNaN(value["notanum"].(float64)), Equals, true)
}

func (s *S) TestUnmarshalDurationInt(c *C) {
	// Don't accept plain ints as durations as it's unclear (issue #200).
	var d time.Duration
	err := yaml.Unmarshal([]byte("123"), &d)
	c.Assert(err, ErrorMatches, "(?s).* line 1: cannot unmarshal !!int `123` into time.Duration")
}

var unmarshalErrorTests = []struct {
	data, error string
}{
	{"v: !!float 'error'", "yaml: cannot decode !!str `error` as a !!float"},
	{"v: [A,", "yaml: line 1: did not find expected node content"},
	{"v:\n- [A,", "yaml: line 2: did not find expected node content"},
	{"a:\n- b: *,", "yaml: line 2: did not find expected alphabetic or numeric character"},
	{"a: *b\n", "yaml: unknown anchor 'b' referenced"},
	{"a: &a\n  b: *a\n", "yaml: anchor 'a' value contains itself"},
	{"value: -", "yaml: block sequence entries are not allowed in this context"},
	{"a: !!binary ==", "yaml: !!binary value contains invalid base64 data"},
	{"{[.]}", `yaml: invalid map key: \[\]interface \{\}\{"\."\}`},
	{"{{.}}", `yaml: invalid map key: map\[string]interface \{\}\{".":interface \{\}\(nil\)\}`},
	{"b: *a\na: &a {c: 1}", `yaml: unknown anchor 'a' referenced`},
	{"%TAG !%79! tag:yaml.org,2002:\n---\nv: !%79!int '1'", "yaml: did not find expected whitespace"},
	{"a:\n  1:\nb\n  2:", ".*could not find expected ':'"},
	{
		"a: &a [00,00,00,00,00,00,00,00,00]\n" +
		"b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\n" +
		"c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]\n" +
		"d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]\n" +
		"e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]\n" +
		"f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]\n" +
		"g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f]\n" +
		"h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g]\n" +
		"i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h]\n",
		"yaml: document contains excessive aliasing",
	},
}

func (s *S) TestUnmarshalErrors(c *C) {
	for i, item := range unmarshalErrorTests {
		c.Logf("test %d: %q", i, item.data)
		var value interface{}
		err := yaml.Unmarshal([]byte(item.data), &value)
		c.Assert(err, ErrorMatches, item.error, Commentf("Partial unmarshal: %#v", value))
	}
}

func (s *S) TestDecoderErrors(c *C) {
	for _, item := range unmarshalErrorTests {
		var value interface{}
		err := yaml.NewDecoder(strings.NewReader(item.data)).Decode(&value)
		c.Assert(err, ErrorMatches, item.error, Commentf("Partial unmarshal: %#v", value))
	}
}

var unmarshalerTests = []struct {
	data, tag string
	value     interface{}
}{
	{"_: {hi: there}", "!!map", map[string]interface{}{"hi": "there"}},
	{"_: [1,A]", "!!seq", []interface{}{1, "A"}},
	{"_: 10", "!!int", 10},
	{"_: null", "!!null", nil},
	{`_: BAR!`, "!!str", "BAR!"},
	{`_: "BAR!"`, "!!str", "BAR!"},
	{"_: !!foo 'BAR!'", "!!foo", "BAR!"},
	{`_: ""`, "!!str", ""},
}

var unmarshalerResult = map[int]error{}

type unmarshalerType struct {
	value interface{}
}

func (o *unmarshalerType) UnmarshalYAML(value *yaml.Node) error {
	if err := value.Decode(&o.value); err != nil {
		return err
	}
	if i, ok := o.value.(int); ok {
		if result, ok := unmarshalerResult[i]; ok {
			return result
		}
	}
	return nil
}

type unmarshalerPointer struct {
	Field *unmarshalerType "_"
}

type unmarshalerValue struct {
	Field unmarshalerType "_"
}

type unmarshalerInlined struct {
	Field   *unmarshalerType "_"
	Inlined unmarshalerType  `yaml:",inline"`
}

type unmarshalerInlinedTwice struct {
	InlinedTwice unmarshalerInlined `yaml:",inline"`
}

type obsoleteUnmarshalerType struct {
	value interface{}
}

func (o *obsoleteUnmarshalerType) UnmarshalYAML(unmarshal func(v interface{}) error) error {
	if err := unmarshal(&o.value); err != nil {
		return err
	}
	if i, ok := o.value.(int); ok {
		if result, ok := unmarshalerResult[i]; ok {
			return result
		}
	}
	return nil
}

type obsoleteUnmarshalerPointer struct {
	Field *obsoleteUnmarshalerType "_"
}

type obsoleteUnmarshalerValue struct {
	Field obsoleteUnmarshalerType "_"
}

func (s *S) TestUnmarshalerPointerField(c *C) {
	for _, item := range unmarshalerTests {
		obj := &unmarshalerPointer{}
		err := yaml.Unmarshal([]byte(item.data), obj)
		c.Assert(err, IsNil)
		if item.value == nil {
			c.Assert(obj.Field, IsNil)
		} else {
			c.Assert(obj.Field, NotNil, Commentf("Pointer not initialized (%#v)", item.value))
			c.Assert(obj.Field.value, DeepEquals, item.value)
		}
	}
	for _, item := range unmarshalerTests {
		obj := &obsoleteUnmarshalerPointer{}
		err := yaml.Unmarshal([]byte(item.data), obj)
		c.Assert(err, IsNil)
		if item.value == nil {
			c.Assert(obj.Field, IsNil)
		} else {
			c.Assert(obj.Field, NotNil, Commentf("Pointer not initialized (%#v)", item.value))
			c.Assert(obj.Field.value, DeepEquals, item.value)
		}
	}
}

func (s *S) TestUnmarshalerValueField(c *C) {
	for _, item := range unmarshalerTests {
		obj := &obsoleteUnmarshalerValue{}
		err := yaml.Unmarshal([]byte(item.data), obj)
		c.Assert(err, IsNil)
		c.Assert(obj.Field, NotNil, Commentf("Pointer not initialized (%#v)", item.value))
		c.Assert(obj.Field.value, DeepEquals, item.value)
	}
}

func (s *S) TestUnmarshalerInlinedField(c *C) {
	obj := &unmarshalerInlined{}
	err := yaml.Unmarshal([]byte("_: a\ninlined: b\n"), obj)
	c.Assert(err, IsNil)
	c.Assert(obj.Field, DeepEquals, &unmarshalerType{"a"})
	c.Assert(obj.Inlined, DeepEquals, unmarshalerType{map[string]interface{}{"_": "a", "inlined": "b"}})

	twc := &unmarshalerInlinedTwice{}
	err = yaml.Unmarshal([]byte("_: a\ninlined: b\n"), twc)
	c.Assert(err, IsNil)
	c.Assert(twc.InlinedTwice.Field, DeepEquals, &unmarshalerType{"a"})
	c.Assert(twc.InlinedTwice.Inlined, DeepEquals, unmarshalerType{map[string]interface{}{"_": "a", "inlined": "b"}})
}

func (s *S) TestUnmarshalerWholeDocument(c *C) {
	obj := &obsoleteUnmarshalerType{}
	err := yaml.Unmarshal([]byte(unmarshalerTests[0].data), obj)
	c.Assert(err, IsNil)
	value, ok := obj.value.(map[string]interface{})
	c.Assert(ok, Equals, true, Commentf("value: %#v", obj.value))
	c.Assert(value["_"], DeepEquals, unmarshalerTests[0].value)
}

func (s *S) TestUnmarshalerTypeError(c *C) {
	unmarshalerResult[2] = &yaml.TypeError{[]string{"foo"}}
	unmarshalerResult[4] = &yaml.TypeError{[]string{"bar"}}
	defer func() {
		delete(unmarshalerResult, 2)
		delete(unmarshalerResult, 4)
	}()

	type T struct {
		Before int
		After  int
		M      map[string]*unmarshalerType
	}
	var v T
	data := `{before: A, m: {abc: 1, def: 2, ghi: 3, jkl: 4}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  foo\n"+
		"  bar\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
	c.Assert(v.M["abc"], NotNil)
	c.Assert(v.M["def"], IsNil)
	c.Assert(v.M["ghi"], NotNil)
	c.Assert(v.M["jkl"], IsNil)

	c.Assert(v.M["abc"].value, Equals, 1)
	c.Assert(v.M["ghi"].value, Equals, 3)
}

func (s *S) TestObsoleteUnmarshalerTypeError(c *C) {
	unmarshalerResult[2] = &yaml.TypeError{[]string{"foo"}}
	unmarshalerResult[4] = &yaml.TypeError{[]string{"bar"}}
	defer func() {
		delete(unmarshalerResult, 2)
		delete(unmarshalerResult, 4)
	}()

	type T struct {
		Before int
		After  int
		M      map[string]*obsoleteUnmarshalerType
	}
	var v T
	data := `{before: A, m: {abc: 1, def: 2, ghi: 3, jkl: 4}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  foo\n"+
		"  bar\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
	c.Assert(v.M["abc"], NotNil)
	c.Assert(v.M["def"], IsNil)
	c.Assert(v.M["ghi"], NotNil)
	c.Assert(v.M["jkl"], IsNil)

	c.Assert(v.M["abc"].value, Equals, 1)
	c.Assert(v.M["ghi"].value, Equals, 3)
}

type proxyTypeError struct{}

func (v *proxyTypeError) UnmarshalYAML(node *yaml.Node) error {
	var s string
	var a int32
	var b int64
	if err := node.Decode(&s); err != nil {
		panic(err)
	}
	if s == "a" {
		if err := node.Decode(&b); err == nil {
			panic("should have failed")
		}
		return node.Decode(&a)
	}
	if err := node.Decode(&a); err == nil {
		panic("should have failed")
	}
	return node.Decode(&b)
}

func (s *S) TestUnmarshalerTypeErrorProxying(c *C) {
	type T struct {
		Before int
		After  int
		M      map[string]*proxyTypeError
	}
	var v T
	data := `{before: A, m: {abc: a, def: b}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  line 1: cannot unmarshal !!str `a` into int32\n"+
		"  line 1: cannot unmarshal !!str `b` into int64\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
}

type obsoleteProxyTypeError struct{}

func (v *obsoleteProxyTypeError) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	var a int32
	var b int64
	if err := unmarshal(&s); err != nil {
		panic(err)
	}
	if s == "a" {
		if err := unmarshal(&b); err == nil {
			panic("should have failed")
		}
		return unmarshal(&a)
	}
	if err := unmarshal(&a); err == nil {
		panic("should have failed")
	}
	return unmarshal(&b)
}

func (s *S) TestObsoleteUnmarshalerTypeErrorProxying(c *C) {
	type T struct {
		Before int
		After  int
		M      map[string]*obsoleteProxyTypeError
	}
	var v T
	data := `{before: A, m: {abc: a, def: b}, after: B}`
	err := yaml.Unmarshal([]byte(data), &v)
	c.Assert(err, ErrorMatches, ""+
		"yaml: unmarshal errors:\n"+
		"  line 1: cannot unmarshal !!str `A` into int\n"+
		"  line 1: cannot unmarshal !!str `a` into int32\n"+
		"  line 1: cannot unmarshal !!str `b` into int64\n"+
		"  line 1: cannot unmarshal !!str `B` into int")
}

var failingErr = errors.New("failingErr")

type failingUnmarshaler struct{}

func (ft *failingUnmarshaler) UnmarshalYAML(node *yaml.Node) error {
	return failingErr
}

func (s *S) TestUnmarshalerError(c *C) {
	err := yaml.Unmarshal([]byte("a: b"), &failingUnmarshaler{})
	c.Assert(err, Equals, failingErr)
}

type obsoleteFailingUnmarshaler struct{}

func (ft *obsoleteFailingUnmarshaler) UnmarshalYAML(unmarshal func(interface{}) error) error {
	return failingErr
}

func (s *S) TestObsoleteUnmarshalerError(c *C) {
	err := yaml.Unmarshal([]byte("a: b"), &obsoleteFailingUnmarshaler{})
	c.Assert(err, Equals, failingErr)
}

type sliceUnmarshaler []int

func (su *sliceUnmarshaler) UnmarshalYAML(node *yaml.Node) error {
	var slice []int
	err := node.Decode(&slice)
	if err == nil {
		*su = slice
		return nil
	}

	var intVal int
	err = node.Decode(&intVal)
	if err == nil {
		*su = []int{intVal}
		return nil
	}

	return err
}

func (s *S) TestUnmarshalerRetry(c *C) {
	var su sliceUnmarshaler
	err := yaml.Unmarshal([]byte("[1, 2, 3]"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, sliceUnmarshaler([]int{1, 2, 3}))

	err = yaml.Unmarshal([]byte("1"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, sliceUnmarshaler([]int{1}))
}

type obsoleteSliceUnmarshaler []int

func (su *obsoleteSliceUnmarshaler) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var slice []int
	err := unmarshal(&slice)
	if err == nil {
		*su = slice
		return nil
	}

	var intVal int
	err = unmarshal(&intVal)
	if err == nil {
		*su = []int{intVal}
		return nil
	}

	return err
}

func (s *S) TestObsoleteUnmarshalerRetry(c *C) {
	var su obsoleteSliceUnmarshaler
	err := yaml.Unmarshal([]byte("[1, 2, 3]"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, obsoleteSliceUnmarshaler([]int{1, 2, 3}))

	err = yaml.Unmarshal([]byte("1"), &su)
	c.Assert(err, IsNil)
	c.Assert(su, DeepEquals, obsoleteSliceUnmarshaler([]int{1}))
}

// From http://yaml.org/type/merge.html
var mergeTests = `
anchors:
  list:
    - &CENTER { "x": 1, "y": 2 }
    - &LEFT   { "x": 0, "y": 2 }
    - &BIG    { "r": 10 }
    - &SMALL  { "r": 1 }

# All the following maps are equal:

plain:
  # Explicit keys
  "x": 1
  "y": 2
  "r": 10
  label: center/big

mergeOne:
  # Merge one map
  << : *CENTER
  "r": 10
  label: center/big

mergeMultiple:
  # Merge multiple maps
  << : [ *CENTER, *BIG ]
  label: center/big

override:
  # Override
  << : [ *BIG, *LEFT, *SMALL ]
  "x": 1
  label: center/big

shortTag:
  # Explicit short merge tag
  !!merge "<<" : [ *CENTER, *BIG ]
  label: center/big

longTag:
  # Explicit merge long tag
  !<tag:yaml.org,2002:merge> "<<" : [ *CENTER, *BIG ]
  label: center/big

inlineMap:
  # Inlined map 
  << : {"x": 1, "y": 2, "r": 10}
  label: center/big

inlineSequenceMap:
  # Inlined map in sequence
  << : [ *CENTER, {"r": 10} ]
  label: center/big
`

func (s *S) TestMerge(c *C) {
	var want = map[interface{}]interface{}{
		"x":     1,
		"y":     2,
		"r":     10,
		"label": "center/big",
	}

	wantStringMap := make(map[string]interface{})
	for k, v := range want {
		wantStringMap[fmt.Sprintf("%v", k)] = v
	}

	var m map[interface{}]interface{}
	err := yaml.Unmarshal([]byte(mergeTests), &m)
	c.Assert(err, IsNil)
	for name, test := range m {
		if name == "anchors" {
			continue
		}
		if name == "plain" {
			c.Assert(test, DeepEquals, wantStringMap, Commentf("test %q failed", name))
			continue
		}
		c.Assert(test, DeepEquals, want, Commentf("test %q failed", name))
	}
}

func (s *S) TestMergeStruct(c *C) {
	type Data struct {
		X, Y, R int
		Label   string
	}
	want := Data{1, 2, 10, "center/big"}

	var m map[string]Data
	err := yaml.Unmarshal([]byte(mergeTests), &m)
	c.Assert(err, IsNil)
	for name, test := range m {
		if name == "anchors" {
			continue
		}
		c.Assert(test, Equals, want, Commentf("test %q failed", name))
	}
}

var unmarshalNullTests = []func() interface{}{
	func() interface{} { var v interface{}; v = "v"; return &v },
	func() interface{} { var s = "s"; return &s },
	func() interface{} { var s = "s"; sptr := &s; return &sptr },
	func() interface{} { var i = 1; return &i },
	func() interface{} { var i = 1; iptr := &i; return &iptr },
	func() interface{} { m := map[string]int{"s": 1}; return &m },
	func() interface{} { m := map[string]int{"s": 1}; return m },
}

func (s *S) TestUnmarshalNull(c *C) {
	for _, test := range unmarshalNullTests {
		pristine := test()
		decoded := test()
		zero := reflect.Zero(reflect.TypeOf(decoded).Elem()).Interface()
		err := yaml.Unmarshal([]byte("null"), decoded)
		c.Assert(err, IsNil)
		switch pristine.(type) {
		case *interface{}, **string, **int, *map[string]int:
			c.Assert(reflect.ValueOf(decoded).Elem().Interface(), DeepEquals, zero)
		default:
			c.Assert(reflect.ValueOf(decoded).Interface(), DeepEquals, pristine)
		}
	}
}

func (s *S) TestUnmarshalPreservesData(c *C) {
	var v struct {
		A, B int
		C int `yaml:"-"`
	}
	v.A = 42
	v.C = 88
	err := yaml.Unmarshal([]byte("---"), &v)
	c.Assert(err, IsNil)
	c.Assert(v.A, Equals, 42)
	c.Assert(v.B, Equals, 0)
	c.Assert(v.C, Equals, 88)

	err = yaml.Unmarshal([]byte("b: 21\nc: 99"), &v)
	c.Assert(err, IsNil)
	c.Assert(v.A, Equals, 42)
	c.Assert(v.B, Equals, 21)
	c.Assert(v.C, Equals, 88)
}

func (s *S) TestUnmarshalSliceOnPreset(c *C) {
	// Issue #48.
	v := struct{ A []int }{[]int{1}}
	yaml.Unmarshal([]byte("a: [2]"), &v)
	c.Assert(v.A, DeepEquals, []int{2})
}

var unmarshalStrictTests = []struct {
	known  bool
	unique bool
	data   string
	value  interface{}
	error  string
}{{
	known: true,
	data:  "a: 1\nc: 2\n",
	value: struct{ A, B int }{A: 1},
	error: `yaml: unmarshal errors:\n  line 2: field c not found in type struct { A int; B int }`,
}, {
	unique: true,
	data:   "a: 1\nb: 2\na: 3\n",
	value:  struct{ A, B int }{A: 3, B: 2},
	error:  `yaml: unmarshal errors:\n  line 3: mapping key "a" already defined at line 1`,
}, {
	unique: true,
	data:   "c: 3\na: 1\nb: 2\nc: 4\n",
	value: struct {
		A       int
		inlineB `yaml:",inline"`
	}{
		A: 1,
		inlineB: inlineB{
			B: 2,
			inlineC: inlineC{
				C: 4,
			},
		},
	},
	error: `yaml: unmarshal errors:\n  line 4: mapping key "c" already defined at line 1`,
}, {
	unique: true,
	data:   "c: 0\na: 1\nb: 2\nc: 1\n",
	value: struct {
		A       int
		inlineB `yaml:",inline"`
	}{
		A: 1,
		inlineB: inlineB{
			B: 2,
			inlineC: inlineC{
				C: 1,
			},
		},
	},
	error: `yaml: unmarshal errors:\n  line 4: mapping key "c" already defined at line 1`,
}, {
	unique: true,
	data:   "c: 1\na: 1\nb: 2\nc: 3\n",
	value: struct {
		A int
		M map[string]interface{} `yaml:",inline"`
	}{
		A: 1,
		M: map[string]interface{}{
			"b": 2,
			"c": 3,
		},
	},
	error: `yaml: unmarshal errors:\n  line 4: mapping key "c" already defined at line 1`,
}, {
	unique: true,
	data:   "a: 1\n9: 2\nnull: 3\n9: 4",
	value: map[interface{}]interface{}{
		"a": 1,
		nil: 3,
		9:   4,
	},
	error: `yaml: unmarshal errors:\n  line 4: mapping key "9" already defined at line 2`,
}}

func (s *S) TestUnmarshalKnownFields(c *C) {
	for i, item := range unmarshalStrictTests {
		c.Logf("test %d: %q", i, item.data)
		// First test that normal Unmarshal unmarshals to the expected value.
		if !item.unique {
			t := reflect.ValueOf(item.value).Type()
			value := reflect.New(t)
			err := yaml.Unmarshal([]byte(item.data), value.Interface())
			c.Assert(err, Equals, nil)
			c.Assert(value.Elem().Interface(), DeepEquals, item.value)
		}

		// Then test that it fails on the same thing with KnownFields on.
		t := reflect.ValueOf(item.value).Type()
		value := reflect.New(t)
		dec := yaml.NewDecoder(bytes.NewBuffer([]byte(item.data)))
		dec.KnownFields(item.known)
		err := dec.Decode(value.Interface())
		c.Assert(err, ErrorMatches, item.error)
	}
}

type textUnmarshaler struct {
	S string
}

func (t *textUnmarshaler) UnmarshalText(s []byte) error {
	t.S = string(s)
	return nil
}

func (s *S) TestFuzzCrashers(c *C) {
	cases := []string{
		// runtime error: index out of range
		"\"\\0\\\r\n",

		// should not happen
		"  0: [\n] 0",
		"? ? \"\n\" 0",
		"    - {\n000}0",
		"0:\n  0: [0\n] 0",
		"    - \"\n000\"0",
		"    - \"\n000\"\"",
		"0:\n    - {\n000}0",
		"0:\n    - \"\n000\"0",
		"0:\n    - \"\n000\"\"",

		// runtime error: index out of range
		" \ufeff\n",
		"? \ufeff\n",
		"? \ufeff:\n",
		"0: \ufeff\n",
		"? \ufeff: \ufeff\n",
	}
	for _, data := range cases {
		var v interface{}
		_ = yaml.Unmarshal([]byte(data), &v)
	}
}

//var data []byte
//func init() {
//	var err error
//	data, err = ioutil.ReadFile("/tmp/file.yaml")
//	if err != nil {
//		panic(err)
//	}
//}
//
//func (s *S) BenchmarkUnmarshal(c *C) {
//	var err error
//	for i := 0; i < c.N; i++ {
//		var v map[string]interface{}
//		err = yaml.Unmarshal(data, &v)
//	}
//	if err != nil {
//		panic(err)
//	}
//}
//
//func (s *S) BenchmarkMarshal(c *C) {
//	var v map[string]interface{}
//	yaml.Unmarshal(data, &v)
//	c.ResetTimer()
//	for i := 0; i < c.N; i++ {
//		yaml.Marshal(&v)
//	}
//}
