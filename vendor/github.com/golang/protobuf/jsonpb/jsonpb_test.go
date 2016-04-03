// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2015 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
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
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

package jsonpb

import (
	"reflect"
	"testing"

	pb "github.com/golang/protobuf/jsonpb/jsonpb_test_proto"
	"github.com/golang/protobuf/proto"
	proto3pb "github.com/golang/protobuf/proto/proto3_proto"
)

var (
	marshaler = Marshaler{}

	marshalerAllOptions = Marshaler{
		Indent: "  ",
	}

	simpleObject = &pb.Simple{
		OInt32:  proto.Int32(-32),
		OInt64:  proto.Int64(-6400000000),
		OUint32: proto.Uint32(32),
		OUint64: proto.Uint64(6400000000),
		OSint32: proto.Int32(-13),
		OSint64: proto.Int64(-2600000000),
		OFloat:  proto.Float32(3.14),
		ODouble: proto.Float64(6.02214179e23),
		OBool:   proto.Bool(true),
		OString: proto.String("hello \"there\""),
		OBytes:  []byte("beep boop"),
	}

	simpleObjectJSON = `{` +
		`"o_bool":true,` +
		`"o_int32":-32,` +
		`"o_int64":"-6400000000",` +
		`"o_uint32":32,` +
		`"o_uint64":"6400000000",` +
		`"o_sint32":-13,` +
		`"o_sint64":"-2600000000",` +
		`"o_float":3.14,` +
		`"o_double":6.02214179e+23,` +
		`"o_string":"hello \"there\"",` +
		`"o_bytes":"YmVlcCBib29w"` +
		`}`

	simpleObjectPrettyJSON = `{
  "o_bool": true,
  "o_int32": -32,
  "o_int64": "-6400000000",
  "o_uint32": 32,
  "o_uint64": "6400000000",
  "o_sint32": -13,
  "o_sint64": "-2600000000",
  "o_float": 3.14,
  "o_double": 6.02214179e+23,
  "o_string": "hello \"there\"",
  "o_bytes": "YmVlcCBib29w"
}`

	repeatsObject = &pb.Repeats{
		RBool:   []bool{true, false, true},
		RInt32:  []int32{-3, -4, -5},
		RInt64:  []int64{-123456789, -987654321},
		RUint32: []uint32{1, 2, 3},
		RUint64: []uint64{6789012345, 3456789012},
		RSint32: []int32{-1, -2, -3},
		RSint64: []int64{-6789012345, -3456789012},
		RFloat:  []float32{3.14, 6.28},
		RDouble: []float64{299792458, 6.62606957e-34},
		RString: []string{"happy", "days"},
		RBytes:  [][]byte{[]byte("skittles"), []byte("m&m's")},
	}

	repeatsObjectJSON = `{` +
		`"r_bool":[true,false,true],` +
		`"r_int32":[-3,-4,-5],` +
		`"r_int64":["-123456789","-987654321"],` +
		`"r_uint32":[1,2,3],` +
		`"r_uint64":["6789012345","3456789012"],` +
		`"r_sint32":[-1,-2,-3],` +
		`"r_sint64":["-6789012345","-3456789012"],` +
		`"r_float":[3.14,6.28],` +
		`"r_double":[2.99792458e+08,6.62606957e-34],` +
		`"r_string":["happy","days"],` +
		`"r_bytes":["c2tpdHRsZXM=","bSZtJ3M="]` +
		`}`

	repeatsObjectPrettyJSON = `{
  "r_bool": [
    true,
    false,
    true
  ],
  "r_int32": [
    -3,
    -4,
    -5
  ],
  "r_int64": [
    "-123456789",
    "-987654321"
  ],
  "r_uint32": [
    1,
    2,
    3
  ],
  "r_uint64": [
    "6789012345",
    "3456789012"
  ],
  "r_sint32": [
    -1,
    -2,
    -3
  ],
  "r_sint64": [
    "-6789012345",
    "-3456789012"
  ],
  "r_float": [
    3.14,
    6.28
  ],
  "r_double": [
    2.99792458e+08,
    6.62606957e-34
  ],
  "r_string": [
    "happy",
    "days"
  ],
  "r_bytes": [
    "c2tpdHRsZXM=",
    "bSZtJ3M="
  ]
}`

	innerSimple   = &pb.Simple{OInt32: proto.Int32(-32)}
	innerSimple2  = &pb.Simple{OInt64: proto.Int64(25)}
	innerRepeats  = &pb.Repeats{RString: []string{"roses", "red"}}
	innerRepeats2 = &pb.Repeats{RString: []string{"violets", "blue"}}
	complexObject = &pb.Widget{
		Color:    pb.Widget_GREEN.Enum(),
		RColor:   []pb.Widget_Color{pb.Widget_RED, pb.Widget_GREEN, pb.Widget_BLUE},
		Simple:   innerSimple,
		RSimple:  []*pb.Simple{innerSimple, innerSimple2},
		Repeats:  innerRepeats,
		RRepeats: []*pb.Repeats{innerRepeats, innerRepeats2},
	}

	complexObjectJSON = `{"color":"GREEN",` +
		`"r_color":["RED","GREEN","BLUE"],` +
		`"simple":{"o_int32":-32},` +
		`"r_simple":[{"o_int32":-32},{"o_int64":"25"}],` +
		`"repeats":{"r_string":["roses","red"]},` +
		`"r_repeats":[{"r_string":["roses","red"]},{"r_string":["violets","blue"]}]` +
		`}`

	complexObjectPrettyJSON = `{
  "color": "GREEN",
  "r_color": [
    "RED",
    "GREEN",
    "BLUE"
  ],
  "simple": {
    "o_int32": -32
  },
  "r_simple": [
    {
      "o_int32": -32
    },
    {
      "o_int64": "25"
    }
  ],
  "repeats": {
    "r_string": [
      "roses",
      "red"
    ]
  },
  "r_repeats": [
    {
      "r_string": [
        "roses",
        "red"
      ]
    },
    {
      "r_string": [
        "violets",
        "blue"
      ]
    }
  ]
}`

	colorPrettyJSON = `{
 "color": 2
}`

	colorListPrettyJSON = `{
  "color": 1000,
  "r_color": [
    "RED"
  ]
}`

	nummyPrettyJSON = `{
  "nummy": {
    "1": 2,
    "3": 4
  }
}`

	objjyPrettyJSON = `{
  "objjy": {
    "1": {
      "dub": 1
    }
  }
}`
	realNumber     = &pb.Real{Value: proto.Float64(3.14159265359)}
	realNumberName = "Pi"
	complexNumber  = &pb.Complex{Imaginary: proto.Float64(0.5772156649)}
	realNumberJSON = `{` +
		`"value":3.14159265359,` +
		`"[jsonpb.Complex.real_extension]":{"imaginary":0.5772156649},` +
		`"[jsonpb.name]":"Pi"` +
		`}`
)

func init() {
	if err := proto.SetExtension(realNumber, pb.E_Name, &realNumberName); err != nil {
		panic(err)
	}
	if err := proto.SetExtension(realNumber, pb.E_Complex_RealExtension, complexNumber); err != nil {
		panic(err)
	}
}

var marshalingTests = []struct {
	desc      string
	marshaler Marshaler
	pb        proto.Message
	json      string
}{
	{"simple flat object", marshaler, simpleObject, simpleObjectJSON},
	{"simple pretty object", marshalerAllOptions, simpleObject, simpleObjectPrettyJSON},
	{"repeated fields flat object", marshaler, repeatsObject, repeatsObjectJSON},
	{"repeated fields pretty object", marshalerAllOptions, repeatsObject, repeatsObjectPrettyJSON},
	{"nested message/enum flat object", marshaler, complexObject, complexObjectJSON},
	{"nested message/enum pretty object", marshalerAllOptions, complexObject, complexObjectPrettyJSON},
	{"enum-string flat object", Marshaler{},
		&pb.Widget{Color: pb.Widget_BLUE.Enum()}, `{"color":"BLUE"}`},
	{"enum-value pretty object", Marshaler{EnumsAsInts: true, Indent: " "},
		&pb.Widget{Color: pb.Widget_BLUE.Enum()}, colorPrettyJSON},
	{"unknown enum value object", marshalerAllOptions,
		&pb.Widget{Color: pb.Widget_Color(1000).Enum(), RColor: []pb.Widget_Color{pb.Widget_RED}}, colorListPrettyJSON},
	{"empty value", marshaler, &pb.Simple3{}, `{}`},
	{"empty value emitted", Marshaler{EmitDefaults: true}, &pb.Simple3{}, `{"dub":0}`},
	{"map<int64, int32>", marshaler, &pb.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}, `{"nummy":{"1":2,"3":4}}`},
	{"map<int64, int32>", marshalerAllOptions, &pb.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}, nummyPrettyJSON},
	{"map<string, string>", marshaler,
		&pb.Mappy{Strry: map[string]string{`"one"`: "two", "three": "four"}},
		`{"strry":{"\"one\"":"two","three":"four"}}`},
	{"map<int32, Object>", marshaler,
		&pb.Mappy{Objjy: map[int32]*pb.Simple3{1: &pb.Simple3{Dub: 1}}}, `{"objjy":{"1":{"dub":1}}}`},
	{"map<int32, Object>", marshalerAllOptions,
		&pb.Mappy{Objjy: map[int32]*pb.Simple3{1: &pb.Simple3{Dub: 1}}}, objjyPrettyJSON},
	{"map<int64, string>", marshaler, &pb.Mappy{Buggy: map[int64]string{1234: "yup"}},
		`{"buggy":{"1234":"yup"}}`},
	{"map<bool, bool>", marshaler, &pb.Mappy{Booly: map[bool]bool{false: true}}, `{"booly":{"false":true}}`},
	{"proto2 map<int64, string>", marshaler, &pb.Maps{MInt64Str: map[int64]string{213: "cat"}},
		`{"m_int64_str":{"213":"cat"}}`},
	{"proto2 map<bool, Object>", marshaler,
		&pb.Maps{MBoolSimple: map[bool]*pb.Simple{true: &pb.Simple{OInt32: proto.Int32(1)}}},
		`{"m_bool_simple":{"true":{"o_int32":1}}}`},
	{"oneof, not set", marshaler, &pb.MsgWithOneof{}, `{}`},
	{"oneof, set", marshaler, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Title{"Grand Poobah"}}, `{"title":"Grand Poobah"}`},
	{"proto2 extension", marshaler, realNumber, realNumberJSON},
}

func TestMarshaling(t *testing.T) {
	for _, tt := range marshalingTests {
		json, err := tt.marshaler.MarshalToString(tt.pb)
		if err != nil {
			t.Errorf("%s: marshaling error: %v", tt.desc, err)
		} else if tt.json != json {
			t.Errorf("%s: got [%v] want [%v]", tt.desc, json, tt.json)
		}
	}
}

var unmarshalingTests = []struct {
	desc string
	json string
	pb   proto.Message
}{
	{"simple flat object", simpleObjectJSON, simpleObject},
	{"simple pretty object", simpleObjectPrettyJSON, simpleObject},
	{"repeated fields flat object", repeatsObjectJSON, repeatsObject},
	{"repeated fields pretty object", repeatsObjectPrettyJSON, repeatsObject},
	{"nested message/enum flat object", complexObjectJSON, complexObject},
	{"nested message/enum pretty object", complexObjectPrettyJSON, complexObject},
	{"enum-string object", `{"color":"BLUE"}`, &pb.Widget{Color: pb.Widget_BLUE.Enum()}},
	{"enum-value object", "{\n \"color\": 2\n}", &pb.Widget{Color: pb.Widget_BLUE.Enum()}},
	{"proto3 enum string", `{"hilarity":"PUNS"}`, &proto3pb.Message{Hilarity: proto3pb.Message_PUNS}},
	{"proto3 enum value", `{"hilarity":1}`, &proto3pb.Message{Hilarity: proto3pb.Message_PUNS}},
	{"unknown enum value object",
		"{\n  \"color\": 1000,\n  \"r_color\": [\n    \"RED\"\n  ]\n}",
		&pb.Widget{Color: pb.Widget_Color(1000).Enum(), RColor: []pb.Widget_Color{pb.Widget_RED}}},
	{"unquoted int64 object", `{"o_int64":-314}`, &pb.Simple{OInt64: proto.Int64(-314)}},
	{"unquoted uint64 object", `{"o_uint64":123}`, &pb.Simple{OUint64: proto.Uint64(123)}},
	{"map<int64, int32>", `{"nummy":{"1":2,"3":4}}`, &pb.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}},
	{"map<string, string>", `{"strry":{"\"one\"":"two","three":"four"}}`, &pb.Mappy{Strry: map[string]string{`"one"`: "two", "three": "four"}}},
	{"map<int32, Object>", `{"objjy":{"1":{"dub":1}}}`, &pb.Mappy{Objjy: map[int32]*pb.Simple3{1: &pb.Simple3{Dub: 1}}}},
	{"oneof", `{"salary":31000}`, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Salary{31000}}},
}

func TestUnmarshaling(t *testing.T) {
	for _, tt := range unmarshalingTests {
		// Make a new instance of the type of our expected object.
		p := reflect.New(reflect.TypeOf(tt.pb).Elem()).Interface().(proto.Message)

		err := UnmarshalString(tt.json, p)
		if err != nil {
			t.Error(err)
			continue
		}

		// For easier diffs, compare text strings of the protos.
		exp := proto.MarshalTextString(tt.pb)
		act := proto.MarshalTextString(p)
		if string(exp) != string(act) {
			t.Errorf("%s: got [%s] want [%s]", tt.desc, act, exp)
		}
	}
}

var unmarshalingShouldError = []struct {
	desc string
	in   string
	pb   proto.Message
}{
	{"a value", "666", new(pb.Simple)},
	{"gibberish", "{adskja123;l23=-=", new(pb.Simple)},
	{"unknown enum name", `{"hilarity":"DAVE"}`, new(proto3pb.Message)},
}

func TestUnmarshalingBadInput(t *testing.T) {
	for _, tt := range unmarshalingShouldError {
		err := UnmarshalString(tt.in, tt.pb)
		if err == nil {
			t.Errorf("an error was expected when parsing %q instead of an object", tt.desc)
		}
	}
}
