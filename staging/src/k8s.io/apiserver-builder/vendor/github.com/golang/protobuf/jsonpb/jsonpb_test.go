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
	"bytes"
	"encoding/json"
	"io"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"

	pb "github.com/golang/protobuf/jsonpb/jsonpb_test_proto"
	proto3pb "github.com/golang/protobuf/proto/proto3_proto"
	anypb "github.com/golang/protobuf/ptypes/any"
	durpb "github.com/golang/protobuf/ptypes/duration"
	stpb "github.com/golang/protobuf/ptypes/struct"
	tspb "github.com/golang/protobuf/ptypes/timestamp"
	wpb "github.com/golang/protobuf/ptypes/wrappers"
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
		`"oBool":true,` +
		`"oInt32":-32,` +
		`"oInt64":"-6400000000",` +
		`"oUint32":32,` +
		`"oUint64":"6400000000",` +
		`"oSint32":-13,` +
		`"oSint64":"-2600000000",` +
		`"oFloat":3.14,` +
		`"oDouble":6.02214179e+23,` +
		`"oString":"hello \"there\"",` +
		`"oBytes":"YmVlcCBib29w"` +
		`}`

	simpleObjectPrettyJSON = `{
  "oBool": true,
  "oInt32": -32,
  "oInt64": "-6400000000",
  "oUint32": 32,
  "oUint64": "6400000000",
  "oSint32": -13,
  "oSint64": "-2600000000",
  "oFloat": 3.14,
  "oDouble": 6.02214179e+23,
  "oString": "hello \"there\"",
  "oBytes": "YmVlcCBib29w"
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
		`"rBool":[true,false,true],` +
		`"rInt32":[-3,-4,-5],` +
		`"rInt64":["-123456789","-987654321"],` +
		`"rUint32":[1,2,3],` +
		`"rUint64":["6789012345","3456789012"],` +
		`"rSint32":[-1,-2,-3],` +
		`"rSint64":["-6789012345","-3456789012"],` +
		`"rFloat":[3.14,6.28],` +
		`"rDouble":[2.99792458e+08,6.62606957e-34],` +
		`"rString":["happy","days"],` +
		`"rBytes":["c2tpdHRsZXM=","bSZtJ3M="]` +
		`}`

	repeatsObjectPrettyJSON = `{
  "rBool": [
    true,
    false,
    true
  ],
  "rInt32": [
    -3,
    -4,
    -5
  ],
  "rInt64": [
    "-123456789",
    "-987654321"
  ],
  "rUint32": [
    1,
    2,
    3
  ],
  "rUint64": [
    "6789012345",
    "3456789012"
  ],
  "rSint32": [
    -1,
    -2,
    -3
  ],
  "rSint64": [
    "-6789012345",
    "-3456789012"
  ],
  "rFloat": [
    3.14,
    6.28
  ],
  "rDouble": [
    2.99792458e+08,
    6.62606957e-34
  ],
  "rString": [
    "happy",
    "days"
  ],
  "rBytes": [
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
		`"rColor":["RED","GREEN","BLUE"],` +
		`"simple":{"oInt32":-32},` +
		`"rSimple":[{"oInt32":-32},{"oInt64":"25"}],` +
		`"repeats":{"rString":["roses","red"]},` +
		`"rRepeats":[{"rString":["roses","red"]},{"rString":["violets","blue"]}]` +
		`}`

	complexObjectPrettyJSON = `{
  "color": "GREEN",
  "rColor": [
    "RED",
    "GREEN",
    "BLUE"
  ],
  "simple": {
    "oInt32": -32
  },
  "rSimple": [
    {
      "oInt32": -32
    },
    {
      "oInt64": "25"
    }
  ],
  "repeats": {
    "rString": [
      "roses",
      "red"
    ]
  },
  "rRepeats": [
    {
      "rString": [
        "roses",
        "red"
      ]
    },
    {
      "rString": [
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
  "rColor": [
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

	anySimple = &pb.KnownTypes{
		An: &anypb.Any{
			TypeUrl: "something.example.com/jsonpb.Simple",
			Value: []byte{
				// &pb.Simple{OBool:true}
				1 << 3, 1,
			},
		},
	}
	anySimpleJSON       = `{"an":{"@type":"something.example.com/jsonpb.Simple","oBool":true}}`
	anySimplePrettyJSON = `{
  "an": {
    "@type": "something.example.com/jsonpb.Simple",
    "oBool": true
  }
}`

	anyWellKnown = &pb.KnownTypes{
		An: &anypb.Any{
			TypeUrl: "type.googleapis.com/google.protobuf.Duration",
			Value: []byte{
				// &durpb.Duration{Seconds: 1, Nanos: 212000000 }
				1 << 3, 1, // seconds
				2 << 3, 0x80, 0xba, 0x8b, 0x65, // nanos
			},
		},
	}
	anyWellKnownJSON       = `{"an":{"@type":"type.googleapis.com/google.protobuf.Duration","value":"1.212s"}}`
	anyWellKnownPrettyJSON = `{
  "an": {
    "@type": "type.googleapis.com/google.protobuf.Duration",
    "value": "1.212s"
  }
}`
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
	{"repeated proto3 enum", Marshaler{},
		&proto3pb.Message{RFunny: []proto3pb.Message_Humour{
			proto3pb.Message_PUNS,
			proto3pb.Message_SLAPSTICK,
		}},
		`{"rFunny":["PUNS","SLAPSTICK"]}`},
	{"repeated proto3 enum as int", Marshaler{EnumsAsInts: true},
		&proto3pb.Message{RFunny: []proto3pb.Message_Humour{
			proto3pb.Message_PUNS,
			proto3pb.Message_SLAPSTICK,
		}},
		`{"rFunny":[1,2]}`},
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
	// TODO: This is broken.
	//{"map<string, enum>", marshaler, &pb.Mappy{Enumy: map[string]pb.Numeral{"XIV": pb.Numeral_ROMAN}}, `{"enumy":{"XIV":"ROMAN"}`},
	{"map<string, enum as int>", Marshaler{EnumsAsInts: true}, &pb.Mappy{Enumy: map[string]pb.Numeral{"XIV": pb.Numeral_ROMAN}}, `{"enumy":{"XIV":2}}`},
	{"proto2 map<int64, string>", marshaler, &pb.Maps{MInt64Str: map[int64]string{213: "cat"}},
		`{"mInt64Str":{"213":"cat"}}`},
	{"proto2 map<bool, Object>", marshaler,
		&pb.Maps{MBoolSimple: map[bool]*pb.Simple{true: &pb.Simple{OInt32: proto.Int32(1)}}},
		`{"mBoolSimple":{"true":{"oInt32":1}}}`},
	{"oneof, not set", marshaler, &pb.MsgWithOneof{}, `{}`},
	{"oneof, set", marshaler, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Title{"Grand Poobah"}}, `{"title":"Grand Poobah"}`},
	{"force orig_name", Marshaler{OrigName: true}, &pb.Simple{OInt32: proto.Int32(4)},
		`{"o_int32":4}`},
	{"proto2 extension", marshaler, realNumber, realNumberJSON},
	{"Any with message", marshaler, anySimple, anySimpleJSON},
	{"Any with message and indent", marshalerAllOptions, anySimple, anySimplePrettyJSON},
	{"Any with WKT", marshaler, anyWellKnown, anyWellKnownJSON},
	{"Any with WKT and indent", marshalerAllOptions, anyWellKnown, anyWellKnownPrettyJSON},
	{"Duration", marshaler, &pb.KnownTypes{Dur: &durpb.Duration{Seconds: 3}}, `{"dur":"3.000s"}`},
	{"Struct", marshaler, &pb.KnownTypes{St: &stpb.Struct{
		Fields: map[string]*stpb.Value{
			"one": &stpb.Value{Kind: &stpb.Value_StringValue{"loneliest number"}},
			"two": &stpb.Value{Kind: &stpb.Value_NullValue{stpb.NullValue_NULL_VALUE}},
		},
	}}, `{"st":{"one":"loneliest number","two":null}}`},
	{"Timestamp", marshaler, &pb.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 21e6}}, `{"ts":"2014-05-13T16:53:20.021Z"}`},

	{"DoubleValue", marshaler, &pb.KnownTypes{Dbl: &wpb.DoubleValue{Value: 1.2}}, `{"dbl":1.2}`},
	{"FloatValue", marshaler, &pb.KnownTypes{Flt: &wpb.FloatValue{Value: 1.2}}, `{"flt":1.2}`},
	{"Int64Value", marshaler, &pb.KnownTypes{I64: &wpb.Int64Value{Value: -3}}, `{"i64":"-3"}`},
	{"UInt64Value", marshaler, &pb.KnownTypes{U64: &wpb.UInt64Value{Value: 3}}, `{"u64":"3"}`},
	{"Int32Value", marshaler, &pb.KnownTypes{I32: &wpb.Int32Value{Value: -4}}, `{"i32":-4}`},
	{"UInt32Value", marshaler, &pb.KnownTypes{U32: &wpb.UInt32Value{Value: 4}}, `{"u32":4}`},
	{"BoolValue", marshaler, &pb.KnownTypes{Bool: &wpb.BoolValue{Value: true}}, `{"bool":true}`},
	{"StringValue", marshaler, &pb.KnownTypes{Str: &wpb.StringValue{Value: "plush"}}, `{"str":"plush"}`},
	{"BytesValue", marshaler, &pb.KnownTypes{Bytes: &wpb.BytesValue{Value: []byte("wow")}}, `{"bytes":"d293"}`},
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
	{"repeated proto3 enum", `{"rFunny":["PUNS","SLAPSTICK"]}`,
		&proto3pb.Message{RFunny: []proto3pb.Message_Humour{
			proto3pb.Message_PUNS,
			proto3pb.Message_SLAPSTICK,
		}}},
	{"repeated proto3 enum as int", `{"rFunny":[1,2]}`,
		&proto3pb.Message{RFunny: []proto3pb.Message_Humour{
			proto3pb.Message_PUNS,
			proto3pb.Message_SLAPSTICK,
		}}},
	{"repeated proto3 enum as mix of strings and ints", `{"rFunny":["PUNS",2]}`,
		&proto3pb.Message{RFunny: []proto3pb.Message_Humour{
			proto3pb.Message_PUNS,
			proto3pb.Message_SLAPSTICK,
		}}},
	{"unquoted int64 object", `{"oInt64":-314}`, &pb.Simple{OInt64: proto.Int64(-314)}},
	{"unquoted uint64 object", `{"oUint64":123}`, &pb.Simple{OUint64: proto.Uint64(123)}},
	{"map<int64, int32>", `{"nummy":{"1":2,"3":4}}`, &pb.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}},
	{"map<string, string>", `{"strry":{"\"one\"":"two","three":"four"}}`, &pb.Mappy{Strry: map[string]string{`"one"`: "two", "three": "four"}}},
	{"map<int32, Object>", `{"objjy":{"1":{"dub":1}}}`, &pb.Mappy{Objjy: map[int32]*pb.Simple3{1: &pb.Simple3{Dub: 1}}}},
	// TODO: This is broken.
	//{"map<string, enum>", `{"enumy":{"XIV":"ROMAN"}`, &pb.Mappy{Enumy: map[string]pb.Numeral{"XIV": pb.Numeral_ROMAN}}},
	{"map<string, enum as int>", `{"enumy":{"XIV":2}}`, &pb.Mappy{Enumy: map[string]pb.Numeral{"XIV": pb.Numeral_ROMAN}}},
	{"oneof", `{"salary":31000}`, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Salary{31000}}},
	{"oneof spec name", `{"country":"Australia"}`, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Country{"Australia"}}},
	{"oneof orig_name", `{"Country":"Australia"}`, &pb.MsgWithOneof{Union: &pb.MsgWithOneof_Country{"Australia"}}},
	{"orig_name input", `{"o_bool":true}`, &pb.Simple{OBool: proto.Bool(true)}},
	{"camelName input", `{"oBool":true}`, &pb.Simple{OBool: proto.Bool(true)}},

	{"Duration", `{"dur":"3.000s"}`, &pb.KnownTypes{Dur: &durpb.Duration{Seconds: 3}}},
	{"Timestamp", `{"ts":"2014-05-13T16:53:20.021Z"}`, &pb.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 21e6}}},

	{"DoubleValue", `{"dbl":1.2}`, &pb.KnownTypes{Dbl: &wpb.DoubleValue{Value: 1.2}}},
	{"FloatValue", `{"flt":1.2}`, &pb.KnownTypes{Flt: &wpb.FloatValue{Value: 1.2}}},
	{"Int64Value", `{"i64":"-3"}`, &pb.KnownTypes{I64: &wpb.Int64Value{Value: -3}}},
	{"UInt64Value", `{"u64":"3"}`, &pb.KnownTypes{U64: &wpb.UInt64Value{Value: 3}}},
	{"Int32Value", `{"i32":-4}`, &pb.KnownTypes{I32: &wpb.Int32Value{Value: -4}}},
	{"UInt32Value", `{"u32":4}`, &pb.KnownTypes{U32: &wpb.UInt32Value{Value: 4}}},
	{"BoolValue", `{"bool":true}`, &pb.KnownTypes{Bool: &wpb.BoolValue{Value: true}}},
	{"StringValue", `{"str":"plush"}`, &pb.KnownTypes{Str: &wpb.StringValue{Value: "plush"}}},
	{"BytesValue", `{"bytes":"d293"}`, &pb.KnownTypes{Bytes: &wpb.BytesValue{Value: []byte("wow")}}},
	// `null` is also a permissible value. Let's just test one.
	{"null DoubleValue", `{"dbl":null}`, &pb.KnownTypes{Dbl: &wpb.DoubleValue{}}},
}

func TestUnmarshaling(t *testing.T) {
	for _, tt := range unmarshalingTests {
		// Make a new instance of the type of our expected object.
		p := reflect.New(reflect.TypeOf(tt.pb).Elem()).Interface().(proto.Message)

		err := UnmarshalString(tt.json, p)
		if err != nil {
			t.Errorf("%s: %v", tt.desc, err)
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

func TestUnmarshalNext(t *testing.T) {
	// We only need to check against a few, not all of them.
	tests := unmarshalingTests[:5]

	// Create a buffer with many concatenated JSON objects.
	var b bytes.Buffer
	for _, tt := range tests {
		b.WriteString(tt.json)
	}

	dec := json.NewDecoder(&b)
	for _, tt := range tests {
		// Make a new instance of the type of our expected object.
		p := reflect.New(reflect.TypeOf(tt.pb).Elem()).Interface().(proto.Message)

		err := UnmarshalNext(dec, p)
		if err != nil {
			t.Errorf("%s: %v", tt.desc, err)
			continue
		}

		// For easier diffs, compare text strings of the protos.
		exp := proto.MarshalTextString(tt.pb)
		act := proto.MarshalTextString(p)
		if string(exp) != string(act) {
			t.Errorf("%s: got [%s] want [%s]", tt.desc, act, exp)
		}
	}

	p := &pb.Simple{}
	err := UnmarshalNext(dec, p)
	if err != io.EOF {
		t.Errorf("eof: got %v, expected io.EOF", err)
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
