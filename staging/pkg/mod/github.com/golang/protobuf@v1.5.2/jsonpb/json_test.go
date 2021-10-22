// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonpb

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"

	pb2 "github.com/golang/protobuf/internal/testprotos/jsonpb_proto"
	pb3 "github.com/golang/protobuf/internal/testprotos/proto3_proto"
	descpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
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

	simpleObject = &pb2.Simple{
		OInt32:     proto.Int32(-32),
		OInt32Str:  proto.Int32(-32),
		OInt64:     proto.Int64(-6400000000),
		OInt64Str:  proto.Int64(-6400000000),
		OUint32:    proto.Uint32(32),
		OUint32Str: proto.Uint32(32),
		OUint64:    proto.Uint64(6400000000),
		OUint64Str: proto.Uint64(6400000000),
		OSint32:    proto.Int32(-13),
		OSint32Str: proto.Int32(-13),
		OSint64:    proto.Int64(-2600000000),
		OSint64Str: proto.Int64(-2600000000),
		OFloat:     proto.Float32(3.14),
		OFloatStr:  proto.Float32(3.14),
		ODouble:    proto.Float64(6.02214179e23),
		ODoubleStr: proto.Float64(6.02214179e23),
		OBool:      proto.Bool(true),
		OString:    proto.String("hello \"there\""),
		OBytes:     []byte("beep boop"),
	}

	simpleObjectInputJSON = `{` +
		`"oBool":true,` +
		`"oInt32":-32,` +
		`"oInt32Str":"-32",` +
		`"oInt64":-6400000000,` +
		`"oInt64Str":"-6400000000",` +
		`"oUint32":32,` +
		`"oUint32Str":"32",` +
		`"oUint64":6400000000,` +
		`"oUint64Str":"6400000000",` +
		`"oSint32":-13,` +
		`"oSint32Str":"-13",` +
		`"oSint64":-2600000000,` +
		`"oSint64Str":"-2600000000",` +
		`"oFloat":3.14,` +
		`"oFloatStr":"3.14",` +
		`"oDouble":6.02214179e+23,` +
		`"oDoubleStr":"6.02214179e+23",` +
		`"oString":"hello \"there\"",` +
		`"oBytes":"YmVlcCBib29w"` +
		`}`

	simpleObjectOutputJSON = `{` +
		`"oBool":true,` +
		`"oInt32":-32,` +
		`"oInt32Str":-32,` +
		`"oInt64":"-6400000000",` +
		`"oInt64Str":"-6400000000",` +
		`"oUint32":32,` +
		`"oUint32Str":32,` +
		`"oUint64":"6400000000",` +
		`"oUint64Str":"6400000000",` +
		`"oSint32":-13,` +
		`"oSint32Str":-13,` +
		`"oSint64":"-2600000000",` +
		`"oSint64Str":"-2600000000",` +
		`"oFloat":3.14,` +
		`"oFloatStr":3.14,` +
		`"oDouble":6.02214179e+23,` +
		`"oDoubleStr":6.02214179e+23,` +
		`"oString":"hello \"there\"",` +
		`"oBytes":"YmVlcCBib29w"` +
		`}`

	simpleObjectInputPrettyJSON = `{
  "oBool": true,
  "oInt32": -32,
  "oInt32Str": "-32",
  "oInt64": -6400000000,
  "oInt64Str": "-6400000000",
  "oUint32": 32,
  "oUint32Str": "32",
  "oUint64": 6400000000,
  "oUint64Str": "6400000000",
  "oSint32": -13,
  "oSint32Str": "-13",
  "oSint64": -2600000000,
  "oSint64Str": "-2600000000",
  "oFloat": 3.14,
  "oFloatStr": "3.14",
  "oDouble": 6.02214179e+23,
  "oDoubleStr": "6.02214179e+23",
  "oString": "hello \"there\"",
  "oBytes": "YmVlcCBib29w"
}`

	simpleObjectOutputPrettyJSON = `{
  "oBool": true,
  "oInt32": -32,
  "oInt32Str": -32,
  "oInt64": "-6400000000",
  "oInt64Str": "-6400000000",
  "oUint32": 32,
  "oUint32Str": 32,
  "oUint64": "6400000000",
  "oUint64Str": "6400000000",
  "oSint32": -13,
  "oSint32Str": -13,
  "oSint64": "-2600000000",
  "oSint64Str": "-2600000000",
  "oFloat": 3.14,
  "oFloatStr": 3.14,
  "oDouble": 6.02214179e+23,
  "oDoubleStr": 6.02214179e+23,
  "oString": "hello \"there\"",
  "oBytes": "YmVlcCBib29w"
}`

	repeatsObject = &pb2.Repeats{
		RBool:   []bool{true, false, true},
		RInt32:  []int32{-3, -4, -5},
		RInt64:  []int64{-123456789, -987654321},
		RUint32: []uint32{1, 2, 3},
		RUint64: []uint64{6789012345, 3456789012},
		RSint32: []int32{-1, -2, -3},
		RSint64: []int64{-6789012345, -3456789012},
		RFloat:  []float32{3.14, 6.28},
		RDouble: []float64{299792458 * 1e20, 6.62606957e-34},
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
		`"rDouble":[2.99792458e+28,6.62606957e-34],` +
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
    2.99792458e+28,
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

	innerSimple   = &pb2.Simple{OInt32: proto.Int32(-32)}
	innerSimple2  = &pb2.Simple{OInt64: proto.Int64(25)}
	innerRepeats  = &pb2.Repeats{RString: []string{"roses", "red"}}
	innerRepeats2 = &pb2.Repeats{RString: []string{"violets", "blue"}}
	complexObject = &pb2.Widget{
		Color:    pb2.Widget_GREEN.Enum(),
		RColor:   []pb2.Widget_Color{pb2.Widget_RED, pb2.Widget_GREEN, pb2.Widget_BLUE},
		Simple:   innerSimple,
		RSimple:  []*pb2.Simple{innerSimple, innerSimple2},
		Repeats:  innerRepeats,
		RRepeats: []*pb2.Repeats{innerRepeats, innerRepeats2},
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
	realNumber     = &pb2.Real{Value: proto.Float64(3.14159265359)}
	realNumberName = "Pi"
	complexNumber  = &pb2.Complex{Imaginary: proto.Float64(0.5772156649)}
	realNumberJSON = `{` +
		`"value":3.14159265359,` +
		`"[jsonpb_test.Complex.real_extension]":{"imaginary":0.5772156649},` +
		`"[jsonpb_test.name]":"Pi"` +
		`}`

	anySimple = &pb2.KnownTypes{
		An: &anypb.Any{
			TypeUrl: "something.example.com/jsonpb_test.Simple",
			Value: []byte{
				// &pb2.Simple{OBool:true}
				1 << 3, 1,
			},
		},
	}
	anySimpleJSON       = `{"an":{"@type":"something.example.com/jsonpb_test.Simple","oBool":true}}`
	anySimplePrettyJSON = `{
  "an": {
    "@type": "something.example.com/jsonpb_test.Simple",
    "oBool": true
  }
}`

	anyWellKnown = &pb2.KnownTypes{
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

	nonFinites = &pb2.NonFinites{
		FNan:  proto.Float32(float32(math.NaN())),
		FPinf: proto.Float32(float32(math.Inf(1))),
		FNinf: proto.Float32(float32(math.Inf(-1))),
		DNan:  proto.Float64(float64(math.NaN())),
		DPinf: proto.Float64(float64(math.Inf(1))),
		DNinf: proto.Float64(float64(math.Inf(-1))),
	}
	nonFinitesJSON = `{` +
		`"fNan":"NaN",` +
		`"fPinf":"Infinity",` +
		`"fNinf":"-Infinity",` +
		`"dNan":"NaN",` +
		`"dPinf":"Infinity",` +
		`"dNinf":"-Infinity"` +
		`}`
)

func init() {
	if err := proto.SetExtension(realNumber, pb2.E_Name, &realNumberName); err != nil {
		panic(err)
	}
	if err := proto.SetExtension(realNumber, pb2.E_Complex_RealExtension, complexNumber); err != nil {
		panic(err)
	}
}

var marshalingTests = []struct {
	desc      string
	marshaler Marshaler
	pb        proto.Message
	json      string
}{
	{"simple flat object", marshaler, simpleObject, simpleObjectOutputJSON},
	{"simple pretty object", marshalerAllOptions, simpleObject, simpleObjectOutputPrettyJSON},
	{"non-finite floats fields object", marshaler, nonFinites, nonFinitesJSON},
	{"repeated fields flat object", marshaler, repeatsObject, repeatsObjectJSON},
	{"repeated fields pretty object", marshalerAllOptions, repeatsObject, repeatsObjectPrettyJSON},
	{"nested message/enum flat object", marshaler, complexObject, complexObjectJSON},
	{"nested message/enum pretty object", marshalerAllOptions, complexObject, complexObjectPrettyJSON},
	{"enum-string flat object", Marshaler{},
		&pb2.Widget{Color: pb2.Widget_BLUE.Enum()}, `{"color":"BLUE"}`},
	{"enum-value pretty object", Marshaler{EnumsAsInts: true, Indent: " "},
		&pb2.Widget{Color: pb2.Widget_BLUE.Enum()}, colorPrettyJSON},
	{"unknown enum value object", marshalerAllOptions,
		&pb2.Widget{Color: pb2.Widget_Color(1000).Enum(), RColor: []pb2.Widget_Color{pb2.Widget_RED}}, colorListPrettyJSON},
	{"repeated proto3 enum", Marshaler{},
		&pb3.Message{RFunny: []pb3.Message_Humour{
			pb3.Message_PUNS,
			pb3.Message_SLAPSTICK,
		}},
		`{"rFunny":["PUNS","SLAPSTICK"]}`},
	{"repeated proto3 enum as int", Marshaler{EnumsAsInts: true},
		&pb3.Message{RFunny: []pb3.Message_Humour{
			pb3.Message_PUNS,
			pb3.Message_SLAPSTICK,
		}},
		`{"rFunny":[1,2]}`},
	{"empty value", marshaler, &pb2.Simple3{}, `{}`},
	{"empty value emitted", Marshaler{EmitDefaults: true}, &pb2.Simple3{}, `{"dub":0}`},
	{"empty repeated emitted", Marshaler{EmitDefaults: true}, &pb2.SimpleSlice3{}, `{"slices":[]}`},
	{"empty map emitted", Marshaler{EmitDefaults: true}, &pb2.SimpleMap3{}, `{"stringy":{}}`},
	{"nested struct null", Marshaler{EmitDefaults: true}, &pb2.SimpleNull3{}, `{"simple":null}`},
	{"map<int64, int32>", marshaler, &pb2.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}, `{"nummy":{"1":2,"3":4}}`},
	{"map<int64, int32>", marshalerAllOptions, &pb2.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}, nummyPrettyJSON},
	{"map<string, string>", marshaler,
		&pb2.Mappy{Strry: map[string]string{`"one"`: "two", "three": "four"}},
		`{"strry":{"\"one\"":"two","three":"four"}}`},
	{"map<int32, Object>", marshaler,
		&pb2.Mappy{Objjy: map[int32]*pb2.Simple3{1: {Dub: 1}}}, `{"objjy":{"1":{"dub":1}}}`},
	{"map<int32, Object>", marshalerAllOptions,
		&pb2.Mappy{Objjy: map[int32]*pb2.Simple3{1: {Dub: 1}}}, objjyPrettyJSON},
	{"map<int64, string>", marshaler, &pb2.Mappy{Buggy: map[int64]string{1234: "yup"}},
		`{"buggy":{"1234":"yup"}}`},
	{"map<bool, bool>", marshaler, &pb2.Mappy{Booly: map[bool]bool{false: true}}, `{"booly":{"false":true}}`},
	{"map<string, enum>", marshaler, &pb2.Mappy{Enumy: map[string]pb2.Numeral{"XIV": pb2.Numeral_ROMAN}}, `{"enumy":{"XIV":"ROMAN"}}`},
	{"map<string, enum as int>", Marshaler{EnumsAsInts: true}, &pb2.Mappy{Enumy: map[string]pb2.Numeral{"XIV": pb2.Numeral_ROMAN}}, `{"enumy":{"XIV":2}}`},
	{"map<int32, bool>", marshaler, &pb2.Mappy{S32Booly: map[int32]bool{1: true, 3: false, 10: true, 12: false}}, `{"s32booly":{"1":true,"3":false,"10":true,"12":false}}`},
	{"map<int64, bool>", marshaler, &pb2.Mappy{S64Booly: map[int64]bool{1: true, 3: false, 10: true, 12: false}}, `{"s64booly":{"1":true,"3":false,"10":true,"12":false}}`},
	{"map<uint32, bool>", marshaler, &pb2.Mappy{U32Booly: map[uint32]bool{1: true, 3: false, 10: true, 12: false}}, `{"u32booly":{"1":true,"3":false,"10":true,"12":false}}`},
	{"map<uint64, bool>", marshaler, &pb2.Mappy{U64Booly: map[uint64]bool{1: true, 3: false, 10: true, 12: false}}, `{"u64booly":{"1":true,"3":false,"10":true,"12":false}}`},
	{"proto2 map<int64, string>", marshaler, &pb2.Maps{MInt64Str: map[int64]string{213: "cat"}},
		`{"mInt64Str":{"213":"cat"}}`},
	{"proto2 map<bool, Object>", marshaler,
		&pb2.Maps{MBoolSimple: map[bool]*pb2.Simple{true: {OInt32: proto.Int32(1)}}},
		`{"mBoolSimple":{"true":{"oInt32":1}}}`},
	{"oneof, not set", marshaler, &pb2.MsgWithOneof{}, `{}`},
	{"oneof, set", marshaler, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_Title{"Grand Poobah"}}, `{"title":"Grand Poobah"}`},
	{"force orig_name", Marshaler{OrigName: true}, &pb2.Simple{OInt32: proto.Int32(4)},
		`{"o_int32":4}`},
	{"proto2 extension", marshaler, realNumber, realNumberJSON},
	{"Any with message", marshaler, anySimple, anySimpleJSON},
	{"Any with message and indent", marshalerAllOptions, anySimple, anySimplePrettyJSON},
	{"Any with WKT", marshaler, anyWellKnown, anyWellKnownJSON},
	{"Any with WKT and indent", marshalerAllOptions, anyWellKnown, anyWellKnownPrettyJSON},
	{"Duration empty", marshaler, &durpb.Duration{}, `"0s"`},
	{"Duration with secs", marshaler, &durpb.Duration{Seconds: 3}, `"3s"`},
	{"Duration with -secs", marshaler, &durpb.Duration{Seconds: -3}, `"-3s"`},
	{"Duration with nanos", marshaler, &durpb.Duration{Nanos: 1e6}, `"0.001s"`},
	{"Duration with -nanos", marshaler, &durpb.Duration{Nanos: -1e6}, `"-0.001s"`},
	{"Duration with large secs", marshaler, &durpb.Duration{Seconds: 1e10, Nanos: 1}, `"10000000000.000000001s"`},
	{"Duration with 6-digit nanos", marshaler, &durpb.Duration{Nanos: 1e4}, `"0.000010s"`},
	{"Duration with 3-digit nanos", marshaler, &durpb.Duration{Nanos: 1e6}, `"0.001s"`},
	{"Duration with -secs -nanos", marshaler, &durpb.Duration{Seconds: -123, Nanos: -450}, `"-123.000000450s"`},
	{"Duration max value", marshaler, &durpb.Duration{Seconds: 315576000000, Nanos: 999999999}, `"315576000000.999999999s"`},
	{"Duration min value", marshaler, &durpb.Duration{Seconds: -315576000000, Nanos: -999999999}, `"-315576000000.999999999s"`},
	{"Struct", marshaler, &pb2.KnownTypes{St: &stpb.Struct{
		Fields: map[string]*stpb.Value{
			"one": {Kind: &stpb.Value_StringValue{"loneliest number"}},
			"two": {Kind: &stpb.Value_NullValue{stpb.NullValue_NULL_VALUE}},
		},
	}}, `{"st":{"one":"loneliest number","two":null}}`},
	{"empty ListValue", marshaler, &pb2.KnownTypes{Lv: &stpb.ListValue{}}, `{"lv":[]}`},
	{"basic ListValue", marshaler, &pb2.KnownTypes{Lv: &stpb.ListValue{Values: []*stpb.Value{
		{Kind: &stpb.Value_StringValue{"x"}},
		{Kind: &stpb.Value_NullValue{}},
		{Kind: &stpb.Value_NumberValue{3}},
		{Kind: &stpb.Value_BoolValue{true}},
	}}}, `{"lv":["x",null,3,true]}`},
	{"Timestamp", marshaler, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 21e6}}, `{"ts":"2014-05-13T16:53:20.021Z"}`},
	{"Timestamp", marshaler, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 0}}, `{"ts":"2014-05-13T16:53:20Z"}`},
	{"number Value", marshaler, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_NumberValue{1}}}, `{"val":1}`},
	{"null Value", marshaler, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_NullValue{stpb.NullValue_NULL_VALUE}}}, `{"val":null}`},
	{"string number value", marshaler, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_StringValue{"9223372036854775807"}}}, `{"val":"9223372036854775807"}`},
	{"list of lists Value", marshaler, &pb2.KnownTypes{Val: &stpb.Value{
		Kind: &stpb.Value_ListValue{&stpb.ListValue{
			Values: []*stpb.Value{
				{Kind: &stpb.Value_StringValue{"x"}},
				{Kind: &stpb.Value_ListValue{&stpb.ListValue{
					Values: []*stpb.Value{
						{Kind: &stpb.Value_ListValue{&stpb.ListValue{
							Values: []*stpb.Value{{Kind: &stpb.Value_StringValue{"y"}}},
						}}},
						{Kind: &stpb.Value_StringValue{"z"}},
					},
				}}},
			},
		}},
	}}, `{"val":["x",[["y"],"z"]]}`},

	{"DoubleValue", marshaler, &pb2.KnownTypes{Dbl: &wpb.DoubleValue{Value: 1.2}}, `{"dbl":1.2}`},
	{"FloatValue", marshaler, &pb2.KnownTypes{Flt: &wpb.FloatValue{Value: 1.2}}, `{"flt":1.2}`},
	{"Int64Value", marshaler, &pb2.KnownTypes{I64: &wpb.Int64Value{Value: -3}}, `{"i64":"-3"}`},
	{"UInt64Value", marshaler, &pb2.KnownTypes{U64: &wpb.UInt64Value{Value: 3}}, `{"u64":"3"}`},
	{"Int32Value", marshaler, &pb2.KnownTypes{I32: &wpb.Int32Value{Value: -4}}, `{"i32":-4}`},
	{"UInt32Value", marshaler, &pb2.KnownTypes{U32: &wpb.UInt32Value{Value: 4}}, `{"u32":4}`},
	{"BoolValue", marshaler, &pb2.KnownTypes{Bool: &wpb.BoolValue{Value: true}}, `{"bool":true}`},
	{"StringValue", marshaler, &pb2.KnownTypes{Str: &wpb.StringValue{Value: "plush"}}, `{"str":"plush"}`},
	{"BytesValue", marshaler, &pb2.KnownTypes{Bytes: &wpb.BytesValue{Value: []byte("wow")}}, `{"bytes":"d293"}`},

	{"required", marshaler, &pb2.MsgWithRequired{Str: proto.String("hello")}, `{"str":"hello"}`},
	{"required bytes", marshaler, &pb2.MsgWithRequiredBytes{Byts: []byte{}}, `{"byts":""}`},
}

func TestMarshaling(t *testing.T) {
	for _, tt := range marshalingTests {
		json, err := tt.marshaler.MarshalToString(tt.pb)
		if err != nil {
			t.Errorf("%s: marshaling error: %v", tt.desc, err)
		} else if tt.json != json {
			t.Errorf("%s:\ngot:  %v\nwant: %v", tt.desc, json, tt.json)
		}
	}
}

func TestMarshalingNil(t *testing.T) {
	var msg *pb2.Simple
	m := &Marshaler{}
	if _, err := m.MarshalToString(msg); err == nil {
		t.Errorf("mashaling nil returned no error")
	}
}

func TestMarshalIllegalTime(t *testing.T) {
	tests := []struct {
		pb   proto.Message
		fail bool
	}{
		{&durpb.Duration{Seconds: 1, Nanos: 0}, false},
		{&durpb.Duration{Seconds: -1, Nanos: 0}, false},
		{&durpb.Duration{Seconds: 1, Nanos: -1}, true},
		{&durpb.Duration{Seconds: -1, Nanos: 1}, true},
		{&durpb.Duration{Seconds: 315576000001}, true},
		{&durpb.Duration{Seconds: -315576000001}, true},
		{&durpb.Duration{Seconds: 1, Nanos: 1000000000}, true},
		{&durpb.Duration{Seconds: -1, Nanos: -1000000000}, true},
		{&tspb.Timestamp{Seconds: 1, Nanos: 1}, false},
		{&tspb.Timestamp{Seconds: 1, Nanos: -1}, true},
		{&tspb.Timestamp{Seconds: 1, Nanos: 1000000000}, true},
	}
	for _, tt := range tests {
		_, err := marshaler.MarshalToString(tt.pb)
		if err == nil && tt.fail {
			t.Errorf("marshaler.MarshalToString(%v) = _, <nil>; want _, <non-nil>", tt.pb)
		}
		if err != nil && !tt.fail {
			t.Errorf("marshaler.MarshalToString(%v) = _, %v; want _, <nil>", tt.pb, err)
		}
	}
}

func TestMarshalJSONPBMarshaler(t *testing.T) {
	rawJson := `{ "foo": "bar", "baz": [0, 1, 2, 3] }`
	msg := dynamicMessage{RawJson: rawJson}
	str, err := new(Marshaler).MarshalToString(&msg)
	if err != nil {
		t.Errorf("an unexpected error while marshaling JSONPBMarshaler: %v", err)
	}
	if str != rawJson {
		t.Errorf("marshaling JSON produced incorrect output: got %s, wanted %s", str, rawJson)
	}
}

func TestMarshalAnyJSONPBMarshaler(t *testing.T) {
	msg := dynamicMessage{RawJson: `{ "foo": "bar", "baz": [0, 1, 2, 3] }`}
	a, err := ptypes.MarshalAny(&msg)
	if err != nil {
		t.Errorf("an unexpected error while marshaling to Any: %v", err)
	}
	str, err := new(Marshaler).MarshalToString(a)
	if err != nil {
		t.Errorf("an unexpected error while marshaling Any to JSON: %v", err)
	}
	// after custom marshaling, it's round-tripped through JSON decoding/encoding already,
	// so the keys are sorted, whitespace is compacted, and "@type" key has been added
	want := `{"@type":"type.googleapis.com/` + dynamicMessageName + `","baz":[0,1,2,3],"foo":"bar"}`
	if str != want {
		t.Errorf("marshaling JSON produced incorrect output: got %s, wanted %s", str, want)
	}
}

func TestMarshalWithCustomValidation(t *testing.T) {
	msg := dynamicMessage{RawJson: `{ "foo": "bar", "baz": [0, 1, 2, 3] }`, Dummy: &dynamicMessage{}}

	js, err := new(Marshaler).MarshalToString(&msg)
	if err != nil {
		t.Errorf("an unexpected error while marshaling to json: %v", err)
	}
	err = Unmarshal(strings.NewReader(js), &msg)
	if err != nil {
		t.Errorf("an unexpected error while unmarshaling from json: %v", err)
	}
}

// Test marshaling message containing unset required fields should produce error.
func TestMarshalUnsetRequiredFields(t *testing.T) {
	msgExt := &pb2.Real{}
	proto.SetExtension(msgExt, pb2.E_Extm, &pb2.MsgWithRequired{})

	tests := []struct {
		desc      string
		marshaler *Marshaler
		pb        proto.Message
	}{
		{
			desc:      "direct required field",
			marshaler: &Marshaler{},
			pb:        &pb2.MsgWithRequired{},
		},
		{
			desc:      "direct required field + emit defaults",
			marshaler: &Marshaler{EmitDefaults: true},
			pb:        &pb2.MsgWithRequired{},
		},
		{
			desc:      "indirect required field",
			marshaler: &Marshaler{},
			pb:        &pb2.MsgWithIndirectRequired{Subm: &pb2.MsgWithRequired{}},
		},
		{
			desc:      "indirect required field + emit defaults",
			marshaler: &Marshaler{EmitDefaults: true},
			pb:        &pb2.MsgWithIndirectRequired{Subm: &pb2.MsgWithRequired{}},
		},
		{
			desc:      "direct required wkt field",
			marshaler: &Marshaler{},
			pb:        &pb2.MsgWithRequiredWKT{},
		},
		{
			desc:      "direct required wkt field + emit defaults",
			marshaler: &Marshaler{EmitDefaults: true},
			pb:        &pb2.MsgWithRequiredWKT{},
		},
		{
			desc:      "direct required bytes field",
			marshaler: &Marshaler{},
			pb:        &pb2.MsgWithRequiredBytes{},
		},
		{
			desc:      "required in map value",
			marshaler: &Marshaler{},
			pb: &pb2.MsgWithIndirectRequired{
				MapField: map[string]*pb2.MsgWithRequired{
					"key": {},
				},
			},
		},
		{
			desc:      "required in repeated item",
			marshaler: &Marshaler{},
			pb: &pb2.MsgWithIndirectRequired{
				SliceField: []*pb2.MsgWithRequired{
					{Str: proto.String("hello")},
					{},
				},
			},
		},
		{
			desc:      "required inside oneof",
			marshaler: &Marshaler{},
			pb: &pb2.MsgWithOneof{
				Union: &pb2.MsgWithOneof_MsgWithRequired{&pb2.MsgWithRequired{}},
			},
		},
		{
			desc:      "required inside extension",
			marshaler: &Marshaler{},
			pb:        msgExt,
		},
	}

	for _, tc := range tests {
		if _, err := tc.marshaler.MarshalToString(tc.pb); err == nil {
			t.Errorf("%s: expected error while marshaling with unset required fields %+v", tc.desc, tc.pb)
		}
	}
}

var unmarshalingTests = []struct {
	desc        string
	unmarshaler Unmarshaler
	json        string
	pb          proto.Message
}{
	{"simple flat object", Unmarshaler{}, simpleObjectInputJSON, simpleObject},
	{"simple pretty object", Unmarshaler{}, simpleObjectInputPrettyJSON, simpleObject},
	{"repeated fields flat object", Unmarshaler{}, repeatsObjectJSON, repeatsObject},
	{"repeated fields pretty object", Unmarshaler{}, repeatsObjectPrettyJSON, repeatsObject},
	{"nested message/enum flat object", Unmarshaler{}, complexObjectJSON, complexObject},
	{"nested message/enum pretty object", Unmarshaler{}, complexObjectPrettyJSON, complexObject},
	{"enum-string object", Unmarshaler{}, `{"color":"BLUE"}`, &pb2.Widget{Color: pb2.Widget_BLUE.Enum()}},
	{"enum-value object", Unmarshaler{}, "{\n \"color\": 2\n}", &pb2.Widget{Color: pb2.Widget_BLUE.Enum()}},
	{"unknown field with allowed option", Unmarshaler{AllowUnknownFields: true}, `{"unknown": "foo"}`, new(pb2.Simple)},
	{"proto3 enum string", Unmarshaler{}, `{"hilarity":"PUNS"}`, &pb3.Message{Hilarity: pb3.Message_PUNS}},
	{"proto3 enum value", Unmarshaler{}, `{"hilarity":1}`, &pb3.Message{Hilarity: pb3.Message_PUNS}},
	{"unknown enum value object",
		Unmarshaler{},
		"{\n  \"color\": 1000,\n  \"r_color\": [\n    \"RED\"\n  ]\n}",
		&pb2.Widget{Color: pb2.Widget_Color(1000).Enum(), RColor: []pb2.Widget_Color{pb2.Widget_RED}}},
	{"repeated proto3 enum", Unmarshaler{}, `{"rFunny":["PUNS","SLAPSTICK"]}`,
		&pb3.Message{RFunny: []pb3.Message_Humour{
			pb3.Message_PUNS,
			pb3.Message_SLAPSTICK,
		}}},
	{"repeated proto3 enum as int", Unmarshaler{}, `{"rFunny":[1,2]}`,
		&pb3.Message{RFunny: []pb3.Message_Humour{
			pb3.Message_PUNS,
			pb3.Message_SLAPSTICK,
		}}},
	{"repeated proto3 enum as mix of strings and ints", Unmarshaler{}, `{"rFunny":["PUNS",2]}`,
		&pb3.Message{RFunny: []pb3.Message_Humour{
			pb3.Message_PUNS,
			pb3.Message_SLAPSTICK,
		}}},
	{"unquoted int64 object", Unmarshaler{}, `{"oInt64":-314}`, &pb2.Simple{OInt64: proto.Int64(-314)}},
	{"unquoted uint64 object", Unmarshaler{}, `{"oUint64":123}`, &pb2.Simple{OUint64: proto.Uint64(123)}},
	{"NaN", Unmarshaler{}, `{"oDouble":"NaN"}`, &pb2.Simple{ODouble: proto.Float64(math.NaN())}},
	{"Inf", Unmarshaler{}, `{"oFloat":"Infinity"}`, &pb2.Simple{OFloat: proto.Float32(float32(math.Inf(1)))}},
	{"-Inf", Unmarshaler{}, `{"oDouble":"-Infinity"}`, &pb2.Simple{ODouble: proto.Float64(math.Inf(-1))}},
	{"map<int64, int32>", Unmarshaler{}, `{"nummy":{"1":2,"3":4}}`, &pb2.Mappy{Nummy: map[int64]int32{1: 2, 3: 4}}},
	{"map<string, string>", Unmarshaler{}, `{"strry":{"\"one\"":"two","three":"four"}}`, &pb2.Mappy{Strry: map[string]string{`"one"`: "two", "three": "four"}}},
	{"map<int32, Object>", Unmarshaler{}, `{"objjy":{"1":{"dub":1}}}`, &pb2.Mappy{Objjy: map[int32]*pb2.Simple3{1: {Dub: 1}}}},
	{"proto2 extension", Unmarshaler{}, realNumberJSON, realNumber},
	{"Any with message", Unmarshaler{}, anySimpleJSON, anySimple},
	{"Any with message and indent", Unmarshaler{}, anySimplePrettyJSON, anySimple},
	{"Any with WKT", Unmarshaler{}, anyWellKnownJSON, anyWellKnown},
	{"Any with WKT and indent", Unmarshaler{}, anyWellKnownPrettyJSON, anyWellKnown},
	{"map<string, enum>", Unmarshaler{}, `{"enumy":{"XIV":"ROMAN"}}`, &pb2.Mappy{Enumy: map[string]pb2.Numeral{"XIV": pb2.Numeral_ROMAN}}},
	{"map<string, enum as int>", Unmarshaler{}, `{"enumy":{"XIV":2}}`, &pb2.Mappy{Enumy: map[string]pb2.Numeral{"XIV": pb2.Numeral_ROMAN}}},
	{"oneof", Unmarshaler{}, `{"salary":31000}`, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_Salary{31000}}},
	{"oneof spec name", Unmarshaler{}, `{"Country":"Australia"}`, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_Country{"Australia"}}},
	{"oneof orig_name", Unmarshaler{}, `{"Country":"Australia"}`, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_Country{"Australia"}}},
	{"oneof spec name2", Unmarshaler{}, `{"homeAddress":"Australia"}`, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_HomeAddress{"Australia"}}},
	{"oneof orig_name2", Unmarshaler{}, `{"home_address":"Australia"}`, &pb2.MsgWithOneof{Union: &pb2.MsgWithOneof_HomeAddress{"Australia"}}},
	{"orig_name input", Unmarshaler{}, `{"o_bool":true}`, &pb2.Simple{OBool: proto.Bool(true)}},
	{"camelName input", Unmarshaler{}, `{"oBool":true}`, &pb2.Simple{OBool: proto.Bool(true)}},

	{"Duration", Unmarshaler{}, `{"dur":"3.000s"}`, &pb2.KnownTypes{Dur: &durpb.Duration{Seconds: 3}}},
	{"Duration", Unmarshaler{}, `{"dur":"4s"}`, &pb2.KnownTypes{Dur: &durpb.Duration{Seconds: 4}}},
	{"Duration with unicode", Unmarshaler{}, `{"dur": "3\u0073"}`, &pb2.KnownTypes{Dur: &durpb.Duration{Seconds: 3}}},
	{"null Duration", Unmarshaler{}, `{"dur":null}`, &pb2.KnownTypes{Dur: nil}},
	{"Timestamp", Unmarshaler{}, `{"ts":"2014-05-13T16:53:20.021Z"}`, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 21e6}}},
	{"Timestamp", Unmarshaler{}, `{"ts":"2014-05-13T16:53:20Z"}`, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 0}}},
	{"Timestamp with unicode", Unmarshaler{}, `{"ts": "2014-05-13T16:53:20\u005a"}`, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: 14e8, Nanos: 0}}},
	{"PreEpochTimestamp", Unmarshaler{}, `{"ts":"1969-12-31T23:59:58.999999995Z"}`, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: -2, Nanos: 999999995}}},
	{"ZeroTimeTimestamp", Unmarshaler{}, `{"ts":"0001-01-01T00:00:00Z"}`, &pb2.KnownTypes{Ts: &tspb.Timestamp{Seconds: -62135596800, Nanos: 0}}},
	{"null Timestamp", Unmarshaler{}, `{"ts":null}`, &pb2.KnownTypes{Ts: nil}},
	{"null Struct", Unmarshaler{}, `{"st": null}`, &pb2.KnownTypes{St: nil}},
	{"empty Struct", Unmarshaler{}, `{"st": {}}`, &pb2.KnownTypes{St: &stpb.Struct{}}},
	{"basic Struct", Unmarshaler{}, `{"st": {"a": "x", "b": null, "c": 3, "d": true}}`, &pb2.KnownTypes{St: &stpb.Struct{Fields: map[string]*stpb.Value{
		"a": {Kind: &stpb.Value_StringValue{"x"}},
		"b": {Kind: &stpb.Value_NullValue{}},
		"c": {Kind: &stpb.Value_NumberValue{3}},
		"d": {Kind: &stpb.Value_BoolValue{true}},
	}}}},
	{"nested Struct", Unmarshaler{}, `{"st": {"a": {"b": 1, "c": [{"d": true}, "f"]}}}`, &pb2.KnownTypes{St: &stpb.Struct{Fields: map[string]*stpb.Value{
		"a": {Kind: &stpb.Value_StructValue{&stpb.Struct{Fields: map[string]*stpb.Value{
			"b": {Kind: &stpb.Value_NumberValue{1}},
			"c": {Kind: &stpb.Value_ListValue{&stpb.ListValue{Values: []*stpb.Value{
				{Kind: &stpb.Value_StructValue{&stpb.Struct{Fields: map[string]*stpb.Value{"d": {Kind: &stpb.Value_BoolValue{true}}}}}},
				{Kind: &stpb.Value_StringValue{"f"}},
			}}}},
		}}}},
	}}}},
	{"null ListValue", Unmarshaler{}, `{"lv": null}`, &pb2.KnownTypes{Lv: nil}},
	{"empty ListValue", Unmarshaler{}, `{"lv": []}`, &pb2.KnownTypes{Lv: &stpb.ListValue{}}},
	{"basic ListValue", Unmarshaler{}, `{"lv": ["x", null, 3, true]}`, &pb2.KnownTypes{Lv: &stpb.ListValue{Values: []*stpb.Value{
		{Kind: &stpb.Value_StringValue{"x"}},
		{Kind: &stpb.Value_NullValue{}},
		{Kind: &stpb.Value_NumberValue{3}},
		{Kind: &stpb.Value_BoolValue{true}},
	}}}},
	{"number Value", Unmarshaler{}, `{"val":1}`, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_NumberValue{1}}}},
	{"null Value", Unmarshaler{}, `{"val":null}`, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_NullValue{stpb.NullValue_NULL_VALUE}}}},
	{"bool Value", Unmarshaler{}, `{"val":true}`, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_BoolValue{true}}}},
	{"string Value", Unmarshaler{}, `{"val":"x"}`, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_StringValue{"x"}}}},
	{"string number value", Unmarshaler{}, `{"val":"9223372036854775807"}`, &pb2.KnownTypes{Val: &stpb.Value{Kind: &stpb.Value_StringValue{"9223372036854775807"}}}},
	{"list of lists Value", Unmarshaler{}, `{"val":["x", [["y"], "z"]]}`, &pb2.KnownTypes{Val: &stpb.Value{
		Kind: &stpb.Value_ListValue{&stpb.ListValue{
			Values: []*stpb.Value{
				{Kind: &stpb.Value_StringValue{"x"}},
				{Kind: &stpb.Value_ListValue{&stpb.ListValue{
					Values: []*stpb.Value{
						{Kind: &stpb.Value_ListValue{&stpb.ListValue{
							Values: []*stpb.Value{{Kind: &stpb.Value_StringValue{"y"}}},
						}}},
						{Kind: &stpb.Value_StringValue{"z"}},
					},
				}}},
			},
		}}}}},

	{"DoubleValue", Unmarshaler{}, `{"dbl":1.2}`, &pb2.KnownTypes{Dbl: &wpb.DoubleValue{Value: 1.2}}},
	{"FloatValue", Unmarshaler{}, `{"flt":1.2}`, &pb2.KnownTypes{Flt: &wpb.FloatValue{Value: 1.2}}},
	{"Int64Value", Unmarshaler{}, `{"i64":"-3"}`, &pb2.KnownTypes{I64: &wpb.Int64Value{Value: -3}}},
	{"UInt64Value", Unmarshaler{}, `{"u64":"3"}`, &pb2.KnownTypes{U64: &wpb.UInt64Value{Value: 3}}},
	{"Int32Value", Unmarshaler{}, `{"i32":-4}`, &pb2.KnownTypes{I32: &wpb.Int32Value{Value: -4}}},
	{"UInt32Value", Unmarshaler{}, `{"u32":4}`, &pb2.KnownTypes{U32: &wpb.UInt32Value{Value: 4}}},
	{"BoolValue", Unmarshaler{}, `{"bool":true}`, &pb2.KnownTypes{Bool: &wpb.BoolValue{Value: true}}},
	{"StringValue", Unmarshaler{}, `{"str":"plush"}`, &pb2.KnownTypes{Str: &wpb.StringValue{Value: "plush"}}},
	{"StringValue containing escaped character", Unmarshaler{}, `{"str":"a\/b"}`, &pb2.KnownTypes{Str: &wpb.StringValue{Value: "a/b"}}},
	{"StructValue containing StringValue's", Unmarshaler{}, `{"escaped": "a\/b", "unicode": "\u00004E16\u0000754C"}`,
		&stpb.Struct{
			Fields: map[string]*stpb.Value{
				"escaped": {Kind: &stpb.Value_StringValue{"a/b"}},
				"unicode": {Kind: &stpb.Value_StringValue{"\u00004E16\u0000754C"}},
			},
		}},
	{"BytesValue", Unmarshaler{}, `{"bytes":"d293"}`, &pb2.KnownTypes{Bytes: &wpb.BytesValue{Value: []byte("wow")}}},

	// Ensure that `null` as a value ends up with a nil pointer instead of a [type]Value struct.
	{"null DoubleValue", Unmarshaler{}, `{"dbl":null}`, &pb2.KnownTypes{Dbl: nil}},
	{"null FloatValue", Unmarshaler{}, `{"flt":null}`, &pb2.KnownTypes{Flt: nil}},
	{"null Int64Value", Unmarshaler{}, `{"i64":null}`, &pb2.KnownTypes{I64: nil}},
	{"null UInt64Value", Unmarshaler{}, `{"u64":null}`, &pb2.KnownTypes{U64: nil}},
	{"null Int32Value", Unmarshaler{}, `{"i32":null}`, &pb2.KnownTypes{I32: nil}},
	{"null UInt32Value", Unmarshaler{}, `{"u32":null}`, &pb2.KnownTypes{U32: nil}},
	{"null BoolValue", Unmarshaler{}, `{"bool":null}`, &pb2.KnownTypes{Bool: nil}},
	{"null StringValue", Unmarshaler{}, `{"str":null}`, &pb2.KnownTypes{Str: nil}},
	{"null BytesValue", Unmarshaler{}, `{"bytes":null}`, &pb2.KnownTypes{Bytes: nil}},

	{"required", Unmarshaler{}, `{"str":"hello"}`, &pb2.MsgWithRequired{Str: proto.String("hello")}},
	{"required bytes", Unmarshaler{}, `{"byts": []}`, &pb2.MsgWithRequiredBytes{Byts: []byte{}}},
}

func TestUnmarshaling(t *testing.T) {
	for _, tt := range unmarshalingTests {
		// Make a new instance of the type of our wanted object.
		p := reflect.New(reflect.TypeOf(tt.pb).Elem()).Interface().(proto.Message)

		err := tt.unmarshaler.Unmarshal(strings.NewReader(tt.json), p)
		if err != nil {
			t.Errorf("unmarshaling %s: %v", tt.desc, err)
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

func TestUnmarshalNullArray(t *testing.T) {
	var repeats pb2.Repeats
	if err := UnmarshalString(`{"rBool":null}`, &repeats); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(&repeats, &pb2.Repeats{}) {
		t.Errorf("got non-nil fields in [%#v]", repeats)
	}
}

func TestUnmarshalNullObject(t *testing.T) {
	var maps pb2.Maps
	if err := UnmarshalString(`{"mInt64Str":null}`, &maps); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(&maps, &pb2.Maps{}) {
		t.Errorf("got non-nil fields in [%#v]", maps)
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
		// Make a new instance of the type of our wanted object.
		p := reflect.New(reflect.TypeOf(tt.pb).Elem()).Interface().(proto.Message)

		err := tt.unmarshaler.UnmarshalNext(dec, p)
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

	p := &pb2.Simple{}
	err := new(Unmarshaler).UnmarshalNext(dec, p)
	if err != io.EOF {
		t.Errorf("eof: got %v, want io.EOF", err)
	}
}

var unmarshalingShouldError = []struct {
	desc string
	in   string
	pb   proto.Message
}{
	{"a value", "666", new(pb2.Simple)},
	{"gibberish", "{adskja123;l23=-=", new(pb2.Simple)},
	{"unknown field", `{"unknown": "foo"}`, new(pb2.Simple)},
	{"unknown enum name", `{"hilarity":"DAVE"}`, new(pb3.Message)},
	{"Duration containing invalid character", `{"dur": "3\U0073"}`, &pb2.KnownTypes{}},
	{"Timestamp containing invalid character", `{"ts": "2014-05-13T16:53:20\U005a"}`, &pb2.KnownTypes{}},
	{"StringValue containing invalid character", `{"str": "\U00004E16\U0000754C"}`, &pb2.KnownTypes{}},
	{"StructValue containing invalid character", `{"str": "\U00004E16\U0000754C"}`, &stpb.Struct{}},
	{"repeated proto3 enum with non array input", `{"rFunny":"PUNS"}`, &pb3.Message{RFunny: []pb3.Message_Humour{}}},
	{"unknown extension field", `{"[ext_unknown]": "value"}`, &pb2.Real{}},
	{"extension field for wrong message", `{"[jsonpb_test.name]": "value"}`, &pb2.Complex{}},
}

func TestUnmarshalingBadInput(t *testing.T) {
	for _, tt := range unmarshalingShouldError {
		err := UnmarshalString(tt.in, tt.pb)
		if err == nil {
			t.Errorf("expected error while parsing %q", tt.desc)
		}
	}
}

type funcResolver func(turl string) (proto.Message, error)

func (fn funcResolver) Resolve(turl string) (proto.Message, error) {
	return fn(turl)
}

func TestAnyWithCustomResolver(t *testing.T) {
	var resolvedTypeUrls []string
	resolver := funcResolver(func(turl string) (proto.Message, error) {
		resolvedTypeUrls = append(resolvedTypeUrls, turl)
		return new(pb2.Simple), nil
	})
	msg := &pb2.Simple{
		OBytes:  []byte{1, 2, 3, 4},
		OBool:   proto.Bool(true),
		OString: proto.String("foobar"),
		OInt64:  proto.Int64(1020304),
	}
	msgBytes, err := proto.Marshal(msg)
	if err != nil {
		t.Errorf("an unexpected error while marshaling message: %v", err)
	}
	// make an Any with a type URL that won't resolve w/out custom resolver
	any := &anypb.Any{
		TypeUrl: "https://foobar.com/some.random.MessageKind",
		Value:   msgBytes,
	}

	m := Marshaler{AnyResolver: resolver}
	js, err := m.MarshalToString(any)
	if err != nil {
		t.Errorf("an unexpected error while marshaling any to JSON: %v", err)
	}
	if len(resolvedTypeUrls) != 1 {
		t.Errorf("custom resolver was not invoked during marshaling")
	} else if resolvedTypeUrls[0] != "https://foobar.com/some.random.MessageKind" {
		t.Errorf("custom resolver was invoked with wrong URL: got %q, wanted %q", resolvedTypeUrls[0], "https://foobar.com/some.random.MessageKind")
	}
	wanted := `{"@type":"https://foobar.com/some.random.MessageKind","oBool":true,"oInt64":"1020304","oString":"foobar","oBytes":"AQIDBA=="}`
	if js != wanted {
		t.Errorf("marshaling JSON produced incorrect output: got %s, wanted %s", js, wanted)
	}

	u := Unmarshaler{AnyResolver: resolver}
	roundTrip := &anypb.Any{}
	err = u.Unmarshal(bytes.NewReader([]byte(js)), roundTrip)
	if err != nil {
		t.Errorf("an unexpected error while unmarshaling any from JSON: %v", err)
	}
	if len(resolvedTypeUrls) != 2 {
		t.Errorf("custom resolver was not invoked during marshaling")
	} else if resolvedTypeUrls[1] != "https://foobar.com/some.random.MessageKind" {
		t.Errorf("custom resolver was invoked with wrong URL: got %q, wanted %q", resolvedTypeUrls[1], "https://foobar.com/some.random.MessageKind")
	}
	if !proto.Equal(any, roundTrip) {
		t.Errorf("message contents not set correctly after unmarshaling JSON: got %s, wanted %s", roundTrip, any)
	}
}

func TestUnmarshalJSONPBUnmarshaler(t *testing.T) {
	rawJson := `{ "foo": "bar", "baz": [0, 1, 2, 3] }`
	var msg dynamicMessage
	if err := Unmarshal(strings.NewReader(rawJson), &msg); err != nil {
		t.Errorf("an unexpected error while parsing into JSONPBUnmarshaler: %v", err)
	}
	if msg.RawJson != rawJson {
		t.Errorf("message contents not set correctly after unmarshaling JSON: got %s, wanted %s", msg.RawJson, rawJson)
	}
}

func TestUnmarshalNullWithJSONPBUnmarshaler(t *testing.T) {
	rawJson := `{"stringField":null}`
	var ptrFieldMsg ptrFieldMessage
	if err := Unmarshal(strings.NewReader(rawJson), &ptrFieldMsg); err != nil {
		t.Errorf("unmarshal error: %v", err)
	}

	want := ptrFieldMessage{StringField: &stringField{IsSet: true, StringValue: "null"}}
	if !proto.Equal(&ptrFieldMsg, &want) {
		t.Errorf("unmarshal result StringField: got %v, want %v", ptrFieldMsg, want)
	}
}

func TestUnmarshalAnyJSONPBUnmarshaler(t *testing.T) {
	rawJson := `{ "@type": "blah.com/` + dynamicMessageName + `", "foo": "bar", "baz": [0, 1, 2, 3] }`
	var got anypb.Any
	if err := Unmarshal(strings.NewReader(rawJson), &got); err != nil {
		t.Errorf("an unexpected error while parsing into JSONPBUnmarshaler: %v", err)
	}

	dm := &dynamicMessage{RawJson: `{"baz":[0,1,2,3],"foo":"bar"}`}
	var want anypb.Any
	if b, err := proto.Marshal(dm); err != nil {
		t.Errorf("an unexpected error while marshaling message: %v", err)
	} else {
		want.TypeUrl = "blah.com/" + dynamicMessageName
		want.Value = b
	}

	if !proto.Equal(&got, &want) {
		t.Errorf("message contents not set correctly after unmarshaling JSON: got %v, wanted %v", &got, &want)
	}
}

const (
	dynamicMessageName = "github_com.golang.protobuf.jsonpb.dynamicMessage"
)

func init() {
	// we register the custom type below so that we can use it in Any types
	proto.RegisterType((*dynamicMessage)(nil), dynamicMessageName)
}

type ptrFieldMessage struct {
	StringField *stringField `protobuf:"bytes,1,opt,name=stringField"`
}

func (m *ptrFieldMessage) Reset() {
}

func (m *ptrFieldMessage) String() string {
	return m.StringField.StringValue
}

func (m *ptrFieldMessage) ProtoMessage() {
}

func (m *ptrFieldMessage) Descriptor() ([]byte, []int) {
	return testMessageFD, []int{0}
}

type stringField struct {
	IsSet       bool   `protobuf:"varint,1,opt,name=isSet"`
	StringValue string `protobuf:"bytes,2,opt,name=stringValue"`
}

func (s *stringField) Reset() {
}

func (s *stringField) String() string {
	return s.StringValue
}

func (s *stringField) ProtoMessage() {
}

func (s *stringField) Descriptor() ([]byte, []int) {
	return testMessageFD, []int{1}
}

func (s *stringField) UnmarshalJSONPB(jum *Unmarshaler, js []byte) error {
	s.IsSet = true
	s.StringValue = string(js)
	return nil
}

// dynamicMessage implements protobuf.Message but is not a normal generated message type.
// It provides implementations of JSONPBMarshaler and JSONPBUnmarshaler for JSON support.
type dynamicMessage struct {
	RawJson string `protobuf:"bytes,1,opt,name=rawJson"`

	// an unexported nested message is present just to ensure that it
	// won't result in a panic (see issue #509)
	Dummy *dynamicMessage `protobuf:"bytes,2,opt,name=dummy"`
}

func (m *dynamicMessage) Reset() {
	m.RawJson = "{}"
}

func (m *dynamicMessage) String() string {
	return m.RawJson
}

func (m *dynamicMessage) ProtoMessage() {
}

func (m *dynamicMessage) Descriptor() ([]byte, []int) {
	return testMessageFD, []int{2}
}

func (m *dynamicMessage) MarshalJSONPB(jm *Marshaler) ([]byte, error) {
	return []byte(m.RawJson), nil
}

func (m *dynamicMessage) UnmarshalJSONPB(jum *Unmarshaler, js []byte) error {
	m.RawJson = string(js)
	return nil
}

var testMessageFD = func() []byte {
	fd := new(descpb.FileDescriptorProto)
	proto.UnmarshalText(`
		name:    "jsonpb.proto"
		package: "github_com.golang.protobuf.jsonpb"
		syntax:  "proto3"
		message_type: [{
			name: "ptrFieldMessage"
			field: [
				{name:"stringField" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".github_com.golang.protobuf.jsonpb.stringField"}
			]
		}, {
			name: "stringField"
			field: [
				{name:"isSet"       number:1 label:LABEL_OPTIONAL type:TYPE_BOOL},
				{name:"stringValue" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}
			]
		}, {
			name: "dynamicMessage"
			field: [
				{name:"rawJson" number:1 label:LABEL_OPTIONAL type:TYPE_BYTES},
				{name:"dummy"   number:2 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".github_com.golang.protobuf.jsonpb.dynamicMessage"}
			]
		}]
	`, fd)
	b, _ := proto.Marshal(fd)
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	zw.Write(b)
	zw.Close()
	return buf.Bytes()
}()

// Test unmarshaling message containing unset required fields should produce error.
func TestUnmarshalUnsetRequiredFields(t *testing.T) {
	tests := []struct {
		desc string
		pb   proto.Message
		json string
	}{
		{
			desc: "direct required field missing",
			pb:   &pb2.MsgWithRequired{},
			json: `{}`,
		},
		{
			desc: "direct required field set to null",
			pb:   &pb2.MsgWithRequired{},
			json: `{"str": null}`,
		},
		{
			desc: "indirect required field missing",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"subm": {}}`,
		},
		{
			desc: "indirect required field set to null",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"subm": {"str": null}}`,
		},
		{
			desc: "direct required bytes field missing",
			pb:   &pb2.MsgWithRequiredBytes{},
			json: `{}`,
		},
		{
			desc: "direct required bytes field set to null",
			pb:   &pb2.MsgWithRequiredBytes{},
			json: `{"byts": null}`,
		},
		{
			desc: "direct required wkt field missing",
			pb:   &pb2.MsgWithRequiredWKT{},
			json: `{}`,
		},
		{
			desc: "direct required wkt field set to null",
			pb:   &pb2.MsgWithRequiredWKT{},
			json: `{"str": null}`,
		},
		{
			desc: "any containing message with required field set to null",
			pb:   &pb2.KnownTypes{},
			json: `{"an": {"@type": "example.com/jsonpb.MsgWithRequired", "str": null}}`,
		},
		{
			desc: "any containing message with missing required field",
			pb:   &pb2.KnownTypes{},
			json: `{"an": {"@type": "example.com/jsonpb.MsgWithRequired"}}`,
		},
		{
			desc: "missing required in map value",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"map_field": {"a": {}, "b": {"str": "hi"}}}`,
		},
		{
			desc: "required in map value set to null",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"map_field": {"a": {"str": "hello"}, "b": {"str": null}}}`,
		},
		{
			desc: "missing required in slice item",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"slice_field": [{}, {"str": "hi"}]}`,
		},
		{
			desc: "required in slice item set to null",
			pb:   &pb2.MsgWithIndirectRequired{},
			json: `{"slice_field": [{"str": "hello"}, {"str": null}]}`,
		},
		{
			desc: "required inside oneof missing",
			pb:   &pb2.MsgWithOneof{},
			json: `{"msgWithRequired": {}}`,
		},
		{
			desc: "required inside oneof set to null",
			pb:   &pb2.MsgWithOneof{},
			json: `{"msgWithRequired": {"str": null}}`,
		},
		{
			desc: "required field in extension missing",
			pb:   &pb2.Real{},
			json: `{"[jsonpb.extm]":{}}`,
		},
		{
			desc: "required field in extension set to null",
			pb:   &pb2.Real{},
			json: `{"[jsonpb.extm]":{"str": null}}`,
		},
	}

	for _, tc := range tests {
		if err := UnmarshalString(tc.json, tc.pb); err == nil {
			t.Errorf("%s: expected error while unmarshaling with unset required fields %s", tc.desc, tc.json)
		}
	}
}
