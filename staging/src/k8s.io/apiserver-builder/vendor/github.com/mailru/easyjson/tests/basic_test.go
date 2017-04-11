package tests

import (
	"reflect"
	"testing"

	"encoding/json"

	"github.com/mailru/easyjson"
	"github.com/mailru/easyjson/jwriter"
)

type testType interface {
	json.Marshaler
	json.Unmarshaler
}

var testCases = []struct {
	Decoded testType
	Encoded string
}{
	{&primitiveTypesValue, primitiveTypesString},
	{&namedPrimitiveTypesValue, namedPrimitiveTypesString},
	{&structsValue, structsString},
	{&omitEmptyValue, omitEmptyString},
	{&snakeStructValue, snakeStructString},
	{&omitEmptyDefaultValue, omitEmptyDefaultString},
	{&optsValue, optsString},
	{&rawValue, rawString},
	{&stdMarshalerValue, stdMarshalerString},
	{&unexportedStructValue, unexportedStructString},
	{&excludedFieldValue, excludedFieldString},
	{&mapsValue, mapsString},
	{&deepNestValue, deepNestString},
	{&IntsValue, IntsString},
}

func TestMarshal(t *testing.T) {
	for i, test := range testCases {
		data, err := test.Decoded.MarshalJSON()
		if err != nil {
			t.Errorf("[%d, %T] MarshalJSON() error: %v", i, test.Decoded, err)
		}

		got := string(data)
		if got != test.Encoded {
			t.Errorf("[%d, %T] MarshalJSON(): got \n%v\n\t\t want \n%v", i, test.Decoded, got, test.Encoded)
		}
	}
}

func TestUnmarshal(t *testing.T) {
	for i, test := range testCases {
		v1 := reflect.New(reflect.TypeOf(test.Decoded).Elem()).Interface()
		v := v1.(testType)

		err := v.UnmarshalJSON([]byte(test.Encoded))
		if err != nil {
			t.Errorf("[%d, %T] UnmarshalJSON() error: %v", i, test.Decoded, err)
		}

		if !reflect.DeepEqual(v, test.Decoded) {
			t.Errorf("[%d, %T] UnmarshalJSON(): got \n%+v\n\t\t want \n%+v", i, test.Decoded, v, test.Decoded)
		}
	}
}

func TestRawMessageSTD(t *testing.T) {
	type T struct {
		F    easyjson.RawMessage
		Fnil easyjson.RawMessage
	}

	val := T{F: easyjson.RawMessage([]byte(`"test"`))}
	str := `{"F":"test","Fnil":null}`

	data, err := json.Marshal(val)
	if err != nil {
		t.Errorf("json.Marshal() error: %v", err)
	}
	got := string(data)
	if got != str {
		t.Errorf("json.Marshal() = %v; want %v", got, str)
	}

	wantV := T{F: easyjson.RawMessage([]byte(`"test"`)), Fnil: easyjson.RawMessage([]byte("null"))}
	var gotV T

	err = json.Unmarshal([]byte(str), &gotV)
	if err != nil {
		t.Errorf("json.Unmarshal() error: %v", err)
	}
	if !reflect.DeepEqual(gotV, wantV) {
		t.Errorf("json.Unmarshal() = %v; want %v", gotV, wantV)
	}
}

func TestParseNull(t *testing.T) {
	var got, want SubStruct
	if err := easyjson.Unmarshal([]byte("null"), &got); err != nil {
		t.Errorf("Unmarshal() error: %v", err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Unmarshal() = %+v; want %+v", got, want)
	}
}

var testSpecialCases = []struct {
	EncodedString string
	Value         string
}{
	{`"Username \u003cuser@example.com\u003e"`, `Username <user@example.com>`},
	{`"Username\ufffd"`, "Username\xc5"},
	{`"тестzтест"`, "тестzтест"},
	{`"тест\ufffdтест"`, "тест\xc5тест"},
	{`"绿茶"`, "绿茶"},
	{`"绿\ufffd茶"`, "绿\xc5茶"},
	{`"тест\u2028"`, "тест\xE2\x80\xA8"},
	{`"\\\r\n\t\""`, "\\\r\n\t\""},
	{`"ü"`, "ü"},
}

func TestSpecialCases(t *testing.T) {
	for i, test := range testSpecialCases {
		w := jwriter.Writer{}
		w.String(test.Value)
		got := string(w.Buffer.BuildBytes())
		if got != test.EncodedString {
			t.Errorf("[%d] Encoded() = %+v; want %+v", i, got, test.EncodedString)
		}
	}
}
