package jlexer

import (
	"bytes"
	"reflect"
	"testing"
)

func TestString(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		want      string
		wantError bool
	}{
		{toParse: `"simple string"`, want: "simple string"},
		{toParse: " \r\r\n\t  " + `"test"`, want: "test"},
		{toParse: `"\n\t\"\/\\\f\r"`, want: "\n\t\"/\\\f\r"},
		{toParse: `"\u0020"`, want: " "},
		{toParse: `"\u0020-\t"`, want: " -\t"},
		{toParse: `"\ufffd\uFFFD"`, want: "\ufffd\ufffd"},
		{toParse: `"\ud83d\ude00"`, want: "😀"},
		{toParse: `"\ud83d\ude08"`, want: "😈"},
		{toParse: `"\ud8"`, wantError: true},

		{toParse: `"test"junk`, want: "test"},

		{toParse: `5`, wantError: true},        // not a string
		{toParse: `"\x"`, wantError: true},     // invalid escape
		{toParse: `"\ud800"`, want: "�"},      // invalid utf-8 char; return replacement char
	} {
		l := Lexer{Data: []byte(test.toParse)}

		got := l.String()
		if got != test.want {
			t.Errorf("[%d, %q] String() = %v; want %v", i, test.toParse, got, test.want)
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] String() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] String() ok; want error", i, test.toParse)
		}
	}
}

func TestBytes(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		want      string
		wantError bool
	}{
		{toParse: `"c2ltcGxlIHN0cmluZw=="`, want: "simple string"},
		{toParse: " \r\r\n\t  " + `"dGVzdA=="`, want: "test"},

		{toParse: `5`, wantError: true},                     // not a JSON string
		{toParse: `"foobar"`, wantError: true},              // not base64 encoded
		{toParse: `"c2ltcGxlIHN0cmluZw="`, wantError: true}, // invalid base64 padding
	} {
		l := Lexer{Data: []byte(test.toParse)}

		got := l.Bytes()
		if bytes.Compare(got, []byte(test.want)) != 0 {
			t.Errorf("[%d, %q] Bytes() = %v; want: %v", i, test.toParse, got, []byte(test.want))
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] Bytes() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] Bytes() ok; want error", i, test.toParse)
		}
	}
}

func TestNumber(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		want      string
		wantError bool
	}{
		{toParse: "123", want: "123"},
		{toParse: "-123", want: "-123"},
		{toParse: "\r\n12.35", want: "12.35"},
		{toParse: "12.35e+1", want: "12.35e+1"},
		{toParse: "12.35e-15", want: "12.35e-15"},
		{toParse: "12.35E-15", want: "12.35E-15"},
		{toParse: "12.35E15", want: "12.35E15"},

		{toParse: `"a"`, wantError: true},
		{toParse: "123junk", wantError: true},
		{toParse: "1.2.3", wantError: true},
		{toParse: "1e2e3", wantError: true},
		{toParse: "1e2.3", wantError: true},
	} {
		l := Lexer{Data: []byte(test.toParse)}

		got := l.number()
		if got != test.want {
			t.Errorf("[%d, %q] number() = %v; want %v", i, test.toParse, got, test.want)
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] number() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] number() ok; want error", i, test.toParse)
		}
	}
}

func TestBool(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		want      bool
		wantError bool
	}{
		{toParse: "true", want: true},
		{toParse: "false", want: false},

		{toParse: "1", wantError: true},
		{toParse: "truejunk", wantError: true},
		{toParse: `false"junk"`, wantError: true},
		{toParse: "True", wantError: true},
		{toParse: "False", wantError: true},
	} {
		l := Lexer{Data: []byte(test.toParse)}

		got := l.Bool()
		if got != test.want {
			t.Errorf("[%d, %q] Bool() = %v; want %v", i, test.toParse, got, test.want)
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] Bool() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] Bool() ok; want error", i, test.toParse)
		}
	}
}

func TestSkipRecursive(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		left      string
		wantError bool
	}{
		{toParse: "5, 4", left: ", 4"},
		{toParse: "[5, 6], 4", left: ", 4"},
		{toParse: "[5, [7,8]]: 4", left: ": 4"},

		{toParse: `{"a":1}, 4`, left: ", 4"},
		{toParse: `{"a":1, "b":{"c": 5}, "e":[12,15]}, 4`, left: ", 4"},

		// array start/end chars in a string
		{toParse: `[5, "]"], 4`, left: ", 4"},
		{toParse: `[5, "\"]"], 4`, left: ", 4"},
		{toParse: `[5, "["], 4`, left: ", 4"},
		{toParse: `[5, "\"["], 4`, left: ", 4"},

		// object start/end chars in a string
		{toParse: `{"a}":1}, 4`, left: ", 4"},
		{toParse: `{"a\"}":1}, 4`, left: ", 4"},
		{toParse: `{"a{":1}, 4`, left: ", 4"},
		{toParse: `{"a\"{":1}, 4`, left: ", 4"},

		// object with double slashes at the end of string
		{toParse: `{"a":"hey\\"}, 4`, left: ", 4"},
	} {
		l := Lexer{Data: []byte(test.toParse)}

		l.SkipRecursive()

		got := string(l.Data[l.pos:])
		if got != test.left {
			t.Errorf("[%d, %q] SkipRecursive() left = %v; want %v", i, test.toParse, got, test.left)
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] SkipRecursive() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] SkipRecursive() ok; want error", i, test.toParse)
		}
	}
}

func TestInterface(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		want      interface{}
		wantError bool
	}{
		{toParse: "null", want: nil},
		{toParse: "true", want: true},
		{toParse: `"a"`, want: "a"},
		{toParse: "5", want: float64(5)},

		{toParse: `{}`, want: map[string]interface{}{}},
		{toParse: `[]`, want: []interface{}(nil)},

		{toParse: `{"a": "b"}`, want: map[string]interface{}{"a": "b"}},
		{toParse: `[5]`, want: []interface{}{float64(5)}},

		{toParse: `{"a":5 , "b" : "string"}`, want: map[string]interface{}{"a": float64(5), "b": "string"}},
		{toParse: `["a", 5 , null, true]`, want: []interface{}{"a", float64(5), nil, true}},

		{toParse: `{"a" "b"}`, wantError: true},
		{toParse: `{"a": "b",}`, wantError: true},
		{toParse: `{"a":"b","c" "b"}`, wantError: true},
		{toParse: `{"a": "b","c":"d",}`, wantError: true},
		{toParse: `{,}`, wantError: true},

		{toParse: `[1, 2,]`, wantError: true},
		{toParse: `[1  2]`, wantError: true},
		{toParse: `[,]`, wantError: true},
	} {
		l := Lexer{Data: []byte(test.toParse)}

		got := l.Interface()
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("[%d, %q] Interface() = %v; want %v", i, test.toParse, got, test.want)
		}
		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] Interface() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] Interface() ok; want error", i, test.toParse)
		}
	}
}

func TestConsumed(t *testing.T) {
	for i, test := range []struct {
		toParse   string
		wantError bool
	}{
		{toParse: "", wantError: false},
		{toParse: "   ", wantError: false},
		{toParse: "\r\n", wantError: false},
		{toParse: "\t\t", wantError: false},

		{toParse: "{", wantError: true},
	} {
		l := Lexer{Data: []byte(test.toParse)}
		l.Consumed()

		err := l.Error()
		if err != nil && !test.wantError {
			t.Errorf("[%d, %q] Consumed() error: %v", i, test.toParse, err)
		} else if err == nil && test.wantError {
			t.Errorf("[%d, %q] Consumed() ok; want error", i, test.toParse)
		}
	}
}
