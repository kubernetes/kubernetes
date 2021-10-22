// Copyright 2015 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"encoding/json"
	"reflect"
	"testing"

	"google.golang.org/api/googleapi"
)

type schema struct {
	// Basic types
	B    bool    `json:"b,omitempty"`
	F    float64 `json:"f,omitempty"`
	I    int64   `json:"i,omitempty"`
	Istr int64   `json:"istr,omitempty,string"`
	Str  string  `json:"str,omitempty"`

	// Pointers to basic types
	PB    *bool    `json:"pb,omitempty"`
	PF    *float64 `json:"pf,omitempty"`
	PI    *int64   `json:"pi,omitempty"`
	PIStr *int64   `json:"pistr,omitempty,string"`
	PStr  *string  `json:"pstr,omitempty"`

	// Other types
	Int64s        googleapi.Int64s         `json:"i64s,omitempty"`
	S             []int                    `json:"s,omitempty"`
	M             map[string]string        `json:"m,omitempty"`
	Any           interface{}              `json:"any,omitempty"`
	Child         *child                   `json:"child,omitempty"`
	MapToAnyArray map[string][]interface{} `json:"maptoanyarray,omitempty"`

	ForceSendFields []string `json:"-"`
	NullFields      []string `json:"-"`
}

type child struct {
	B bool `json:"childbool,omitempty"`
}

type testCase struct {
	s    schema
	want string
}

func TestBasics(t *testing.T) {
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"B", "F", "I", "Istr", "Str", "PB", "PF", "PI", "PIStr", "PStr"},
			},
			want: `{"b":false,"f":0.0,"i":0,"istr":"0","str":""}`,
		},
		{
			s: schema{
				NullFields: []string{"B", "F", "I", "Istr", "Str", "PB", "PF", "PI", "PIStr", "PStr"},
			},
			want: `{"b":null,"f":null,"i":null,"istr":null,"str":null,"pb":null,"pf":null,"pi":null,"pistr":null,"pstr":null}`,
		},
		{
			s: schema{
				B:     true,
				F:     1.2,
				I:     1,
				Istr:  2,
				Str:   "a",
				PB:    googleapi.Bool(true),
				PF:    googleapi.Float64(1.2),
				PI:    googleapi.Int64(int64(1)),
				PIStr: googleapi.Int64(int64(2)),
				PStr:  googleapi.String("a"),
			},
			want: `{"b":true,"f":1.2,"i":1,"istr":"2","str":"a","pb":true,"pf":1.2,"pi":1,"pistr":"2","pstr":"a"}`,
		},
		{
			s: schema{
				B:     false,
				F:     0.0,
				I:     0,
				Istr:  0,
				Str:   "",
				PB:    googleapi.Bool(false),
				PF:    googleapi.Float64(0.0),
				PI:    googleapi.Int64(int64(0)),
				PIStr: googleapi.Int64(int64(0)),
				PStr:  googleapi.String(""),
			},
			want: `{"pb":false,"pf":0.0,"pi":0,"pistr":"0","pstr":""}`,
		},
		{
			s: schema{
				B:               false,
				F:               0.0,
				I:               0,
				Istr:            0,
				Str:             "",
				PB:              googleapi.Bool(false),
				PF:              googleapi.Float64(0.0),
				PI:              googleapi.Int64(int64(0)),
				PIStr:           googleapi.Int64(int64(0)),
				PStr:            googleapi.String(""),
				ForceSendFields: []string{"B", "F", "I", "Istr", "Str", "PB", "PF", "PI", "PIStr", "PStr"},
			},
			want: `{"b":false,"f":0.0,"i":0,"istr":"0","str":"","pb":false,"pf":0.0,"pi":0,"pistr":"0","pstr":""}`,
		},
		{
			s: schema{
				B:          false,
				F:          0.0,
				I:          0,
				Istr:       0,
				Str:        "",
				PB:         googleapi.Bool(false),
				PF:         googleapi.Float64(0.0),
				PI:         googleapi.Int64(int64(0)),
				PIStr:      googleapi.Int64(int64(0)),
				PStr:       googleapi.String(""),
				NullFields: []string{"B", "F", "I", "Istr", "Str"},
			},
			want: `{"b":null,"f":null,"i":null,"istr":null,"str":null,"pb":false,"pf":0.0,"pi":0,"pistr":"0","pstr":""}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

func TestSliceFields(t *testing.T) {
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s:    schema{S: []int{}, Int64s: googleapi.Int64s{}},
			want: `{}`,
		},
		{
			s:    schema{S: []int{1}, Int64s: googleapi.Int64s{1}},
			want: `{"s":[1],"i64s":["1"]}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"S", "Int64s"},
			},
			want: `{"s":[],"i64s":[]}`,
		},
		{
			s: schema{
				S:               []int{},
				Int64s:          googleapi.Int64s{},
				ForceSendFields: []string{"S", "Int64s"},
			},
			want: `{"s":[],"i64s":[]}`,
		},
		{
			s: schema{
				S:               []int{1},
				Int64s:          googleapi.Int64s{1},
				ForceSendFields: []string{"S", "Int64s"},
			},
			want: `{"s":[1],"i64s":["1"]}`,
		},
		{
			s: schema{
				NullFields: []string{"S", "Int64s"},
			},
			want: `{"s":null,"i64s":null}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

func TestMapField(t *testing.T) {
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s:    schema{M: make(map[string]string)},
			want: `{}`,
		},
		{
			s:    schema{M: map[string]string{"a": "b"}},
			want: `{"m":{"a":"b"}}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"M"},
			},
			want: `{"m":{}}`,
		},
		{
			s: schema{
				NullFields: []string{"M"},
			},
			want: `{"m":null}`,
		},
		{
			s: schema{
				M:               make(map[string]string),
				ForceSendFields: []string{"M"},
			},
			want: `{"m":{}}`,
		},
		{
			s: schema{
				M:          make(map[string]string),
				NullFields: []string{"M"},
			},
			want: `{"m":null}`,
		},
		{
			s: schema{
				M:               map[string]string{"a": "b"},
				ForceSendFields: []string{"M"},
			},
			want: `{"m":{"a":"b"}}`,
		},
		{
			s: schema{
				M:          map[string]string{"a": "b"},
				NullFields: []string{"M.a", "M."},
			},
			want: `{"m": {"a": null, "":null}}`,
		},
		{
			s: schema{
				M:          map[string]string{"a": "b"},
				NullFields: []string{"M.c"},
			},
			want: `{"m": {"a": "b", "c": null}}`,
		},
		{
			s: schema{
				NullFields:      []string{"M.a"},
				ForceSendFields: []string{"M"},
			},
			want: `{"m": {"a": null}}`,
		},
		{
			s: schema{
				NullFields: []string{"M.a"},
			},
			want: `{}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

func TestMapToAnyArray(t *testing.T) {
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s:    schema{MapToAnyArray: make(map[string][]interface{})},
			want: `{}`,
		},
		{
			s: schema{
				MapToAnyArray: map[string][]interface{}{
					"a": {2, "b"},
				},
			},
			want: `{"maptoanyarray":{"a":[2, "b"]}}`,
		},
		{
			s: schema{
				MapToAnyArray: map[string][]interface{}{
					"a": nil,
				},
			},
			want: `{"maptoanyarray":{"a": null}}`,
		},
		{
			s: schema{
				MapToAnyArray: map[string][]interface{}{
					"a": {nil},
				},
			},
			want: `{"maptoanyarray":{"a":[null]}}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"MapToAnyArray"},
			},
			want: `{"maptoanyarray":{}}`,
		},
		{
			s: schema{
				NullFields: []string{"MapToAnyArray"},
			},
			want: `{"maptoanyarray":null}`,
		},
		{
			s: schema{
				MapToAnyArray:   make(map[string][]interface{}),
				ForceSendFields: []string{"MapToAnyArray"},
			},
			want: `{"maptoanyarray":{}}`,
		},
		{
			s: schema{
				MapToAnyArray: map[string][]interface{}{
					"a": {2, "b"},
				},
				ForceSendFields: []string{"MapToAnyArray"},
			},
			want: `{"maptoanyarray":{"a":[2, "b"]}}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

type anyType struct {
	Field int
}

func (a anyType) MarshalJSON() ([]byte, error) {
	return []byte(`"anyType value"`), nil
}

func TestAnyField(t *testing.T) {
	// ForceSendFields has no effect on nil interfaces and interfaces that contain nil pointers.
	var nilAny *anyType
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s:    schema{Any: nilAny},
			want: `{"any": null}`,
		},
		{
			s:    schema{Any: &anyType{}},
			want: `{"any":"anyType value"}`,
		},
		{
			s:    schema{Any: anyType{}},
			want: `{"any":"anyType value"}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"Any"},
			},
			want: `{}`,
		},
		{
			s: schema{
				NullFields: []string{"Any"},
			},
			want: `{"any":null}`,
		},
		{
			s: schema{
				Any:             nilAny,
				ForceSendFields: []string{"Any"},
			},
			want: `{"any": null}`,
		},
		{
			s: schema{
				Any:             &anyType{},
				ForceSendFields: []string{"Any"},
			},
			want: `{"any":"anyType value"}`,
		},
		{
			s: schema{
				Any:             anyType{},
				ForceSendFields: []string{"Any"},
			},
			want: `{"any":"anyType value"}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

func TestSubschema(t *testing.T) {
	// Subschemas are always stored as pointers, so ForceSendFields has no effect on them.
	for _, tc := range []testCase{
		{
			s:    schema{},
			want: `{}`,
		},
		{
			s: schema{
				ForceSendFields: []string{"Child"},
			},
			want: `{}`,
		},
		{
			s: schema{
				NullFields: []string{"Child"},
			},
			want: `{"child":null}`,
		},
		{
			s:    schema{Child: &child{}},
			want: `{"child":{}}`,
		},
		{
			s: schema{
				Child:           &child{},
				ForceSendFields: []string{"Child"},
			},
			want: `{"child":{}}`,
		},
		{
			s:    schema{Child: &child{B: true}},
			want: `{"child":{"childbool":true}}`,
		},

		{
			s: schema{
				Child:           &child{B: true},
				ForceSendFields: []string{"Child"},
			},
			want: `{"child":{"childbool":true}}`,
		},
	} {
		checkMarshalJSON(t, tc)
	}
}

// checkMarshalJSON verifies that calling schemaToMap on tc.s yields a result which is equivalent to tc.want.
func checkMarshalJSON(t *testing.T, tc testCase) {
	doCheckMarshalJSON(t, tc.s, tc.s.ForceSendFields, tc.s.NullFields, tc.want)
	if len(tc.s.ForceSendFields) == 0 && len(tc.s.NullFields) == 0 {
		// verify that the code path used when ForceSendFields and NullFields
		// are non-empty produces the same output as the fast path that is used
		// when they are empty.
		doCheckMarshalJSON(t, tc.s, []string{"dummy"}, []string{"dummy"}, tc.want)
	}
}

func doCheckMarshalJSON(t *testing.T, s schema, forceSendFields, nullFields []string, wantJSON string) {
	encoded, err := MarshalJSON(s, forceSendFields, nullFields)
	if err != nil {
		t.Fatalf("encoding json:\n got err: %v", err)
	}

	// The expected and obtained JSON can differ in field ordering, so unmarshal before comparing.
	var got interface{}
	var want interface{}
	err = json.Unmarshal(encoded, &got)
	if err != nil {
		t.Fatalf("decoding json:\n got err: %v", err)
	}
	err = json.Unmarshal([]byte(wantJSON), &want)
	if err != nil {
		t.Fatalf("decoding json:\n got err: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("schemaToMap:\ngot :%v\nwant: %v", got, want)
	}
}

func TestParseJSONTag(t *testing.T) {
	for _, tc := range []struct {
		tag  string
		want jsonTag
	}{
		{
			tag:  "-",
			want: jsonTag{ignore: true},
		}, {
			tag:  "name,omitempty",
			want: jsonTag{apiName: "name"},
		}, {
			tag:  "name,omitempty,string",
			want: jsonTag{apiName: "name", stringFormat: true},
		},
	} {
		got, err := parseJSONTag(tc.tag)
		if err != nil {
			t.Fatalf("parsing json:\n got err: %v\ntag: %q", err, tc.tag)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseJSONTage:\ngot :%v\nwant:%v", got, tc.want)
		}
	}
}
func TestParseMalformedJSONTag(t *testing.T) {
	for _, tag := range []string{
		"",
		"name",
		"name,",
		"name,blah",
		"name,blah,string",
		",omitempty",
		",omitempty,string",
		"name,omitempty,string,blah",
	} {
		_, err := parseJSONTag(tag)
		if err == nil {
			t.Fatalf("parsing json: expected err, got nil for tag: %v", tag)
		}
	}
}
