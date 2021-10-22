// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fields

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"cloud.google.com/go/internal/testutil"
	"github.com/google/go-cmp/cmp"
)

type embed1 struct {
	Em1    int
	Dup    int // annihilates with embed2.Dup
	Shadow int
	embed3
}

type embed2 struct {
	Dup int
	embed3
	embed4
}

type embed3 struct {
	Em3 int // annihilated because embed3 is in both embed1 and embed2
	embed5
}

type embed4 struct {
	Em4     int
	Dup     int // annihilation of Dup in embed1, embed2 hides this Dup
	*embed1     // ignored because it occurs at a higher level
}

type embed5 struct {
	x int
}

type Anonymous int

type S1 struct {
	Exported   int
	unexported int
	Shadow     int // shadows S1.Shadow
	embed1
	*embed2
	Anonymous
}

type Time struct {
	time.Time
}

var intType = reflect.TypeOf(int(0))

func field(name string, tval interface{}, index ...int) *Field {
	return &Field{
		Name:      name,
		Type:      reflect.TypeOf(tval),
		Index:     index,
		ParsedTag: []string(nil),
	}
}

func tfield(name string, tval interface{}, index ...int) *Field {
	return &Field{
		Name:        name,
		Type:        reflect.TypeOf(tval),
		Index:       index,
		NameFromTag: true,
		ParsedTag:   []string(nil),
	}
}

func TestFieldsNoTags(t *testing.T) {
	c := NewCache(nil, nil, nil)
	got, err := c.Fields(reflect.TypeOf(S1{}))
	if err != nil {
		t.Fatal(err)
	}
	want := []*Field{
		field("Exported", int(0), 0),
		field("Shadow", int(0), 2),
		field("Em1", int(0), 3, 0),
		field("Em4", int(0), 4, 2, 0),
		field("Anonymous", Anonymous(0), 5),
	}
	for _, f := range want {
		f.ParsedTag = nil
	}
	if msg, ok := compareFields(got, want); !ok {
		t.Error(msg)
	}
}

func TestAgainstJSONEncodingNoTags(t *testing.T) {
	// Demonstrates that this package produces the same set of fields as encoding/json.
	s1 := S1{
		Exported:   1,
		unexported: 2,
		Shadow:     3,
		embed1: embed1{
			Em1:    4,
			Dup:    5,
			Shadow: 6,
			embed3: embed3{
				Em3:    7,
				embed5: embed5{x: 8},
			},
		},
		embed2: &embed2{
			Dup: 9,
			embed3: embed3{
				Em3:    10,
				embed5: embed5{x: 11},
			},
			embed4: embed4{
				Em4:    12,
				Dup:    13,
				embed1: &embed1{Em1: 14},
			},
		},
		Anonymous: Anonymous(15),
	}
	var want S1
	want.embed2 = &embed2{} // need this because reflection won't create it
	jsonRoundTrip(t, s1, &want)
	var got S1
	got.embed2 = &embed2{}
	fields, err := NewCache(nil, nil, nil).Fields(reflect.TypeOf(got))
	if err != nil {
		t.Fatal(err)
	}
	setFields(fields, &got, s1)
	if !testutil.Equal(got, want,
		cmp.AllowUnexported(S1{}, embed1{}, embed2{}, embed3{}, embed4{}, embed5{})) {
		t.Errorf("got\n%+v\nwant\n%+v", got, want)
	}
}

// Tests use of LeafTypes parameter to NewCache
func TestAgainstJSONEncodingEmbeddedTime(t *testing.T) {
	timeLeafFn := func(t reflect.Type) bool {
		return t == reflect.TypeOf(time.Time{})
	}
	// Demonstrates that this package can produce the same set of
	// fields as encoding/json for a struct with an embedded time.Time.
	now := time.Now().UTC()
	myt := Time{
		now,
	}
	var want Time
	jsonRoundTrip(t, myt, &want)
	var got Time
	fields, err := NewCache(nil, nil, timeLeafFn).Fields(reflect.TypeOf(got))
	if err != nil {
		t.Fatal(err)
	}
	setFields(fields, &got, myt)
	if !testutil.Equal(got, want) {
		t.Errorf("got\n%+v\nwant\n%+v", got, want)
	}
}

type S2 struct {
	NoTag     int
	XXX       int           `json:"tag"` // tag name takes precedence
	Anonymous `json:"anon"` // anonymous non-structs also get their name from the tag
	Embed     `json:"em"`   // embedded structs with tags become fields
	Tag       int
	YYY       int `json:"Tag"` // tag takes precedence over untagged field of the same name
	Empty     int `json:""`    // empty tag is noop
	tEmbed1
	tEmbed2
}

type Embed struct {
	Em int
}

type tEmbed1 struct {
	Dup int
	X   int `json:"Dup2"`
}

type tEmbed2 struct {
	Y int `json:"Dup"`  // takes precedence over tEmbed1.Dup because it is tagged
	Z int `json:"Dup2"` // same name as tEmbed1.X and both tagged, so ignored
}

func jsonTagParser(t reflect.StructTag) (name string, keep bool, other interface{}, err error) {
	return ParseStandardTag("json", t)
}

func validateFunc(t reflect.Type) (err error) {
	if t.Kind() != reflect.Struct {
		return errors.New("non-struct type used")
	}

	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).Type.Kind() == reflect.Slice {
			return fmt.Errorf("slice field found at field %s on struct %s", t.Field(i).Name, t.Name())
		}
	}

	return nil
}

func TestFieldsWithTags(t *testing.T) {
	got, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S2{}))
	if err != nil {
		t.Fatal(err)
	}
	want := []*Field{
		field("NoTag", int(0), 0),
		tfield("tag", int(0), 1),
		tfield("anon", Anonymous(0), 2),
		tfield("em", Embed{}, 4),
		tfield("Tag", int(0), 6),
		field("Empty", int(0), 7),
		tfield("Dup", int(0), 8, 0),
	}
	if msg, ok := compareFields(got, want); !ok {
		t.Error(msg)
	}
}

func TestAgainstJSONEncodingWithTags(t *testing.T) {
	// Demonstrates that this package produces the same set of fields as encoding/json.
	s2 := S2{
		NoTag:     1,
		XXX:       2,
		Anonymous: 3,
		Embed: Embed{
			Em: 4,
		},
		tEmbed1: tEmbed1{
			Dup: 5,
			X:   6,
		},
		tEmbed2: tEmbed2{
			Y: 7,
			Z: 8,
		},
	}
	var want S2
	jsonRoundTrip(t, s2, &want)
	var got S2
	fields, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(got))
	if err != nil {
		t.Fatal(err)
	}
	setFields(fields, &got, s2)
	if !testutil.Equal(got, want, cmp.AllowUnexported(S2{})) {
		t.Errorf("got\n%+v\nwant\n%+v", got, want)
	}
}

func TestUnexportedAnonymousNonStruct(t *testing.T) {
	// An unexported anonymous non-struct field should not be recorded.
	// This is currently a bug in encoding/json.
	// https://github.com/golang/go/issues/18009
	type S struct{}

	got, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %d fields, want 0", len(got))
	}
}

func TestUnexportedAnonymousStruct(t *testing.T) {
	// An unexported anonymous struct with a tag is ignored.
	// This is currently a bug in encoding/json.
	// https://github.com/golang/go/issues/18009
	type (
		s1 struct{ X int }
		S2 struct {
			s1 `json:"Y"`
		}
	)
	got, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S2{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %d fields, want 0", len(got))
	}
}

func TestDominantField(t *testing.T) {
	// With fields sorted by index length and then by tag presence,
	// the dominant field is always the first. Make sure all error
	// cases are caught.
	for _, test := range []struct {
		fields []Field
		wantOK bool
	}{
		// A single field is OK.
		{[]Field{{Index: []int{0}}}, true},
		{[]Field{{Index: []int{0}, NameFromTag: true}}, true},
		// A single field at top level is OK.
		{[]Field{{Index: []int{0}}, {Index: []int{1, 0}}}, true},
		{[]Field{{Index: []int{0}}, {Index: []int{1, 0}, NameFromTag: true}}, true},
		{[]Field{{Index: []int{0}, NameFromTag: true}, {Index: []int{1, 0}, NameFromTag: true}}, true},
		// A single tagged field is OK.
		{[]Field{{Index: []int{0}, NameFromTag: true}, {Index: []int{1}}}, true},
		// Two untagged fields at the same level is an error.
		{[]Field{{Index: []int{0}}, {Index: []int{1}}}, false},
		// Two tagged fields at the same level is an error.
		{[]Field{{Index: []int{0}, NameFromTag: true}, {Index: []int{1}, NameFromTag: true}}, false},
	} {
		_, gotOK := dominantField(test.fields)
		if gotOK != test.wantOK {
			t.Errorf("%v: got %t, want %t", test.fields, gotOK, test.wantOK)
		}
	}
}

func TestIgnore(t *testing.T) {
	type S struct {
		X int `json:"-"`
	}
	got, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %d fields, want 0", len(got))
	}
}

func TestParsedTag(t *testing.T) {
	type S struct {
		X int `json:"name,omitempty"`
	}
	got, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S{}))
	if err != nil {
		t.Fatal(err)
	}
	want := []*Field{
		{Name: "name", NameFromTag: true, Type: intType,
			Index: []int{0}, ParsedTag: []string{"omitempty"}},
	}
	if msg, ok := compareFields(got, want); !ok {
		t.Error(msg)
	}
}

func TestValidateFunc(t *testing.T) {
	type MyInvalidStruct struct {
		A string
		B []int
	}

	_, err := NewCache(nil, validateFunc, nil).Fields(reflect.TypeOf(MyInvalidStruct{}))
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	type MyValidStruct struct {
		A string
		B int
	}
	_, err = NewCache(nil, validateFunc, nil).Fields(reflect.TypeOf(MyValidStruct{}))
	if err != nil {
		t.Fatalf("expected nil, got error: %s\n", err)
	}
}

func compareFields(got []Field, want []*Field) (msg string, ok bool) {
	if len(got) != len(want) {
		return fmt.Sprintf("got %d fields, want %d", len(got), len(want)), false
	}
	for i, g := range got {
		w := *want[i]
		if !fieldsEqual(&g, &w) {
			return fmt.Sprintf("got\n%+v\nwant\n%+v", g, w), false
		}
	}
	return "", true
}

// Need this because Field contains a function, which cannot be compared even
// by testutil.Equal.
func fieldsEqual(f1, f2 *Field) bool {
	if f1 == nil || f2 == nil {
		return f1 == f2
	}
	return f1.Name == f2.Name &&
		f1.NameFromTag == f2.NameFromTag &&
		f1.Type == f2.Type &&
		testutil.Equal(f1.ParsedTag, f2.ParsedTag)
}

// Set the fields of dst from those of src.
// dst must be a pointer to a struct value.
// src must be a struct value.
func setFields(fields []Field, dst, src interface{}) {
	vsrc := reflect.ValueOf(src)
	vdst := reflect.ValueOf(dst).Elem()
	for _, f := range fields {
		fdst := vdst.FieldByIndex(f.Index)
		fsrc := vsrc.FieldByIndex(f.Index)
		fdst.Set(fsrc)
	}
}

func jsonRoundTrip(t *testing.T, in, out interface{}) {
	bytes, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(bytes, out); err != nil {
		t.Fatal(err)
	}
}

type S3 struct {
	S4
	Abc        int
	AbC        int
	Tag        int
	X          int `json:"Tag"`
	unexported int
}

type S4 struct {
	ABc int
	Y   int `json:"Abc"` // ignored because of top-level Abc
}

func TestMatchingField(t *testing.T) {
	fields, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S3{}))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		want *Field
	}{
		// Exact match wins.
		{"Abc", field("Abc", int(0), 1)},
		{"AbC", field("AbC", int(0), 2)},
		{"ABc", field("ABc", int(0), 0, 0)},
		// If there are multiple matches but no exact match or tag,
		// the first field wins, lexicographically by index.
		// Here, "ABc" is at a deeper embedding level, but since S4 appears
		// first in S3, its index precedes the other fields of S3.
		{"abc", field("ABc", int(0), 0, 0)},
		// Tag name takes precedence over untagged field of the same name.
		{"Tag", tfield("Tag", int(0), 4)},
		// Unexported fields disappear.
		{"unexported", nil},
		// Untagged embedded structs disappear.
		{"S4", nil},
	} {
		if got := fields.Match(test.name); !fieldsEqual(got, test.want) {
			t.Errorf("match %q:\ngot  %+v\nwant %+v", test.name, got, test.want)
		}
	}
}

func TestAgainstJSONMatchingField(t *testing.T) {
	s3 := S3{
		S4:         S4{ABc: 1, Y: 2},
		Abc:        3,
		AbC:        4,
		Tag:        5,
		X:          6,
		unexported: 7,
	}
	var want S3
	jsonRoundTrip(t, s3, &want)
	v := reflect.ValueOf(want)
	fields, err := NewCache(jsonTagParser, nil, nil).Fields(reflect.TypeOf(S3{}))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		got  int
	}{
		{"Abc", 3},
		{"AbC", 4},
		{"ABc", 1},
		{"abc", 1},
		{"Tag", 6},
	} {
		f := fields.Match(test.name)
		if f == nil {
			t.Fatalf("%s: no match", test.name)
		}
		w := v.FieldByIndex(f.Index).Interface()
		if test.got != w {
			t.Errorf("%s: got %d, want %d", test.name, test.got, w)
		}
	}
}

func TestTagErrors(t *testing.T) {
	called := false
	c := NewCache(func(t reflect.StructTag) (string, bool, interface{}, error) {
		called = true
		s := t.Get("f")
		if s == "bad" {
			return "", false, nil, errors.New("error")
		}
		return s, true, nil, nil
	}, nil, nil)

	type T struct {
		X int `f:"ok"`
		Y int `f:"bad"`
	}

	_, err := c.Fields(reflect.TypeOf(T{}))
	if !called {
		t.Fatal("tag parser not called")
	}
	if err == nil {
		t.Error("want error, got nil")
	}
	// Second time, we should cache the error.
	called = false
	_, err = c.Fields(reflect.TypeOf(T{}))
	if called {
		t.Fatal("tag parser called on second time")
	}
	if err == nil {
		t.Error("want error, got nil")
	}
}
