// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
	"reflect"
	"testing"
)

var (
	goodIdentifiers = []string{
		"asdf",
		"foo-bar-baz",
		"zab_rab_oof",
		"database",
		"my~database",
		"example.com/database",
		"example.com/~bob/database",
		"example.com/ourapp-1.0.0",
		"sub-domain.example.com/org/product/release-1.0.0",
		"sub-domain.example.com/org/product/~alice/release-1.0.0",
	}
	badIdentifiers = []string{
		"",
		"BAR",
		"foo#",
		"EXAMPLE.com",
		"foo.com/BAR",
		"/app",
		"app/",
		"-app",
		"app-",
		".app",
		"app.",
		"_app",
		"app_",
		"~app",
		"app~",
	}
)

func TestNewACIdentifier(t *testing.T) {
	for i, in := range goodIdentifiers {
		l, err := NewACIdentifier(in)
		if err != nil {
			t.Errorf("#%d: got err=%v, want nil", i, err)
		}
		if l == nil {
			t.Errorf("#%d: got l=nil, want non-nil", i)
		}
	}
}

func TestNewACIdentifierBad(t *testing.T) {
	for i, in := range badIdentifiers {
		l, err := NewACIdentifier(in)
		if l != nil {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}

func TestMustACIdentifier(t *testing.T) {
	for i, in := range goodIdentifiers {
		l := MustACIdentifier(in)
		if l == nil {
			t.Errorf("#%d: got l=nil, want non-nil", i)
		}
	}
}

func expectPanicMustACIdentifier(i int, in string, t *testing.T) {
	defer func() {
		recover()
	}()
	_ = MustACIdentifier(in)
	t.Errorf("#%d: panic expected", i)
}

func TestMustACIdentifierBad(t *testing.T) {
	for i, in := range badIdentifiers {
		expectPanicMustACIdentifier(i, in, t)
	}
}

func TestSanitizeACIdentifier(t *testing.T) {
	tests := map[string]string{
		"foo#":                                                    "foo",
		"FOO":                                                     "foo",
		"EXAMPLE.com":                                             "example.com",
		"foo.com/BAR":                                             "foo.com/bar",
		"/app":                                                    "app",
		"app/":                                                    "app",
		"-app":                                                    "app",
		"app-":                                                    "app",
		".app":                                                    "app",
		"app.":                                                    "app",
		"_app":                                                    "app",
		"app_":                                                    "app",
		"~app":                                                    "app",
		"app~":                                                    "app",
		"app///":                                                  "app",
		"-/.app..":                                                "app",
		"-/app.name-test/-/":                                      "app.name-test",
		"sub-domain.example.com/org/product/~alice/release-1.0.0": "sub-domain.example.com/org/product/~alice/release-1.0.0",
	}
	for in, ex := range tests {
		o, err := SanitizeACIdentifier(in)
		if err != nil {
			t.Errorf("got err=%v, want nil", err)
		}
		if o != ex {
			t.Errorf("got l=%s, want %s", o, ex)
		}
	}
}

func TestACIdentifierSetGood(t *testing.T) {
	tests := map[string]ACIdentifier{
		"blargh":               ACIdentifier("blargh"),
		"example-ourapp-1-0-0": ACIdentifier("example-ourapp-1-0-0"),
	}
	for in, w := range tests {
		// Ensure an empty name is set appropriately
		var a ACIdentifier
		err := a.Set(in)
		if err != nil {
			t.Errorf("%v: got err=%v, want nil", in, err)
			continue
		}
		if !reflect.DeepEqual(a, w) {
			t.Errorf("%v: a=%v, want %v", in, a, w)
		}

		// Ensure an existing name is overwritten
		var b ACIdentifier = ACIdentifier("orig")
		err = b.Set(in)
		if err != nil {
			t.Errorf("%v: got err=%v, want nil", in, err)
			continue
		}
		if !reflect.DeepEqual(b, w) {
			t.Errorf("%v: b=%v, want %v", in, b, w)
		}
	}
}

func TestACIdentifierSetBad(t *testing.T) {
	for i, in := range badIdentifiers {
		// Ensure an empty name stays empty
		var a ACIdentifier
		err := a.Set(in)
		if err == nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
			continue
		}
		if w := ACIdentifier(""); !reflect.DeepEqual(a, w) {
			t.Errorf("%d: a=%v, want %v", i, a, w)
		}

		// Ensure an existing name is not overwritten
		var b ACIdentifier = ACIdentifier("orig")
		err = b.Set(in)
		if err == nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
			continue
		}
		if w := ACIdentifier("orig"); !reflect.DeepEqual(b, w) {
			t.Errorf("%d: b=%v, want %v", i, b, w)
		}
	}
}

func TestSanitizeACIdentifierBad(t *testing.T) {
	tests := []string{
		"__",
		"..",
		"//",
		"",
		".//-"}
	for i, in := range tests {
		l, err := SanitizeACIdentifier(in)
		if l != "" {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}

func TestACIdentifierUnmarshalBad(t *testing.T) {
	tests := []string{
		"",
		"garbage",
		`""`,
		`"EXAMPLE"`,
		`"example.com/app#1"`,
		`"~example.com/app1"`,
	}
	for i, in := range tests {
		var a, b ACIdentifier
		err := a.UnmarshalJSON([]byte(in))
		if err == nil {
			t.Errorf("#%d: err=nil, want non-nil", i)
		} else if !reflect.DeepEqual(a, b) {
			t.Errorf("#%d: a=%v, want empty", i, a)
		}
	}
}

func TestACIdentifierUnmarshalGood(t *testing.T) {
	tests := map[string]ACIdentifier{
		`"example"`: ACIdentifier("example"),
		`"foo-bar"`: ACIdentifier("foo-bar"),
	}
	for in, w := range tests {
		var a ACIdentifier
		err := json.Unmarshal([]byte(in), &a)
		if err != nil {
			t.Errorf("%v: err=%v, want nil", in, err)
		} else if !reflect.DeepEqual(a, w) {
			t.Errorf("%v: a=%v, want %v", in, a, w)
		}
	}
}

func TestACIdentifierMarshalBad(t *testing.T) {
	tests := map[string]error{
		"Foo":      ErrInvalidCharInACIdentifier,
		"foo#":     ErrInvalidCharInACIdentifier,
		"-foo":     ErrInvalidEdgeInACIdentifier,
		"example-": ErrInvalidEdgeInACIdentifier,
		"":         ErrEmptyACIdentifier,
	}
	for in, werr := range tests {
		a := ACIdentifier(in)
		b, gerr := json.Marshal(a)
		if b != nil {
			t.Errorf("ACIdentifier(%q): want b=nil, got %v", in, b)
		}
		if jerr, ok := gerr.(*json.MarshalerError); !ok {
			t.Errorf("expected JSONMarshalerError")
		} else {
			if e := jerr.Err; e != werr {
				t.Errorf("err=%#v, want %#v", e, werr)
			}
		}
	}
}

func TestACIdentifierMarshalGood(t *testing.T) {
	for i, in := range goodIdentifiers {
		a := ACIdentifier(in)
		b, err := json.Marshal(a)
		if !reflect.DeepEqual(b, []byte(`"`+in+`"`)) {
			t.Errorf("#%d: marshalled=%v, want %v", i, b, []byte(in))
		}
		if err != nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
		}
	}
}
