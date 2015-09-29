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
	goodNames = []string{
		"asdf",
		"foo-bar-baz",
		"database",
	}
	badNames = []string{
		"",
		"foo#",
		"example.com",
		"EXAMPLE.com",
		"example/database",
		"example/database-1.0.0",
		"foo.com/BAR",
		"example.com/app_1",
		"/app",
		"app/",
		"-app",
		"app-",
		".app",
		"app.",
	}
)

func TestNewACName(t *testing.T) {
	for i, in := range goodNames {
		l, err := NewACName(in)
		if err != nil {
			t.Errorf("#%d: got err=%v, want nil", i, err)
		}
		if l == nil {
			t.Errorf("#%d: got l=nil, want non-nil", i)
		}
	}
}

func TestNewACNameBad(t *testing.T) {
	for i, in := range badNames {
		l, err := NewACName(in)
		if l != nil {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}

func TestMustACName(t *testing.T) {
	for i, in := range goodNames {
		l := MustACName(in)
		if l == nil {
			t.Errorf("#%d: got l=nil, want non-nil", i)
		}
	}
}

func expectPanicMustACName(i int, in string, t *testing.T) {
	defer func() {
		recover()
	}()
	_ = MustACName(in)
	t.Errorf("#%d: panic expected", i)
}

func TestMustACNameBad(t *testing.T) {
	for i, in := range badNames {
		expectPanicMustACName(i, in, t)
	}
}

func TestSanitizeACName(t *testing.T) {
	tests := map[string]string{
		"foo#":                                             "foo",
		"EXAMPLE.com":                                      "example-com",
		"foo.com/BAR":                                      "foo-com-bar",
		"example.com/app_1":                                "example-com-app-1",
		"/app":                                             "app",
		"app/":                                             "app",
		"-app":                                             "app",
		"app-":                                             "app",
		".app":                                             "app",
		"app.":                                             "app",
		"app///":                                           "app",
		"-/.app..":                                         "app",
		"-/app.name-test/-/":                               "app-name-test",
		"sub-domain.example.com/org/product/release-1.0.0": "sub-domain-example-com-org-product-release-1-0-0",
	}
	for in, ex := range tests {
		o, err := SanitizeACName(in)
		if err != nil {
			t.Errorf("got err=%v, want nil", err)
		}
		if o != ex {
			t.Errorf("got l=%s, want %s", o, ex)
		}
	}
}

func TestACNameSetGood(t *testing.T) {
	tests := map[string]ACName{
		"blargh":               ACName("blargh"),
		"example-ourapp-1-0-0": ACName("example-ourapp-1-0-0"),
	}
	for in, w := range tests {
		// Ensure an empty name is set appropriately
		var a ACName
		err := a.Set(in)
		if err != nil {
			t.Errorf("%v: got err=%v, want nil", in, err)
			continue
		}
		if !reflect.DeepEqual(a, w) {
			t.Errorf("%v: a=%v, want %v", in, a, w)
		}

		// Ensure an existing name is overwritten
		var b ACName = ACName("orig")
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

func TestACNameSetBad(t *testing.T) {
	for i, in := range badNames {
		// Ensure an empty name stays empty
		var a ACName
		err := a.Set(in)
		if err == nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
			continue
		}
		if w := ACName(""); !reflect.DeepEqual(a, w) {
			t.Errorf("%d: a=%v, want %v", i, a, w)
		}

		// Ensure an existing name is not overwritten
		var b ACName = ACName("orig")
		err = b.Set(in)
		if err == nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
			continue
		}
		if w := ACName("orig"); !reflect.DeepEqual(b, w) {
			t.Errorf("%d: b=%v, want %v", i, b, w)
		}
	}
}

func TestSanitizeACNameBad(t *testing.T) {
	tests := []string{
		"__",
		"..",
		"//",
		"",
		".//-"}
	for i, in := range tests {
		l, err := SanitizeACName(in)
		if l != "" {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}

func TestACNameUnmarshalBad(t *testing.T) {
	tests := []string{
		"",
		"garbage",
		`""`,
		`"EXAMPLE"`,
		`"example.com/app_1"`,
	}
	for i, in := range tests {
		var a, b ACName
		err := a.UnmarshalJSON([]byte(in))
		if err == nil {
			t.Errorf("#%d: err=nil, want non-nil", i)
		} else if !reflect.DeepEqual(a, b) {
			t.Errorf("#%d: a=%v, want empty", i, a)
		}
	}
}

func TestACNameUnmarshalGood(t *testing.T) {
	tests := map[string]ACName{
		`"example"`: ACName("example"),
		`"foo-bar"`: ACName("foo-bar"),
	}
	for in, w := range tests {
		var a ACName
		err := json.Unmarshal([]byte(in), &a)
		if err != nil {
			t.Errorf("%v: err=%v, want nil", in, err)
		} else if !reflect.DeepEqual(a, w) {
			t.Errorf("%v: a=%v, want %v", in, a, w)
		}
	}
}

func TestACNameMarshalBad(t *testing.T) {
	tests := map[string]error{
		"Foo":      ErrInvalidCharInACName,
		"foo#":     ErrInvalidCharInACName,
		"-foo":     ErrInvalidEdgeInACName,
		"example-": ErrInvalidEdgeInACName,
		"":         ErrEmptyACName,
	}
	for in, werr := range tests {
		a := ACName(in)
		b, gerr := json.Marshal(a)
		if b != nil {
			t.Errorf("ACName(%q): want b=nil, got %v", in, b)
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

func TestACNameMarshalGood(t *testing.T) {
	for i, in := range goodNames {
		a := ACName(in)
		b, err := json.Marshal(a)
		if !reflect.DeepEqual(b, []byte(`"`+in+`"`)) {
			t.Errorf("#%d: marshalled=%v, want %v", i, b, []byte(in))
		}
		if err != nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
		}
	}
}
