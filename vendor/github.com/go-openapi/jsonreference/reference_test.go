// Copyright 2013 sigu-399 ( https://github.com/sigu-399 )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author       sigu-399
// author-github  https://github.com/sigu-399
// author-mail    sigu.399@gmail.com
//
// repository-name  jsonreference
// repository-desc  An implementation of JSON Reference - Go language
//
// description    Automated tests on package.
//
// created        03-03-2013

package jsonreference

import (
	"testing"

	"github.com/go-openapi/jsonpointer"
	"github.com/stretchr/testify/assert"
)

func TestIsRoot(t *testing.T) {
	in := "#"
	r1, err := New(in)
	assert.NoError(t, err)
	assert.True(t, r1.IsRoot())

	in = "#/ok"
	r1 = MustCreateRef(in)
	assert.False(t, r1.IsRoot())

	assert.Panics(t, assert.PanicTestFunc(func() {
		MustCreateRef("%2")
	}))
}

func TestFull(t *testing.T) {

	in := "http://host/path/a/b/c#/f/a/b"

	r1, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %s", in, err.Error())
	}

	if in != r1.String() {
		t.Errorf("New(%v) = %v, expect %v", in, r1.String(), in)
	}

	if r1.HasFragmentOnly != false {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in, r1.HasFragmentOnly, false)
	}

	if r1.HasFullURL != true {
		t.Errorf("New(%v)::HasFullURL %v expect %v", in, r1.HasFullURL, true)
	}

	if r1.HasURLPathOnly != false {
		t.Errorf("New(%v)::HasURLPathOnly %v expect %v", in, r1.HasURLPathOnly, false)
	}

	if r1.HasFileScheme != false {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in, r1.HasFileScheme, false)
	}

	if r1.GetPointer().String() != "/f/a/b" {
		t.Errorf("New(%v)::GetPointer() %v expect %v", in, r1.GetPointer().String(), "/f/a/b")
	}
}

func TestFullURL(t *testing.T) {

	in := "http://host/path/a/b/c"

	r1, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %s", in, err.Error())
	}

	if in != r1.String() {
		t.Errorf("New(%v) = %v, expect %v", in, r1.String(), in)
	}

	if r1.HasFragmentOnly != false {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in, r1.HasFragmentOnly, false)
	}

	if r1.HasFullURL != true {
		t.Errorf("New(%v)::HasFullURL %v expect %v", in, r1.HasFullURL, true)
	}

	if r1.HasURLPathOnly != false {
		t.Errorf("New(%v)::HasURLPathOnly %v expect %v", in, r1.HasURLPathOnly, false)
	}

	if r1.HasFileScheme != false {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in, r1.HasFileScheme, false)
	}

	if r1.GetPointer().String() != "" {
		t.Errorf("New(%v)::GetPointer() %v expect %v", in, r1.GetPointer().String(), "")
	}
}

func TestFragmentOnly(t *testing.T) {

	in := "#/fragment/only"

	r1, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %s", in, err.Error())
	}

	if in != r1.String() {
		t.Errorf("New(%v) = %v, expect %v", in, r1.String(), in)
	}

	if r1.HasFragmentOnly != true {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in, r1.HasFragmentOnly, true)
	}

	if r1.HasFullURL != false {
		t.Errorf("New(%v)::HasFullURL %v expect %v", in, r1.HasFullURL, false)
	}

	if r1.HasURLPathOnly != false {
		t.Errorf("New(%v)::HasURLPathOnly %v expect %v", in, r1.HasURLPathOnly, false)
	}

	if r1.HasFileScheme != false {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in, r1.HasFileScheme, false)
	}

	if r1.GetPointer().String() != "/fragment/only" {
		t.Errorf("New(%v)::GetPointer() %v expect %v", in, r1.GetPointer().String(), "/fragment/only")
	}

	p, _ := jsonpointer.New(r1.referenceURL.Fragment)
	r2 := Ref{referencePointer: p, HasFragmentOnly: true}
	assert.Equal(t, r2.String(), in)

	r3 := Ref{referencePointer: p, HasFragmentOnly: false}
	assert.Equal(t, r3.String(), in[1:])
}

func TestURLPathOnly(t *testing.T) {

	in := "/documents/document.json"

	r1, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %s", in, err.Error())
	}

	if in != r1.String() {
		t.Errorf("New(%v) = %v, expect %v", in, r1.String(), in)
	}

	if r1.HasFragmentOnly != false {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in, r1.HasFragmentOnly, false)
	}

	if r1.HasFullURL != false {
		t.Errorf("New(%v)::HasFullURL %v expect %v", in, r1.HasFullURL, false)
	}

	if r1.HasURLPathOnly != true {
		t.Errorf("New(%v)::HasURLPathOnly %v expect %v", in, r1.HasURLPathOnly, true)
	}

	if r1.HasFileScheme != false {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in, r1.HasFileScheme, false)
	}

	if r1.GetPointer().String() != "" {
		t.Errorf("New(%v)::GetPointer() %v expect %v", in, r1.GetPointer().String(), "")
	}
}

func TestURLRelativePathOnly(t *testing.T) {

	in := "document.json"

	r1, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %s", in, err.Error())
	}

	if in != r1.String() {
		t.Errorf("New(%v) = %v, expect %v", in, r1.String(), in)
	}

	if r1.HasFragmentOnly != false {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in, r1.HasFragmentOnly, false)
	}

	if r1.HasFullURL != false {
		t.Errorf("New(%v)::HasFullURL %v expect %v", in, r1.HasFullURL, false)
	}

	if r1.HasURLPathOnly != true {
		t.Errorf("New(%v)::HasURLPathOnly %v expect %v", in, r1.HasURLPathOnly, true)
	}

	if r1.HasFileScheme != false {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in, r1.HasFileScheme, false)
	}

	if r1.GetPointer().String() != "" {
		t.Errorf("New(%v)::GetPointer() %v expect %v", in, r1.GetPointer().String(), "")
	}
}

func TestInheritsInValid(t *testing.T) {
	in1 := "http://www.test.com/doc.json"
	in2 := "#/a/b"

	r1, _ := New(in1)
	r2 := Ref{}
	result, err := r1.Inherits(r2)
	assert.Error(t, err)
	assert.Nil(t, result)

	r1 = Ref{}
	r2, _ = New(in2)
	result, err = r1.Inherits(r2)
	assert.NoError(t, err)
	assert.Equal(t, r2, *result)
}

func TestInheritsValid(t *testing.T) {

	in1 := "http://www.test.com/doc.json"
	in2 := "#/a/b"
	out := in1 + in2

	r1, _ := New(in1)
	r2, _ := New(in2)

	result, err := r1.Inherits(r2)
	if err != nil {
		t.Errorf("Inherits(%s,%s) error %s", r1.String(), r2.String(), err.Error())
	}

	if result.String() != out {
		t.Errorf("Inherits(%s,%s) = %s, expect %s", r1.String(), r2.String(), result.String(), out)
	}

	if result.GetPointer().String() != "/a/b" {
		t.Errorf("result(%v)::GetPointer() %v expect %v", result.String(), result.GetPointer().String(), "/a/b")
	}
}

func TestInheritsDifferentHost(t *testing.T) {

	in1 := "http://www.test.com/doc.json"
	in2 := "http://www.test2.com/doc.json#bla"

	r1, _ := New(in1)
	r2, _ := New(in2)

	result, err := r1.Inherits(r2)

	if err != nil {
		t.Errorf("Inherits(%s,%s) should not fail. Error: %s", r1.String(), r2.String(), err.Error())
	}

	if result.String() != in2 {
		t.Errorf("Inherits(%s,%s) should be %s but is %s", in1, in2, in2, result)
	}

	if result.GetPointer().String() != "" {
		t.Errorf("result(%v)::GetPointer() %v expect %v", result.String(), result.GetPointer().String(), "")
	}
}

func TestFileScheme(t *testing.T) {

	in1 := "file:///Users/mac/1.json#a"
	in2 := "file:///Users/mac/2.json#b"

	r1, _ := New(in1)
	r2, _ := New(in2)

	if r1.HasFragmentOnly != false {
		t.Errorf("New(%v)::HasFragmentOnly %v expect %v", in1, r1.HasFragmentOnly, false)
	}

	if r1.HasFileScheme != true {
		t.Errorf("New(%v)::HasFileScheme %v expect %v", in1, r1.HasFileScheme, true)
	}

	if r1.HasFullFilePath != true {
		t.Errorf("New(%v)::HasFullFilePath %v expect %v", in1, r1.HasFullFilePath, true)
	}

	if r1.IsCanonical() != true {
		t.Errorf("New(%v)::IsCanonical %v expect %v", in1, r1.IsCanonical, true)
	}

	result, err := r1.Inherits(r2)
	if err != nil {
		t.Errorf("Inherits(%s,%s) should not fail. Error: %s", r1.String(), r2.String(), err.Error())
	}
	if result.String() != in2 {
		t.Errorf("Inherits(%s,%s) should be %s but is %s", in1, in2, in2, result)
	}

	if result.GetPointer().String() != "" {
		t.Errorf("result(%v)::GetPointer() %v expect %v", result.String(), result.GetPointer().String(), "")
	}
}

func TestReferenceResolution(t *testing.T) {

	// 5.4. Reference Resolution Examples
	// http://tools.ietf.org/html/rfc3986#section-5.4

	base := "http://a/b/c/d;p?q"
	baseRef, err := New(base)

	if err != nil {
		t.Errorf("New(%s) failed error: %s", base, err.Error())
	}
	if baseRef.String() != base {
		t.Errorf("New(%s) %s expected %s", base, baseRef.String(), base)
	}

	checks := []string{
		// 5.4.1. Normal Examples
		// http://tools.ietf.org/html/rfc3986#section-5.4.1

		"g:h", "g:h",
		"g", "http://a/b/c/g",
		"./g", "http://a/b/c/g",
		"g/", "http://a/b/c/g/",
		"/g", "http://a/g",
		"//g", "http://g",
		"?y", "http://a/b/c/d;p?y",
		"g?y", "http://a/b/c/g?y",
		"#s", "http://a/b/c/d;p?q#s",
		"g#s", "http://a/b/c/g#s",
		"g?y#s", "http://a/b/c/g?y#s",
		";x", "http://a/b/c/;x",
		"g;x", "http://a/b/c/g;x",
		"g;x?y#s", "http://a/b/c/g;x?y#s",
		"", "http://a/b/c/d;p?q",
		".", "http://a/b/c/",
		"./", "http://a/b/c/",
		"..", "http://a/b/",
		"../", "http://a/b/",
		"../g", "http://a/b/g",
		"../..", "http://a/",
		"../../", "http://a/",
		"../../g", "http://a/g",

		// 5.4.2. Abnormal Examples
		// http://tools.ietf.org/html/rfc3986#section-5.4.2

		"../../../g", "http://a/g",
		"../../../../g", "http://a/g",

		"/./g", "http://a/g",
		"/../g", "http://a/g",
		"g.", "http://a/b/c/g.",
		".g", "http://a/b/c/.g",
		"g..", "http://a/b/c/g..",
		"..g", "http://a/b/c/..g",

		"./../g", "http://a/b/g",
		"./g/.", "http://a/b/c/g/",
		"g/./h", "http://a/b/c/g/h",
		"g/../h", "http://a/b/c/h",
		"g;x=1/./y", "http://a/b/c/g;x=1/y",
		"g;x=1/../y", "http://a/b/c/y",

		"g?y/./x", "http://a/b/c/g?y/./x",
		"g?y/../x", "http://a/b/c/g?y/../x",
		"g#s/./x", "http://a/b/c/g#s/./x",
		"g#s/../x", "http://a/b/c/g#s/../x",

		"http:g", "http:g", // for strict parsers
		//"http:g", "http://a/b/c/g", // for backward compatibility

	}
	for i := 0; i < len(checks); i += 2 {
		child := checks[i]
		expected := checks[i+1]
		// fmt.Printf("%d:   %v  ->  %v\n", i/2, child, expected)

		childRef, e := New(child)
		if e != nil {
			t.Errorf("%d: New(%s) failed error: %s", i/2, child, e.Error())
		}

		res, e := baseRef.Inherits(childRef)
		if res == nil {
			t.Errorf("%d: Inherits(%s, %s) nil not expected", i/2, base, child)
		}
		if e != nil {
			t.Errorf("%d: Inherits(%s) failed error: %s", i/2, child, e.Error())
		}
		if res.String() != expected {
			t.Errorf("%d: Inherits(%s, %s) %s expected %s", i/2, base, child, res.String(), expected)
		}
	}
}
