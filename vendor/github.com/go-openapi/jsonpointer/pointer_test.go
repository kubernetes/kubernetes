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
// repository-name  jsonpointer
// repository-desc  An implementation of JSON Pointer - Go language
//
// description    Automated tests on package.
//
// created        03-03-2013

package jsonpointer

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	TestDocumentNBItems = 11
	TestNodeObjNBItems  = 4
	TestDocumentString  = `{
"foo": ["bar", "baz"],
"obj": { "a":1, "b":2, "c":[3,4], "d":[ {"e":9}, {"f":[50,51]} ] },
"": 0,
"a/b": 1,
"c%d": 2,
"e^f": 3,
"g|h": 4,
"i\\j": 5,
"k\"l": 6,
" ": 7,
"m~n": 8
}`
)

var testDocumentJSON interface{}

type testStructJSON struct {
	Foo []string `json:"foo"`
	Obj struct {
		A int   `json:"a"`
		B int   `json:"b"`
		C []int `json:"c"`
		D []struct {
			E int   `json:"e"`
			F []int `json:"f"`
		} `json:"d"`
	} `json:"obj"`
}

type aliasedMap map[string]interface{}

var testStructJSONDoc testStructJSON
var testStructJSONPtr *testStructJSON

func init() {
	json.Unmarshal([]byte(TestDocumentString), &testDocumentJSON)
	json.Unmarshal([]byte(TestDocumentString), &testStructJSONDoc)
	testStructJSONPtr = &testStructJSONDoc
}

func TestEscaping(t *testing.T) {

	ins := []string{`/`, `/`, `/a~1b`, `/a~1b`, `/c%d`, `/e^f`, `/g|h`, `/i\j`, `/k"l`, `/ `, `/m~0n`}
	outs := []float64{0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8}

	for i := range ins {

		p, err := New(ins[i])
		if err != nil {
			t.Errorf("New(%v) error %v", ins[i], err.Error())
		}

		result, _, err := p.Get(testDocumentJSON)
		if err != nil {
			t.Errorf("Get(%v) error %v", ins[i], err.Error())
		}

		if result != outs[i] {
			t.Errorf("Get(%v) = %v, expect %v", ins[i], result, outs[i])
		}
	}

}

func TestFullDocument(t *testing.T) {

	in := ``

	p, err := New(in)
	if err != nil {
		t.Errorf("New(%v) error %v", in, err.Error())
	}

	result, _, err := p.Get(testDocumentJSON)
	if err != nil {
		t.Errorf("Get(%v) error %v", in, err.Error())
	}

	if len(result.(map[string]interface{})) != TestDocumentNBItems {
		t.Errorf("Get(%v) = %v, expect full document", in, result)
	}

	result, _, err = p.get(testDocumentJSON, nil)
	if err != nil {
		t.Errorf("Get(%v) error %v", in, err.Error())
	}

	if len(result.(map[string]interface{})) != TestDocumentNBItems {
		t.Errorf("Get(%v) = %v, expect full document", in, result)
	}
}

func TestDecodedTokens(t *testing.T) {
	p, err := New("/obj/a~1b")
	assert.NoError(t, err)
	assert.Equal(t, []string{"obj", "a/b"}, p.DecodedTokens())
}

func TestIsEmpty(t *testing.T) {
	p, err := New("")
	assert.NoError(t, err)
	assert.True(t, p.IsEmpty())
	p, err = New("/obj")
	assert.NoError(t, err)
	assert.False(t, p.IsEmpty())
}

func TestGetSingle(t *testing.T) {
	in := `/obj`

	_, err := New(in)
	assert.NoError(t, err)
	result, _, err := GetForToken(testDocumentJSON, "obj")
	assert.NoError(t, err)
	assert.Len(t, result, TestNodeObjNBItems)

	result, _, err = GetForToken(testStructJSONDoc, "Obj")
	assert.Error(t, err)
	assert.Nil(t, result)

	result, _, err = GetForToken(testStructJSONDoc, "Obj2")
	assert.Error(t, err)
	assert.Nil(t, result)
}

type pointableImpl struct {
	a string
}

func (p pointableImpl) JSONLookup(token string) (interface{}, error) {
	if token == "some" {
		return p.a, nil
	}
	return nil, fmt.Errorf("object has no field %q", token)
}

func TestPointableInterface(t *testing.T) {
	p := &pointableImpl{"hello"}

	result, _, err := GetForToken(p, "some")
	assert.NoError(t, err)
	assert.Equal(t, p.a, result)

	result, _, err = GetForToken(p, "something")
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestGetNode(t *testing.T) {

	in := `/obj`

	p, err := New(in)
	assert.NoError(t, err)
	result, _, err := p.Get(testDocumentJSON)
	assert.NoError(t, err)
	assert.Len(t, result, TestNodeObjNBItems)

	result, _, err = p.Get(aliasedMap(testDocumentJSON.(map[string]interface{})))
	assert.NoError(t, err)
	assert.Len(t, result, TestNodeObjNBItems)

	result, _, err = p.Get(testStructJSONDoc)
	assert.NoError(t, err)
	assert.Equal(t, testStructJSONDoc.Obj, result)

	result, _, err = p.Get(testStructJSONPtr)
	assert.NoError(t, err)
	assert.Equal(t, testStructJSONDoc.Obj, result)
}

func TestArray(t *testing.T) {

	ins := []string{`/foo/0`, `/foo/0`, `/foo/1`}
	outs := []string{"bar", "bar", "baz"}

	for i := range ins {
		p, err := New(ins[i])
		assert.NoError(t, err)

		result, _, err := p.Get(testStructJSONDoc)
		assert.NoError(t, err)
		assert.Equal(t, outs[i], result)

		result, _, err = p.Get(testStructJSONPtr)
		assert.NoError(t, err)
		assert.Equal(t, outs[i], result)

		result, _, err = p.Get(testDocumentJSON)
		assert.NoError(t, err)
		assert.Equal(t, outs[i], result)
	}
}

func TestOtherThings(t *testing.T) {
	_, err := New("abc")
	assert.Error(t, err)

	p, err := New("")
	assert.NoError(t, err)
	assert.Equal(t, "", p.String())

	p, err = New("/obj/a")
	assert.Equal(t, "/obj/a", p.String())

	s := Escape("m~n")
	assert.Equal(t, "m~0n", s)
	s = Escape("m/n")
	assert.Equal(t, "m~1n", s)

	p, err = New("/foo/3")
	assert.NoError(t, err)
	_, _, err = p.Get(testDocumentJSON)
	assert.Error(t, err)

	p, err = New("/foo/a")
	assert.NoError(t, err)
	_, _, err = p.Get(testDocumentJSON)
	assert.Error(t, err)

	p, err = New("/notthere")
	assert.NoError(t, err)
	_, _, err = p.Get(testDocumentJSON)
	assert.Error(t, err)

	p, err = New("/invalid")
	assert.NoError(t, err)
	_, _, err = p.Get(1234)
	assert.Error(t, err)

	p, err = New("/foo/1")
	assert.NoError(t, err)
	expected := "hello"
	bbb := testDocumentJSON.(map[string]interface{})["foo"]
	bbb.([]interface{})[1] = "hello"

	v, _, err := p.Get(testDocumentJSON)
	assert.NoError(t, err)
	assert.Equal(t, expected, v)

	esc := Escape("a/")
	assert.Equal(t, "a~1", esc)
	unesc := Unescape(esc)
	assert.Equal(t, "a/", unesc)

	unesc = Unescape("~01")
	assert.Equal(t, "~1", unesc)
	assert.Equal(t, "~0~1", Escape("~/"))
	assert.Equal(t, "~/", Unescape("~0~1"))
}

func TestObject(t *testing.T) {

	ins := []string{`/obj/a`, `/obj/b`, `/obj/c/0`, `/obj/c/1`, `/obj/c/1`, `/obj/d/1/f/0`}
	outs := []float64{1, 2, 3, 4, 4, 50}

	for i := range ins {

		p, err := New(ins[i])
		assert.NoError(t, err)

		result, _, err := p.Get(testDocumentJSON)
		assert.NoError(t, err)
		assert.Equal(t, outs[i], result)

		result, _, err = p.Get(testStructJSONDoc)
		assert.NoError(t, err)
		assert.EqualValues(t, outs[i], result)

		result, _, err = p.Get(testStructJSONPtr)
		assert.NoError(t, err)
		assert.EqualValues(t, outs[i], result)
	}
}
