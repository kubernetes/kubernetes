/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package generators

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/parser"
	"k8s.io/gengo/types"
)

func construct(t *testing.T, files map[string]string, testNamer namer.Namer) (*parser.Builder, types.Universe, []*types.Type) {
	b := parser.New()
	for name, src := range files {
		if err := b.AddFile(filepath.Dir(name), name, []byte(src)); err != nil {
			t.Fatal(err)
		}
	}
	u, err := b.FindTypes()
	if err != nil {
		t.Fatal(err)
	}
	orderer := namer.Orderer{Namer: testNamer}
	o := orderer.OrderUniverse(u)
	return b, u, o
}

func testOpenAPITypeWritter(t *testing.T, code string) (error, *assert.Assertions, *bytes.Buffer) {
	assert := assert.New(t)
	var testFiles = map[string]string{
		"base/foo/bar.go": code,
	}
	rawNamer := namer.NewRawNamer("o", nil)
	namers := namer.NameSystems{
		"raw": namer.NewRawNamer("", nil),
	}
	builder, universe, _ := construct(t, testFiles, rawNamer)
	context, err := generator.NewContext(builder, namers, "raw")
	if err != nil {
		t.Fatal(err)
	}
	buffer := &bytes.Buffer{}
	sw := generator.NewSnippetWriter(buffer, context, "$", "$")
	blahT := universe.Type(types.Name{Package: "base/foo", Name: "Blah"})
	return newOpenAPITypeWriter(sw).generate(blahT), assert, buffer
}

func TestSimple(t *testing.T) {
	err, assert, buffer := testOpenAPITypeWritter(t, `
package foo

import (
	"time"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// Blah is a test.
// +k8s:openapi=true
type Blah struct {
	// A simple string
	String string
	// A simple int
	Int int `+"`"+`json:",omitempty"`+"`"+`
	// An int considered string simple int
	IntString int `+"`"+`json:",string"`+"`"+`
	// A simple int64
	Int64 int64
	// A simple int32
	Int32 int32
	// A simple int16
	Int16 int16
	// A simple int8
	Int8 int8
	// A simple int
	Uint uint
	// A simple int64
	Uint64 uint64
	// A simple int32
	Uint32 uint32
	// A simple int16
	Uint16 uint16
	// A simple int8
	Uint8 uint8
	// A simple byte
	Byte byte
	// A simple boolean
	Bool bool
	// A simple float64
	Float64 float64
	// A simple float32
	Float32 float32
	// A simple time
	Time time.Time
	// a base64 encoded characters
	ByteArray []byte
	// an int or string type
	IntOrString intstr.IntOrString
}
		`)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(`"foo.Blah": {
Schema: spec.Schema{
SchemaProps: spec.SchemaProps{
Description: "Blah is a test.",
Properties: map[string]spec.Schema{
"String": {
SchemaProps: spec.SchemaProps{
Description: "A simple string",
Type: []string{"string"},
Format: "",
},
},
"Int64": {
SchemaProps: spec.SchemaProps{
Description: "A simple int64",
Type: []string{"integer"},
Format: "int64",
},
},
"Int32": {
SchemaProps: spec.SchemaProps{
Description: "A simple int32",
Type: []string{"integer"},
Format: "int32",
},
},
"Int16": {
SchemaProps: spec.SchemaProps{
Description: "A simple int16",
Type: []string{"integer"},
Format: "int32",
},
},
"Int8": {
SchemaProps: spec.SchemaProps{
Description: "A simple int8",
Type: []string{"integer"},
Format: "byte",
},
},
"Uint": {
SchemaProps: spec.SchemaProps{
Description: "A simple int",
Type: []string{"integer"},
Format: "int32",
},
},
"Uint64": {
SchemaProps: spec.SchemaProps{
Description: "A simple int64",
Type: []string{"integer"},
Format: "int64",
},
},
"Uint32": {
SchemaProps: spec.SchemaProps{
Description: "A simple int32",
Type: []string{"integer"},
Format: "int64",
},
},
"Uint16": {
SchemaProps: spec.SchemaProps{
Description: "A simple int16",
Type: []string{"integer"},
Format: "int32",
},
},
"Uint8": {
SchemaProps: spec.SchemaProps{
Description: "A simple int8",
Type: []string{"integer"},
Format: "byte",
},
},
"Byte": {
SchemaProps: spec.SchemaProps{
Description: "A simple byte",
Type: []string{"integer"},
Format: "byte",
},
},
"Bool": {
SchemaProps: spec.SchemaProps{
Description: "A simple boolean",
Type: []string{"boolean"},
Format: "",
},
},
"Float64": {
SchemaProps: spec.SchemaProps{
Description: "A simple float64",
Type: []string{"number"},
Format: "double",
},
},
"Float32": {
SchemaProps: spec.SchemaProps{
Description: "A simple float32",
Type: []string{"number"},
Format: "float",
},
},
"Time": {
SchemaProps: spec.SchemaProps{
Description: "A simple time",
Type: []string{"string"},
Format: "date-time",
},
},
"ByteArray": {
SchemaProps: spec.SchemaProps{
Description: "a base64 encoded characters",
Type: []string{"string"},
Format: "byte",
},
},
"IntOrString": {
SchemaProps: spec.SchemaProps{
Description: "an int or string type",
Ref: spec.MustCreateRef("#/definitions/intstr.IntOrString"),
},
},
},
Required: []string{"String","Int64","Int32","Int16","Int8","Uint","Uint64","Uint32","Uint16","Uint8","Byte","Bool","Float64","Float32","Time","ByteArray","IntOrString"},
},
},
Dependencies: []string{
"intstr.IntOrString",},
},
`, buffer.String())
}

func TestFailingSample1(t *testing.T) {
	err, assert, _ := testOpenAPITypeWritter(t, `
package foo

// Map sample tests openAPIGen.generateMapProperty method.
type Blah struct {
	// A sample String to String map
	StringToArray map[string]map[string]string
}
	`)
	if assert.Error(err, "An error was expected") {
		assert.Equal(err, fmt.Errorf("map Element kind Map is not supported in map[string]map[string]string"))
	}
}

func TestFailingSample2(t *testing.T) {
	err, assert, _ := testOpenAPITypeWritter(t, `
package foo

// Map sample tests openAPIGen.generateMapProperty method.
type Blah struct {
	// A sample String to String map
	StringToArray map[int]string
}	`)
	if assert.Error(err, "An error was expected") {
		assert.Equal(err, fmt.Errorf("map with non-string keys are not supported by OpenAPI in map[int]string"))
	}
}

func TestPointer(t *testing.T) {
	err, assert, buffer := testOpenAPITypeWritter(t, `
package foo

// PointerSample demonstrate pointer's properties
type Blah struct {
	// A string pointer
	StringPointer *string
	// A struct pointer
	StructPointer *Blah
	// A slice pointer
	SlicePointer *[]string
	// A map pointer
	MapPointer *map[string]string
}
	`)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(`"foo.Blah": {
Schema: spec.Schema{
SchemaProps: spec.SchemaProps{
Description: "PointerSample demonstrate pointer's properties",
Properties: map[string]spec.Schema{
"StringPointer": {
SchemaProps: spec.SchemaProps{
Description: "A string pointer",
Type: []string{"string"},
Format: "",
},
},
"StructPointer": {
SchemaProps: spec.SchemaProps{
Description: "A struct pointer",
Ref: spec.MustCreateRef("#/definitions/foo.Blah"),
},
},
"SlicePointer": {
SchemaProps: spec.SchemaProps{
Description: "A slice pointer",
Type: []string{"array"},
Items: &spec.SchemaOrArray{
Schema: &spec.Schema{
SchemaProps: spec.SchemaProps{
Type: []string{"string"},
Format: "",
},
},
},
},
},
"MapPointer": {
SchemaProps: spec.SchemaProps{
Description: "A map pointer",
Type: []string{"object"},
AdditionalProperties: &spec.SchemaOrBool{
Schema: &spec.Schema{
SchemaProps: spec.SchemaProps{
Type: []string{"string"},
Format: "",
},
},
},
},
},
},
Required: []string{"StringPointer","StructPointer","SlicePointer","MapPointer"},
},
},
Dependencies: []string{
"foo.Blah",},
},
`, buffer.String())
}
