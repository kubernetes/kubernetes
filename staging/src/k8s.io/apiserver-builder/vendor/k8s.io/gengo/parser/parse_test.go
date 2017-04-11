/*
Copyright 2015 The Kubernetes Authors.

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

package parser_test

import (
	"bytes"
	"path/filepath"
	"reflect"
	"testing"
	"text/template"

	"k8s.io/gengo/args"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/parser"
	"k8s.io/gengo/types"
)

func TestRecursive(t *testing.T) {
	d := args.Default()
	d.InputDirs = []string{"k8s.io/gengo/testdata/a/..."}
	b, err := d.NewBuilder()
	if err != nil {
		t.Fatalf("Fail making builder: %v", err)
	}
	_, err = b.FindTypes()
	if err != nil {
		t.Fatalf("Fail finding types: %v", err)
	}
	foundB := false
	for _, p := range b.FindPackages() {
		t.Logf("Package: %v", p)
		if p == "k8s.io/gengo/testdata/a/b" {
			foundB = true
		}
	}
	if !foundB {
		t.Errorf("Expected to find packages a and b")
	}
}

type file struct {
	path     string
	contents string
}

// Pass files in topological order - deps first!
func construct(t *testing.T, files []file, testNamer namer.Namer) (*parser.Builder, types.Universe, []*types.Type) {
	b := parser.New()
	for _, f := range files {
		if err := b.AddFileForTest(filepath.Dir(f.path), f.path, []byte(f.contents)); err != nil {
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

func TestBuilder(t *testing.T) {
	var testFiles = []file{
		{
			path: "base/common/proto/common.go", contents: `
                package common

                type Object struct {
    	            ID int64
                }
                `,
		}, {
			path: "base/foo/proto/foo.go", contents: `
                package foo

                import (
	                "base/common/proto"
                )

                type Blah struct {
    	            common.Object
    	            Count int64
    	            Frobbers map[string]*Frobber
    	            Baz []Object
    	            Nickname *string
    	            NumberIsAFavorite map[int]bool
                }

                type Frobber struct {
	                Name string
	                Amount int64
                }

                type Object struct {
	                common.Object
                }

                func AFunc(obj1 common.Object, obj2 Object) Frobber {
                }

                var AVar Frobber
    
                var (
	                AnotherVar = Frobber{}
                )
                `,
		},
	}

	var tmplText = `
package o
{{define "Struct"}}type {{Name .}} interface { {{range $m := .Members}}{{$n := Name $m.Type}}
	{{if $m.Embedded}}{{$n}}{{else}}{{$m.Name}}() {{$n}}{{if $m.Type.Elem}}{{else}}
	Set{{$m.Name}}({{$n}}){{end}}{{end}}{{end}}
}

{{end}}
{{define "Func"}}{{$s := .Underlying.Signature}}var {{Name .}} func({{range $index,$elem := $s.Parameters}}{{if $index}}, {{end}}{{Raw $elem}}{{end}}) {{if $s.Results|len |gt 1}}({{end}}{{range $index,$elem := $s.Results}}{{if $index}}, {{end}}{{Raw .}}{{end}}{{if $s.Results|len |gt 1}}){{end}} = {{Raw .}}

{{end}}
{{define "Var"}}{{$t := .Underlying}}var {{Name .}} {{Raw $t}} = {{Raw .}}

{{end}}
{{range $t := .}}{{if eq $t.Kind "Struct"}}{{template "Struct" $t}}{{end}}{{end}}
{{range $t := .}}{{if eq $t.Kind "DeclarationOf"}}{{if eq $t.Underlying.Kind "Func"}}{{template "Func" $t}}{{end}}{{end}}{{end}}
{{range $t := .}}{{if eq $t.Kind "DeclarationOf"}}{{if ne $t.Underlying.Kind "Func"}}{{template "Var" $t}}{{end}}{{end}}{{end}}`

	var expect = `
package o



type CommonObject interface { 
	ID() Int64
	SetID(Int64)
}

type FooBlah interface { 
	CommonObject
	Count() Int64
	SetCount(Int64)
	Frobbers() MapStringToPointerFooFrobber
	Baz() SliceFooObject
	Nickname() PointerString
	NumberIsAFavorite() MapIntToBool
}

type FooFrobber interface { 
	Name() String
	SetName(String)
	Amount() Int64
	SetAmount(Int64)
}

type FooObject interface { 
	CommonObject
}


var FooAFunc func(proto.Object, proto.Object) proto.Frobber = proto.AFunc


var FooAVar proto.Frobber = proto.AVar

var FooAnotherVar proto.Frobber = proto.AnotherVar

`
	testNamer := namer.NewPublicNamer(1, "proto")
	rawNamer := namer.NewRawNamer("o", nil)
	_, u, o := construct(t, testFiles, testNamer)
	t.Logf("\n%v\n\n", o)
	args := map[string]interface{}{
		"Name": testNamer.Name,
		"Raw":  rawNamer.Name,
	}
	tmpl := template.Must(
		template.New("").
			Funcs(args).
			Parse(tmplText),
	)
	buf := &bytes.Buffer{}
	tmpl.Execute(buf, o)
	if e, a := expect, buf.String(); e != a {
		t.Errorf("Wanted, got:\n%v\n-----\n%v\n", e, a)
	}
	if p := u.Package("base/foo/proto"); !p.HasImport("base/common/proto") {
		t.Errorf("Unexpected lack of import line: %s", p.Imports)
	}
}

func TestStructParse(t *testing.T) {
	var structTest = file{
		path: "base/foo/proto/foo.go",
		contents: `
            package foo

            // Blah is a test.
            // A test, I tell you.
            type Blah struct {
	            // A is the first field.
	            A int64 ` + "`" + `json:"a"` + "`" + `
            
	            // B is the second field.
	            // Multiline comments work.
	            B string ` + "`" + `json:"b"` + "`" + `
            }
            `,
	}

	_, u, o := construct(t, []file{structTest}, namer.NewPublicNamer(0))
	t.Logf("%#v", o)
	blahT := u.Type(types.Name{Package: "base/foo/proto", Name: "Blah"})
	if blahT == nil {
		t.Fatal("type not found")
	}
	if e, a := types.Struct, blahT.Kind; e != a {
		t.Errorf("struct kind wrong, wanted %v, got %v", e, a)
	}
	if e, a := []string{"Blah is a test.", "A test, I tell you."}, blahT.CommentLines; !reflect.DeepEqual(e, a) {
		t.Errorf("struct comment wrong, wanted %q, got %q", e, a)
	}
	m := types.Member{
		Name:         "B",
		Embedded:     false,
		CommentLines: []string{"B is the second field.", "Multiline comments work."},
		Tags:         `json:"b"`,
		Type:         types.String,
	}
	if e, a := m, blahT.Members[1]; !reflect.DeepEqual(e, a) {
		t.Errorf("wanted, got:\n%#v\n%#v", e, a)
	}
}

func TestParseSecondClosestCommentLines(t *testing.T) {
	const fileName = "base/foo/proto/foo.go"
	testCases := []struct {
		testFile file
		expected []string
	}{
		{
			testFile: file{
				path: fileName, contents: `
				    package foo
                    // Blah's SecondClosestCommentLines.
                    // Another line.

                    // Blah is a test.
                    // A test, I tell you.
                    type Blah struct {
	                    a int
                    }
                    `},
			expected: []string{"Blah's SecondClosestCommentLines.", "Another line."},
		},
		{
			testFile: file{
				path: fileName, contents: `
				    package foo
                    // Blah's SecondClosestCommentLines.
                    // Another line.
                    
                    type Blah struct {
	                    a int
                    }
                    `},
			expected: []string{"Blah's SecondClosestCommentLines.", "Another line."},
		},
	}
	for _, test := range testCases {
		_, u, o := construct(t, []file{test.testFile}, namer.NewPublicNamer(0))
		t.Logf("%#v", o)
		blahT := u.Type(types.Name{Package: "base/foo/proto", Name: "Blah"})
		if e, a := test.expected, blahT.SecondClosestCommentLines; !reflect.DeepEqual(e, a) {
			t.Errorf("struct second closest comment wrong, wanted %q, got %q", e, a)
		}
	}
}

func TestTypeKindParse(t *testing.T) {
	var testFiles = []file{
		{path: "a/foo.go", contents: "package a\ntype Test string\n"},
		{path: "b/foo.go", contents: "package b\ntype Test map[int]string\n"},
		{path: "c/foo.go", contents: "package c\ntype Test []string\n"},
		{path: "d/foo.go", contents: "package d\ntype Test struct{a int; b struct{a int}; c map[int]string; d *string}\n"},
		{path: "e/foo.go", contents: "package e\ntype Test *string\n"},
		{path: "f/foo.go", contents: `
            package f
            import (
	            "a"
	            "b"
            )
            type Test []a.Test
            type Test2 *a.Test
            type Test3 map[a.Test]b.Test
            type Test4 struct {
	            a struct {a a.Test; b b.Test}
	            b map[a.Test]b.Test
	            c *a.Test
	            d []a.Test
	            e []string
            }
            `},
		{path: "g/foo.go", contents: `
            package g
            type Test func(a, b string) (c, d string)
            func (t Test) Method(a, b string) (c, d string) { return t(a, b) }
            type Interface interface{Method(a, b string) (c, d string)}
            `},
	}

	// Check that the right types are found, and the namers give the expected names.

	assertions := []struct {
		Package, Name string
		k             types.Kind
		names         []string
	}{
		{
			Package: "a", Name: "Test", k: types.Alias,
			names: []string{"Test", "ATest", "test", "aTest", "a.Test"},
		},
		{
			Package: "b", Name: "Test", k: types.Map,
			names: []string{"Test", "BTest", "test", "bTest", "b.Test"},
		},
		{
			Package: "c", Name: "Test", k: types.Slice,
			names: []string{"Test", "CTest", "test", "cTest", "c.Test"},
		},
		{
			Package: "d", Name: "Test", k: types.Struct,
			names: []string{"Test", "DTest", "test", "dTest", "d.Test"},
		},
		{
			Package: "e", Name: "Test", k: types.Pointer,
			names: []string{"Test", "ETest", "test", "eTest", "e.Test"},
		},
		{
			Package: "f", Name: "Test", k: types.Slice,
			names: []string{"Test", "FTest", "test", "fTest", "f.Test"},
		},
		{
			Package: "g", Name: "Test", k: types.Func,
			names: []string{"Test", "GTest", "test", "gTest", "g.Test"},
		},
		{
			Package: "g", Name: "Interface", k: types.Interface,
			names: []string{"Interface", "GInterface", "interface", "gInterface", "g.Interface"},
		},
		{
			Package: "", Name: "string", k: types.Builtin,
			names: []string{"String", "String", "string", "string", "string"},
		},
		{
			Package: "", Name: "int", k: types.Builtin,
			names: []string{"Int", "Int", "int", "int", "int"},
		},
		{
			Package: "", Name: "struct{a int}", k: types.Struct,
			names: []string{"StructInt", "StructInt", "structInt", "structInt", "struct{a int}"},
		},
		{
			Package: "", Name: "struct{a a.Test; b b.Test}", k: types.Struct,
			names: []string{"StructTestTest", "StructATestBTest", "structTestTest", "structATestBTest", "struct{a a.Test; b b.Test}"},
		},
		{
			Package: "", Name: "map[int]string", k: types.Map,
			names: []string{"MapIntToString", "MapIntToString", "mapIntToString", "mapIntToString", "map[int]string"},
		},
		{
			Package: "", Name: "map[a.Test]b.Test", k: types.Map,
			names: []string{"MapTestToTest", "MapATestToBTest", "mapTestToTest", "mapATestToBTest", "map[a.Test]b.Test"},
		},
		{
			Package: "", Name: "[]string", k: types.Slice,
			names: []string{"SliceString", "SliceString", "sliceString", "sliceString", "[]string"},
		},
		{
			Package: "", Name: "[]a.Test", k: types.Slice,
			names: []string{"SliceTest", "SliceATest", "sliceTest", "sliceATest", "[]a.Test"},
		},
		{
			Package: "", Name: "*string", k: types.Pointer,
			names: []string{"PointerString", "PointerString", "pointerString", "pointerString", "*string"},
		},
		{
			Package: "", Name: "*a.Test", k: types.Pointer,
			names: []string{"PointerTest", "PointerATest", "pointerTest", "pointerATest", "*a.Test"},
		},
	}

	namers := []namer.Namer{
		namer.NewPublicNamer(0),
		namer.NewPublicNamer(1),
		namer.NewPrivateNamer(0),
		namer.NewPrivateNamer(1),
		namer.NewRawNamer("", nil),
	}

	for nameIndex, namer := range namers {
		_, u, _ := construct(t, testFiles, namer)
		t.Logf("Found types:\n")
		for pkgName, pkg := range u {
			for typeName, cur := range pkg.Types {
				t.Logf("%q-%q: %s %s", pkgName, typeName, cur.Name, cur.Kind)
			}
		}
		t.Logf("\n\n")

		for _, item := range assertions {
			n := types.Name{Package: item.Package, Name: item.Name}
			thisType := u.Type(n)
			if thisType == nil {
				t.Errorf("type %s not found", n)
				continue
			}
			underlyingType := thisType
			if item.k != types.Alias && thisType.Kind == types.Alias {
				underlyingType = thisType.Underlying
				if underlyingType == nil {
					t.Errorf("underlying type %s not found", n)
					continue
				}
			}
			if e, a := item.k, underlyingType.Kind; e != a {
				t.Errorf("%v-%s: type kind wrong, wanted %v, got %v (%#v)", nameIndex, n, e, a, underlyingType)
			}
			if e, a := item.names[nameIndex], namer.Name(thisType); e != a {
				t.Errorf("%v-%s: Expected %q, got %q", nameIndex, n, e, a)
			}
		}

		// Also do some one-off checks
		gtest := u.Type(types.Name{Package: "g", Name: "Test"})
		if e, a := 1, len(gtest.Methods); e != a {
			t.Errorf("expected %v but found %v methods: %#v", e, a, gtest)
		}
		iface := u.Type(types.Name{Package: "g", Name: "Interface"})
		if e, a := 1, len(iface.Methods); e != a {
			t.Errorf("expected %v but found %v methods: %#v", e, a, iface)
		}
	}
}
