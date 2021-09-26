// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protogen

import (
	"flag"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

func init() {
	warnings = false // avoid spam in tests
}

func TestPluginParameters(t *testing.T) {
	var flags flag.FlagSet
	value := flags.Int("integer", 0, "")
	const params = "integer=2"
	_, err := Options{
		ParamFunc: flags.Set,
	}.New(&pluginpb.CodeGeneratorRequest{
		Parameter: proto.String(params),
	})
	if err != nil {
		t.Errorf("New(generator parameters %q): %v", params, err)
	}
	if *value != 2 {
		t.Errorf("New(generator parameters %q): integer=%v, want 2", params, *value)
	}
}

func TestPluginParameterErrors(t *testing.T) {
	for _, parameter := range []string{
		"unknown=1",
		"boolean=error",
	} {
		var flags flag.FlagSet
		flags.Bool("boolean", false, "")
		_, err := Options{
			ParamFunc: flags.Set,
		}.New(&pluginpb.CodeGeneratorRequest{
			Parameter: proto.String(parameter),
		})
		if err == nil {
			t.Errorf("New(generator parameters %q): want error, got nil", parameter)
		}
	}
}

func TestNoGoPackage(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("testdata/go_package/no_go_package.proto"),
				Syntax:  proto.String(protoreflect.Proto3.String()),
				Package: proto.String("goproto.testdata"),
			},
			{
				Name:       proto.String("testdata/go_package/no_go_package_import.proto"),
				Syntax:     proto.String(protoreflect.Proto3.String()),
				Package:    proto.String("goproto.testdata"),
				Dependency: []string{"testdata/go_package/no_go_package.proto"},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	for i, f := range gen.Files {
		if got, want := string(f.GoPackageName), "goproto_testdata"; got != want {
			t.Errorf("gen.Files[%d].GoPackageName = %v, want %v", i, got, want)
		}
		if got, want := string(f.GoImportPath), "testdata/go_package"; got != want {
			t.Errorf("gen.Files[%d].GoImportPath = %v, want %v", i, got, want)
		}
	}
}

func TestPackageNamesAndPaths(t *testing.T) {
	const (
		filename         = "dir/filename.proto"
		protoPackageName = "proto.package"
	)
	for _, test := range []struct {
		desc            string
		parameter       string
		goPackageOption string
		generate        bool
		wantPackageName GoPackageName
		wantImportPath  GoImportPath
		wantFilename    string
	}{
		{
			desc:            "no parameters, no go_package option",
			generate:        true,
			wantPackageName: "proto_package",
			wantImportPath:  "dir",
			wantFilename:    "dir/filename",
		},
		{
			desc:            "go_package option sets import path",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "go_package option sets import path and package",
			goPackageOption: "golang.org/x/foo;bar",
			generate:        true,
			wantPackageName: "bar",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "go_package option sets package",
			goPackageOption: "foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "dir",
			wantFilename:    "dir/filename",
		},
		{
			desc:            "command line sets import path for a file",
			parameter:       "Mdir/filename.proto=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "command line sets import path for a file with package name specified",
			parameter:       "Mdir/filename.proto=golang.org/x/bar;bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "bar",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "import_path parameter sets import path of generated files",
			parameter:       "import_path=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "import_path parameter does not set import path of dependencies",
			parameter:       "import_path=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        false,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "module option set",
			parameter:       "module=golang.org/x",
			goPackageOption: "golang.org/x/foo",
			generate:        false,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "foo/filename",
		},
		{
			desc:            "paths=import uses import path from command line",
			parameter:       "paths=import,Mdir/filename.proto=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/bar/filename",
		},
		{
			desc:            "module option implies paths=import",
			parameter:       "module=golang.org/x,Mdir/filename.proto=golang.org/x/foo",
			generate:        false,
			wantPackageName: "proto_package",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "foo/filename",
		},
	} {
		context := fmt.Sprintf(`
TEST: %v
  --go_out=%v:.
  file %q: generate=%v
  option go_package = %q;

  `,
			test.desc, test.parameter, filename, test.generate, test.goPackageOption)

		req := &pluginpb.CodeGeneratorRequest{
			Parameter: proto.String(test.parameter),
			ProtoFile: []*descriptorpb.FileDescriptorProto{
				{
					Name:    proto.String(filename),
					Package: proto.String(protoPackageName),
					Options: &descriptorpb.FileOptions{
						GoPackage: proto.String(test.goPackageOption),
					},
				},
			},
		}
		if test.generate {
			req.FileToGenerate = []string{filename}
		}
		gen, err := Options{}.New(req)
		if err != nil {
			t.Errorf("%vNew(req) = %v", context, err)
			continue
		}
		gotFile, ok := gen.FilesByPath[filename]
		if !ok {
			t.Errorf("%v%v: missing file info", context, filename)
			continue
		}
		if got, want := gotFile.GoPackageName, test.wantPackageName; got != want {
			t.Errorf("%vGoPackageName=%v, want %v", context, got, want)
		}
		if got, want := gotFile.GoImportPath, test.wantImportPath; got != want {
			t.Errorf("%vGoImportPath=%v, want %v", context, got, want)
		}
		gen.NewGeneratedFile(gotFile.GeneratedFilenamePrefix, "")
		resp := gen.Response()
		if got, want := resp.File[0].GetName(), test.wantFilename; got != want {
			t.Errorf("%vgenerated filename=%v, want %v", context, got, want)
		}
	}
}

func TestPackageNameInference(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("dir/file1.proto"),
				Package: proto.String("proto.package"),
			},
			{
				Name:    proto.String("dir/file2.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("foo"),
				},
			},
		},
		FileToGenerate: []string{"dir/file1.proto", "dir/file2.proto"},
	})
	if err != nil {
		t.Fatalf("New(req) = %v", err)
	}
	if f1, ok := gen.FilesByPath["dir/file1.proto"]; !ok {
		t.Errorf("missing file info for dir/file1.proto")
	} else if f1.GoPackageName != "foo" {
		t.Errorf("dir/file1.proto: GoPackageName=%v, want foo; package name should be derived from dir/file2.proto", f1.GoPackageName)
	}
}

func TestInconsistentPackageNames(t *testing.T) {
	_, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("dir/file1.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("golang.org/x/foo"),
				},
			},
			{
				Name:    proto.String("dir/file2.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("golang.org/x/foo;bar"),
				},
			},
		},
		FileToGenerate: []string{"dir/file1.proto", "dir/file2.proto"},
	})
	if err == nil {
		t.Fatalf("inconsistent package names for the same import path: New(req) = nil, want error")
	}
}

func TestImports(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{})
	if err != nil {
		t.Fatal(err)
	}
	g := gen.NewGeneratedFile("foo.go", "golang.org/x/foo")
	g.P("package foo")
	g.P()
	for _, importPath := range []GoImportPath{
		"golang.org/x/foo",
		// Multiple references to the same package.
		"golang.org/x/bar",
		"golang.org/x/bar",
		// Reference to a different package with the same basename.
		"golang.org/y/bar",
		"golang.org/x/baz",
		// Reference to a package conflicting with a predeclared identifier.
		"golang.org/z/string",
	} {
		g.P("var _ = ", GoIdent{GoName: "X", GoImportPath: importPath}, " // ", importPath)
	}
	want := `package foo

import (
	bar "golang.org/x/bar"
	baz "golang.org/x/baz"
	bar1 "golang.org/y/bar"
	string1 "golang.org/z/string"
)

var _ = X         // "golang.org/x/foo"
var _ = bar.X     // "golang.org/x/bar"
var _ = bar.X     // "golang.org/x/bar"
var _ = bar1.X    // "golang.org/y/bar"
var _ = baz.X     // "golang.org/x/baz"
var _ = string1.X // "golang.org/z/string"
`
	got, err := g.Content()
	if err != nil {
		t.Fatalf("g.Content() = %v", err)
	}
	if diff := cmp.Diff(string(want), string(got)); diff != "" {
		t.Fatalf("content mismatch (-want +got):\n%s", diff)
	}
}

func TestImportRewrites(t *testing.T) {
	gen, err := Options{
		ImportRewriteFunc: func(i GoImportPath) GoImportPath {
			return "prefix/" + i
		},
	}.New(&pluginpb.CodeGeneratorRequest{})
	if err != nil {
		t.Fatal(err)
	}
	g := gen.NewGeneratedFile("foo.go", "golang.org/x/foo")
	g.P("package foo")
	g.P("var _ = ", GoIdent{GoName: "X", GoImportPath: "golang.org/x/bar"})
	want := `package foo

import (
	bar "prefix/golang.org/x/bar"
)

var _ = bar.X
`
	got, err := g.Content()
	if err != nil {
		t.Fatalf("g.Content() = %v", err)
	}
	if diff := cmp.Diff(string(want), string(got)); diff != "" {
		t.Fatalf("content mismatch (-want +got):\n%s", diff)
	}
}
