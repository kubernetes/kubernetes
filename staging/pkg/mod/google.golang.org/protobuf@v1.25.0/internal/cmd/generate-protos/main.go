// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run . -execute

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	gengo "google.golang.org/protobuf/cmd/protoc-gen-go/internal_gengo"
	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/internal/detrand"
)

// Override the location of the Go package for various source files.
// TOOD: Commit these changes upstream.
var protoPackages = map[string]string{
	"google/protobuf/any.proto":                  "google.golang.org/protobuf/types/known/anypb;anypb",
	"google/protobuf/api.proto":                  "google.golang.org/protobuf/types/known/apipb;apipb",
	"google/protobuf/duration.proto":             "google.golang.org/protobuf/types/known/durationpb;durationpb",
	"google/protobuf/empty.proto":                "google.golang.org/protobuf/types/known/emptypb;emptypb",
	"google/protobuf/field_mask.proto":           "google.golang.org/protobuf/types/known/fieldmaskpb;fieldmaskpb",
	"google/protobuf/source_context.proto":       "google.golang.org/protobuf/types/known/sourcecontextpb;sourcecontextpb",
	"google/protobuf/struct.proto":               "google.golang.org/protobuf/types/known/structpb;structpb",
	"google/protobuf/timestamp.proto":            "google.golang.org/protobuf/types/known/timestamppb;timestamppb",
	"google/protobuf/type.proto":                 "google.golang.org/protobuf/types/known/typepb;typepb",
	"google/protobuf/wrappers.proto":             "google.golang.org/protobuf/types/known/wrapperspb;wrapperspb",
	"google/protobuf/descriptor.proto":           "google.golang.org/protobuf/types/descriptorpb;descriptorpb",
	"google/protobuf/compiler/plugin.proto":      "google.golang.org/protobuf/types/pluginpb;pluginpb",
	"conformance/conformance.proto":              "google.golang.org/protobuf/internal/testprotos/conformance;conformance",
	"google/protobuf/test_messages_proto2.proto": "google.golang.org/protobuf/internal/testprotos/conformance;conformance",
	"google/protobuf/test_messages_proto3.proto": "google.golang.org/protobuf/internal/testprotos/conformance;conformance",

	"benchmarks.proto": "google.golang.org/protobuf/internal/testprotos/benchmarks;benchmarks",
	"datasets/google_message1/proto2/benchmark_message1_proto2.proto": "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message1/proto2;proto2",
	"datasets/google_message1/proto3/benchmark_message1_proto3.proto": "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message1/proto3;proto3",
	"datasets/google_message2/benchmark_message2.proto":               "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message2;google_message2",
	"datasets/google_message3/benchmark_message3.proto":               "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_1.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_2.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_3.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_4.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_5.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_6.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_7.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message3/benchmark_message3_8.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message3;google_message3",
	"datasets/google_message4/benchmark_message4.proto":               "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message4;google_message4",
	"datasets/google_message4/benchmark_message4_1.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message4;google_message4",
	"datasets/google_message4/benchmark_message4_2.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message4;google_message4",
	"datasets/google_message4/benchmark_message4_3.proto":             "google.golang.org/protobuf/internal/testprotos/benchmarks/datasets/google_message4;google_message4",

	"cmd/protoc-gen-go/testdata/nopackage/nopackage.proto": "google.golang.org/protobuf/cmd/protoc-gen-go/testdata/nopackage",
}

func init() {
	// Determine repository root path.
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").CombinedOutput()
	check(err)
	repoRoot = strings.TrimSpace(string(out))

	// Determine the module path.
	cmd := exec.Command("go", "list", "-m", "-f", "{{.Path}}")
	cmd.Dir = repoRoot
	out, err = cmd.CombinedOutput()
	check(err)
	modulePath = strings.TrimSpace(string(out))

	// When the environment variable RUN_AS_PROTOC_PLUGIN is set,
	// we skip running main and instead act as a protoc plugin.
	// This allows the binary to pass itself to protoc.
	if plugin := os.Getenv("RUN_AS_PROTOC_PLUGIN"); plugin != "" {
		// Disable deliberate output instability for generated files.
		// This is reasonable since we fully control the output.
		detrand.Disable()

		protogen.Options{}.Run(func(gen *protogen.Plugin) error {
			for _, file := range gen.Files {
				if file.Generate {
					gengo.GenerateVersionMarkers = false
					gengo.GenerateFile(gen, file)
					generateIdentifiers(gen, file)
				}
			}
			gen.SupportedFeatures = gengo.SupportedFeatures
			return nil
		})
		os.Exit(0)
	}
}

var (
	run        bool
	protoRoot  string
	repoRoot   string
	modulePath string

	generatedPreamble = []string{
		"// Copyright 2019 The Go Authors. All rights reserved.",
		"// Use of this source code is governed by a BSD-style",
		"// license that can be found in the LICENSE file.",
		"",
		"// Code generated by generate-protos. DO NOT EDIT.",
		"",
	}
)

func main() {
	flag.BoolVar(&run, "execute", false, "Write generated files to destination.")
	flag.StringVar(&protoRoot, "protoroot", os.Getenv("PROTOBUF_ROOT"), "The root of the protobuf source tree.")
	flag.Parse()
	if protoRoot == "" {
		panic("protobuf source root is not set")
	}

	generateLocalProtos()
	generateRemoteProtos()
}

func generateLocalProtos() {
	tmpDir, err := ioutil.TempDir(repoRoot, "tmp")
	check(err)
	defer os.RemoveAll(tmpDir)

	// Generate all local proto files (except version-locked files).
	dirs := []struct {
		path        string
		annotateFor map[string]bool
		exclude     map[string]bool
	}{
		{path: "cmd/protoc-gen-go/testdata", annotateFor: map[string]bool{
			"cmd/protoc-gen-go/testdata/annotations/annotations.proto": true},
		},
		{path: "internal/testprotos", exclude: map[string]bool{
			"internal/testprotos/irregular/irregular.proto": true,
		}},
	}
	excludeRx := regexp.MustCompile(`legacy/.*/`)
	for _, d := range dirs {
		subDirs := map[string]bool{}

		dstDir := tmpDir
		check(os.MkdirAll(dstDir, 0775))

		srcDir := filepath.Join(repoRoot, filepath.FromSlash(d.path))
		filepath.Walk(srcDir, func(srcPath string, _ os.FileInfo, _ error) error {
			if !strings.HasSuffix(srcPath, ".proto") || excludeRx.MatchString(srcPath) {
				return nil
			}
			relPath, err := filepath.Rel(repoRoot, srcPath)
			check(err)

			srcRelPath, err := filepath.Rel(srcDir, srcPath)
			check(err)
			subDirs[filepath.Dir(srcRelPath)] = true

			if d.exclude[filepath.ToSlash(relPath)] {
				return nil
			}

			opts := "module=" + modulePath
			opts += "," + protoMapOpt()

			// Emit a .meta file for certain files.
			if d.annotateFor[filepath.ToSlash(relPath)] {
				opts += ",annotate_code"
			}

			protoc("-I"+filepath.Join(protoRoot, "src"), "-I"+repoRoot, "--go_out="+opts+":"+dstDir, relPath)
			return nil
		})

		// For directories in testdata, generate a test that links in all
		// generated packages to ensure that it builds and initializes properly.
		// This is done because "go build ./..." does not build sub-packages
		// under testdata.
		if filepath.Base(d.path) == "testdata" {
			var imports []string
			for sd := range subDirs {
				imports = append(imports, fmt.Sprintf("_ %q", path.Join(modulePath, d.path, filepath.ToSlash(sd))))
			}
			sort.Strings(imports)

			s := strings.Join(append(generatedPreamble, []string{
				"package main",
				"",
				"import (" + strings.Join(imports, "\n") + ")",
			}...), "\n")
			b, err := format.Source([]byte(s))
			check(err)
			check(ioutil.WriteFile(filepath.Join(tmpDir, filepath.FromSlash(d.path+"/gen_test.go")), b, 0664))
		}
	}

	syncOutput(repoRoot, tmpDir)
}

func generateRemoteProtos() {
	tmpDir, err := ioutil.TempDir(repoRoot, "tmp")
	check(err)
	defer os.RemoveAll(tmpDir)

	// Generate all remote proto files.
	files := []struct{ prefix, path string }{
		{"", "conformance/conformance.proto"},
		{"benchmarks", "benchmarks.proto"},
		{"benchmarks", "datasets/google_message1/proto2/benchmark_message1_proto2.proto"},
		{"benchmarks", "datasets/google_message1/proto3/benchmark_message1_proto3.proto"},
		{"benchmarks", "datasets/google_message2/benchmark_message2.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_1.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_2.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_3.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_4.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_5.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_6.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_7.proto"},
		{"benchmarks", "datasets/google_message3/benchmark_message3_8.proto"},
		{"benchmarks", "datasets/google_message4/benchmark_message4.proto"},
		{"benchmarks", "datasets/google_message4/benchmark_message4_1.proto"},
		{"benchmarks", "datasets/google_message4/benchmark_message4_2.proto"},
		{"benchmarks", "datasets/google_message4/benchmark_message4_3.proto"},
		{"src", "google/protobuf/any.proto"},
		{"src", "google/protobuf/api.proto"},
		{"src", "google/protobuf/compiler/plugin.proto"},
		{"src", "google/protobuf/descriptor.proto"},
		{"src", "google/protobuf/duration.proto"},
		{"src", "google/protobuf/empty.proto"},
		{"src", "google/protobuf/field_mask.proto"},
		{"src", "google/protobuf/source_context.proto"},
		{"src", "google/protobuf/struct.proto"},
		{"src", "google/protobuf/test_messages_proto2.proto"},
		{"src", "google/protobuf/test_messages_proto3.proto"},
		{"src", "google/protobuf/timestamp.proto"},
		{"src", "google/protobuf/type.proto"},
		{"src", "google/protobuf/wrappers.proto"},
	}
	for _, f := range files {
		protoc("-I"+filepath.Join(protoRoot, f.prefix), "--go_out=paths=import,"+protoMapOpt()+":"+tmpDir, f.path)
	}

	syncOutput(repoRoot, filepath.Join(tmpDir, modulePath))

	// Sanity check for unsynchronized files.
	os.RemoveAll(filepath.Join(tmpDir, modulePath))
	check(filepath.Walk(tmpDir, func(path string, fi os.FileInfo, err error) error {
		if !fi.IsDir() {
			return fmt.Errorf("unsynchronized generated file: %v", strings.TrimPrefix(path, tmpDir))
		}
		return err
	}))
}

func protoc(args ...string) {
	// TODO: Remove --experimental_allow_proto3_optional flag.
	cmd := exec.Command("protoc", "--plugin=protoc-gen-go="+os.Args[0], "--experimental_allow_proto3_optional")
	cmd.Args = append(cmd.Args, args...)
	cmd.Env = append(os.Environ(), "RUN_AS_PROTOC_PLUGIN=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("executing: %v\n%s\n", strings.Join(cmd.Args, " "), out)
	}
	check(err)
}

// generateIdentifiers generates an internal package for descriptor.proto
// and well-known types.
func generateIdentifiers(gen *protogen.Plugin, file *protogen.File) {
	if file.Desc.Package() != "google.protobuf" {
		return
	}

	importPath := modulePath + "/internal/genid"
	base := strings.TrimSuffix(path.Base(file.Desc.Path()), ".proto")
	g := gen.NewGeneratedFile(importPath+"/"+base+"_gen.go", protogen.GoImportPath(importPath))
	for _, s := range generatedPreamble {
		g.P(s)
	}
	g.P("package ", path.Base(importPath))
	g.P()

	g.P("const ", file.GoDescriptorIdent.GoName, " = ", strconv.Quote(file.Desc.Path()))
	g.P()

	var processEnums func([]*protogen.Enum)
	var processMessages func([]*protogen.Message)
	const protoreflectPackage = protogen.GoImportPath("google.golang.org/protobuf/reflect/protoreflect")
	processEnums = func(enums []*protogen.Enum) {
		for _, enum := range enums {
			g.P("// Full and short names for ", enum.Desc.FullName(), ".")
			g.P("const (")
			g.P(enum.GoIdent.GoName, "_enum_fullname = ", strconv.Quote(string(enum.Desc.FullName())))
			g.P(enum.GoIdent.GoName, "_enum_name = ", strconv.Quote(string(enum.Desc.Name())))
			g.P(")")
			g.P()
		}
	}
	processMessages = func(messages []*protogen.Message) {
		for _, message := range messages {
			g.P("// Names for ", message.Desc.FullName(), ".")
			g.P("const (")
			g.P(message.GoIdent.GoName, "_message_name ", protoreflectPackage.Ident("Name"), " = ", strconv.Quote(string(message.Desc.Name())))
			g.P(message.GoIdent.GoName, "_message_fullname ", protoreflectPackage.Ident("FullName"), " = ", strconv.Quote(string(message.Desc.FullName())))
			g.P(")")
			g.P()

			if len(message.Fields) > 0 {
				g.P("// Field names for ", message.Desc.FullName(), ".")
				g.P("const (")
				for _, field := range message.Fields {
					g.P(message.GoIdent.GoName, "_", field.GoName, "_field_name ", protoreflectPackage.Ident("Name"), " = ", strconv.Quote(string(field.Desc.Name())))
				}
				g.P()
				for _, field := range message.Fields {
					g.P(message.GoIdent.GoName, "_", field.GoName, "_field_fullname ", protoreflectPackage.Ident("FullName"), " = ", strconv.Quote(string(field.Desc.FullName())))
				}
				g.P(")")
				g.P()

				g.P("// Field numbers for ", message.Desc.FullName(), ".")
				g.P("const (")
				for _, field := range message.Fields {
					g.P(message.GoIdent.GoName, "_", field.GoName, "_field_number ", protoreflectPackage.Ident("FieldNumber"), " = ", field.Desc.Number())
				}
				g.P(")")
				g.P()
			}

			if len(message.Oneofs) > 0 {
				g.P("// Oneof names for ", message.Desc.FullName(), ".")
				g.P("const (")
				for _, oneof := range message.Oneofs {
					g.P(message.GoIdent.GoName, "_", oneof.GoName, "_oneof_name ", protoreflectPackage.Ident("Name"), " = ", strconv.Quote(string(oneof.Desc.Name())))
				}
				g.P()
				for _, oneof := range message.Oneofs {
					g.P(message.GoIdent.GoName, "_", oneof.GoName, "_oneof_fullname ", protoreflectPackage.Ident("FullName"), " = ", strconv.Quote(string(oneof.Desc.FullName())))
				}
				g.P(")")
				g.P()
			}

			processEnums(message.Enums)
			processMessages(message.Messages)
		}
	}
	processEnums(file.Enums)
	processMessages(file.Messages)
}

func syncOutput(dstDir, srcDir string) {
	filepath.Walk(srcDir, func(srcPath string, _ os.FileInfo, _ error) error {
		if !strings.HasSuffix(srcPath, ".go") && !strings.HasSuffix(srcPath, ".meta") {
			return nil
		}
		relPath, err := filepath.Rel(srcDir, srcPath)
		check(err)
		dstPath := filepath.Join(dstDir, relPath)

		if run {
			if copyFile(dstPath, srcPath) {
				fmt.Println("#", relPath)
			}
		} else {
			cmd := exec.Command("diff", dstPath, srcPath, "-N", "-u")
			cmd.Stdout = os.Stdout
			cmd.Run()
		}
		return nil
	})
}

func copyFile(dstPath, srcPath string) (changed bool) {
	src, err := ioutil.ReadFile(srcPath)
	check(err)
	check(os.MkdirAll(filepath.Dir(dstPath), 0775))
	dst, _ := ioutil.ReadFile(dstPath)
	if bytes.Equal(src, dst) {
		return false
	}
	check(ioutil.WriteFile(dstPath, src, 0664))
	return true
}

func protoMapOpt() string {
	var opts []string
	for k, v := range protoPackages {
		opts = append(opts, fmt.Sprintf("M%v=%v", k, v))
	}
	return strings.Join(opts, ",")
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
