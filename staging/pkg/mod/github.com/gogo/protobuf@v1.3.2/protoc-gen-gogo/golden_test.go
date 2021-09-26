package main

import (
	"bytes"
	"flag"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// Set --regenerate to regenerate the golden files.
var regenerate = flag.Bool("regenerate", false, "regenerate golden files")

// When the environment variable RUN_AS_PROTOC_GEN_GO is set, we skip running
// tests and instead act as protoc-gen-gogo. This allows the test binary to
// pass itself to protoc.
func init() {
	if os.Getenv("RUN_AS_PROTOC_GEN_GO") != "" {
		main()
		os.Exit(0)
	}
}

func TestGolden(t *testing.T) {
	workdir, err := ioutil.TempDir("", "proto-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(workdir)

	// Find all the proto files we need to compile. We assume that each directory
	// contains the files for a single package.
	packages := map[string][]string{}
	err = filepath.Walk("testdata", func(path string, info os.FileInfo, err error) error {
		if !strings.HasSuffix(path, ".proto") {
			return nil
		}
		dir := filepath.Dir(path)
		packages[dir] = append(packages[dir], path)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	// Compile each package, using this binary as protoc-gen-gogo.
	for _, sources := range packages {
		args := []string{"-Itestdata", "--gogo_out=plugins=grpc,paths=source_relative:" + workdir}
		args = append(args, sources...)
		protoc(t, args)
	}

	// Compare each generated file to the golden version.
	filepath.Walk(workdir, func(genPath string, info os.FileInfo, _ error) error {
		if info.IsDir() {
			return nil
		}

		// For each generated file, figure out the path to the corresponding
		// golden file in the testdata directory.
		relPath, rerr := filepath.Rel(workdir, genPath)
		if rerr != nil {
			t.Errorf("filepath.Rel(%q, %q): %v", workdir, genPath, rerr)
			return nil
		}
		if filepath.SplitList(relPath)[0] == ".." {
			t.Errorf("generated file %q is not relative to %q", genPath, workdir)
		}
		goldenPath := filepath.Join("testdata", relPath)

		got, gerr := ioutil.ReadFile(genPath)
		if gerr != nil {
			t.Error(gerr)
			return nil
		}
		if *regenerate {
			// If --regenerate set, just rewrite the golden files.
			err := ioutil.WriteFile(goldenPath, got, 0666)
			if err != nil {
				t.Error(err)
			}
			return nil
		}

		want, err := ioutil.ReadFile(goldenPath)
		if err != nil {
			t.Error(err)
			return nil
		}

		want = fdescRE.ReplaceAll(want, nil)
		got = fdescRE.ReplaceAll(got, nil)
		if bytes.Equal(got, want) {
			return nil
		}

		cmd := exec.Command("diff", "-u", goldenPath, genPath)
		out, _ := cmd.CombinedOutput()
		t.Errorf("golden file differs: %v\n%v", relPath, string(out))
		return nil
	})
}

var fdescRE = regexp.MustCompile(`(?ms)^var fileDescriptor.*}`)

// Source files used by TestParameters.
const (
	aProto = `
syntax = "proto3";
package test.alpha;
option go_package = "package/alpha";
import "beta/b.proto";
message M { test.beta.M field = 1; }`

	bProto = `
syntax = "proto3";
package test.beta;
// no go_package option
message M {}`
)

func TestParameters(t *testing.T) {
	for _, test := range []struct {
		parameters   string
		wantFiles    map[string]bool
		wantImportsA map[string]bool
		wantPackageA string
		wantPackageB string
	}{{
		parameters: "",
		wantFiles: map[string]bool{
			"package/alpha/a.pb.go": true,
			"beta/b.pb.go":          true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
		wantImportsA: map[string]bool{
			"github.com/gogo/protobuf/proto": true,
			"beta":                           true,
		},
	}, {
		parameters: "import_prefix=prefix",
		wantFiles: map[string]bool{
			"package/alpha/a.pb.go": true,
			"beta/b.pb.go":          true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
		wantImportsA: map[string]bool{
			// This really doesn't seem like useful behavior.
			"prefixgithub.com/gogo/protobuf/proto": true,
			"prefixbeta":                           true,
		},
	}, {
		// import_path only affects the 'package' line.
		parameters:   "import_path=import/path/of/pkg",
		wantPackageA: "alpha",
		wantPackageB: "pkg",
		wantFiles: map[string]bool{
			"package/alpha/a.pb.go": true,
			"beta/b.pb.go":          true,
		},
	}, {
		parameters: "Mbeta/b.proto=package/gamma",
		wantFiles: map[string]bool{
			"package/alpha/a.pb.go": true,
			"beta/b.pb.go":          true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
		wantImportsA: map[string]bool{
			"github.com/gogo/protobuf/proto": true,
			// Rewritten by the M parameter.
			"package/gamma": true,
		},
	}, {
		parameters: "import_prefix=prefix,Mbeta/b.proto=package/gamma",
		wantFiles: map[string]bool{
			"package/alpha/a.pb.go": true,
			"beta/b.pb.go":          true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
		wantImportsA: map[string]bool{
			// import_prefix applies after M.
			"prefixpackage/gamma": true,
		},
	}, {
		parameters: "paths=source_relative",
		wantFiles: map[string]bool{
			"alpha/a.pb.go": true,
			"beta/b.pb.go":  true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
	}, {
		parameters: "paths=source_relative,import_prefix=prefix",
		wantFiles: map[string]bool{
			// import_prefix doesn't affect filenames.
			"alpha/a.pb.go": true,
			"beta/b.pb.go":  true,
		},
		wantPackageA: "alpha",
		wantPackageB: "test_beta",
	}} {
		name := test.parameters
		if name == "" {
			name = "defaults"
		}
		// TODO: Switch to t.Run when we no longer support Go 1.6.
		t.Logf("TEST: %v", name)
		workdir, werr := ioutil.TempDir("", "proto-test")
		if werr != nil {
			t.Fatal(werr)
		}
		defer os.RemoveAll(workdir)

		for _, dir := range []string{"alpha", "beta", "out"} {
			if err := os.MkdirAll(filepath.Join(workdir, dir), 0777); err != nil {
				t.Fatal(err)
			}
		}

		if err := ioutil.WriteFile(filepath.Join(workdir, "alpha", "a.proto"), []byte(aProto), 0666); err != nil {
			t.Fatal(err)
		}

		if err := ioutil.WriteFile(filepath.Join(workdir, "beta", "b.proto"), []byte(bProto), 0666); err != nil {
			t.Fatal(err)
		}

		protoc(t, []string{
			"-I" + workdir,
			"--gogo_out=" + test.parameters + ":" + filepath.Join(workdir, "out"),
			filepath.Join(workdir, "alpha", "a.proto"),
		})
		protoc(t, []string{
			"-I" + workdir,
			"--gogo_out=" + test.parameters + ":" + filepath.Join(workdir, "out"),
			filepath.Join(workdir, "beta", "b.proto"),
		})

		contents := make(map[string]string)
		gotFiles := make(map[string]bool)
		outdir := filepath.Join(workdir, "out")
		filepath.Walk(outdir, func(p string, info os.FileInfo, _ error) error {
			if info.IsDir() {
				return nil
			}
			base := filepath.Base(p)
			if base == "a.pb.go" || base == "b.pb.go" {
				b, err := ioutil.ReadFile(p)
				if err != nil {
					t.Fatal(err)
				}
				contents[base] = string(b)
			}
			relPath, _ := filepath.Rel(outdir, p)
			gotFiles[relPath] = true
			return nil
		})
		for got := range gotFiles {
			if runtime.GOOS == "windows" {
				got = filepath.ToSlash(got)
			}
			if !test.wantFiles[got] {
				t.Skipf("unexpected output file: %v", got)
			}
		}
		for want := range test.wantFiles {
			if runtime.GOOS == "windows" {
				want = filepath.FromSlash(want)
			}
			if !gotFiles[want] {
				t.Skipf("missing output file:    %v", want)
			}
		}
		gotPackageA, gotImports, err := parseFile(contents["a.pb.go"])
		if err != nil {
			t.Fatal(err)
		}
		gotPackageB, _, err := parseFile(contents["b.pb.go"])
		if err != nil {
			t.Fatal(err)
		}
		if got, want := gotPackageA, test.wantPackageA; want != got {
			t.Errorf("output file a.pb.go is package %q, want %q", got, want)
		}
		if got, want := gotPackageB, test.wantPackageB; want != got {
			t.Errorf("output file b.pb.go is package %q, want %q", got, want)
		}
		missingImport := false
	WantImport:
		for want := range test.wantImportsA {
			for _, imp := range gotImports {
				if `"`+want+`"` == imp {
					continue WantImport
				}
			}
			t.Errorf("output file a.pb.go does not contain expected import %q", want)
			missingImport = true
		}
		if missingImport {
			t.Error("got imports:")
			for _, imp := range gotImports {
				t.Errorf("  %v", imp)
			}
		}
	}
}

// parseFile returns a file's package name and a list of all packages it imports.
func parseFile(source string) (packageName string, imports []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "<source>", source, parser.ImportsOnly)
	if err != nil {
		return "", nil, err
	}
	for _, imp := range f.Imports {
		imports = append(imports, imp.Path.Value)
	}
	return f.Name.Name, imports, nil
}

func protoc(t *testing.T, args []string) {
	cmd := exec.Command("protoc-min-version", "--version=3.0.0")
	cmd.Args = append(cmd.Args, args...)
	// We set the RUN_AS_PROTOC_GEN_GO environment variable to indicate that
	// the subprocess should act as a proto compiler rather than a test.
	cmd.Env = append(os.Environ(), "RUN_AS_PROTOC_GEN_GO=1")
	out, err := cmd.CombinedOutput()
	if len(out) > 0 || err != nil {
		t.Log("RUNNING: ", strings.Join(cmd.Args, " "))
	}
	if len(out) > 0 {
		t.Log(string(out))
	}
	if err != nil {
		t.Fatalf("protoc: %v", err)
	}
}
