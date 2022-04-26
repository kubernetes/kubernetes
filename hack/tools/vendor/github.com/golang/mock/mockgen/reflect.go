// Copyright 2012 Google Inc.
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

package main

// This file contains the model construction by reflection.

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"

	"github.com/golang/mock/mockgen/model"
)

var (
	progOnly   = flag.Bool("prog_only", false, "(reflect mode) Only generate the reflection program; write it to stdout and exit.")
	execOnly   = flag.String("exec_only", "", "(reflect mode) If set, execute this reflection program.")
	buildFlags = flag.String("build_flags", "", "(reflect mode) Additional flags for go build.")
)

// reflectMode generates mocks via reflection on an interface.
func reflectMode(importPath string, symbols []string) (*model.Package, error) {
	if *execOnly != "" {
		return run(*execOnly)
	}

	program, err := writeProgram(importPath, symbols)
	if err != nil {
		return nil, err
	}

	if *progOnly {
		if _, err := os.Stdout.Write(program); err != nil {
			return nil, err
		}
		os.Exit(0)
	}

	wd, _ := os.Getwd()

	// Try to run the reflection program  in the current working directory.
	if p, err := runInDir(program, wd); err == nil {
		return p, nil
	}

	// Try to run the program in the same directory as the input package.
	if p, err := build.Import(importPath, wd, build.FindOnly); err == nil {
		dir := p.Dir
		if p, err := runInDir(program, dir); err == nil {
			return p, nil
		}
	}

	// Try to run it in a standard temp directory.
	return runInDir(program, "")
}

func writeProgram(importPath string, symbols []string) ([]byte, error) {
	var program bytes.Buffer
	data := reflectData{
		ImportPath: importPath,
		Symbols:    symbols,
	}
	if err := reflectProgram.Execute(&program, &data); err != nil {
		return nil, err
	}
	return program.Bytes(), nil
}

// run the given program and parse the output as a model.Package.
func run(program string) (*model.Package, error) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, err
	}

	filename := f.Name()
	defer os.Remove(filename)
	if err := f.Close(); err != nil {
		return nil, err
	}

	// Run the program.
	cmd := exec.Command(program, "-output", filename)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	f, err = os.Open(filename)
	if err != nil {
		return nil, err
	}

	// Process output.
	var pkg model.Package
	if err := gob.NewDecoder(f).Decode(&pkg); err != nil {
		return nil, err
	}

	if err := f.Close(); err != nil {
		return nil, err
	}

	return &pkg, nil
}

// runInDir writes the given program into the given dir, runs it there, and
// parses the output as a model.Package.
func runInDir(program []byte, dir string) (*model.Package, error) {
	// We use TempDir instead of TempFile so we can control the filename.
	tmpDir, err := ioutil.TempDir(dir, "gomock_reflect_")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			log.Printf("failed to remove temp directory: %s", err)
		}
	}()
	const progSource = "prog.go"
	var progBinary = "prog.bin"
	if runtime.GOOS == "windows" {
		// Windows won't execute a program unless it has a ".exe" suffix.
		progBinary += ".exe"
	}

	if err := ioutil.WriteFile(filepath.Join(tmpDir, progSource), program, 0600); err != nil {
		return nil, err
	}

	cmdArgs := []string{}
	cmdArgs = append(cmdArgs, "build")
	if *buildFlags != "" {
		cmdArgs = append(cmdArgs, strings.Split(*buildFlags, " ")...)
	}
	cmdArgs = append(cmdArgs, "-o", progBinary, progSource)

	// Build the program.
	buf := bytes.NewBuffer(nil)
	cmd := exec.Command("go", cmdArgs...)
	cmd.Dir = tmpDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, buf)
	if err := cmd.Run(); err != nil {
		sErr := buf.String()
		if strings.Contains(sErr, `cannot find package "."`) &&
			strings.Contains(sErr, "github.com/golang/mock/mockgen/model") {
			fmt.Fprint(os.Stderr, "Please reference the steps in the README to fix this error:\n\thttps://github.com/golang/mock#reflect-vendoring-error.")
			return nil, err
		}
		return nil, err
	}

	return run(filepath.Join(tmpDir, progBinary))
}

type reflectData struct {
	ImportPath string
	Symbols    []string
}

// This program reflects on an interface value, and prints the
// gob encoding of a model.Package to standard output.
// JSON doesn't work because of the model.Type interface.
var reflectProgram = template.Must(template.New("program").Parse(`
package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"os"
	"path"
	"reflect"

	"github.com/golang/mock/mockgen/model"

	pkg_ {{printf "%q" .ImportPath}}
)

var output = flag.String("output", "", "The output file name, or empty to use stdout.")

func main() {
	flag.Parse()

	its := []struct{
		sym string
		typ reflect.Type
	}{
		{{range .Symbols}}
		{ {{printf "%q" .}}, reflect.TypeOf((*pkg_.{{.}})(nil)).Elem()},
		{{end}}
	}
	pkg := &model.Package{
		// NOTE: This behaves contrary to documented behaviour if the
		// package name is not the final component of the import path.
		// The reflect package doesn't expose the package name, though.
		Name: path.Base({{printf "%q" .ImportPath}}),
	}

	for _, it := range its {
		intf, err := model.InterfaceFromInterfaceType(it.typ)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Reflection: %v\n", err)
			os.Exit(1)
		}
		intf.Name = it.sym
		pkg.Interfaces = append(pkg.Interfaces, intf)
	}

	outfile := os.Stdout
	if len(*output) != 0 {
		var err error
		outfile, err = os.Create(*output)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to open output file %q", *output)
		}
		defer func() {
			if err := outfile.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "failed to close output file %q", *output)
				os.Exit(1)
			}
		}()
	}

	if err := gob.NewEncoder(outfile).Encode(pkg); err != nil {
		fmt.Fprintf(os.Stderr, "gob encode: %v\n", err)
		os.Exit(1)
	}
}
`))
