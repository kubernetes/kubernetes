// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

// codecgen generates codec.Selfer implementations for a set of types.
package main

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"text/template"
	"time"
)

const genFrunMainTmpl = `//+build ignore

package main
{{ if .Types }}import "{{ .ImportPath }}"{{ end }}
func main() {
	{{ $.PackageName }}.CodecGenTempWrite{{ .RandString }}()
}
`

// const genFrunPkgTmpl = `//+build codecgen
const genFrunPkgTmpl = `
package {{ $.PackageName }}

import (
	{{ if not .CodecPkgFiles }}{{ .CodecPkgName }} "{{ .CodecImportPath }}"{{ end }}
{{/*
	{{ if .Types }}"{{ .ImportPath }}"{{ end }}
	"io"
*/}}
	"os"
	"reflect"
	"bytes"
	"go/format"
)

{{/* This is not used anymore. Remove it.
func write(w io.Writer, s string) {
	if _, err := io.WriteString(w, s); err != nil {
		panic(err)
	}
}
*/}}

func CodecGenTempWrite{{ .RandString }}() {
	fout, err := os.Create("{{ .OutFile }}")
	if err != nil {
		panic(err)
	}
	defer fout.Close()
	var out bytes.Buffer
	
	var typs []reflect.Type 
{{ range $index, $element := .Types }}
	var t{{ $index }} {{ . }}
	typs = append(typs, reflect.TypeOf(t{{ $index }}))
{{ end }}
	{{ if not .CodecPkgFiles }}{{ .CodecPkgName }}.{{ end }}Gen(&out, "{{ .BuildTag }}", "{{ .PackageName }}", {{ .UseUnsafe }}, typs...)
	bout, err := format.Source(out.Bytes())
	if err != nil {
		fout.Write(out.Bytes())
		panic(err)
	}
	fout.Write(bout)
}

`

// Generate is given a list of *.go files to parse, and an output file (fout).
//
// It finds all types T in the files, and it creates 2 tmp files (frun).
//   - main package file passed to 'go run'
//   - package level file which calls *genRunner.Selfer to write Selfer impls for each T.
// We use a package level file so that it can reference unexported types in the package being worked on.
// Tool then executes: "go run __frun__" which creates fout.
// fout contains Codec(En|De)codeSelf implementations for every type T.
//
func Generate(outfile, buildTag, codecPkgPath string, useUnsafe bool, goRunTag string,
	regexName *regexp.Regexp, deleteTempFile bool, infiles ...string) (err error) {
	// For each file, grab AST, find each type, and write a call to it.
	if len(infiles) == 0 {
		return
	}
	if outfile == "" || codecPkgPath == "" {
		err = errors.New("outfile and codec package path cannot be blank")
		return
	}
	// We have to parse dir for package, before opening the temp file for writing (else ImportDir fails).
	// Also, ImportDir(...) must take an absolute path.
	lastdir := filepath.Dir(outfile)
	absdir, err := filepath.Abs(lastdir)
	if err != nil {
		return
	}
	pkg, err := build.Default.ImportDir(absdir, build.AllowBinary)
	if err != nil {
		return
	}
	type tmplT struct {
		CodecPkgName    string
		CodecImportPath string
		ImportPath      string
		OutFile         string
		PackageName     string
		RandString      string
		BuildTag        string
		Types           []string
		CodecPkgFiles   bool
		UseUnsafe       bool
	}
	tv := tmplT{
		CodecPkgName:    "codec1978",
		OutFile:         outfile,
		CodecImportPath: codecPkgPath,
		BuildTag:        buildTag,
		UseUnsafe:       useUnsafe,
		RandString:      strconv.FormatInt(time.Now().UnixNano(), 10),
	}
	tv.ImportPath = pkg.ImportPath
	if tv.ImportPath == tv.CodecImportPath {
		tv.CodecPkgFiles = true
		tv.CodecPkgName = "codec"
	}
	astfiles := make([]*ast.File, len(infiles))
	for i, infile := range infiles {
		if filepath.Dir(infile) != lastdir {
			err = errors.New("in files must all be in same directory as outfile")
			return
		}
		fset := token.NewFileSet()
		astfiles[i], err = parser.ParseFile(fset, infile, nil, 0)
		if err != nil {
			return
		}
		if i == 0 {
			tv.PackageName = astfiles[i].Name.Name
			if tv.PackageName == "main" {
				// codecgen cannot be run on types in the 'main' package.
				// A temporary 'main' package must be created, and should reference the fully built
				// package containing the types.
				// Also, the temporary main package will conflict with the main package which already has a main method.
				err = errors.New("codecgen cannot be run on types in the 'main' package")
				return
			}
		}
	}

	for _, f := range astfiles {
		for _, d := range f.Decls {
			if gd, ok := d.(*ast.GenDecl); ok {
				for _, dd := range gd.Specs {
					if td, ok := dd.(*ast.TypeSpec); ok {
						// if len(td.Name.Name) == 0 || td.Name.Name[0] > 'Z' || td.Name.Name[0] < 'A' {
						if len(td.Name.Name) == 0 {
							continue
						}

						// only generate for:
						//   struct: StructType
						//   primitives (numbers, bool, string): Ident
						//   map: MapType
						//   slice, array: ArrayType
						//   chan: ChanType
						// do not generate:
						//   FuncType, InterfaceType, StarExpr (ptr), etc
						switch td.Type.(type) {
						case *ast.StructType, *ast.Ident, *ast.MapType, *ast.ArrayType, *ast.ChanType:
							if regexName.FindStringIndex(td.Name.Name) != nil {
								tv.Types = append(tv.Types, td.Name.Name)
							}
						}
					}
				}
			}
		}
	}

	if len(tv.Types) == 0 {
		return
	}

	// we cannot use ioutil.TempFile, because we cannot guarantee the file suffix (.go).
	// Also, we cannot create file in temp directory,
	// because go run will not work (as it needs to see the types here).
	// Consequently, create the temp file in the current directory, and remove when done.

	// frun, err = ioutil.TempFile("", "codecgen-")
	// frunName := filepath.Join(os.TempDir(), "codecgen-"+strconv.FormatInt(time.Now().UnixNano(), 10)+".go")

	frunMainName := "codecgen-main-" + tv.RandString + ".generated.go"
	frunPkgName := "codecgen-pkg-" + tv.RandString + ".generated.go"
	if deleteTempFile {
		defer os.Remove(frunMainName)
		defer os.Remove(frunPkgName)
	}
	// var frunMain, frunPkg *os.File
	if _, err = gen1(frunMainName, genFrunMainTmpl, &tv); err != nil {
		return
	}
	if _, err = gen1(frunPkgName, genFrunPkgTmpl, &tv); err != nil {
		return
	}

	// remove outfile, so "go run ..." will not think that types in outfile already exist.
	os.Remove(outfile)

	// execute go run frun
	cmd := exec.Command("go", "run", "-tags="+goRunTag, frunMainName) //, frunPkg.Name())
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	if err = cmd.Run(); err != nil {
		err = fmt.Errorf("error running 'go run %s': %v, console: %s",
			frunMainName, err, buf.Bytes())
		return
	}
	os.Stdout.Write(buf.Bytes())
	return
}

func gen1(frunName, tmplStr string, tv interface{}) (frun *os.File, err error) {
	os.Remove(frunName)
	if frun, err = os.Create(frunName); err != nil {
		return
	}
	defer frun.Close()

	t := template.New("")
	if t, err = t.Parse(tmplStr); err != nil {
		return
	}
	bw := bufio.NewWriter(frun)
	if err = t.Execute(bw, tv); err != nil {
		return
	}
	if err = bw.Flush(); err != nil {
		return
	}
	return
}

func main() {
	o := flag.String("o", "", "out file")
	c := flag.String("c", genCodecPath, "codec path")
	t := flag.String("t", "", "build tag to put in file")
	r := flag.String("r", ".*", "regex for type name to match")
	rt := flag.String("rt", "", "tags for go run")
	x := flag.Bool("x", false, "keep temp file")
	u := flag.Bool("u", false, "Use unsafe, e.g. to avoid unnecessary allocation on []byte->string")

	flag.Parse()
	if err := Generate(*o, *t, *c, *u, *rt,
		regexp.MustCompile(*r), !*x, flag.Args()...); err != nil {
		fmt.Fprintf(os.Stderr, "codecgen error: %v\n", err)
		os.Exit(1)
	}
}
