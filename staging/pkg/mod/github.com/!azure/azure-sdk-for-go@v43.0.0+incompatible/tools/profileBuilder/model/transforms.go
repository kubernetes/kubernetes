// +build go1.9

// Copyright 2018 Microsoft Corporation and contributors
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

// Package model holds the business logic for the operations made available by
// profileBuilder.
//
// This package is not governed by the SemVer associated with the rest of the
// Azure-SDK-for-Go.
package model

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/tools/internal/modinfo"

	"golang.org/x/tools/imports"
)

// ListDefinition represents a JSON file that contains a list of packages to include
type ListDefinition struct {
	Include      []string          `json:"include"`
	PathOverride map[string]string `json:"pathOverride"`
}

const (
	armPathModifier = "mgmt"
	aliasFileName   = "models.go"
)

// BuildProfile takes a list of packages and creates a profile
func BuildProfile(packageList ListDefinition, name, outputLocation string, outputLog, errLog *log.Logger, recursive, modules bool) {
	// limit the number of concurrent calls to parser.ParseDir()
	semLimit := 32
	if runtime.GOOS == "darwin" {
		// set a lower limit for darwin as it runs out of file handles
		semLimit = 16
	}
	sem := make(chan struct{}, semLimit)
	wg := &sync.WaitGroup{}
	wg.Add(len(packageList.Include))
	for _, pkgDir := range packageList.Include {
		if !filepath.IsAbs(pkgDir) {
			abs, err := filepath.Abs(pkgDir)
			if err != nil {
				errLog.Fatalf("failed to convert to absolute path: %v", err)
			}
			pkgDir = abs
		}
		go func(pd string) {
			filepath.Walk(pd, func(path string, info os.FileInfo, err error) error {
				if !info.IsDir() {
					return nil
				}
				fs := token.NewFileSet()
				sem <- struct{}{}
				packages, err := parser.ParseDir(fs, path, func(f os.FileInfo) bool {
					// exclude test files
					return !strings.HasSuffix(f.Name(), "_test.go")
				}, 0)
				<-sem
				if err != nil {
					errLog.Fatalf("failed to parse '%s': %v", path, err)
				}
				if len(packages) < 1 {
					errLog.Fatalf("didn't find any packages in '%s'", path)
				}
				if len(packages) > 1 {
					errLog.Fatalf("found more than one package in '%s'", path)
				}
				for pn := range packages {
					p := packages[pn]
					// trim any non-exported nodes
					if exp := ast.PackageExports(p); !exp {
						errLog.Fatalf("package '%s' doesn't contain any exports", pn)
					}
					// construct the import path from the outputLocation
					// e.g. D:\work\src\github.com\Azure\azure-sdk-for-go\profiles\2017-03-09\compute\mgmt\compute
					// becomes github.com/Azure/azure-sdk-for-go/profiles/2017-03-09/compute/mgmt/compute
					i := strings.Index(path, "github.com")
					if i == -1 {
						errLog.Fatalf("didn't find 'github.com' in '%s'", path)
					}
					importPath := strings.Replace(path[i:], "\\", "/", -1)
					ap, err := NewAliasPackage(p, importPath)
					if err != nil {
						errLog.Fatalf("failed to create alias package: %v", err)
					}
					updateAliasPackageUserAgent(ap, name)
					// build the profile output directory, if there's an override path use that
					var aliasPath string
					var ok bool
					if aliasPath, ok = packageList.PathOverride[importPath]; !ok {
						var err error
						if modules && modinfo.HasVersionSuffix(path) {
							// strip off the major version dir so it's not included in the alias path
							path = filepath.Dir(path)
						}
						aliasPath, err = getAliasPath(path)
						if err != nil {
							errLog.Fatalf("failed to calculate alias directory: %v", err)
						}
					}
					aliasPath = filepath.Join(outputLocation, aliasPath)
					if _, err := os.Stat(aliasPath); os.IsNotExist(err) {
						err = os.MkdirAll(aliasPath, os.ModeDir|0755)
						if err != nil {
							errLog.Fatalf("failed to create alias directory: %v", err)
						}
					}
					writeAliasPackage(ap, aliasPath, outputLog, errLog)
				}
				if !recursive {
					return filepath.SkipDir
				}
				return nil
			})
			wg.Done()
		}(pkgDir)
	}
	wg.Wait()
	close(sem)
	outputLog.Print(len(packageList.Include), " packages generated.")
}

// getAliasPath takes an existing API Version path and converts the path to a path which uses the new profile layout.
func getAliasPath(packageDir string) (string, error) {
	// we want to transform this:
	//  .../services/compute/mgmt/2016-03-30/compute
	// into this:
	//  compute/mgmt/compute
	// i.e. remove everything to the left of /services along with the API version
	pi, err := DeconstructPath(packageDir)
	if err != nil {
		return "", err
	}

	output := []string{
		pi.Provider,
	}

	if pi.IsArm {
		output = append(output, armPathModifier)
	}
	output = append(output, pi.Group)
	if pi.APIPkg != "" {
		output = append(output, pi.APIPkg)
	}

	return filepath.Join(output...), nil
}

// updateAliasPackageUserAgent updates the "UserAgent" function in the generated profile, if it is present.
func updateAliasPackageUserAgent(ap *AliasPackage, profileName string) {
	var userAgent *ast.FuncDecl
	for _, decl := range ap.Files[aliasFileName].Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == "UserAgent" {
			userAgent = fd
			break
		}
	}
	if userAgent == nil {
		return
	}

	// Grab the expression being returned.
	retResults := &userAgent.Body.List[0].(*ast.ReturnStmt).Results[0]

	// Append a string literal to the result
	updated := &ast.BinaryExpr{
		Op: token.ADD,
		X:  *retResults,
		Y: &ast.BasicLit{
			Value: fmt.Sprintf(`" profiles/%s"`, profileName),
		},
	}
	*retResults = updated
}

// writeAliasPackage adds the MSFT Copyright Header, then writes the alias package to disk.
func writeAliasPackage(ap *AliasPackage, outputPath string, outputLog, errLog *log.Logger) {
	files := token.NewFileSet()

	err := os.MkdirAll(path.Dir(outputPath), 0755|os.ModeDir)
	if err != nil {
		errLog.Fatalf("error creating directory: %v", err)
	}

	aliasFile := filepath.Join(outputPath, aliasFileName)
	outputFile, err := os.Create(aliasFile)
	if err != nil {
		errLog.Fatalf("error creating file: %v", err)
	}

	// TODO: This should really be added by the `goalias` package itself. Doing it here is a work around
	fmt.Fprintln(outputFile, "// +build go1.9")
	fmt.Fprintln(outputFile)

	generatorStampBuilder := new(bytes.Buffer)

	fmt.Fprintf(generatorStampBuilder, "// Copyright %4d Microsoft Corporation\n", time.Now().Year())
	fmt.Fprintln(generatorStampBuilder, `//
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
// limitations under the License.`)

	fmt.Fprintln(outputFile, generatorStampBuilder.String())

	generatorStampBuilder.Reset()

	fmt.Fprintln(generatorStampBuilder, "// This code was auto-generated by:")
	fmt.Fprintln(generatorStampBuilder, "// github.com/Azure/azure-sdk-for-go/tools/profileBuilder")

	fmt.Fprintln(generatorStampBuilder)
	fmt.Fprint(outputFile, generatorStampBuilder.String())

	outputLog.Printf("Writing File: %s", aliasFile)

	file := ap.ModelFile()

	var b bytes.Buffer
	printer.Fprint(&b, files, file)
	res, err := imports.Process(aliasFile, b.Bytes(), nil)
	if err != nil {
		errLog.Fatalf("failed to process imports: %v", err)
	}
	fmt.Fprintf(outputFile, "%s", res)
	outputFile.Close()

	// be sure to specify the file for formatting not the directory; this is to
	// avoid race conditions when formatting parent/child directories (foo and foo/fooapi)
	if err := exec.Command("gofmt", "-w", aliasFile).Run(); err != nil {
		errLog.Fatalf("error formatting profile '%s': %v", aliasFile, err)
	}
}
