/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package codeinspector

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"strings"
	"unicode"
)

// GetPublicFunctions lists all public functions (not methods) from a golang source file.
func GetPublicFunctions(filePath string) ([]string, error) {
	var functionNames []string

	// Create the AST by parsing src.
	fset := token.NewFileSet() // positions are relative to fset
	f, err := parser.ParseFile(fset, filePath, nil, 0)
	if err != nil {
		return nil, fmt.Errorf("failed parse file to list functions: %v", err)
	}

	// Inspect the AST and print all identifiers and literals.
	ast.Inspect(f, func(n ast.Node) bool {
		var s string
		switch x := n.(type) {
		case *ast.FuncDecl:
			s = x.Name.Name
			// It's a function (not method), and is public, record it.
			if x.Recv == nil && isPublic(s) {
				functionNames = append(functionNames, s)
			}
		}
		return true
	})

	return functionNames, nil
}

// isPublic checks if a given string is a public function name.
func isPublic(myString string) bool {
	a := []rune(myString)
	a[0] = unicode.ToUpper(a[0])
	return myString == string(a)
}

// GetSourceCodeFiles lists golang source code files from directory, excluding sub-directory and tests files.
func GetSourceCodeFiles(dir string) ([]string, error) {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var filenames []string

	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".go") && !strings.HasSuffix(file.Name(), "_test.go") {
			filenames = append(filenames, file.Name())
		}
	}

	return filenames, nil
}
