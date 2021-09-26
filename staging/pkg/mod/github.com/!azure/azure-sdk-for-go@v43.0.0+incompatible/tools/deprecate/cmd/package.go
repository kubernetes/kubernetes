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

package cmd

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

var packageCmd = &cobra.Command{
	Use:   "package <package dir> <message>",
	Short: "Adds a deprecation comment to all exported types and functions.",
	Long: `The package command adds a deprecation comment to all exported types and functions
in the specified package.  The comment is of the form "// Deprecated: <message>".`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return thePackageCmd(args)
	},
}

func init() {
	rootCmd.AddCommand(packageCmd)
}

func thePackageCmd(args []string) error {
	if len(args) < 2 {
		return errors.New("not enough arguments were supplied")
	}

	fileInfos, err := ioutil.ReadDir(args[0])
	if err != nil {
		return fmt.Errorf("failed to read directory: %v", err)
	}

	for _, fileInfo := range fileInfos {
		if fileInfo.IsDir() {
			continue
		}
		err := addDeprecationToFile(filepath.Join(args[0], fileInfo.Name()), args[1])
		if err != nil {
			return fmt.Errorf("failed to add deprecation comments: %v", err)
		}
	}
	return nil
}

func addDeprecationToFile(file, message string) error {
	vprintf("adding deprecation comments to %s\n", file)
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
	if err != nil {
		return err
	}

	// walk the AST, for each exported type and func add deprecated message
	ast.Inspect(node, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.GenDecl:
			if isExportedSpec(x.Specs) {
				addDeprecationComment(x.Doc, message)
			}
		case *ast.FuncDecl:
			if x.Name.IsExported() {
				addDeprecationComment(x.Doc, message)
			}
			// return false as we don't care about the function body.
			return false
		}
		return true
	})

	// write the updated content
	f, err := os.OpenFile(file, os.O_WRONLY|os.O_TRUNC, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	return printer.Fprint(f, fset, node)
}

func addDeprecationComment(cg *ast.CommentGroup, message string) {
	if cg == nil {
		return
	}
	// create a new comment and add it to the beginning of the comment group
	nd := []*ast.Comment{
		{
			Text:  fmt.Sprintf("// Deprecated: %s", message),
			Slash: cg.Pos() - 1,
		},
	}
	cg.List = append(nd, cg.List...)
}

func isExportedSpec(list []ast.Spec) bool {
	for _, l := range list {
		if s, ok := l.(*ast.TypeSpec); ok {
			return s.Name.IsExported()
		}
	}
	// assume it's exported
	return true
}
