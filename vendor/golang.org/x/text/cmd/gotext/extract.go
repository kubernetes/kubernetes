// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/constant"
	"go/format"
	"go/parser"
	"go/types"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/loader"
)

// TODO:
// - merge information into existing files
// - handle different file formats (PO, XLIFF)
// - handle features (gender, plural)
// - message rewriting

var cmdExtract = &Command{
	Run:       runExtract,
	UsageLine: "extract <package>*",
	Short:     "extract strings to be translated from code",
}

func runExtract(cmd *Command, args []string) error {
	if len(args) == 0 {
		args = []string{"."}
	}

	conf := loader.Config{
		Build:      &build.Default,
		ParserMode: parser.ParseComments,
	}

	// Use the initial packages from the command line.
	args, err := conf.FromArgs(args, false)
	if err != nil {
		return err
	}

	// Load, parse and type-check the whole program.
	iprog, err := conf.Load()
	if err != nil {
		return err
	}

	// print returns Go syntax for the specified node.
	print := func(n ast.Node) string {
		var buf bytes.Buffer
		format.Node(&buf, conf.Fset, n)
		return buf.String()
	}

	var translations []Translation

	for _, info := range iprog.InitialPackages() {
		for _, f := range info.Files {
			// Associate comments with nodes.
			cmap := ast.NewCommentMap(iprog.Fset, f, f.Comments)
			getComment := func(n ast.Node) string {
				cs := cmap.Filter(n).Comments()
				if len(cs) > 0 {
					return strings.TrimSpace(cs[0].Text())
				}
				return ""
			}

			// Find function calls.
			ast.Inspect(f, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}

				// Skip calls of functions other than
				// (*message.Printer).{Sp,Fp,P}rintf.
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				meth := info.Selections[sel]
				if meth == nil || meth.Kind() != types.MethodVal {
					return true
				}
				// TODO: remove cheap hack and check if the type either
				// implements some interface or is specifically of type
				// "golang.org/x/text/message".Printer.
				m, ok := extractFuncs[path.Base(meth.Recv().String())]
				if !ok {
					return true
				}

				// argn is the index of the format string.
				argn, ok := m[meth.Obj().Name()]
				if !ok || argn >= len(call.Args) {
					return true
				}

				// Skip calls with non-constant format string.
				fmtstr := info.Types[call.Args[argn]].Value
				if fmtstr == nil || fmtstr.Kind() != constant.String {
					return true
				}

				posn := conf.Fset.Position(call.Lparen)
				filepos := fmt.Sprintf("%s:%d:%d", filepath.Base(posn.Filename), posn.Line, posn.Column)

				// TODO: identify the type of the format argument. If it is not
				// a string, multiple keys may be defined.
				var key []string

				// TODO: replace substitutions (%v) with a translator friendly
				// notation. For instance:
				//     "%d files remaining" -> "{numFiles} files remaining", or
				//     "%d files remaining" -> "{arg1} files remaining"
				// Alternatively, this could be done at a later stage.
				msg := constant.StringVal(fmtstr)

				// Construct a Translation unit.
				c := Translation{
					Key:              key,
					Position:         filepath.Join(info.Pkg.Path(), filepos),
					Original:         Text{Msg: msg},
					ExtractedComment: getComment(call.Args[0]),
					// TODO(fix): this doesn't get the before comment.
					// Comment: getComment(call),
				}

				for i, arg := range call.Args[argn+1:] {
					var val string
					if v := info.Types[arg].Value; v != nil {
						val = v.ExactString()
					}
					posn := conf.Fset.Position(arg.Pos())
					filepos := fmt.Sprintf("%s:%d:%d", filepath.Base(posn.Filename), posn.Line, posn.Column)
					c.Args = append(c.Args, Argument{
						ID:             i + 1,
						Type:           info.Types[arg].Type.String(),
						UnderlyingType: info.Types[arg].Type.Underlying().String(),
						Expr:           print(arg),
						Value:          val,
						Comment:        getComment(arg),
						Position:       filepath.Join(info.Pkg.Path(), filepos),
						// TODO report whether it implements
						// interfaces plural.Interface,
						// gender.Interface.
					})
				}

				translations = append(translations, c)
				return true
			})
		}
	}

	data, err := json.MarshalIndent(translations, "", "    ")
	if err != nil {
		return err
	}
	for _, tag := range getLangs() {
		// TODO: merge with existing files, don't overwrite.
		os.MkdirAll(*dir, 0744)
		file := filepath.Join(*dir, fmt.Sprintf("gotext_%v.out.json", tag))
		if err := ioutil.WriteFile(file, data, 0744); err != nil {
			return fmt.Errorf("could not create file: %v", err)
		}
	}
	return nil
}

// extractFuncs indicates the types and methods for which to extract strings,
// and which argument to extract.
// TODO: use the types in conf.Import("golang.org/x/text/message") to extract
// the correct instances.
var extractFuncs = map[string]map[string]int{
	// TODO: Printer -> *golang.org/x/text/message.Printer
	"message.Printer": {
		"Printf":  0,
		"Sprintf": 0,
		"Fprintf": 1,
	},
}
