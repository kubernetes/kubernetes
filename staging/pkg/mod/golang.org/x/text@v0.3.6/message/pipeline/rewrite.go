// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pipeline

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/constant"
	"go/format"
	"go/token"
	"io"
	"os"
	"strings"

	"golang.org/x/tools/go/loader"
)

const printerType = "golang.org/x/text/message.Printer"

// Rewrite rewrites the Go files in a single package to use the localization
// machinery and rewrites strings to adopt best practices when possible.
// If w is not nil the generated files are written to it, each files with a
// "--- <filename>" header. Otherwise the files are overwritten.
func Rewrite(w io.Writer, args ...string) error {
	conf := &loader.Config{
		AllowErrors: true, // Allow unused instances of message.Printer.
	}
	prog, err := loadPackages(conf, args)
	if err != nil {
		return wrap(err, "")
	}

	for _, info := range prog.InitialPackages() {
		for _, f := range info.Files {
			// Associate comments with nodes.

			// Pick up initialized Printers at the package level.
			r := rewriter{info: info, conf: conf}
			for _, n := range info.InitOrder {
				if t := r.info.Types[n.Rhs].Type.String(); strings.HasSuffix(t, printerType) {
					r.printerVar = n.Lhs[0].Name()
				}
			}

			ast.Walk(&r, f)

			w := w
			if w == nil {
				var err error
				if w, err = os.Create(conf.Fset.File(f.Pos()).Name()); err != nil {
					return wrap(err, "open failed")
				}
			} else {
				fmt.Fprintln(w, "---", conf.Fset.File(f.Pos()).Name())
			}

			if err := format.Node(w, conf.Fset, f); err != nil {
				return wrap(err, "go format failed")
			}
		}
	}

	return nil
}

type rewriter struct {
	info       *loader.PackageInfo
	conf       *loader.Config
	printerVar string
}

// print returns Go syntax for the specified node.
func (r *rewriter) print(n ast.Node) string {
	var buf bytes.Buffer
	format.Node(&buf, r.conf.Fset, n)
	return buf.String()
}

func (r *rewriter) Visit(n ast.Node) ast.Visitor {
	// Save the state by scope.
	if _, ok := n.(*ast.BlockStmt); ok {
		r := *r
		return &r
	}
	// Find Printers created by assignment.
	stmt, ok := n.(*ast.AssignStmt)
	if ok {
		for _, v := range stmt.Lhs {
			if r.printerVar == r.print(v) {
				r.printerVar = ""
			}
		}
		for i, v := range stmt.Rhs {
			if t := r.info.Types[v].Type.String(); strings.HasSuffix(t, printerType) {
				r.printerVar = r.print(stmt.Lhs[i])
				return r
			}
		}
	}
	// Find Printers created by variable declaration.
	spec, ok := n.(*ast.ValueSpec)
	if ok {
		for _, v := range spec.Names {
			if r.printerVar == r.print(v) {
				r.printerVar = ""
			}
		}
		for i, v := range spec.Values {
			if t := r.info.Types[v].Type.String(); strings.HasSuffix(t, printerType) {
				r.printerVar = r.print(spec.Names[i])
				return r
			}
		}
	}
	if r.printerVar == "" {
		return r
	}
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return r
	}

	// TODO: Handle literal values?
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return r
	}
	meth := r.info.Selections[sel]

	source := r.print(sel.X)
	fun := r.print(sel.Sel)
	if meth != nil {
		source = meth.Recv().String()
		fun = meth.Obj().Name()
	}

	// TODO: remove cheap hack and check if the type either
	// implements some interface or is specifically of type
	// "golang.org/x/text/message".Printer.
	m, ok := rewriteFuncs[source]
	if !ok {
		return r
	}

	rewriteType, ok := m[fun]
	if !ok {
		return r
	}
	ident := ast.NewIdent(r.printerVar)
	ident.NamePos = sel.X.Pos()
	sel.X = ident
	if rewriteType.method != "" {
		sel.Sel.Name = rewriteType.method
	}

	// Analyze arguments.
	argn := rewriteType.arg
	if rewriteType.format || argn >= len(call.Args) {
		return r
	}
	hasConst := false
	for _, a := range call.Args[argn:] {
		if v := r.info.Types[a].Value; v != nil && v.Kind() == constant.String {
			hasConst = true
			break
		}
	}
	if !hasConst {
		return r
	}
	sel.Sel.Name = rewriteType.methodf

	// We are done if there is only a single string that does not need to be
	// escaped.
	if len(call.Args) == 1 {
		s, ok := constStr(r.info, call.Args[0])
		if ok && !strings.Contains(s, "%") && !rewriteType.newLine {
			return r
		}
	}

	// Rewrite arguments as format string.
	expr := &ast.BasicLit{
		ValuePos: call.Lparen,
		Kind:     token.STRING,
	}
	newArgs := append(call.Args[:argn:argn], expr)
	newStr := []string{}
	for i, a := range call.Args[argn:] {
		if s, ok := constStr(r.info, a); ok {
			newStr = append(newStr, strings.Replace(s, "%", "%%", -1))
		} else {
			newStr = append(newStr, "%v")
			newArgs = append(newArgs, call.Args[argn+i])
		}
	}
	s := strings.Join(newStr, rewriteType.sep)
	if rewriteType.newLine {
		s += "\n"
	}
	expr.Value = fmt.Sprintf("%q", s)

	call.Args = newArgs

	// TODO: consider creating an expression instead of a constant string and
	// then wrapping it in an escape function or so:
	// call.Args[argn+i] = &ast.CallExpr{
	// 		Fun: &ast.SelectorExpr{
	// 			X:   ast.NewIdent("message"),
	// 			Sel: ast.NewIdent("Lookup"),
	// 		},
	// 		Args: []ast.Expr{a},
	// 	}
	// }

	return r
}

type rewriteType struct {
	// method is the name of the equivalent method on a printer, or "" if it is
	// the same.
	method string

	// methodf is the method to use if the arguments can be rewritten as a
	// arguments to a printf-style call.
	methodf string

	// format is true if the method takes a formatting string followed by
	// substitution arguments.
	format bool

	// arg indicates the position of the argument to extract. If all is
	// positive, all arguments from this argument onwards needs to be extracted.
	arg int

	sep     string
	newLine bool
}

// rewriteFuncs list functions that can be directly mapped to the printer
// functions of the message package.
var rewriteFuncs = map[string]map[string]rewriteType{
	// TODO: Printer -> *golang.org/x/text/message.Printer
	"fmt": {
		"Print":  rewriteType{methodf: "Printf"},
		"Sprint": rewriteType{methodf: "Sprintf"},
		"Fprint": rewriteType{methodf: "Fprintf"},

		"Println":  rewriteType{methodf: "Printf", sep: " ", newLine: true},
		"Sprintln": rewriteType{methodf: "Sprintf", sep: " ", newLine: true},
		"Fprintln": rewriteType{methodf: "Fprintf", sep: " ", newLine: true},

		"Printf":  rewriteType{method: "Printf", format: true},
		"Sprintf": rewriteType{method: "Sprintf", format: true},
		"Fprintf": rewriteType{method: "Fprintf", format: true},
	},
}

func constStr(info *loader.PackageInfo, e ast.Expr) (s string, ok bool) {
	v := info.Types[e].Value
	if v == nil || v.Kind() != constant.String {
		return "", false
	}
	return constant.StringVal(v), true
}
