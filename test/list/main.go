/*
Copyright 2016 The Kubernetes Authors.

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

// list all unit and ginkgo test names that will be run
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var (
	dumpTree     = flag.Bool("dump", false, "print AST")
	dumpJson     = flag.Bool("json", false, "output test list as JSON")
	dumpPrefixes = flag.Bool("prefixes", false, "output owner prefixes as JSON")
	warn         = flag.Bool("warn", false, "print warnings")

	ownerRe           = regexp.MustCompile(`(?m)^//\s?OWNER\s?=\s?(sig/\S*)`)
	normalizeBracesRe = regexp.MustCompile(`\[.*?\]\s*|\{.*?\}\s*`)
)

type Test struct {
	Loc      string
	Name     string
	TestName string
	Owner    string `json:omitempty`
}

// collect extracts test metadata from a file.
// If src is nil, it reads filename for the code, otherwise it
// uses src (which may be a string, byte[], or io.Reader).
func collect(filename string, src interface{}) []Test {
	// Create the AST by parsing src.
	fset := token.NewFileSet() // positions are relative to fset
	f, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		panic(err)
	}

	if *dumpTree {
		ast.Print(fset, f)
	}

	tests := make([]Test, 0)

	ast.Walk(makeWalker("[k8s.io]", fset, &tests), f)

	// Unit tests are much simpler to enumerate!
	if strings.HasSuffix(filename, "_test.go") {
		packageName := f.Name.Name
		dirName, _ := filepath.Split(filename)
		if filepath.Base(dirName) != packageName && *warn {
			log.Printf("Warning: strange path/package mismatch %s %s\n", filename, packageName)
		}
		testPath := "k8s.io/kubernetes/" + dirName[:len(dirName)-1]
		for _, decl := range f.Decls {
			funcdecl, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			name := funcdecl.Name.Name
			if strings.HasPrefix(name, "Test") {
				tests = append(tests, Test{
					Loc:      fset.Position(funcdecl.Pos()).String(),
					Name:     testPath,
					TestName: name,
				})
			}
		}
	}

	// Look for owner in package-level comments.
	var owner string
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			m := ownerRe.FindStringSubmatch(c.Text)
			if m != nil {
				owner = m[1]
			}
		}
	}
	if owner != "" {
		for i := 0; i < len(tests); i++ {
			tests[i].Owner = owner
		}
	}

	return tests
}

// funcName converts a selectorExpr with two idents into a string,
// x.y -> "x.y"
func funcName(n ast.Expr) string {
	if sel, ok := n.(*ast.SelectorExpr); ok {
		if x, ok := sel.X.(*ast.Ident); ok {
			return x.String() + "." + sel.Sel.String()
		}
	}
	return ""
}

// isSprintf returns whether the given node is a call to fmt.Sprintf
func isSprintf(n ast.Expr) bool {
	call, ok := n.(*ast.CallExpr)
	return ok && funcName(call.Fun) == "fmt.Sprintf" && len(call.Args) != 0
}

type walker struct {
	path  string
	fset  *token.FileSet
	tests *[]Test
	vals  map[string]string
}

func makeWalker(path string, fset *token.FileSet, tests *[]Test) *walker {
	return &walker{path, fset, tests, make(map[string]string)}
}

// clone creates a new walker with the given string extending the path.
func (w *walker) clone(ext string) *walker {
	return &walker{w.path + " " + ext, w.fset, w.tests, w.vals}
}

// firstArg attempts to statically determine the value of the first
// argument. It only handles strings, and converts any unknown values
// (fmt.Sprintf interpolations) into *.
func (w *walker) firstArg(n *ast.CallExpr) string {
	if len(n.Args) == 0 {
		return ""
	}
	var lit *ast.BasicLit
	if isSprintf(n.Args[0]) {
		return w.firstArg(n.Args[0].(*ast.CallExpr))
	}
	lit, ok := n.Args[0].(*ast.BasicLit)
	if ok && lit.Kind == token.STRING {
		v, err := strconv.Unquote(lit.Value)
		if err != nil {
			panic(err)
		}
		if strings.Contains(v, "%") {
			v = strings.Replace(v, "%d", "*", -1)
			v = strings.Replace(v, "%v", "*", -1)
			v = strings.Replace(v, "%s", "*", -1)
		}
		return v
	}
	if ident, ok := n.Args[0].(*ast.Ident); ok {
		if val, ok := w.vals[ident.String()]; ok {
			return val
		}
	}
	if *warn {
		log.Printf("Warning: dynamic arg value: %v\n", w.fset.Position(n.Args[0].Pos()))
	}
	return "*"
}

// describeName returns the first argument of a function if it's
// a Ginkgo-relevant function (Describe/KubeDescribe/Context),
// and the empty string otherwise.
func (w *walker) describeName(n *ast.CallExpr) string {
	switch x := n.Fun.(type) {
	case *ast.SelectorExpr:
		if x.Sel.Name != "KubeDescribe" {
			return ""
		}
	case *ast.Ident:
		if x.Name != "Describe" && x.Name != "Context" {
			return ""
		}
	default:
		return ""
	}
	return w.firstArg(n)
}

// itName returns the first argument if it's a call to It(), else "".
func (w *walker) itName(n *ast.CallExpr) string {
	if fun, ok := n.Fun.(*ast.Ident); ok && fun.Name == "It" {
		return w.firstArg(n)
	}
	return ""
}

// Visit walks the AST, following Ginkgo context and collecting tests.
// See the documentation for ast.Walk for more details.
func (w *walker) Visit(n ast.Node) ast.Visitor {
	switch x := n.(type) {
	case *ast.CallExpr:
		name := w.describeName(x)
		if name != "" && len(x.Args) >= 2 {
			// If calling (Kube)Describe/Context, make a new
			// walker to recurse with the description added.
			return w.clone(name)
		}
		name = w.itName(x)
		if name != "" {
			// We've found an It() call, the full test name
			// can be determined now.
			if w.path == "[k8s.io]" && *warn {
				log.Printf("It without matching Describe: %s\n", w.fset.Position(n.Pos()))
			}
			*w.tests = append(*w.tests, Test{
				Loc:      w.fset.Position(n.Pos()).String(),
				Name:     w.path,
				TestName: name,
			})
			return nil // Stop walking
		}
	case *ast.AssignStmt:
		// Attempt to track literals that might be used as
		// arguments. This analysis is very unsound, and ignores
		// both scope and program flow, but is sufficient for
		// our minor use case.
		ident, ok := x.Lhs[0].(*ast.Ident)
		if ok {
			if isSprintf(x.Rhs[0]) {
				// x := fmt.Sprintf("something", args)
				w.vals[ident.String()] = w.firstArg(x.Rhs[0].(*ast.CallExpr))
			}
			if lit, ok := x.Rhs[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
				// x := "a literal string"
				v, err := strconv.Unquote(lit.Value)
				if err != nil {
					panic(err)
				}
				w.vals[ident.String()] = v
			}
		}
	}
	return w // Continue walking
}

type testList struct {
	tests []Test
}

// handlePath walks the filesystem recursively, collecting tests
// from files with paths *e2e*.go and *_test.go, ignoring third_party
// and staging directories.
func (t *testList) handlePath(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if strings.Contains(path, "third_party") ||
		strings.Contains(path, "staging") ||
		strings.Contains(path, "_output") {
		return filepath.SkipDir
	}
	if strings.HasSuffix(path, ".go") && strings.Contains(path, "e2e") ||
		strings.HasSuffix(path, "_test.go") {
		tests := collect(path, nil)
		t.tests = append(t.tests, tests...)
	}
	return nil
}

// Given a test name, strip unnecessary extra braces.
func normalize(name string) string {
	return normalizeBracesRe.ReplaceAllString(name, "")
}

// Given a list of tests with owners, return a list of prefixes that uniquely
// assign each test to an owner.
func makePrefixes(tests []Test) map[string][]string {
	owners := make(map[string]string)
	for _, test := range tests {
		if test.Owner != "" {
			owners[normalize(test.Name+" "+test.TestName)] = test.Owner
		}
	}
	// Returns empty slice if any tests with this prefix have a different owner.
	FindMatches := func(prefix, prefixOwner string) []string {
		r := []string{}
		for name, owner := range owners {
			if strings.HasPrefix(name, prefix) {
				if owner != prefixOwner {
					return []string{}
				} else {
					r = append(r, name)
				}
			}
		}
		return r
	}

	for changed := true; changed; {
		changed = false
		for name, owner := range owners {
			// try removing the last word in the name, use it if all tests beginning
			// with this shorter name share the same owner.
			lastIndex := strings.LastIndex(name, " ")
			if lastIndex > 0 {
				newPrefix := name[:lastIndex]
				matches := FindMatches(newPrefix, owner)
				if len(matches) > 0 {
					changed = true
					for _, matchName := range matches {
						delete(owners, matchName)
					}
					owners[newPrefix] = owner
					break
				}
			}
		}
	}

	ret := make(map[string][]string)
	for name, owner := range owners {
		ret[owner] = append(ret[owner], name)
	}
	return ret
}

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		args = append(args, ".")
	}
	tests := testList{}
	for _, arg := range args {
		err := filepath.Walk(arg, tests.handlePath)
		if err != nil {
			log.Fatalf("Error walking: %v", err)
		}
	}
	if *dumpPrefixes {
		json, err := json.MarshalIndent(makePrefixes(tests.tests), "", "    ")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(string(json))
	} else if *dumpJson {
		json, err := json.Marshal(tests.tests)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(string(json))
	} else {
		for _, t := range tests.tests {
			fmt.Println(t)
		}
	}
}
