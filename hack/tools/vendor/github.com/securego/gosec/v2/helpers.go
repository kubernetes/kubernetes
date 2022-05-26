// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
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

package gosec

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// MatchCallByPackage ensures that the specified package is imported,
// adjusts the name for any aliases and ignores cases that are
// initialization only imports.
//
// Usage:
// 	node, matched := MatchCallByPackage(n, ctx, "math/rand", "Read")
//
func MatchCallByPackage(n ast.Node, c *Context, pkg string, names ...string) (*ast.CallExpr, bool) {
	importedName, found := GetImportedName(pkg, c)
	if !found {
		return nil, false
	}

	if callExpr, ok := n.(*ast.CallExpr); ok {
		packageName, callName, err := GetCallInfo(callExpr, c)
		if err != nil {
			return nil, false
		}
		if packageName == importedName {
			for _, name := range names {
				if callName == name {
					return callExpr, true
				}
			}
		}
	}
	return nil, false
}

// MatchCompLit will match an ast.CompositeLit based on the supplied type
func MatchCompLit(n ast.Node, ctx *Context, required string) *ast.CompositeLit {
	if complit, ok := n.(*ast.CompositeLit); ok {
		typeOf := ctx.Info.TypeOf(complit)
		if typeOf.String() == required {
			return complit
		}
	}
	return nil
}

// GetInt will read and return an integer value from an ast.BasicLit
func GetInt(n ast.Node) (int64, error) {
	if node, ok := n.(*ast.BasicLit); ok && node.Kind == token.INT {
		return strconv.ParseInt(node.Value, 0, 64)
	}
	return 0, fmt.Errorf("Unexpected AST node type: %T", n)
}

// GetFloat will read and return a float value from an ast.BasicLit
func GetFloat(n ast.Node) (float64, error) {
	if node, ok := n.(*ast.BasicLit); ok && node.Kind == token.FLOAT {
		return strconv.ParseFloat(node.Value, 64)
	}
	return 0.0, fmt.Errorf("Unexpected AST node type: %T", n)
}

// GetChar will read and return a char value from an ast.BasicLit
func GetChar(n ast.Node) (byte, error) {
	if node, ok := n.(*ast.BasicLit); ok && node.Kind == token.CHAR {
		return node.Value[0], nil
	}
	return 0, fmt.Errorf("Unexpected AST node type: %T", n)
}

// GetString will read and return a string value from an ast.BasicLit
func GetString(n ast.Node) (string, error) {
	if node, ok := n.(*ast.BasicLit); ok && node.Kind == token.STRING {
		return strconv.Unquote(node.Value)
	}
	return "", fmt.Errorf("Unexpected AST node type: %T", n)
}

// GetCallObject returns the object and call expression and associated
// object for a given AST node. nil, nil will be returned if the
// object cannot be resolved.
func GetCallObject(n ast.Node, ctx *Context) (*ast.CallExpr, types.Object) {
	switch node := n.(type) {
	case *ast.CallExpr:
		switch fn := node.Fun.(type) {
		case *ast.Ident:
			return node, ctx.Info.Uses[fn]
		case *ast.SelectorExpr:
			return node, ctx.Info.Uses[fn.Sel]
		}
	}
	return nil, nil
}

// GetCallInfo returns the package or type and name  associated with a
// call expression.
func GetCallInfo(n ast.Node, ctx *Context) (string, string, error) {
	switch node := n.(type) {
	case *ast.CallExpr:
		switch fn := node.Fun.(type) {
		case *ast.SelectorExpr:
			switch expr := fn.X.(type) {
			case *ast.Ident:
				if expr.Obj != nil && expr.Obj.Kind == ast.Var {
					t := ctx.Info.TypeOf(expr)
					if t != nil {
						return t.String(), fn.Sel.Name, nil
					}
					return "undefined", fn.Sel.Name, fmt.Errorf("missing type info")
				}
				return expr.Name, fn.Sel.Name, nil
			case *ast.SelectorExpr:
				if expr.Sel != nil {
					t := ctx.Info.TypeOf(expr.Sel)
					if t != nil {
						return t.String(), fn.Sel.Name, nil
					}
					return "undefined", fn.Sel.Name, fmt.Errorf("missing type info")
				}
			case *ast.CallExpr:
				switch call := expr.Fun.(type) {
				case *ast.Ident:
					if call.Name == "new" {
						t := ctx.Info.TypeOf(expr.Args[0])
						if t != nil {
							return t.String(), fn.Sel.Name, nil
						}
						return "undefined", fn.Sel.Name, fmt.Errorf("missing type info")
					}
					if call.Obj != nil {
						switch decl := call.Obj.Decl.(type) {
						case *ast.FuncDecl:
							ret := decl.Type.Results
							if ret != nil && len(ret.List) > 0 {
								ret1 := ret.List[0]
								if ret1 != nil {
									t := ctx.Info.TypeOf(ret1.Type)
									if t != nil {
										return t.String(), fn.Sel.Name, nil
									}
									return "undefined", fn.Sel.Name, fmt.Errorf("missing type info")
								}
							}
						}
					}
				}
			}
		case *ast.Ident:
			return ctx.Pkg.Name(), fn.Name, nil
		}
	}

	return "", "", fmt.Errorf("unable to determine call info")
}

// GetCallStringArgsValues returns the values of strings arguments if they can be resolved
func GetCallStringArgsValues(n ast.Node, ctx *Context) []string {
	values := []string{}
	switch node := n.(type) {
	case *ast.CallExpr:
		for _, arg := range node.Args {
			switch param := arg.(type) {
			case *ast.BasicLit:
				value, err := GetString(param)
				if err == nil {
					values = append(values, value)
				}
			case *ast.Ident:
				values = append(values, GetIdentStringValues(param)...)
			}
		}
	}
	return values
}

// GetIdentStringValues return the string values of an Ident if they can be resolved
func GetIdentStringValues(ident *ast.Ident) []string {
	values := []string{}
	obj := ident.Obj
	if obj != nil {
		switch decl := obj.Decl.(type) {
		case *ast.ValueSpec:
			for _, v := range decl.Values {
				value, err := GetString(v)
				if err == nil {
					values = append(values, value)
				}
			}
		case *ast.AssignStmt:
			for _, v := range decl.Rhs {
				value, err := GetString(v)
				if err == nil {
					values = append(values, value)
				}
			}
		}
	}
	return values
}

// GetBinaryExprOperands returns all operands of a binary expression by traversing
// the expression tree
func GetBinaryExprOperands(be *ast.BinaryExpr) []ast.Node {
	var traverse func(be *ast.BinaryExpr)
	result := []ast.Node{}
	traverse = func(be *ast.BinaryExpr) {
		if lhs, ok := be.X.(*ast.BinaryExpr); ok {
			traverse(lhs)
		} else {
			result = append(result, be.X)
		}
		if rhs, ok := be.Y.(*ast.BinaryExpr); ok {
			traverse(rhs)
		} else {
			result = append(result, be.Y)
		}
	}
	traverse(be)
	return result
}

// GetImportedName returns the name used for the package within the
// code. It will resolve aliases and ignores initialization only imports.
func GetImportedName(path string, ctx *Context) (string, bool) {
	importName, imported := ctx.Imports.Imported[path]
	if !imported {
		return "", false
	}

	if _, initonly := ctx.Imports.InitOnly[path]; initonly {
		return "", false
	}

	if alias, ok := ctx.Imports.Aliased[path]; ok {
		importName = alias
	}
	return importName, true
}

// GetImportPath resolves the full import path of an identifier based on
// the imports in the current context.
func GetImportPath(name string, ctx *Context) (string, bool) {
	for path := range ctx.Imports.Imported {
		if imported, ok := GetImportedName(path, ctx); ok && imported == name {
			return path, true
		}
	}
	return "", false
}

// GetLocation returns the filename and line number of an ast.Node
func GetLocation(n ast.Node, ctx *Context) (string, int) {
	fobj := ctx.FileSet.File(n.Pos())
	return fobj.Name(), fobj.Line(n.Pos())
}

// Gopath returns all GOPATHs
func Gopath() []string {
	defaultGoPath := runtime.GOROOT()
	if u, err := user.Current(); err == nil {
		defaultGoPath = filepath.Join(u.HomeDir, "go")
	}
	path := Getenv("GOPATH", defaultGoPath)
	paths := strings.Split(path, string(os.PathListSeparator))
	for idx, path := range paths {
		if abs, err := filepath.Abs(path); err == nil {
			paths[idx] = abs
		}
	}
	return paths
}

// Getenv returns the values of the environment variable, otherwise
// returns the default if variable is not set
func Getenv(key, userDefault string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return userDefault
}

// GetPkgRelativePath returns the Go relative relative path derived
// form the given path
func GetPkgRelativePath(path string) (string, error) {
	abspath, err := filepath.Abs(path)
	if err != nil {
		abspath = path
	}
	if strings.HasSuffix(abspath, ".go") {
		abspath = filepath.Dir(abspath)
	}
	for _, base := range Gopath() {
		projectRoot := filepath.FromSlash(fmt.Sprintf("%s/src/", base))
		if strings.HasPrefix(abspath, projectRoot) {
			return strings.TrimPrefix(abspath, projectRoot), nil
		}
	}
	return "", errors.New("no project relative path found")
}

// GetPkgAbsPath returns the Go package absolute path derived from
// the given path
func GetPkgAbsPath(pkgPath string) (string, error) {
	absPath, err := filepath.Abs(pkgPath)
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return "", errors.New("no project absolute path found")
	}
	return absPath, nil
}

// ConcatString recursively concatenates strings from a binary expression
func ConcatString(n *ast.BinaryExpr) (string, bool) {
	var s string
	// sub expressions are found in X object, Y object is always last BasicLit
	if rightOperand, ok := n.Y.(*ast.BasicLit); ok {
		if str, err := GetString(rightOperand); err == nil {
			s = str + s
		}
	} else {
		return "", false
	}
	if leftOperand, ok := n.X.(*ast.BinaryExpr); ok {
		if recursion, ok := ConcatString(leftOperand); ok {
			s = recursion + s
		}
	} else if leftOperand, ok := n.X.(*ast.BasicLit); ok {
		if str, err := GetString(leftOperand); err == nil {
			s = str + s
		}
	} else {
		return "", false
	}
	return s, true
}

// FindVarIdentities returns array of all variable identities in a given binary expression
func FindVarIdentities(n *ast.BinaryExpr, c *Context) ([]*ast.Ident, bool) {
	identities := []*ast.Ident{}
	// sub expressions are found in X object, Y object is always the last term
	if rightOperand, ok := n.Y.(*ast.Ident); ok {
		obj := c.Info.ObjectOf(rightOperand)
		if _, ok := obj.(*types.Var); ok && !TryResolve(rightOperand, c) {
			identities = append(identities, rightOperand)
		}
	}
	if leftOperand, ok := n.X.(*ast.BinaryExpr); ok {
		if leftIdentities, ok := FindVarIdentities(leftOperand, c); ok {
			identities = append(identities, leftIdentities...)
		}
	} else {
		if leftOperand, ok := n.X.(*ast.Ident); ok {
			obj := c.Info.ObjectOf(leftOperand)
			if _, ok := obj.(*types.Var); ok && !TryResolve(leftOperand, c) {
				identities = append(identities, leftOperand)
			}
		}
	}

	if len(identities) > 0 {
		return identities, true
	}
	// if nil or error, return false
	return nil, false
}

// PackagePaths returns a slice with all packages path at given root directory
func PackagePaths(root string, excludes []*regexp.Regexp) ([]string, error) {
	if strings.HasSuffix(root, "...") {
		root = root[0 : len(root)-3]
	} else {
		return []string{root}, nil
	}
	paths := map[string]bool{}
	err := filepath.Walk(root, func(path string, f os.FileInfo, err error) error {
		if filepath.Ext(path) == ".go" {
			path = filepath.Dir(path)
			if isExcluded(filepath.ToSlash(path), excludes) {
				return nil
			}
			paths[path] = true
		}
		return nil
	})
	if err != nil {
		return []string{}, err
	}

	result := []string{}
	for path := range paths {
		result = append(result, path)
	}
	return result, nil
}

// isExcluded checks if a string matches any of the exclusion regexps
func isExcluded(str string, excludes []*regexp.Regexp) bool {
	if excludes == nil {
		return false
	}
	for _, exclude := range excludes {
		if exclude != nil && exclude.MatchString(str) {
			return true
		}
	}
	return false
}

// ExcludedDirsRegExp builds the regexps for a list of excluded dirs provided as strings
func ExcludedDirsRegExp(excludedDirs []string) []*regexp.Regexp {
	var exps []*regexp.Regexp
	for _, excludedDir := range excludedDirs {
		str := fmt.Sprintf(`([\\/])?%s([\\/])?`, strings.ReplaceAll(filepath.ToSlash(excludedDir), "/", `\/`))
		r := regexp.MustCompile(str)
		exps = append(exps, r)
	}
	return exps
}

// RootPath returns the absolute root path of a scan
func RootPath(root string) (string, error) {
	root = strings.TrimSuffix(root, "...")
	return filepath.Abs(root)
}
