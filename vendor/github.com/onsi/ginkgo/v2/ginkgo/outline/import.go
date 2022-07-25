// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Most of the required functions were available in the
// "golang.org/x/tools/go/ast/astutil" package, but not exported.
// They were copied from https://github.com/golang/tools/blob/2b0845dc783e36ae26d683f4915a5840ef01ab0f/go/ast/astutil/imports.go

package outline

import (
	"go/ast"
	"strconv"
	"strings"
)

// packageNameForImport returns the package name for the package. If the package
// is not imported, it returns nil. "Package name" refers to `pkgname` in the
// call expression `pkgname.ExportedIdentifier`. Examples:
// (import path not found) -> nil
// "import example.com/pkg/foo" -> "foo"
// "import fooalias example.com/pkg/foo" -> "fooalias"
// "import . example.com/pkg/foo" -> ""
func packageNameForImport(f *ast.File, path string) *string {
	spec := importSpec(f, path)
	if spec == nil {
		return nil
	}
	name := spec.Name.String()
	if name == "<nil>" {
		// If the package name is not explicitly specified,
		// make an educated guess. This is not guaranteed to be correct.
		lastSlash := strings.LastIndex(path, "/")
		if lastSlash == -1 {
			name = path
		} else {
			name = path[lastSlash+1:]
		}
	}
	if name == "." {
		name = ""
	}
	return &name
}

// importSpec returns the import spec if f imports path,
// or nil otherwise.
func importSpec(f *ast.File, path string) *ast.ImportSpec {
	for _, s := range f.Imports {
		if importPath(s) == path {
			return s
		}
	}
	return nil
}

// importPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func importPath(s *ast.ImportSpec) string {
	t, err := strconv.Unquote(s.Path.Value)
	if err != nil {
		return ""
	}
	return t
}
