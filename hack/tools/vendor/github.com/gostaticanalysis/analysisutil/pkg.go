package analysisutil

import (
	"go/types"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// RemoVendor removes vendoring information from import path.
func RemoveVendor(path string) string {
	i := strings.Index(path, "vendor/")
	if i >= 0 {
		return path[i+len("vendor/"):]
	}
	return path
}

// LookupFromImports finds an object from import paths.
func LookupFromImports(imports []*types.Package, path, name string) types.Object {
	path = RemoveVendor(path)
	for i := range imports {
		if path == RemoveVendor(imports[i].Path()) {
			return imports[i].Scope().Lookup(name)
		}
	}
	return nil
}

// Imported returns true when the given pass imports the pkg.
func Imported(pkgPath string, pass *analysis.Pass) bool {
	fs := pass.Files
	if len(fs) == 0 {
		return false
	}
	for _, f := range fs {
		for _, i := range f.Imports {
			path, err := strconv.Unquote(i.Path.Value)
			if err != nil {
				continue
			}
			if RemoveVendor(path) == pkgPath {
				return true
			}
		}
	}
	return false
}
