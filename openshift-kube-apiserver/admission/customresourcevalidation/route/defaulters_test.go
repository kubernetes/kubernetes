package route

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestDuplicatedDefaulters(t *testing.T) {
	expected, err := findDefaultersInPackage("../../../../vendor/github.com/openshift/library-go/pkg/route/defaulting")
	if err != nil {
		t.Fatalf("error finding expected manual defaulters: %v", err)
	}

	actual, err := findDefaultersInPackage(".")
	if err != nil {
		t.Fatalf("error finding actual manual defaulters: %v", err)
	}

	for _, missing := range expected.Difference(actual).List() {
		t.Errorf("missing local duplicate of library-go defaulter %q", missing)
	}

	for _, extra := range actual.Difference(expected).List() {
		t.Errorf("found local defaulter %q without library-go counterpart", extra)
	}
}

// findDefaultersInPackage parses the source of the Go package at the given path and returns the
// names of all manual defaulter functions it declares. Package function declarations can't be
// enumerated using reflection.
func findDefaultersInPackage(path string) (sets.String, error) {
	pkgs, err := parser.ParseDir(token.NewFileSet(), path, func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to parse source of package at %q: %v", path, err)
	}
	if len(pkgs) != 1 {
		return nil, fmt.Errorf("expected exactly 1 package for all sources in %q, got %d", path, len(pkgs))
	}

	defaulters := sets.NewString()
	for _, pkg := range pkgs {
		ast.Inspect(pkg, func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.Package, *ast.File:
				return true
			case *ast.FuncDecl:
				if typed.Recv == nil && strings.HasPrefix(typed.Name.Name, "SetDefaults_") {
					defaulters.Insert(typed.Name.Name)
				}
				return false
			default:
				return false
			}
		})
	}
	return defaulters, nil
}
