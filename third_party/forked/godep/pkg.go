package main

import (
	"fmt"
	"go/build"
	"os"
	"regexp"
	"sort"
	"strings"
)

// Package represents a Go package.
type Package struct {
	Dir        string
	Root       string
	ImportPath string
	Deps       []string
	Standard   bool
	Processed  bool

	GoFiles        []string
	CgoFiles       []string
	IgnoredGoFiles []string

	TestGoFiles  []string
	TestImports  []string
	XTestGoFiles []string
	XTestImports []string

	Error struct {
		Err string
	}

	// --- New stuff for now
	Imports      []string
	Dependencies []build.Package
}

// LoadPackages loads the named packages
// Unlike the go tool, an empty argument list is treated as an empty list; "."
// must be given explicitly if desired.
// IgnoredGoFiles will be processed and their dependencies resolved recursively
func LoadPackages(names ...string) (a []*Package, err error) {
	debugln("LoadPackages", names)
	if len(names) == 0 {
		return nil, nil
	}

	pkgs := strings.Split(ignorePackages, ",")
	sort.Strings(pkgs)
	for _, i := range importPaths(names) {
		p, err := listPackage(i)
		if err != nil {
			if len(pkgs) > 0 {
				idx := sort.SearchStrings(pkgs, i)
				if idx < len(pkgs) && pkgs[idx] == i {
					fmt.Fprintf(os.Stderr, "warning: ignoring package %q \n", i)
					continue
				}
			}
			return nil, err
		}
		a = append(a, p)
	}
	return a, nil
}

func (p *Package) allGoFiles() []string {
	var a []string
	a = append(a, p.GoFiles...)
	a = append(a, p.CgoFiles...)
	a = append(a, p.TestGoFiles...)
	a = append(a, p.XTestGoFiles...)
	a = append(a, p.IgnoredGoFiles...)
	return a
}

// matchPattern(pattern)(name) reports whether
// name matches pattern.  Pattern is a limited glob
// pattern in which '...' means 'any string' and there
// is no other special syntax.
// Taken from $GOROOT/src/cmd/go/main.go.
func matchPattern(pattern string) func(name string) bool {
	re := regexp.QuoteMeta(pattern)
	re = strings.Replace(re, `\.\.\.`, `.*`, -1)
	// Special case: foo/... matches foo too.
	if strings.HasSuffix(re, `/.*`) {
		re = re[:len(re)-len(`/.*`)] + `(/.*)?`
	}
	reg := regexp.MustCompile(`^` + re + `$`)
	return func(name string) bool {
		return reg.MatchString(name)
	}
}
