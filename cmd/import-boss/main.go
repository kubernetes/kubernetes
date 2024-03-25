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

// import-boss enforces import restrictions in a given repository.
package main

import (
	"flag"
	"os"

	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/spf13/pflag"
	"golang.org/x/tools/go/packages"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

const (
	rulesFileName = ".import-restrictions"
	goModFile     = "go.mod"
)

func main() {
	klog.InitFlags(nil)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()

	pkgs, err := loadPkgs(pflag.Args()...)
	if err != nil {
		klog.Errorf("failed to load packages: %v", err)
	}

	pkgs = massage(pkgs)
	boss := newBoss(pkgs)

	var allErrs []error
	for _, pkg := range pkgs {
		if pkgErrs := boss.Verify(pkg); pkgErrs != nil {
			allErrs = append(allErrs, pkgErrs...)
		}
	}

	fail := false
	for _, err := range allErrs {
		if lister, ok := err.(interface{ Unwrap() []error }); ok {
			for _, err := range lister.Unwrap() {
				fmt.Printf("ERROR: %v\n", err)
			}
		} else {
			fmt.Printf("ERROR: %v\n", err)
		}
		fail = true
	}

	if fail {
		os.Exit(1)
	}

	klog.V(2).Info("Completed successfully.")
}

func loadPkgs(patterns ...string) ([]*packages.Package, error) {
	cfg := packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedImports |
			packages.NeedDeps | packages.NeedModule,
		Tests: true,
	}

	klog.V(1).Infof("loading: %v", patterns)
	tBefore := time.Now()
	pkgs, err := packages.Load(&cfg, patterns...)
	if err != nil {
		return nil, err
	}
	klog.V(2).Infof("loaded %d pkg(s) in %v", len(pkgs), time.Since(tBefore))

	var allErrs []error
	for _, pkg := range pkgs {
		var errs []error
		for _, e := range pkg.Errors {
			if e.Kind == packages.ListError || e.Kind == packages.ParseError {
				errs = append(errs, e)
			}
		}
		if len(errs) > 0 {
			allErrs = append(allErrs, fmt.Errorf("error(s) in %q: %w", pkg.PkgPath, errors.Join(errs...)))
		}
	}
	if len(allErrs) > 0 {
		return nil, errors.Join(allErrs...)
	}

	return pkgs, nil
}

func massage(in []*packages.Package) []*packages.Package {
	out := []*packages.Package{}

	for _, pkg := range in {
		klog.V(2).Infof("considering pkg: %q", pkg.PkgPath)

		// Discard packages which represent the <pkg>.test result.  They don't seem
		// to hold any interesting source info.
		if strings.HasSuffix(pkg.PkgPath, ".test") {
			klog.V(3).Infof("ignoring testbin pkg: %q", pkg.PkgPath)
			continue
		}

		// Packages which end in "_test" have tests which use the special "_test"
		// package suffix.  Packages which have test files must be tests.  Don't
		// ask me, this is what packages.Load produces.
		if strings.HasSuffix(pkg.PkgPath, "_test") || hasTestFiles(pkg.GoFiles) {
			// NOTE: This syntax can be undone with unmassage().
			pkg.PkgPath = strings.TrimSuffix(pkg.PkgPath, "_test") + " ((tests:" + pkg.Name + "))"
			klog.V(3).Infof("renamed to: %q", pkg.PkgPath)
		}
		out = append(out, pkg)
	}

	return out
}

func unmassage(str string) string {
	idx := strings.LastIndex(str, " ((")
	if idx == -1 {
		return str
	}
	return str[0:idx]
}

type ImportBoss struct {
	// incomingImports holds all the packages importing the key.
	incomingImports map[string][]string

	// transitiveIncomingImports holds the transitive closure of
	// incomingImports.
	transitiveIncomingImports map[string][]string
}

func newBoss(pkgs []*packages.Package) *ImportBoss {
	boss := &ImportBoss{
		incomingImports:           map[string][]string{},
		transitiveIncomingImports: map[string][]string{},
	}

	for _, pkg := range pkgs {
		// Accumulate imports
		for imp := range pkg.Imports {
			boss.incomingImports[imp] = append(boss.incomingImports[imp], pkg.PkgPath)
		}
	}

	boss.transitiveIncomingImports = transitiveClosure(boss.incomingImports)

	return boss
}

func hasTestFiles(files []string) bool {
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			return true
		}
	}
	return false
}

func (boss *ImportBoss) Verify(pkg *packages.Package) []error {
	pkgDir := packageDir(pkg)
	if pkgDir == "" {
		// This Package has no usable files, e.g. only tests, which are modelled in
		// a distinct Package.
		return nil
	}

	restrictionFiles, err := recursiveRead(filepath.Join(pkgDir, rulesFileName))
	if err != nil {
		return []error{fmt.Errorf("error finding rules file: %w", err)}
	}
	if len(restrictionFiles) == 0 {
		return nil
	}

	klog.V(2).Infof("verifying pkg %q (%s)", pkg.PkgPath, pkgDir)
	var errs []error
	errs = append(errs, boss.verifyRules(pkg, restrictionFiles)...)
	errs = append(errs, boss.verifyInverseRules(pkg, restrictionFiles)...)
	return errs
}

// packageDir tries to figure out the directory of the specified package.
func packageDir(pkg *packages.Package) string {
	if len(pkg.GoFiles) > 0 {
		return filepath.Dir(pkg.GoFiles[0])
	}
	if len(pkg.IgnoredFiles) > 0 {
		return filepath.Dir(pkg.IgnoredFiles[0])
	}
	return ""
}

type FileFormat struct {
	Rules        []Rule
	InverseRules []Rule

	path string
}

// A single import restriction rule.
type Rule struct {
	// All import paths that match this regexp...
	SelectorRegexp string
	// ... must have one of these prefixes ...
	AllowedPrefixes []string
	// ... and must not have one of these prefixes.
	ForbiddenPrefixes []string
	// True if the rule is to be applied to transitive imports.
	Transitive bool
}

// Disposition represents a decision or non-decision.
type Disposition int

const (
	// DepForbidden means the dependency was explicitly forbidden by a rule.
	DepForbidden Disposition = iota
	// DepAllowed means the dependency was explicitly allowed by a rule.
	DepAllowed
	// DepAllowed means the dependency did not match any rule.
	DepUnknown
)

// Evaluate considers this rule and decides if this dependency is allowed.
func (r Rule) Evaluate(imp string) Disposition {
	// To pass, an import muct be allowed and not forbidden.
	// Check forbidden first.
	for _, forbidden := range r.ForbiddenPrefixes {
		klog.V(5).Infof("checking %q against forbidden prefix %q", imp, forbidden)
		if hasPathPrefix(imp, forbidden) {
			klog.V(5).Infof("this import of %q is forbidden", imp)
			return DepForbidden
		}
	}
	for _, allowed := range r.AllowedPrefixes {
		klog.V(5).Infof("checking %q against allowed prefix %q", imp, allowed)
		if hasPathPrefix(imp, allowed) {
			klog.V(5).Infof("this import of %q is allowed", imp)
			return DepAllowed
		}
	}
	return DepUnknown
}

// recursiveRead collects all '.import-restriction' files, between the current directory,
// and the module root.
func recursiveRead(path string) ([]*FileFormat, error) {
	restrictionFiles := make([]*FileFormat, 0)

	for {
		if _, err := os.Stat(path); err == nil {
			rules, err := readFile(path)
			if err != nil {
				return nil, err
			}

			restrictionFiles = append(restrictionFiles, rules)
		}

		nextPath, removedDir := removeLastDir(path)
		if nextPath == path || isGoModRoot(path) || removedDir == "src" {
			break
		}

		path = nextPath
	}

	return restrictionFiles, nil
}

func readFile(path string) (*FileFormat, error) {
	currentBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't read %v: %w", path, err)
	}

	var current FileFormat
	err = yaml.Unmarshal(currentBytes, &current)
	if err != nil {
		return nil, fmt.Errorf("couldn't unmarshal %v: %w", path, err)
	}
	current.path = path
	return &current, nil
}

// isGoModRoot checks if a directory is the root directory for a package
// by checking for the existence of a 'go.mod' file in that directory.
func isGoModRoot(path string) bool {
	_, err := os.Stat(filepath.Join(filepath.Dir(path), goModFile))
	return err == nil
}

// removeLastDir removes the last directory, but leaves the file name
// unchanged. It returns the new path and the removed directory. So:
// "a/b/c/file" -> ("a/b/file", "c")
func removeLastDir(path string) (newPath, removedDir string) {
	dir, file := filepath.Split(path)
	dir = strings.TrimSuffix(dir, string(filepath.Separator))
	return filepath.Join(filepath.Dir(dir), file), filepath.Base(dir)
}

func (boss *ImportBoss) verifyRules(pkg *packages.Package, restrictionFiles []*FileFormat) []error {
	klog.V(3).Infof("verifying pkg %q rules", pkg.PkgPath)

	// compile all Selector regex in all restriction files
	selectors := make([][]*regexp.Regexp, len(restrictionFiles))
	for i, restrictionFile := range restrictionFiles {
		for _, r := range restrictionFile.Rules {
			re, err := regexp.Compile(r.SelectorRegexp)
			if err != nil {
				return []error{
					fmt.Errorf("regexp `%s` in file %q doesn't compile: %w", r.SelectorRegexp, restrictionFile.path, err),
				}
			}

			selectors[i] = append(selectors[i], re)
		}
	}

	realPkgPath := unmassage(pkg.PkgPath)

	direct, indirect := transitiveImports(pkg)
	isDirect := map[string]bool{}
	for _, imp := range direct {
		isDirect[imp] = true
	}
	relate := func(imp string) string {
		if isDirect[imp] {
			return "->"
		}
		return "-->"
	}

	var errs []error
	for _, imp := range uniq(direct, indirect) {
		if unmassage(imp) == realPkgPath {
			// Tests in package "foo_test" depend on the test package for
			// "foo" (if both exist in a giver directory).
			continue
		}
		klog.V(4).Infof("considering import %q %s %q", pkg.PkgPath, relate(imp), imp)
		matched := false
		decided := false
		for i, file := range restrictionFiles {
			klog.V(4).Infof("rules file %s", file.path)
			for j, rule := range file.Rules {
				if !rule.Transitive && !isDirect[imp] {
					continue
				}
				matching := selectors[i][j].MatchString(imp)
				if !matching {
					continue
				}
				matched = true
				klog.V(6).Infof("selector %v matches %q", rule.SelectorRegexp, imp)

				disp := rule.Evaluate(imp)
				if disp == DepAllowed {
					decided = true
					break // no further rules, next file
				} else if disp == DepForbidden {
					errs = append(errs, fmt.Errorf("%q %s %q is forbidden by %s", pkg.PkgPath, relate(imp), imp, file.path))
					decided = true
					break // no further rules, next file
				}
			}
			if decided {
				break // no further files, next import
			}
		}
		if matched && !decided {
			klog.V(5).Infof("%q %s %q did not match any rule", pkg, relate(imp), imp)
			errs = append(errs, fmt.Errorf("%q %s %q did not match any rule", pkg.PkgPath, relate(imp), imp))
		}
	}

	if len(errs) > 0 {
		return errs
	}

	return nil
}

func uniq(slices ...[]string) []string {
	m := map[string]bool{}
	for _, sl := range slices {
		for _, str := range sl {
			m[str] = true
		}
	}
	ret := []string{}
	for str := range m {
		ret = append(ret, str)
	}
	sort.Strings(ret)
	return ret
}

func hasPathPrefix(path, prefix string) bool {
	if prefix == "" || path == prefix {
		return true
	}
	if !strings.HasSuffix(path, string(filepath.Separator)) {
		prefix += string(filepath.Separator)
	}
	return strings.HasPrefix(path, prefix)
}

func transitiveImports(pkg *packages.Package) ([]string, []string) {
	direct := []string{}
	indirect := []string{}
	seen := map[string]bool{}
	for _, imp := range pkg.Imports {
		direct = append(direct, imp.PkgPath)
		dfsImports(&indirect, seen, imp)
	}
	return direct, indirect
}

func dfsImports(dest *[]string, seen map[string]bool, p *packages.Package) {
	for _, p2 := range p.Imports {
		if seen[p2.PkgPath] {
			continue
		}
		seen[p2.PkgPath] = true
		*dest = append(*dest, p2.PkgPath)
		dfsImports(dest, seen, p2)
	}
}

// verifyInverseRules checks that all packages that import a package are allowed to import it.
func (boss *ImportBoss) verifyInverseRules(pkg *packages.Package, restrictionFiles []*FileFormat) []error {
	klog.V(3).Infof("verifying pkg %q inverse-rules", pkg.PkgPath)

	// compile all Selector regex in all restriction files
	selectors := make([][]*regexp.Regexp, len(restrictionFiles))
	for i, restrictionFile := range restrictionFiles {
		for _, r := range restrictionFile.InverseRules {
			re, err := regexp.Compile(r.SelectorRegexp)
			if err != nil {
				return []error{
					fmt.Errorf("regexp `%s` in file %q doesn't compile: %w", r.SelectorRegexp, restrictionFile.path, err),
				}
			}

			selectors[i] = append(selectors[i], re)
		}
	}

	realPkgPath := unmassage(pkg.PkgPath)

	isDirect := map[string]bool{}
	for _, imp := range boss.incomingImports[pkg.PkgPath] {
		isDirect[imp] = true
	}
	relate := func(imp string) string {
		if isDirect[imp] {
			return "<-"
		}
		return "<--"
	}

	var errs []error
	for _, imp := range boss.transitiveIncomingImports[pkg.PkgPath] {
		if unmassage(imp) == realPkgPath {
			// Tests in package "foo_test" depend on the test package for
			// "foo" (if both exist in a giver directory).
			continue
		}
		klog.V(4).Infof("considering import %q %s %q", pkg.PkgPath, relate(imp), imp)
		matched := false
		decided := false
		for i, file := range restrictionFiles {
			klog.V(4).Infof("rules file %s", file.path)
			for j, rule := range file.InverseRules {
				if !rule.Transitive && !isDirect[imp] {
					continue
				}
				matching := selectors[i][j].MatchString(imp)
				if !matching {
					continue
				}
				matched = true
				klog.V(6).Infof("selector %v matches %q", rule.SelectorRegexp, imp)

				disp := rule.Evaluate(imp)
				if disp == DepAllowed {
					decided = true
					break // no further rules, next file
				} else if disp == DepForbidden {
					errs = append(errs, fmt.Errorf("%q %s %q is forbidden by %s", pkg.PkgPath, relate(imp), imp, file.path))
					decided = true
					break // no further rules, next file
				}
			}
			if decided {
				break // no further files, next import
			}
		}
		if matched && !decided {
			klog.V(5).Infof("%q %s %q did not match any rule", pkg.PkgPath, relate(imp), imp)
			errs = append(errs, fmt.Errorf("%q %s %q did not match any rule", pkg.PkgPath, relate(imp), imp))
		}
	}

	if len(errs) > 0 {
		return errs
	}

	return nil
}

func transitiveClosure(in map[string][]string) map[string][]string {
	type edge struct {
		from string
		to   string
	}

	adj := make(map[edge]bool)
	imports := make(map[string]struct{})
	for from, tos := range in {
		for _, to := range tos {
			adj[edge{from, to}] = true
			imports[to] = struct{}{}
		}
	}

	// Warshal's algorithm
	for k := range in {
		for i := range in {
			if !adj[edge{i, k}] {
				continue
			}
			for j := range imports {
				if adj[edge{i, j}] {
					continue
				}
				if adj[edge{k, j}] {
					adj[edge{i, j}] = true
				}
			}
		}
	}

	out := make(map[string][]string, len(in))
	for i := range in {
		for j := range imports {
			if adj[edge{i, j}] {
				out[i] = append(out[i], j)
			}
		}

		sort.Strings(out[i])
	}

	return out
}
