/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"

	bzl "github.com/bazelbuild/buildtools/build"

	"k8s.io/klog"
)

const (
	vendorPath     = "vendor/"
	automanagedTag = "automanaged"
)

var (
	root      = flag.String("root", ".", "root of go source")
	dryRun    = flag.Bool("dry-run", false, "run in dry mode")
	printDiff = flag.Bool("print-diff", false, "print diff to stdout")
	validate  = flag.Bool("validate", false, "run in dry mode and exit nonzero if any BUILD files need to be updated")
	cfgPath   = flag.String("cfg-path", ".kazelcfg.json", "path to kazel config (relative paths interpreted relative to -repo.")
)

func main() {
	flag.Parse()
	flag.Set("alsologtostderr", "true")
	if *root == "" {
		klog.Fatalf("-root argument is required")
	}
	if *validate {
		*dryRun = true
	}
	v, err := newVendorer(*root, *cfgPath, *dryRun)
	if err != nil {
		klog.Fatalf("unable to build vendorer: %v", err)
	}
	if err = os.Chdir(v.root); err != nil {
		klog.Fatalf("cannot chdir into root %q: %v", v.root, err)
	}

	if v.cfg.ManageGoRules {
		if err = v.walkVendor(); err != nil {
			klog.Fatalf("err walking vendor: %v", err)
		}
		if err = v.walkRepo(); err != nil {
			klog.Fatalf("err walking repo: %v", err)
		}
	}
	if err = v.walkGenerated(); err != nil {
		klog.Fatalf("err walking generated: %v", err)
	}
	if _, err = v.walkSource("."); err != nil {
		klog.Fatalf("err walking source: %v", err)
	}
	written := 0
	if written, err = v.reconcileAllRules(); err != nil {
		klog.Fatalf("err reconciling rules: %v", err)
	}
	if *validate && written > 0 {
		fmt.Fprintf(os.Stderr, "\n%d BUILD files not up-to-date.\n", written)
		os.Exit(1)
	}
}

// Vendorer collects context, configuration, and cache while walking the tree.
type Vendorer struct {
	ctx                 *build.Context
	icache              map[icacheKey]icacheVal
	skippedPaths        []*regexp.Regexp
	skippedOpenAPIPaths []*regexp.Regexp
	dryRun              bool
	root                string
	cfg                 *Cfg
	newRules            map[string][]*bzl.Rule // package path -> list of rules to add or update
	managedAttrs        []string
}

func newVendorer(root, cfgPath string, dryRun bool) (*Vendorer, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("could not get absolute path: %v", err)
	}
	if !filepath.IsAbs(cfgPath) {
		cfgPath = filepath.Join(absRoot, cfgPath)
	}
	cfg, err := ReadCfg(cfgPath)
	if err != nil {
		return nil, err
	}

	v := Vendorer{
		ctx:          context(),
		dryRun:       dryRun,
		root:         absRoot,
		icache:       map[icacheKey]icacheVal{},
		cfg:          cfg,
		newRules:     make(map[string][]*bzl.Rule),
		managedAttrs: []string{"srcs", "deps", "library"},
	}

	builtIn, err := compileSkippedPaths([]string{"^\\.git", "^bazel-*"})
	if err != nil {
		return nil, err
	}

	sp, err := compileSkippedPaths(cfg.SkippedPaths)
	if err != nil {
		return nil, err
	}
	sp = append(builtIn, sp...)
	v.skippedPaths = sp

	sop, err := compileSkippedPaths(cfg.SkippedOpenAPIGenPaths)
	if err != nil {
		return nil, err
	}
	v.skippedOpenAPIPaths = append(sop, sp...)

	return &v, nil

}

type icacheKey struct {
	path, srcDir string
}

type icacheVal struct {
	pkg *build.Package
	err error
}

func (v *Vendorer) importPkg(path string, srcDir string) (*build.Package, error) {
	k := icacheKey{path: path, srcDir: srcDir}
	if val, ok := v.icache[k]; ok {
		return val.pkg, val.err
	}

	// cache miss
	pkg, err := v.ctx.Import(path, srcDir, build.ImportComment)
	v.icache[k] = icacheVal{pkg: pkg, err: err}
	return pkg, err
}

func writeHeaders(file *bzl.File) {
	pkgRule := bzl.Rule{
		Call: &bzl.CallExpr{
			X: &bzl.LiteralExpr{Token: "package"},
		},
	}
	pkgRule.SetAttr("default_visibility", asExpr([]string{"//visibility:public"}))

	file.Stmt = append(file.Stmt,
		[]bzl.Expr{
			pkgRule.Call,
			&bzl.CallExpr{
				X: &bzl.LiteralExpr{Token: "load"},
				List: asExpr([]string{
					"@io_bazel_rules_go//go:def.bzl",
				}).(*bzl.ListExpr).List,
			},
		}...,
	)
}

func writeRules(file *bzl.File, rules []*bzl.Rule) {
	for _, rule := range rules {
		file.Stmt = append(file.Stmt, rule.Call)
	}
}

func (v *Vendorer) resolve(ipath string) Label {
	if ipath == v.cfg.GoPrefix {
		return Label{
			tag: "go_default_library",
		}
	} else if strings.HasPrefix(ipath, v.cfg.GoPrefix) {
		return Label{
			pkg: strings.TrimPrefix(ipath, v.cfg.GoPrefix+"/"),
			tag: "go_default_library",
		}
	}
	if v.cfg.VendorMultipleBuildFiles {
		return Label{
			pkg: "vendor/" + ipath,
			tag: "go_default_library",
		}
	}
	return Label{
		pkg: "vendor",
		tag: ipath,
	}
}

func (v *Vendorer) walk(root string, f func(path, ipath string, pkg *build.Package) error) error {
	skipVendor := true
	if root == vendorPath {
		skipVendor = false
	}
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return nil
		}
		if skipVendor && strings.HasPrefix(path, vendorPath) {
			return filepath.SkipDir
		}
		for _, r := range v.skippedPaths {
			if r.MatchString(path) {
				return filepath.SkipDir
			}
		}
		ipath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		pkg, err := v.importPkg(".", filepath.Join(v.root, path))
		if err != nil {
			if _, ok := err.(*build.NoGoError); err != nil && ok {
				return nil
			}
			return err
		}

		return f(path, ipath, pkg)
	})
}

func (v *Vendorer) walkRepo() error {
	for _, root := range v.cfg.SrcDirs {
		if err := v.walk(root, v.updatePkg); err != nil {
			return err
		}
	}
	return nil
}

func (v *Vendorer) updateSinglePkg(path string) error {
	pkg, err := v.importPkg(".", "./"+path)
	if err != nil {
		if _, ok := err.(*build.NoGoError); err != nil && ok {
			return nil
		}
		return err
	}
	return v.updatePkg(path, "", pkg)
}

type ruleType int

// The RuleType* constants enumerate the bazel rules supported by this tool.
const (
	RuleTypeGoBinary ruleType = iota
	RuleTypeGoLibrary
	RuleTypeGoTest
	RuleTypeGoXTest
	RuleTypeCGoGenrule
	RuleTypeFileGroup
	RuleTypeOpenAPILibrary
)

// RuleKind converts a value of the RuleType* enum into the BUILD string.
func (rt ruleType) RuleKind() string {
	switch rt {
	case RuleTypeGoBinary:
		return "go_binary"
	case RuleTypeGoLibrary:
		return "go_library"
	case RuleTypeGoTest:
		return "go_test"
	case RuleTypeGoXTest:
		return "go_test"
	case RuleTypeCGoGenrule:
		return "cgo_genrule"
	case RuleTypeFileGroup:
		return "filegroup"
	case RuleTypeOpenAPILibrary:
		return "openapi_library"
	}
	panic("unreachable")
}

// NamerFunc is a function that returns the appropriate name for the rule for the provided RuleType.
type NamerFunc func(ruleType) string

func (v *Vendorer) updatePkg(path, _ string, pkg *build.Package) error {

	srcNameMap := func(srcs ...[]string) *bzl.ListExpr {
		return asExpr(merge(srcs...)).(*bzl.ListExpr)
	}

	srcs := srcNameMap(pkg.GoFiles, pkg.SFiles)
	cgoSrcs := srcNameMap(pkg.CgoFiles, pkg.CFiles, pkg.CXXFiles, pkg.HFiles)
	testSrcs := srcNameMap(pkg.TestGoFiles)
	xtestSrcs := srcNameMap(pkg.XTestGoFiles)

	v.addRules(path, v.emit(srcs, cgoSrcs, testSrcs, xtestSrcs, pkg, func(rt ruleType) string {
		switch rt {
		case RuleTypeGoBinary:
			return filepath.Base(pkg.Dir)
		case RuleTypeGoLibrary:
			return "go_default_library"
		case RuleTypeGoTest:
			return "go_default_test"
		case RuleTypeGoXTest:
			return "go_default_xtest"
		case RuleTypeCGoGenrule:
			return "cgo_codegen"
		}
		panic("unreachable")
	}))

	return nil
}

func (v *Vendorer) emit(srcs, cgoSrcs, testSrcs, xtestSrcs *bzl.ListExpr, pkg *build.Package, namer NamerFunc) []*bzl.Rule {
	var goLibAttrs = make(Attrs)
	var rules []*bzl.Rule

	deps := v.extractDeps(pkg.Imports)

	if len(srcs.List) >= 0 {
		goLibAttrs.Set("srcs", srcs)
	} else if len(cgoSrcs.List) == 0 {
		return nil
	}

	if len(deps.List) > 0 {
		goLibAttrs.SetList("deps", deps)
	}

	if pkg.IsCommand() {
		rules = append(rules, newRule(RuleTypeGoBinary, namer, map[string]bzl.Expr{
			"library": asExpr(":" + namer(RuleTypeGoLibrary)),
		}))
	}

	addGoDefaultLibrary := len(cgoSrcs.List) > 0 || len(srcs.List) > 0
	if len(cgoSrcs.List) != 0 {
		cgoRuleAttrs := make(Attrs)

		cgoRuleAttrs.SetList("srcs", cgoSrcs)
		cgoRuleAttrs.SetList("clinkopts", asExpr([]string{"-lz", "-lm", "-lpthread", "-ldl"}).(*bzl.ListExpr))

		rules = append(rules, newRule(RuleTypeCGoGenrule, namer, cgoRuleAttrs))

		goLibAttrs.Set("library", asExpr(":"+namer(RuleTypeCGoGenrule)))
	}

	if len(testSrcs.List) != 0 {
		testRuleAttrs := make(Attrs)

		testRuleAttrs.SetList("srcs", testSrcs)
		testRuleAttrs.SetList("deps", v.extractDeps(pkg.TestImports))

		if addGoDefaultLibrary {
			testRuleAttrs.Set("library", asExpr(":"+namer(RuleTypeGoLibrary)))
		}
		rules = append(rules, newRule(RuleTypeGoTest, namer, testRuleAttrs))
	}

	if addGoDefaultLibrary {
		rules = append(rules, newRule(RuleTypeGoLibrary, namer, goLibAttrs))
	}

	if len(xtestSrcs.List) != 0 {
		xtestRuleAttrs := make(Attrs)

		xtestRuleAttrs.SetList("srcs", xtestSrcs)
		xtestRuleAttrs.SetList("deps", v.extractDeps(pkg.XTestImports))

		rules = append(rules, newRule(RuleTypeGoXTest, namer, xtestRuleAttrs))
	}

	return rules
}

func (v *Vendorer) addRules(pkgPath string, rules []*bzl.Rule) {
	cleanPath := filepath.Clean(pkgPath)
	v.newRules[cleanPath] = append(v.newRules[cleanPath], rules...)
}

func (v *Vendorer) walkVendor() error {
	var rules []*bzl.Rule
	updateFunc := func(path, ipath string, pkg *build.Package) error {
		srcNameMap := func(srcs ...[]string) *bzl.ListExpr {
			return asExpr(
				apply(
					merge(srcs...),
					mapper(func(s string) string {
						return strings.TrimPrefix(filepath.Join(path, s), "vendor/")
					}),
				),
			).(*bzl.ListExpr)
		}

		srcs := srcNameMap(pkg.GoFiles, pkg.SFiles)
		cgoSrcs := srcNameMap(pkg.CgoFiles, pkg.CFiles, pkg.CXXFiles, pkg.HFiles)
		testSrcs := srcNameMap(pkg.TestGoFiles)
		xtestSrcs := srcNameMap(pkg.XTestGoFiles)

		tagBase := v.resolve(ipath).tag

		rules = append(rules, v.emit(srcs, cgoSrcs, testSrcs, xtestSrcs, pkg, func(rt ruleType) string {
			switch rt {
			case RuleTypeGoBinary:
				return tagBase + "_bin"
			case RuleTypeGoLibrary:
				return tagBase
			case RuleTypeGoTest:
				return tagBase + "_test"
			case RuleTypeGoXTest:
				return tagBase + "_xtest"
			case RuleTypeCGoGenrule:
				return tagBase + "_cgo"
			}
			panic("unreachable")
		})...)

		return nil
	}
	if v.cfg.VendorMultipleBuildFiles {
		updateFunc = v.updatePkg
	}
	if err := v.walk(vendorPath, updateFunc); err != nil {
		return err
	}
	v.addRules(vendorPath, rules)

	return nil
}

func (v *Vendorer) extractDeps(deps []string) *bzl.ListExpr {
	return asExpr(
		apply(
			merge(deps),
			filterer(func(s string) bool {
				pkg, err := v.importPkg(s, v.root)
				if err != nil {
					if strings.Contains(err.Error(), `cannot find package "C"`) ||
						// added in go1.7
						strings.Contains(err.Error(), `cannot find package "context"`) ||
						strings.Contains(err.Error(), `cannot find package "net/http/httptrace"`) {
						return false
					}
					fmt.Fprintf(os.Stderr, "extract err: %v\n", err)
					return false
				}
				if pkg.Goroot {
					return false
				}
				return true
			}),
			mapper(func(s string) string {
				return v.resolve(s).String()
			}),
		),
	).(*bzl.ListExpr)
}

func (v *Vendorer) reconcileAllRules() (int, error) {
	var paths []string
	for path := range v.newRules {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	written := 0
	for _, path := range paths {
		w, err := ReconcileRules(path, v.newRules[path], v.managedAttrs, v.dryRun, v.cfg.ManageGoRules)
		if w {
			written++
		}
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// Attrs collects the attributes for a rule.
type Attrs map[string]bzl.Expr

// Set sets the named attribute to the provided bazel expression.
func (a Attrs) Set(name string, expr bzl.Expr) {
	a[name] = expr
}

// SetList sets the named attribute to the provided bazel expression list.
func (a Attrs) SetList(name string, expr *bzl.ListExpr) {
	if len(expr.List) == 0 {
		return
	}
	a[name] = expr
}

// Label defines a bazel label.
type Label struct {
	pkg, tag string
}

func (l Label) String() string {
	return fmt.Sprintf("//%v:%v", l.pkg, l.tag)
}

func asExpr(e interface{}) bzl.Expr {
	rv := reflect.ValueOf(e)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &bzl.LiteralExpr{Token: fmt.Sprintf("%d", e)}
	case reflect.Float32, reflect.Float64:
		return &bzl.LiteralExpr{Token: fmt.Sprintf("%f", e)}
	case reflect.String:
		return &bzl.StringExpr{Value: e.(string)}
	case reflect.Slice, reflect.Array:
		var list []bzl.Expr
		for i := 0; i < rv.Len(); i++ {
			list = append(list, asExpr(rv.Index(i).Interface()))
		}
		return &bzl.ListExpr{List: list}
	default:
		klog.Fatalf("Uh oh")
		return nil
	}
}

type sed func(s []string) []string

func mapString(in []string, f func(string) string) []string {
	var out []string
	for _, s := range in {
		out = append(out, f(s))
	}
	return out
}

func mapper(f func(string) string) sed {
	return func(in []string) []string {
		return mapString(in, f)
	}
}

func filterString(in []string, f func(string) bool) []string {
	var out []string
	for _, s := range in {
		if f(s) {
			out = append(out, s)
		}
	}
	return out
}

func filterer(f func(string) bool) sed {
	return func(in []string) []string {
		return filterString(in, f)
	}
}

func apply(stream []string, seds ...sed) []string {
	for _, sed := range seds {
		stream = sed(stream)
	}
	return stream
}

func merge(streams ...[]string) []string {
	var out []string
	for _, stream := range streams {
		out = append(out, stream...)
	}
	return out
}

func newRule(rt ruleType, namer NamerFunc, attrs map[string]bzl.Expr) *bzl.Rule {
	rule := &bzl.Rule{
		Call: &bzl.CallExpr{
			X: &bzl.LiteralExpr{Token: rt.RuleKind()},
		},
	}
	rule.SetAttr("name", asExpr(namer(rt)))
	for k, v := range attrs {
		rule.SetAttr(k, v)
	}
	rule.SetAttr("tags", asExpr([]string{automanagedTag}))
	return rule
}

// findBuildFile determines the name of a preexisting BUILD file, returning
// a default if no such file exists.
func findBuildFile(pkgPath string) (bool, string) {
	options := []string{"BUILD.bazel", "BUILD"}
	for _, b := range options {
		path := filepath.Join(pkgPath, b)
		info, err := os.Stat(path)
		if err == nil && !info.IsDir() {
			return true, path
		}
	}
	return false, filepath.Join(pkgPath, "BUILD")
}

// ReconcileRules reconciles, simplifies, and writes the rules for the specified package, adding
// additional dependency rules as needed.
func ReconcileRules(pkgPath string, rules []*bzl.Rule, managedAttrs []string, dryRun bool, manageGoRules bool) (bool, error) {
	_, path := findBuildFile(pkgPath)
	info, err := os.Stat(path)
	if err != nil && os.IsNotExist(err) {
		f := &bzl.File{}
		writeHeaders(f)
		if manageGoRules {
			reconcileLoad(f, rules)
		}
		writeRules(f, rules)
		return writeFile(path, f, false, dryRun)
	} else if err != nil {
		return false, err
	}
	if info.IsDir() {
		return false, fmt.Errorf("%q cannot be a directory", path)
	}
	b, err := ioutil.ReadFile(path)
	if err != nil {
		return false, err
	}
	f, err := bzl.Parse(path, b)
	if err != nil {
		return false, err
	}
	oldRules := make(map[string]*bzl.Rule)
	for _, r := range f.Rules("") {
		oldRules[r.Name()] = r
	}
	for _, r := range rules {
		o, ok := oldRules[r.Name()]
		if !ok {
			f.Stmt = append(f.Stmt, r.Call)
			continue
		}
		if !RuleIsManaged(o, manageGoRules) {
			continue
		}
		reconcileAttr := func(o, n *bzl.Rule, name string) {
			if e := n.Attr(name); e != nil {
				o.SetAttr(name, e)
			} else {
				o.DelAttr(name)
			}
		}
		for _, attr := range managedAttrs {
			reconcileAttr(o, r, attr)
		}
		delete(oldRules, r.Name())
	}

	for _, r := range oldRules {
		if !RuleIsManaged(r, manageGoRules) {
			continue
		}
		f.DelRules(r.Kind(), r.Name())
	}
	if manageGoRules {
		reconcileLoad(f, f.Rules(""))
	}

	return writeFile(path, f, true, dryRun)
}

func reconcileLoad(f *bzl.File, rules []*bzl.Rule) {
	usedRuleKindsMap := map[string]bool{}
	for _, r := range rules {
		// Select only the Go rules we need to import, excluding builtins like filegroup.
		// TODO: make less fragile
		switch r.Kind() {
		case "go_prefix", "go_library", "go_binary", "go_test", "go_proto_library", "cgo_genrule", "cgo_library":
			usedRuleKindsMap[r.Kind()] = true
		}
	}

	usedRuleKindsList := []string{}
	for k := range usedRuleKindsMap {
		usedRuleKindsList = append(usedRuleKindsList, k)
	}
	sort.Strings(usedRuleKindsList)

	for _, r := range f.Rules("load") {
		const goRulesLabel = "@io_bazel_rules_go//go:def.bzl"
		args := bzl.Strings(&bzl.ListExpr{List: r.Call.List})
		if len(args) == 0 {
			continue
		}
		if args[0] != goRulesLabel {
			continue
		}
		if len(usedRuleKindsList) == 0 {
			f.DelRules(r.Kind(), r.Name())
			continue
		}
		r.Call.List = asExpr(append(
			[]string{goRulesLabel}, usedRuleKindsList...,
		)).(*bzl.ListExpr).List
		break
	}
}

// RuleIsManaged returns whether the provided rule is managed by this tool,
// based on the tags set on the rule.
func RuleIsManaged(r *bzl.Rule, manageGoRules bool) bool {
	var automanaged bool
	if !manageGoRules && (strings.HasPrefix(r.Kind(), "go_") || strings.HasPrefix(r.Kind(), "cgo_")) {
		return false
	}
	for _, tag := range r.AttrStrings("tags") {
		if tag == automanagedTag {
			automanaged = true
			break
		}
	}
	return automanaged
}

func writeFile(path string, f *bzl.File, exists, dryRun bool) (bool, error) {
	var info bzl.RewriteInfo
	bzl.Rewrite(f, &info)
	out := bzl.Format(f)
	if exists {
		orig, err := ioutil.ReadFile(path)
		if err != nil {
			return false, err
		}
		if bytes.Compare(orig, out) == 0 {
			return false, nil
		}
		if *printDiff {
			Diff(orig, out)
		}
	}
	if dryRun {
		fmt.Fprintf(os.Stderr, "DRY-RUN: wrote %q\n", path)
		return true, nil
	}
	werr := ioutil.WriteFile(path, out, 0644)
	if werr == nil {
		fmt.Fprintf(os.Stderr, "wrote %q\n", path)
	}
	return werr == nil, werr
}

func context() *build.Context {
	return &build.Context{
		GOARCH:      "amd64",
		GOOS:        "linux",
		GOROOT:      build.Default.GOROOT,
		GOPATH:      build.Default.GOPATH,
		ReleaseTags: []string{"go1.1", "go1.2", "go1.3", "go1.4", "go1.5", "go1.6", "go1.7", "go1.8"},
		Compiler:    runtime.Compiler,
		CgoEnabled:  true,
	}
}

func walk(root string, walkFn filepath.WalkFunc) error {
	return nil
}

func compileSkippedPaths(skippedPaths []string) ([]*regexp.Regexp, error) {
	regexPaths := []*regexp.Regexp{}

	for _, sp := range skippedPaths {
		r, err := regexp.Compile(sp)
		if err != nil {
			return nil, err
		}
		regexPaths = append(regexPaths, r)
	}
	return regexPaths, nil
}
