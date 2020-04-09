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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"

	"github.com/bazelbuild/buildtools/build"

	"k8s.io/klog/v2"
)

const (
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
		klog.Fatalf("kazel no longer supports managing Go rules")
	}

	wroteGenerated := false
	if wroteGenerated, err = v.walkGenerated(); err != nil {
		klog.Fatalf("err walking generated: %v", err)
	}
	if _, err = v.walkSource("."); err != nil {
		klog.Fatalf("err walking source: %v", err)
	}
	written := 0
	if written, err = v.reconcileAllRules(); err != nil {
		klog.Fatalf("err reconciling rules: %v", err)
	}
	if wroteGenerated {
		written++
	}
	if *validate && written > 0 {
		fmt.Fprintf(os.Stderr, "\n%d BUILD files not up-to-date.\n", written)
		os.Exit(1)
	}
}

// Vendorer collects context, configuration, and cache while walking the tree.
type Vendorer struct {
	skippedPaths           []*regexp.Regexp
	skippedK8sCodegenPaths []*regexp.Regexp
	dryRun                 bool
	root                   string
	cfg                    *Cfg
	newRules               map[string][]*build.Rule // package path -> list of rules to add or update
	managedAttrs           []string                 // which rule attributes kazel will overwrite
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
		dryRun:       dryRun,
		root:         absRoot,
		cfg:          cfg,
		newRules:     make(map[string][]*build.Rule),
		managedAttrs: []string{"srcs"},
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

	sop, err := compileSkippedPaths(cfg.SkippedK8sCodegenPaths)
	if err != nil {
		return nil, err
	}
	v.skippedK8sCodegenPaths = append(sop, sp...)

	return &v, nil

}

func writeRules(file *build.File, rules []*build.Rule) {
	for _, rule := range rules {
		file.Stmt = append(file.Stmt, rule.Call)
	}
}

func (v *Vendorer) addRules(pkgPath string, rules []*build.Rule) {
	cleanPath := filepath.Clean(pkgPath)
	v.newRules[cleanPath] = append(v.newRules[cleanPath], rules...)
}

func (v *Vendorer) reconcileAllRules() (int, error) {
	var paths []string
	for path := range v.newRules {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	written := 0
	for _, path := range paths {
		w, err := ReconcileRules(path, v.newRules[path], v.managedAttrs, v.dryRun)
		if w {
			written++
		}
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// addCommentBefore adds a whole-line comment before the provided Expr.
func addCommentBefore(e build.Expr, comment string) {
	c := e.Comment()
	c.Before = append(c.Before, build.Comment{Token: fmt.Sprintf("# %s", comment)})
}

// varExpr creates a variable expression of the form "name = expr".
// v will be converted into an appropriate Expr using asExpr.
// The optional description will be included as a comment before the expression.
func varExpr(name, desc string, v interface{}) build.Expr {
	e := &build.BinaryExpr{
		X:  &build.LiteralExpr{Token: name},
		Op: "=",
		Y:  asExpr(v),
	}
	if desc != "" {
		addCommentBefore(e, desc)
	}
	return e
}

// rvSliceLessFunc returns a function that can be used with sort.Slice() or sort.SliceStable()
// to sort a slice of reflect.Values.
// It sorts ints and floats as their native kinds, and everything else as a string.
func rvSliceLessFunc(k reflect.Kind, vs []reflect.Value) func(int, int) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return func(i, j int) bool { return vs[i].Int() < vs[j].Int() }
	case reflect.Float32, reflect.Float64:
		return func(i, j int) bool { return vs[i].Float() < vs[j].Float() }
	default:
		return func(i, j int) bool {
			return fmt.Sprintf("%v", vs[i]) < fmt.Sprintf("%v", vs[j])
		}
	}
}

// asExpr converts a native Go type into the equivalent Starlark expression using reflection.
// The keys of maps will be sorted for reproducibility.
func asExpr(e interface{}) build.Expr {
	rv := reflect.ValueOf(e)
	switch rv.Kind() {
	case reflect.Bool:
		return &build.LiteralExpr{Token: fmt.Sprintf("%t", e)}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &build.LiteralExpr{Token: fmt.Sprintf("%d", e)}
	case reflect.Float32, reflect.Float64:
		return &build.LiteralExpr{Token: fmt.Sprintf("%g", e)}
	case reflect.String:
		return &build.StringExpr{Value: e.(string)}
	case reflect.Slice, reflect.Array:
		var list []build.Expr
		for i := 0; i < rv.Len(); i++ {
			list = append(list, asExpr(rv.Index(i).Interface()))
		}
		return &build.ListExpr{List: list}
	case reflect.Map:
		var list []build.Expr
		keys := rv.MapKeys()
		sort.SliceStable(keys, rvSliceLessFunc(rv.Type().Key().Kind(), keys))
		for _, key := range keys {
			list = append(list, &build.KeyValueExpr{
				Key:   asExpr(key.Interface()),
				Value: asExpr(rv.MapIndex(key).Interface()),
			})
		}
		return &build.DictExpr{List: list}
	default:
		klog.Fatalf("unhandled kind: %q for value: %q", rv.Kind(), rv)
		return nil
	}
}

func newRule(rt, name string, attrs map[string]build.Expr) *build.Rule {
	rule := &build.Rule{
		Call: &build.CallExpr{
			X: &build.LiteralExpr{Token: rt},
		},
	}
	rule.SetAttr("name", asExpr(name))
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
	return false, filepath.Join(pkgPath, "BUILD.bazel")
}

// ReconcileRules reconciles, simplifies, and writes the rules for the specified package, adding
// additional dependency rules as needed.
func ReconcileRules(pkgPath string, rules []*build.Rule, managedAttrs []string, dryRun bool) (bool, error) {
	_, path := findBuildFile(pkgPath)
	info, err := os.Stat(path)
	if err != nil && os.IsNotExist(err) {
		f := &build.File{}
		writeRules(f, rules)
		return writeFile(path, f, nil, false, dryRun)
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
	f, err := build.Parse(path, b)
	if err != nil {
		return false, err
	}
	oldRules := make(map[string]*build.Rule)
	for _, r := range f.Rules("") {
		oldRules[r.Name()] = r
	}
	for _, r := range rules {
		o, ok := oldRules[r.Name()]
		if !ok {
			f.Stmt = append(f.Stmt, r.Call)
			continue
		}
		if !RuleIsManaged(o) {
			continue
		}
		reconcileAttr := func(o, n *build.Rule, name string) {
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
		if !RuleIsManaged(r) {
			continue
		}
		f.DelRules(r.Kind(), r.Name())
	}

	return writeFile(path, f, nil, true, dryRun)
}

// RuleIsManaged returns whether the provided rule is managed by this tool,
// based on the tags set on the rule.
func RuleIsManaged(r *build.Rule) bool {
	for _, tag := range r.AttrStrings("tags") {
		if tag == automanagedTag {
			return true
		}
	}
	return false
}

// writeFile writes out f to path, prepending boilerplate to the output.
// If exists is true, compares against the existing file specified by path,
// returning false if there are no changes.
// Otherwise, returns true.
// If dryRun is false, no files are actually changed; otherwise, the file will be written.
func writeFile(path string, f *build.File, boilerplate []byte, exists, dryRun bool) (bool, error) {
	var info build.RewriteInfo
	build.Rewrite(f, &info)
	var out []byte
	out = append(out, boilerplate...)
	// double format the source file as our modification logic sometimes uses
	// LiteralExpr where it should use other types of expressions, and this
	// prevents issues where kazel thus formats structures incorrectly.
	// :this_is_fine:
	outData := build.Format(f)
	var err error
	f, err = build.Parse(path, outData)
	if err != nil {
		return false, fmt.Errorf("internal error occurred formatting file: %v", err)
	}
	// also call Rewrite again to run Buildifier against the results as
	// visibility rules are not ordered correctly for some reason
	build.Rewrite(f, &info)
	out = append(out, build.Format(f)...)
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
