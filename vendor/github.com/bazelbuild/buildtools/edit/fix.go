/*
Copyright 2016 Google Inc. All Rights Reserved.
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
// Functions to clean and fix BUILD files

package edit

import (
	"regexp"
	"sort"
	"strings"

	"github.com/bazelbuild/buildtools/build"
)

// splitOptionsWithSpaces is a cleanup function.
// It splits options strings that contain a space. This change
// should be safe as Blaze is splitting those strings, but we will
// eventually get rid of this misfeature.
//   eg. it converts from:
//     copts = ["-Dfoo -Dbar"]
//   to:
//     copts = ["-Dfoo", "-Dbar"]
func splitOptionsWithSpaces(_ *build.File, r *build.Rule, _ string) bool {
	var attrToRewrite = []string{
		"copts",
		"linkopts",
	}
	fixed := false
	for _, attrName := range attrToRewrite {
		attr := r.Attr(attrName)
		if attr != nil {
			for _, li := range AllLists(attr) {
				fixed = splitStrings(li) || fixed
			}
		}
	}
	return fixed
}

func splitStrings(list *build.ListExpr) bool {
	var all []build.Expr
	fixed := false
	for _, e := range list.List {
		str, ok := e.(*build.StringExpr)
		if !ok {
			all = append(all, e)
			continue
		}
		if strings.Contains(str.Value, " ") && !strings.Contains(str.Value, "'\"") {
			fixed = true
			for i, substr := range strings.Fields(str.Value) {
				item := &build.StringExpr{Value: substr}
				if i == 0 {
					item.Comments = str.Comments
				}
				all = append(all, item)
			}
		} else {
			all = append(all, str)
		}
	}
	list.List = all
	return fixed
}

// shortenLabels rewrites the labels in the rule using the short notation.
func shortenLabels(_ *build.File, r *build.Rule, pkg string) bool {
	fixed := false
	for _, attr := range r.AttrKeys() {
		e := r.Attr(attr)
		if !ContainsLabels(attr) {
			continue
		}
		for _, li := range AllLists(e) {
			for _, elem := range li.List {
				str, ok := elem.(*build.StringExpr)
				if ok && str.Value != ShortenLabel(str.Value, pkg) {
					str.Value = ShortenLabel(str.Value, pkg)
					fixed = true
				}
			}
		}
	}
	return fixed
}

// removeVisibility removes useless visibility attributes.
func removeVisibility(f *build.File, r *build.Rule, pkg string) bool {
	pkgDecl := PackageDeclaration(f)
	defaultVisibility := pkgDecl.AttrStrings("default_visibility")

	// If no default_visibility is given, it is implicitly private.
	if len(defaultVisibility) == 0 {
		defaultVisibility = []string{"//visibility:private"}
	}

	visibility := r.AttrStrings("visibility")
	if len(visibility) == 0 || len(visibility) != len(defaultVisibility) {
		return false
	}
	sort.Strings(defaultVisibility)
	sort.Strings(visibility)
	for i, vis := range visibility {
		if vis != defaultVisibility[i] {
			return false
		}
	}
	r.DelAttr("visibility")
	return true
}

// removeTestOnly removes the useless testonly attributes.
func removeTestOnly(f *build.File, r *build.Rule, pkg string) bool {
	pkgDecl := PackageDeclaration(f)

	def := strings.HasSuffix(r.Kind(), "_test") || r.Kind() == "test_suite"
	if !def {
		if pkgDecl.Attr("default_testonly") == nil {
			def = strings.HasPrefix(pkg, "javatests/")
		} else if pkgDecl.AttrLiteral("default_testonly") == "1" {
			def = true
		} else if pkgDecl.AttrLiteral("default_testonly") != "0" {
			// Non-literal value: it's not safe to do a change.
			return false
		}
	}

	testonly := r.AttrLiteral("testonly")
	if def && testonly == "1" {
		r.DelAttr("testonly")
		return true
	}
	if !def && testonly == "0" {
		r.DelAttr("testonly")
		return true
	}
	return false
}

func genruleRenameDepsTools(_ *build.File, r *build.Rule, _ string) bool {
	return r.Kind() == "genrule" && RenameAttribute(r, "deps", "tools") == nil
}

// explicitHeuristicLabels adds $(location ...) for each label in the string s.
func explicitHeuristicLabels(s string, labels map[string]bool) string {
	// Regexp comes from LABEL_CHAR_MATCHER in
	//   java/com/google/devtools/build/lib/analysis/LabelExpander.java
	re := regexp.MustCompile("[a-zA-Z0-9:/_.+-]+|[^a-zA-Z0-9:/_.+-]+")
	parts := re.FindAllString(s, -1)
	changed := false
	canChange := true
	for i, part := range parts {
		// We don't want to add $(location when it's already present.
		// So we skip the next label when we see location(s).
		if part == "location" || part == "locations" {
			canChange = false
		}
		if !labels[part] {
			if labels[":"+part] { // leading colon is often missing
				part = ":" + part
			} else {
				continue
			}
		}

		if !canChange {
			canChange = true
			continue
		}
		parts[i] = "$(location " + part + ")"
		changed = true
	}
	if changed {
		return strings.Join(parts, "")
	}
	return s
}

func addLabels(r *build.Rule, attr string, labels map[string]bool) {
	a := r.Attr(attr)
	if a == nil {
		return
	}
	for _, li := range AllLists(a) {
		for _, item := range li.List {
			if str, ok := item.(*build.StringExpr); ok {
				labels[str.Value] = true
			}
		}
	}
}

// genruleFixHeuristicLabels modifies the cmd attribute of genrules, so
// that they don't rely on heuristic label expansion anymore.
// Label expansion is made explicit with the $(location ...) command.
func genruleFixHeuristicLabels(_ *build.File, r *build.Rule, _ string) bool {
	if r.Kind() != "genrule" {
		return false
	}

	cmd := r.Attr("cmd")
	if cmd == nil {
		return false
	}
	labels := make(map[string]bool)
	addLabels(r, "tools", labels)
	addLabels(r, "srcs", labels)

	fixed := false
	for _, str := range AllStrings(cmd) {
		newVal := explicitHeuristicLabels(str.Value, labels)
		if newVal != str.Value {
			fixed = true
			str.Value = newVal
		}
	}
	return fixed
}

// sortExportsFiles sorts the first argument of exports_files if it is a list.
func sortExportsFiles(_ *build.File, r *build.Rule, _ string) bool {
	if r.Kind() != "exports_files" || len(r.Call.List) == 0 {
		return false
	}
	build.SortStringList(r.Call.List[0])
	return true
}

// removeVarref replaces all varref('x') with '$(x)'.
// The goal is to eventually remove varref from the build language.
func removeVarref(_ *build.File, r *build.Rule, _ string) bool {
	fixed := false
	EditFunction(r.Call, "varref", func(call *build.CallExpr, stk []build.Expr) build.Expr {
		if len(call.List) != 1 {
			return nil
		}
		str, ok := (call.List[0]).(*build.StringExpr)
		if !ok {
			return nil
		}
		fixed = true
		str.Value = "$(" + str.Value + ")"
		// Preserve suffix comments from the function call
		str.Comment().Suffix = append(str.Comment().Suffix, call.Comment().Suffix...)
		return str
	})
	return fixed
}

// sortGlob sorts the list argument to glob.
func sortGlob(_ *build.File, r *build.Rule, _ string) bool {
	fixed := false
	EditFunction(r.Call, "glob", func(call *build.CallExpr, stk []build.Expr) build.Expr {
		if len(call.List) == 0 {
			return nil
		}
		build.SortStringList(call.List[0])
		fixed = true
		return call
	})
	return fixed
}

func evaluateListConcatenation(expr build.Expr) build.Expr {
	if _, ok := expr.(*build.ListExpr); ok {
		return expr
	}
	bin, ok := expr.(*build.BinaryExpr)
	if !ok || bin.Op != "+" {
		return expr
	}
	li1, ok1 := evaluateListConcatenation(bin.X).(*build.ListExpr)
	li2, ok2 := evaluateListConcatenation(bin.Y).(*build.ListExpr)
	if !ok1 || !ok2 {
		return expr
	}
	res := *li1
	res.List = append(li1.List, li2.List...)
	return &res
}

// mergeLiteralLists evaluates the concatenation of two literal lists.
// e.g. [1, 2] + [3, 4]  ->  [1, 2, 3, 4]
func mergeLiteralLists(_ *build.File, r *build.Rule, _ string) bool {
	fixed := false
	build.Edit(r.Call, func(expr build.Expr, stk []build.Expr) build.Expr {
		newexpr := evaluateListConcatenation(expr)
		fixed = fixed || (newexpr != expr)
		return newexpr
	})
	return fixed
}

// usePlusEqual replaces uses of extend and append with the += operator.
// e.g. foo.extend(bar)  =>  foo += bar
//      foo.append(bar)  =>  foo += [bar]
func usePlusEqual(f *build.File) bool {
	fixed := false
	for i, stmt := range f.Stmt {
		call, ok := stmt.(*build.CallExpr)
		if !ok {
			continue
		}
		dot, ok := call.X.(*build.DotExpr)
		if !ok || len(call.List) != 1 {
			continue
		}
		obj, ok := dot.X.(*build.Ident)
		if !ok {
			continue
		}

		var fix *build.AssignExpr
		if dot.Name == "extend" {
			fix = &build.AssignExpr{LHS: obj, Op: "+=", RHS: call.List[0]}
		} else if dot.Name == "append" {
			list := &build.ListExpr{List: []build.Expr{call.List[0]}}
			fix = &build.AssignExpr{LHS: obj, Op: "+=", RHS: list}
		} else {
			continue
		}
		fix.Comments = call.Comments // Keep original comments
		f.Stmt[i] = fix
		fixed = true
	}
	return fixed
}

func isNonemptyComment(comment *build.Comments) bool {
	return len(comment.Before)+len(comment.Suffix)+len(comment.After) > 0
}

// Checks whether a load statement or any of its arguments have a comment
func hasComment(load *build.LoadStmt) bool {
	if isNonemptyComment(load.Comment()) {
		return true
	}
	if isNonemptyComment(load.Module.Comment()) {
		return true
	}

	for i := range load.From {
		if isNonemptyComment(load.From[i].Comment()) || isNonemptyComment(load.To[i].Comment()) {
			return true
		}
	}
	return false
}

// cleanUnusedLoads removes symbols from load statements that are not used in the file.
// It also cleans symbols loaded multiple times, sorts symbol list, and removes load
// statements when the list is empty.
func cleanUnusedLoads(f *build.File) bool {
	symbols := UsedSymbols(f)
	fixed := false

	var all []build.Expr
	for _, stmt := range f.Stmt {
		load, ok := stmt.(*build.LoadStmt)
		if !ok || hasComment(load) {
			all = append(all, stmt)
			continue
		}
		var fromSymbols, toSymbols []*build.Ident
		for i := range load.From {
			fromSymbol := load.From[i]
			toSymbol := load.To[i]
			if symbols[toSymbol.Name] {
				// The symbol is actually used
				fromSymbols = append(fromSymbols, fromSymbol)
				toSymbols = append(toSymbols, toSymbol)
				// If the same symbol is loaded twice, we'll remove it.
				delete(symbols, toSymbol.Name)
			} else {
				fixed = true
			}
		}
		if len(toSymbols) > 0 { // Keep the load statement if it loads at least one symbol.
			sort.Sort(loadArgs{fromSymbols, toSymbols})
			load.From = fromSymbols
			load.To = toSymbols
			all = append(all, load)
		} else {
			fixed = true
		}
	}
	f.Stmt = all
	return fixed
}

// movePackageDeclarationToTheTop ensures that the call to package() is done
// before everything else (except comments).
func movePackageDeclarationToTheTop(f *build.File) bool {
	pkg := ExistingPackageDeclaration(f)
	if pkg == nil {
		return false
	}
	all := []build.Expr{}
	inserted := false // true when the package declaration has been inserted
	for _, stmt := range f.Stmt {
		_, isComment := stmt.(*build.CommentBlock)
		_, isString := stmt.(*build.StringExpr)     // typically a docstring
		_, isAssignExpr := stmt.(*build.AssignExpr) // e.g. variable declaration
		_, isLoad := stmt.(*build.LoadStmt)
		if isComment || isString || isAssignExpr || isLoad {
			all = append(all, stmt)
			continue
		}
		if stmt == pkg.Call {
			if inserted {
				// remove the old package
				continue
			}
			return false // the file was ok
		}
		if !inserted {
			all = append(all, pkg.Call)
			inserted = true
		}
		all = append(all, stmt)
	}
	f.Stmt = all
	return true
}

// moveToPackage is an auxilliary function used by moveLicensesAndDistribs.
// The function shouldn't appear more than once in the file (depot cleanup has
// been done).
func moveToPackage(f *build.File, attrname string) bool {
	var all []build.Expr
	fixed := false
	for _, stmt := range f.Stmt {
		rule, ok := ExprToRule(stmt, attrname)
		if !ok || len(rule.Call.List) != 1 {
			all = append(all, stmt)
			continue
		}
		pkgDecl := PackageDeclaration(f)
		pkgDecl.SetAttr(attrname, rule.Call.List[0])
		pkgDecl.AttrDefn(attrname).Comments = *stmt.Comment()
		fixed = true
	}
	f.Stmt = all
	return fixed
}

// moveLicensesAndDistribs replaces the 'licenses' and 'distribs' functions
// with an attribute in package.
// Before:  licenses(["notice"])
// After:   package(licenses = ["notice"])
func moveLicensesAndDistribs(f *build.File) bool {
	fixed1 := moveToPackage(f, "licenses")
	fixed2 := moveToPackage(f, "distribs")
	return fixed1 || fixed2
}

// AllRuleFixes is a list of all Buildozer fixes that can be applied on a rule.
var AllRuleFixes = []struct {
	Name    string
	Fn      func(file *build.File, rule *build.Rule, pkg string) bool
	Message string
}{
	{"sortGlob", sortGlob,
		"Sort the list in a call to glob"},
	{"splitOptions", splitOptionsWithSpaces,
		"Each option should be given separately in the list"},
	{"shortenLabels", shortenLabels,
		"Style: Use the canonical label notation"},
	{"removeVisibility", removeVisibility,
		"This visibility attribute is useless (it corresponds to the default value)"},
	{"removeTestOnly", removeTestOnly,
		"This testonly attribute is useless (it corresponds to the default value)"},
	{"genruleRenameDepsTools", genruleRenameDepsTools,
		"'deps' attribute in genrule has been renamed 'tools'"},
	{"genruleFixHeuristicLabels", genruleFixHeuristicLabels,
		"$(location) should be called explicitely"},
	{"sortExportsFiles", sortExportsFiles,
		"Files in exports_files should be sorted"},
	{"varref", removeVarref,
		"All varref('foo') should be replaced with '$foo'"},
	{"mergeLiteralLists", mergeLiteralLists,
		"Remove useless list concatenation"},
}

// FileLevelFixes is a list of all Buildozer fixes that apply on the whole file.
var FileLevelFixes = []struct {
	Name    string
	Fn      func(file *build.File) bool
	Message string
}{
	{"movePackageToTop", movePackageDeclarationToTheTop,
		"The package declaration should be the first rule in a file"},
	{"usePlusEqual", usePlusEqual,
		"Prefer '+=' over 'extend' or 'append'"},
	{"unusedLoads", cleanUnusedLoads,
		"Remove unused symbols from load statements"},
	{"moveLicensesAndDistribs", moveLicensesAndDistribs,
		"Move licenses and distribs to the package function"},
}

// FixRule aims to fix errors in BUILD files, remove deprecated features, and
// simplify the code.
func FixRule(f *build.File, pkg string, rule *build.Rule, fixes []string) *build.File {
	fixesAsMap := make(map[string]bool)
	for _, fix := range fixes {
		fixesAsMap[fix] = true
	}
	fixed := false
	for _, fix := range AllRuleFixes {
		if len(fixes) == 0 || fixesAsMap[fix.Name] {
			fixed = fix.Fn(f, rule, pkg) || fixed
		}
	}
	if !fixed {
		return nil
	}
	return f
}

// FixFile fixes everything it can in the BUILD file.
func FixFile(f *build.File, pkg string, fixes []string) *build.File {
	fixesAsMap := make(map[string]bool)
	for _, fix := range fixes {
		fixesAsMap[fix] = true
	}
	fixed := false
	for _, rule := range f.Rules("") {
		res := FixRule(f, pkg, rule, fixes)
		if res != nil {
			fixed = true
			f = res
		}
	}
	for _, fix := range FileLevelFixes {
		if len(fixes) == 0 || fixesAsMap[fix.Name] {
			fixed = fix.Fn(f) || fixed
		}
	}
	if !fixed {
		return nil
	}
	return f
}

// A wrapper for a LoadStmt's From and To slices for consistent sorting of their contents.
// It's assumed that the following slices have the same length, the contents are sorted by
// the `To` attribute, the items of `From` are swapped exactly the same way as the items of `To`.
type loadArgs struct {
	From []*build.Ident
	To   []*build.Ident
}

func (args loadArgs) Len() int {
	return len(args.From)
}

func (args loadArgs) Swap(i, j int) {
	args.From[i], args.From[j] = args.From[j], args.From[i]
	args.To[i], args.To[j] = args.To[j], args.To[i]
}

func (args loadArgs) Less(i, j int) bool {
	return args.To[i].Name < args.To[j].Name
}
