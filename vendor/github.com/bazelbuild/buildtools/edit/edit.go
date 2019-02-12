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

// Package edit provides high-level auxiliary functions for AST manipulation
// on BUILD files.
package edit

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/bazelbuild/buildtools/build"
	"github.com/bazelbuild/buildtools/wspace"
)

var (
	// ShortenLabelsFlag if true converts added labels to short form , e.g. //foo:bar => :bar
	ShortenLabelsFlag = true
	// DeleteWithComments if true a list attribute will be be deleted in ListDelete, even if there is a comment attached to it
	DeleteWithComments = true
)

// ParseLabel parses a Blaze label (eg. //devtools/buildozer:rule), and returns
// the repo name ("" for the main repo), package (with leading slashes trimmed)
// and rule name (e.g. ["", "devtools/buildozer", "rule"]).
func ParseLabel(target string) (string, string, string) {
	repo := ""
	if strings.HasPrefix(target, "@") {
		target = strings.TrimLeft(target, "@")
		parts := strings.SplitN(target, "/", 2)
		if len(parts) == 1 {
			// "@foo" -> "foo", "", "foo" (ie @foo//:foo)
			return target, "", target
		}
		repo = parts[0]
		target = "/" + parts[1]
	}
	// TODO(bazel-team): check if the next line can now be deleted
	target = strings.TrimRight(target, ":") // labels can end with ':'
	parts := strings.SplitN(target, ":", 2)
	parts[0] = strings.TrimPrefix(parts[0], "//")
	if len(parts) == 1 {
		if strings.HasPrefix(target, "//") {
			// "//absolute/pkg" -> "absolute/pkg", "pkg"
			return repo, parts[0], path.Base(parts[0])
		}
		// "relative/label" -> "", "relative/label"
		return repo, "", parts[0]
	}
	return repo, parts[0], parts[1]
}

// ShortenLabel rewrites labels to use the canonical form (the form
// recommended by build-style).  This behavior can be disabled using the
// --noshorten_labels flag for projects that consistently use long-form labels.
// "//foo/bar:bar" => "//foo/bar", or ":bar" when possible.
func ShortenLabel(label string, pkg string) string {
	if !ShortenLabelsFlag {
		return label
	}
	if !strings.HasPrefix(label, "//") {
		// It doesn't look like a long label, so we preserve it.
		return label
	}
	repo, labelPkg, rule := ParseLabel(label)
	if repo == "" && labelPkg == pkg { // local label
		return ":" + rule
	}
	slash := strings.LastIndex(labelPkg, "/")
	if (slash >= 0 && labelPkg[slash+1:] == rule) || labelPkg == rule {
		return "//" + labelPkg
	}
	return label
}

// LabelsEqual returns true if label1 and label2 are equal. The function
// takes care of the optional ":" prefix and differences between long-form
// labels and local labels.
func LabelsEqual(label1, label2, pkg string) bool {
	str1 := strings.TrimPrefix(ShortenLabel(label1, pkg), ":")
	str2 := strings.TrimPrefix(ShortenLabel(label2, pkg), ":")
	return str1 == str2
}

// isFile returns true if the path refers to a regular file after following
// symlinks.
func isFile(path string) bool {
	path, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false
	}
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.Mode().IsRegular()
}

// InterpretLabelForWorkspaceLocation returns the name of the BUILD file to
// edit, the full package name, and the rule. It takes a workspace-rooted
// directory to use.
func InterpretLabelForWorkspaceLocation(root string, target string) (buildFile string, pkg string, rule string) {
	repo, pkg, rule := ParseLabel(target)
	rootDir, relativePath := wspace.FindWorkspaceRoot(root)
	if repo != "" {
		files, err := wspace.FindRepoBuildFiles(rootDir)
		if err == nil {
			if buildFile, ok := files[repo]; ok {
				return buildFile, pkg, rule
			}
		}
		// TODO(rodrigoq): report error for other repos
	}

	if strings.HasPrefix(target, "//") {
		buildFile = path.Join(rootDir, pkg, "BUILD")
		return
	}
	if isFile(pkg) {
		// allow operation on other files like WORKSPACE
		buildFile = pkg
		pkg = path.Join(relativePath, filepath.Dir(pkg))
		return
	}
	if pkg != "" {
		buildFile = pkg + "/BUILD"
	} else {
		buildFile = "BUILD"
	}
	pkg = path.Join(relativePath, pkg)
	return
}

// InterpretLabel returns the name of the BUILD file to edit, the full
// package name, and the rule. It uses the pwd for resolving workspace file paths.
func InterpretLabel(target string) (buildFile string, pkg string, rule string) {
	return InterpretLabelForWorkspaceLocation("", target)
}

// ExprToRule returns a Rule from an Expr.
// The boolean is false iff the Expr is not a function call, or does not have
// the expected kind.
func ExprToRule(expr build.Expr, kind string) (*build.Rule, bool) {
	call, ok := expr.(*build.CallExpr)
	if !ok {
		return nil, false
	}
	k, ok := call.X.(*build.LiteralExpr)
	if !ok || k.Token != kind {
		return nil, false
	}
	return &build.Rule{Call: call}, true
}

// ExistingPackageDeclaration returns the package declaration, or nil if there is none.
func ExistingPackageDeclaration(f *build.File) *build.Rule {
	for _, stmt := range f.Stmt {
		if rule, ok := ExprToRule(stmt, "package"); ok {
			return rule
		}
	}
	return nil
}

// PackageDeclaration returns the package declaration. If it doesn't
// exist, it is created at the top of the BUILD file, after leading
// comments.
func PackageDeclaration(f *build.File) *build.Rule {
	if pkg := ExistingPackageDeclaration(f); pkg != nil {
		return pkg
	}
	all := []build.Expr{}
	added := false
	call := &build.CallExpr{X: &build.LiteralExpr{Token: "package"}}
	// Skip CommentBlocks and find a place to insert the package declaration.
	for _, stmt := range f.Stmt {
		_, ok := stmt.(*build.CommentBlock)
		if !ok && !added {
			all = append(all, call)
			added = true
		}
		all = append(all, stmt)
	}
	if !added { // In case the file is empty.
		all = append(all, call)
	}
	f.Stmt = all
	return &build.Rule{Call: call}
}

// RemoveEmptyPackage removes empty package declarations from the file, i.e.:
//    package()
// This might appear because of a buildozer transformation (e.g. when removing a package
// attribute). Removing it is required for the file to be valid.
func RemoveEmptyPackage(f *build.File) *build.File {
	var all []build.Expr
	for _, stmt := range f.Stmt {
		if call, ok := stmt.(*build.CallExpr); ok {
			functionName, ok := call.X.(*build.LiteralExpr)
			if ok && functionName.Token == "package" && len(call.List) == 0 {
				continue
			}
		}
		all = append(all, stmt)
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all}
}

// InsertAfter inserts an expression after index i.
func InsertAfter(i int, stmt []build.Expr, expr build.Expr) []build.Expr {
	i = i + 1 // index after the element at i
	result := make([]build.Expr, len(stmt)+1)
	copy(result[0:i], stmt[0:i])
	result[i] = expr
	copy(result[i+1:], stmt[i:])
	return result
}

// IndexOfLast finds the index of the last expression of a specific kind.
func IndexOfLast(stmt []build.Expr, Kind string) int {
	lastIndex := -1
	for i, s := range stmt {
		sAsCallExpr, ok := s.(*build.CallExpr)
		if !ok {
			continue
		}
		literal, ok := sAsCallExpr.X.(*build.LiteralExpr)
		if ok && literal.Token == Kind {
			lastIndex = i
		}
	}
	return lastIndex
}

// InsertAfterLastOfSameKind inserts an expression after the last expression of the same kind.
func InsertAfterLastOfSameKind(stmt []build.Expr, expr *build.CallExpr) []build.Expr {
	index := IndexOfLast(stmt, expr.X.(*build.LiteralExpr).Token)
	if index == -1 {
		return InsertAtEnd(stmt, expr)
	}
	return InsertAfter(index, stmt, expr)
}

// InsertAtEnd inserts an expression at the end of a list, before trailing comments.
func InsertAtEnd(stmt []build.Expr, expr build.Expr) []build.Expr {
	var i int
	for i = len(stmt) - 1; i >= 0; i-- {
		_, ok := stmt[i].(*build.CommentBlock)
		if !ok {
			break
		}
	}
	return InsertAfter(i, stmt, expr)
}

// FindRuleByName returns the rule in the file that has the given name.
// If the name is "__pkg__", it returns the global package declaration.
func FindRuleByName(f *build.File, name string) *build.Rule {
	if name == "__pkg__" {
		return PackageDeclaration(f)
	}
	i := IndexOfRuleByName(f, name)
	if i != -1 {
		return &build.Rule{Call: f.Stmt[i].(*build.CallExpr)}
	}
	return nil
}

// UseImplicitName returns the rule in the file if it meets these conditions:
// - It is the only unnamed rule in the file.
// - The file path's ending directory name and the passed rule name match.
// In the Pants Build System, by pantsbuild, the use of an implicit name makes
// creating targets easier. This function implements such names.
func UseImplicitName(f *build.File, rule string) *build.Rule {
	// We disallow empty names
	if f.Path == "BUILD" {
		return nil
	}
	ruleCount := 0
	var temp, found *build.Rule
	pkg := filepath.Base(filepath.Dir(f.Path))

	for _, stmt := range f.Stmt {
		call, ok := stmt.(*build.CallExpr)
		if !ok {
			continue
		}
		temp = &build.Rule{Call: call}
		if temp.Kind() != "" && temp.Name() == "" {
			ruleCount++
			found = temp
		}
	}

	if ruleCount == 1 {
		if rule == pkg {
			return found
		}
	}
	return nil
}

// IndexOfRuleByName returns the index (in f.Stmt) of the CallExpr which defines a rule named `name`, or -1 if it doesn't exist.
func IndexOfRuleByName(f *build.File, name string) int {
	linenum := -1
	if strings.HasPrefix(name, "%") {
		// "%<LINENUM>" will match the rule which begins at LINENUM.
		// This is for convenience, "%" is not a valid character in bazel targets.
		if result, err := strconv.Atoi(name[1:]); err == nil {
			linenum = result
		}
	}

	for i, stmt := range f.Stmt {
		call, ok := stmt.(*build.CallExpr)
		if !ok {
			continue
		}
		r := &build.Rule{Call: call}
		start, _ := call.X.Span()
		if r.Name() == name || start.Line == linenum {
			return i
		}
	}
	return -1
}

// FindExportedFile returns the first exports_files call which contains the
// file 'name', or nil if not found
func FindExportedFile(f *build.File, name string) *build.Rule {
	for _, r := range f.Rules("exports_files") {
		if len(r.Call.List) == 0 {
			continue
		}
		pkg := "" // Files are not affected by the package name
		if ListFind(r.Call.List[0], name, pkg) != nil {
			return r
		}
	}
	return nil
}

// DeleteRule returns the AST without the specified rule
func DeleteRule(f *build.File, rule *build.Rule) *build.File {
	var all []build.Expr
	for _, stmt := range f.Stmt {
		if stmt == rule.Call {
			continue
		}
		all = append(all, stmt)
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all}
}

// DeleteRuleByName returns the AST without the rules that have the
// given name.
func DeleteRuleByName(f *build.File, name string) *build.File {
	var all []build.Expr
	for _, stmt := range f.Stmt {
		call, ok := stmt.(*build.CallExpr)
		if !ok {
			all = append(all, stmt)
			continue
		}
		r := &build.Rule{Call: call}
		if r.Name() != name {
			all = append(all, stmt)
		}
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all}
}

// DeleteRuleByKind removes the rules of the specified kind from the AST.
// Returns an updated copy of f.
func DeleteRuleByKind(f *build.File, kind string) *build.File {
	var all []build.Expr
	for _, stmt := range f.Stmt {
		call, ok := stmt.(*build.CallExpr)
		if !ok {
			all = append(all, stmt)
			continue
		}
		k, ok := call.X.(*build.LiteralExpr)
		if !ok || k.Token != kind {
			all = append(all, stmt)
		}
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all}
}

// AllLists returns all the lists concatenated in an expression.
// For example, in: glob(["*.go"]) + [":rule"]
// the function will return [[":rule"]].
func AllLists(e build.Expr) []*build.ListExpr {
	switch e := e.(type) {
	case *build.ListExpr:
		return []*build.ListExpr{e}
	case *build.BinaryExpr:
		if e.Op == "+" {
			return append(AllLists(e.X), AllLists(e.Y)...)
		}
	}
	return nil
}

// FirstList works in the same way as AllLists, except that it
// returns only one list, or nil.
func FirstList(e build.Expr) *build.ListExpr {
	switch e := e.(type) {
	case *build.ListExpr:
		return e
	case *build.BinaryExpr:
		if e.Op == "+" {
			li := FirstList(e.X)
			if li == nil {
				return FirstList(e.Y)
			}
			return li
		}
	}
	return nil
}

// AllStrings returns all the string literals concatenated in an expression.
// For example, in: "foo" + x + "bar"
// the function will return ["foo", "bar"].
func AllStrings(e build.Expr) []*build.StringExpr {
	switch e := e.(type) {
	case *build.StringExpr:
		return []*build.StringExpr{e}
	case *build.BinaryExpr:
		if e.Op == "+" {
			return append(AllStrings(e.X), AllStrings(e.Y)...)
		}
	}
	return nil
}

// ListFind looks for a string in the list expression (which may be a
// concatenation of lists). It returns the element if it is found. nil
// otherwise.
func ListFind(e build.Expr, item string, pkg string) *build.StringExpr {
	item = ShortenLabel(item, pkg)
	for _, li := range AllLists(e) {
		for _, elem := range li.List {
			str, ok := elem.(*build.StringExpr)
			if ok && LabelsEqual(str.Value, item, pkg) {
				return str
			}
		}
	}
	return nil
}

// hasComments returns whether the StringExpr literal has a comment attached to it.
func hasComments(literal *build.StringExpr) bool {
	return len(literal.Before) > 0 || len(literal.Suffix) > 0
}

// ContainsComments returns whether the expr has a comment that includes str.
func ContainsComments(expr build.Expr, str string) bool {
	str = strings.ToLower(str)
	com := expr.Comment()
	comments := append(com.Before, com.Suffix...)
	comments = append(comments, com.After...)
	for _, c := range comments {
		if strings.Contains(strings.ToLower(c.Token), str) {
			return true
		}
	}
	return false
}

// ListDelete deletes the item from a list expression in e and returns
// the StringExpr deleted, or nil otherwise.
func ListDelete(e build.Expr, item, pkg string) (deleted *build.StringExpr) {
	deleted = nil
	item = ShortenLabel(item, pkg)
	for _, li := range AllLists(e) {
		var all []build.Expr
		for _, elem := range li.List {
			if str, ok := elem.(*build.StringExpr); ok {
				if LabelsEqual(str.Value, item, pkg) && (DeleteWithComments || !hasComments(str)) {
					deleted = str
					continue
				}
			}
			all = append(all, elem)
		}
		li.List = all
	}
	return deleted
}

// ListAttributeDelete deletes string item from list attribute attr, deletes attr if empty,
// and returns the StringExpr deleted, or nil otherwise.
func ListAttributeDelete(rule *build.Rule, attr, item, pkg string) *build.StringExpr {
	deleted := ListDelete(rule.Attr(attr), item, pkg)
	if deleted != nil {
		if listExpr, ok := rule.Attr(attr).(*build.ListExpr); ok && len(listExpr.List) == 0 {
			rule.DelAttr(attr)
		}
	}
	return deleted
}

// ListReplace replaces old with value in all lists in e and returns a Boolean
// to indicate whether the replacement was successful.
func ListReplace(e build.Expr, old, value, pkg string) bool {
	replaced := false
	old = ShortenLabel(old, pkg)
	for _, li := range AllLists(e) {
		for k, elem := range li.List {
			str, ok := elem.(*build.StringExpr)
			if !ok || !LabelsEqual(str.Value, old, pkg) {
				continue
			}
			li.List[k] = &build.StringExpr{Value: ShortenLabel(value, pkg), Comments: *elem.Comment()}
			replaced = true
		}
	}
	return replaced
}

// isExprLessThan compares two Expr statements. Currently, only labels are supported.
func isExprLessThan(x1, x2 build.Expr) bool {
	str1, ok1 := x1.(*build.StringExpr)
	str2, ok2 := x2.(*build.StringExpr)
	if ok1 != ok2 {
		return ok2
	}
	if ok1 && ok2 {
		// Labels starting with // are put at the end.
		pre1 := strings.HasPrefix(str1.Value, "//")
		pre2 := strings.HasPrefix(str2.Value, "//")
		if pre1 != pre2 {
			return pre2
		}
		return str1.Value < str2.Value
	}
	return false
}

func sortedInsert(list []build.Expr, item build.Expr) []build.Expr {
	i := 0
	for ; i < len(list); i++ {
		if isExprLessThan(item, list[i]) {
			break
		}
	}
	res := make([]build.Expr, 0, len(list)+1)
	res = append(res, list[:i]...)
	res = append(res, item)
	res = append(res, list[i:]...)
	return res
}

// attributeMustNotBeSorted returns true if the list in the attribute cannot be
// sorted. For some attributes, it makes sense to try to do a sorted insert
// (e.g. deps), even when buildifier will not sort it for conservative reasons.
// For a few attributes, sorting will never make sense.
func attributeMustNotBeSorted(rule, attr string) bool {
	// TODO(bazel-team): Come up with a more complete list.
	return attr == "args"
}

// getVariable returns the binary expression that assignes a variable to expr, if expr is
// an identifier of a variable that vars contains a mapping for.
func getVariable(expr build.Expr, vars *map[string]*build.BinaryExpr) (varAssignment *build.BinaryExpr) {
	if vars == nil {
		return nil
	}

	if literal, ok := expr.(*build.LiteralExpr); ok {
		if varAssignment = (*vars)[literal.Token]; varAssignment != nil {
			return varAssignment
		}
	}
	return nil
}

// AddValueToList adds a value to a list. If the expression is
// not a list, a list with a single element is appended to the original
// expression.
func AddValueToList(oldList build.Expr, pkg string, item build.Expr, sorted bool) build.Expr {
	if oldList == nil {
		return &build.ListExpr{List: []build.Expr{item}}
	}

	str, ok := item.(*build.StringExpr)
	if ok && ListFind(oldList, str.Value, pkg) != nil {
		// The value is already in the list.
		return oldList
	}
	li := FirstList(oldList)
	if li != nil {
		if sorted {
			li.List = sortedInsert(li.List, item)
		} else {
			li.List = append(li.List, item)
		}
		return oldList
	}
	list := &build.ListExpr{List: []build.Expr{item}}
	concat := &build.BinaryExpr{Op: "+", X: oldList, Y: list}
	return concat
}

// AddValueToListAttribute adds the given item to the list attribute identified by name and pkg.
func AddValueToListAttribute(r *build.Rule, name string, pkg string, item build.Expr, vars *map[string]*build.BinaryExpr) {
	old := r.Attr(name)
	sorted := !attributeMustNotBeSorted(r.Kind(), name)
	if varAssignment := getVariable(old, vars); varAssignment != nil {
		varAssignment.Y = AddValueToList(varAssignment.Y, pkg, item, sorted)
	} else {
		r.SetAttr(name, AddValueToList(old, pkg, item, sorted))
	}
}

// MoveAllListAttributeValues moves all values from list attribute oldAttr to newAttr,
// and deletes oldAttr.
func MoveAllListAttributeValues(rule *build.Rule, oldAttr, newAttr, pkg string, vars *map[string]*build.BinaryExpr) error {
	if rule.Attr(oldAttr) == nil {
		return fmt.Errorf("no attribute %s found in %s", oldAttr, rule.Name())
	}
	if rule.Attr(newAttr) == nil {
		RenameAttribute(rule, oldAttr, newAttr)
		return nil
	}
	if listExpr, ok := rule.Attr(oldAttr).(*build.ListExpr); ok {
		for _, val := range listExpr.List {
			AddValueToListAttribute(rule, newAttr, pkg, val, vars)
		}
		rule.DelAttr(oldAttr)
		return nil
	}
	return fmt.Errorf("%s already exists and %s is not a simple list", newAttr, oldAttr)
}

// DictionarySet looks for the key in the dictionary expression. If value is not nil,
// it replaces the current value with it. In all cases, it returns the current value.
func DictionarySet(dict *build.DictExpr, key string, value build.Expr) build.Expr {
	for _, e := range dict.List {
		kv, _ := e.(*build.KeyValueExpr)
		if k, ok := kv.Key.(*build.StringExpr); ok && k.Value == key {
			if value != nil {
				kv.Value = value
			}
			return kv.Value
		}
	}
	if value != nil {
		kv := &build.KeyValueExpr{Key: &build.StringExpr{Value: key}, Value: value}
		dict.List = append(dict.List, kv)
	}
	return nil
}

// RenameAttribute renames an attribute in a rule.
func RenameAttribute(r *build.Rule, oldName, newName string) error {
	if r.Attr(newName) != nil {
		return fmt.Errorf("attribute %s already exists in rule %s", newName, r.Name())
	}
	for _, kv := range r.Call.List {
		as, ok := kv.(*build.BinaryExpr)
		if !ok || as.Op != "=" {
			continue
		}
		k, ok := as.X.(*build.LiteralExpr)
		if !ok || k.Token != oldName {
			continue
		}
		k.Token = newName
		return nil
	}
	return fmt.Errorf("no attribute %s found in rule %s", oldName, r.Name())
}

// EditFunction is a wrapper around build.Edit. The callback is called only on
// functions 'name'.
func EditFunction(v build.Expr, name string, f func(x *build.CallExpr, stk []build.Expr) build.Expr) build.Expr {
	return build.Edit(v, func(expr build.Expr, stk []build.Expr) build.Expr {
		call, ok := expr.(*build.CallExpr)
		if !ok {
			return nil
		}
		fct, ok := call.X.(*build.LiteralExpr)
		if !ok || fct.Token != name {
			return nil
		}
		return f(call, stk)
	})
}

// UsedSymbols returns the set of symbols used in the BUILD file (variables, function names).
func UsedSymbols(f *build.File) map[string]bool {
	symbols := make(map[string]bool)
	build.Walk(f, func(expr build.Expr, stack []build.Expr) {
		literal, ok := expr.(*build.LiteralExpr)
		if !ok {
			return
		}
		// Check if we are on the left-side of an assignment
		for _, e := range stack {
			if as, ok := e.(*build.BinaryExpr); ok {
				if as.Op == "=" && as.X == expr {
					return
				}
			}
		}
		symbols[literal.Token] = true
	})
	return symbols
}

func newLoad(args []string) *build.CallExpr {
	load := &build.CallExpr{
		X: &build.LiteralExpr{
			Token: "load",
		},
		List:         []build.Expr{},
		ForceCompact: true,
	}
	for _, a := range args {
		load.List = append(load.List, &build.StringExpr{Value: a})
	}
	return load
}

// appendLoad tries to find an existing load location and append symbols to it.
func appendLoad(stmts []build.Expr, args []string) bool {
	if len(args) == 0 {
		return false
	}
	location := args[0]
	symbolsToLoad := make(map[string]bool)
	for _, s := range args[1:] {
		symbolsToLoad[s] = true
	}
	var lastLoad *build.CallExpr
	for _, s := range stmts {
		call, ok := s.(*build.CallExpr)
		if !ok {
			continue
		}
		if l, ok := call.X.(*build.LiteralExpr); !ok || l.Token != "load" {
			continue
		}
		if len(call.List) < 2 {
			continue
		}
		if s, ok := call.List[0].(*build.StringExpr); !ok || s.Value != location {
			continue // Loads a different file.
		}
		for _, arg := range call.List[1:] {
			if s, ok := arg.(*build.StringExpr); ok {
				delete(symbolsToLoad, s.Value) // Already loaded.
			}
		}
		// Remember the last insert location, but potentially remove more symbols
		// that are already loaded in other subsequent calls.
		lastLoad = call
	}

	if lastLoad == nil {
		return false
	}

	// Append the remaining loads to the last load location.
	sortedSymbols := []string{}
	for s := range symbolsToLoad {
		sortedSymbols = append(sortedSymbols, s)
	}
	sort.Strings(sortedSymbols)
	for _, s := range sortedSymbols {
		lastLoad.List = append(lastLoad.List, &build.StringExpr{Value: s})
	}
	return true
}

// InsertLoad inserts a load statement at the top of the list of statements.
// The load statement is constructed using args. Symbols that are already loaded
// from the given filepath are ignored. If stmts already contains a load for the
// location in arguments, appends the symbols to load to it.
func InsertLoad(stmts []build.Expr, args []string) []build.Expr {
	if appendLoad(stmts, args) {
		return stmts
	}

	load := newLoad(args)

	var all []build.Expr
	added := false
	for _, stmt := range stmts {
		_, isComment := stmt.(*build.CommentBlock)
		if isComment || added {
			all = append(all, stmt)
			continue
		}
		all = append(all, load)
		all = append(all, stmt)
		added = true
	}
	if !added { // Empty file or just comments.
		all = append(all, load)
	}
	return all
}
