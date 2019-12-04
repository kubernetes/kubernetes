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
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/bazelbuild/buildtools/build"
	"github.com/bazelbuild/buildtools/tables"
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
		if strings.HasPrefix(target, "//") || tables.StripLabelLeadingSlashes {
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
	if !strings.Contains(label, "//") {
		// It doesn't look like a long label, so we preserve it.
		return label
	}
	repo, labelPkg, rule := ParseLabel(label)
	if repo == "" && labelPkg == pkg { // local label
		return ":" + rule
	}
	slash := strings.LastIndex(labelPkg, "/")
	if (slash >= 0 && labelPkg[slash+1:] == rule) || labelPkg == rule {
		if repo == "" {
			return "//" + labelPkg
		}
		return "@" + repo + "//" + labelPkg
	}
	if strings.HasPrefix(label, "@") && repo == rule && labelPkg == "" {
		return "@" + repo
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
	k, ok := call.X.(*build.Ident)
	if !ok || k.Name != kind {
		return nil, false
	}
	return &build.Rule{call, ""}, true
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
// exist, it is created at the top of the BUILD file, after optional
// docstring, comments, and load statements.
func PackageDeclaration(f *build.File) *build.Rule {
	if pkg := ExistingPackageDeclaration(f); pkg != nil {
		return pkg
	}
	all := []build.Expr{}
	added := false
	call := &build.CallExpr{X: &build.Ident{Name: "package"}}
	for _, stmt := range f.Stmt {
		switch stmt.(type) {
		case *build.CommentBlock, *build.LoadStmt, *build.StringExpr:
			// Skip docstring, comments, and load statements to
			// find a place to insert the package declaration.
		default:
			if !added {
				all = append(all, call)
				added = true
			}
		}
		all = append(all, stmt)
	}
	if !added { // In case the file is empty.
		all = append(all, call)
	}
	f.Stmt = all
	return &build.Rule{call, ""}
}

// RemoveEmptyPackage removes empty package declarations from the file, i.e.:
//    package()
// This might appear because of a buildozer transformation (e.g. when removing a package
// attribute). Removing it is required for the file to be valid.
func RemoveEmptyPackage(f *build.File) *build.File {
	var all []build.Expr
	for _, stmt := range f.Stmt {
		if call, ok := stmt.(*build.CallExpr); ok {
			functionName, ok := call.X.(*build.Ident)
			if ok && functionName.Name == "package" && len(call.List) == 0 {
				continue
			}
		}
		all = append(all, stmt)
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all, Type: build.TypeBuild}
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
		literal, ok := sAsCallExpr.X.(*build.Ident)
		if ok && literal.Name == Kind {
			lastIndex = i
		}
	}
	return lastIndex
}

// InsertAfterLastOfSameKind inserts an expression after the last expression of the same kind.
func InsertAfterLastOfSameKind(stmt []build.Expr, expr *build.CallExpr) []build.Expr {
	index := IndexOfLast(stmt, expr.X.(*build.Ident).Name)
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
	_, rule := IndexOfRuleByName(f, name)
	return rule
}

// IndexOfRuleByName returns the index (in f.Stmt) of the CallExpr which defines a rule named `name`, or -1 if it doesn't exist.
func IndexOfRuleByName(f *build.File, name string) (int, *build.Rule) {
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
		r := f.Rule(call)
		start, _ := call.X.Span()
		if r.Name() == name || start.Line == linenum {
			return i, r
		}
	}
	return -1, nil
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
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all, Type: build.TypeBuild}
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
		r := f.Rule(call)
		if r.Name() != name {
			all = append(all, stmt)
		}
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all, Type: build.TypeBuild}
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
		k, ok := call.X.(*build.Ident)
		if !ok || k.Name != kind {
			all = append(all, stmt)
		}
	}
	return &build.File{Path: f.Path, Comments: f.Comments, Stmt: all, Type: build.TypeBuild}
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

// AllSelects returns all the selects concatenated in an expression.
func AllSelects(e build.Expr) []*build.CallExpr {
	switch e := e.(type) {
	case *build.BinaryExpr:
		if e.Op == "+" {
			return append(AllSelects(e.X), AllSelects(e.Y)...)
		}
	case *build.CallExpr:
		if x, ok := e.X.(*build.Ident); ok && x.Name == "select" {
			return []*build.CallExpr{e}
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

// RemoveEmptySelectsAndConcatLists iterates the tree in order to turn
// empty selects into empty lists and adjacent lists are concatenated
func RemoveEmptySelectsAndConcatLists(e build.Expr) build.Expr {
	switch e := e.(type) {
	case *build.BinaryExpr:
		if e.Op == "+" {
			e.X = RemoveEmptySelectsAndConcatLists(e.X)
			e.Y = RemoveEmptySelectsAndConcatLists(e.Y)

			x, xIsList := e.X.(*build.ListExpr)
			y, yIsList := e.Y.(*build.ListExpr)

			if xIsList && yIsList {
				return &build.ListExpr{List: append(x.List, y.List...)}
			}

			if xIsList && len(x.List) == 0 {
				return e.Y
			}

			if yIsList && len(y.List) == 0 {
				return e.X
			}
		}
	case *build.CallExpr:
		if x, ok := e.X.(*build.Ident); ok && x.Name == "select" {
			if len(e.List) == 0 {
				return &build.ListExpr{List: []build.Expr{}}
			}

			if dict, ok := e.List[0].(*build.DictExpr); ok {
				for _, keyVal := range dict.List {
					if keyVal, ok := keyVal.(*build.KeyValueExpr); ok {
						val, ok := keyVal.Value.(*build.ListExpr)
						if !ok || len(val.List) > 0 {
							return e
						}
					} else {
						return e
					}
				}

				return &build.ListExpr{List: []build.Expr{}}
			}
		}
	}

	return e
}

// ComputeIntersection returns the intersection of the two lists given as parameters;
// if the containing elements are not build.StringExpr, the result will be nil.
func ComputeIntersection(list1, list2 []build.Expr) []build.Expr {
	if list1 == nil || list2 == nil {
		return nil
	}

	if len(list2) == 0 {
		return []build.Expr{}
	}

	i := 0
	for j, common := range list1 {
		if common, ok := common.(*build.StringExpr); ok {
			found := false
			for _, elem := range list2 {
				if str, ok := elem.(*build.StringExpr); ok {
					if str.Value == common.Value {
						found = true
						break
					}
				} else {
					return nil
				}
			}

			if found {
				list1[i] = list1[j]
				i++
			}
		} else {
			return nil
		}
	}
	return list1[:i]
}

// SelectListsIntersection returns the intersection of the lists of strings inside
// the dictionary argument of the select expression given as a parameter
func SelectListsIntersection(sel *build.CallExpr, pkg string) (intersection []build.Expr) {
	if len(sel.List) == 0 || len(sel.List) > 1 {
		return nil
	}

	dict, ok := sel.List[0].(*build.DictExpr)
	if !ok || len(dict.List) == 0 {
		return nil
	}

	if keyVal, ok := dict.List[0].(*build.KeyValueExpr); ok {
		if val, ok := keyVal.Value.(*build.ListExpr); ok {
			intersection = make([]build.Expr, len(val.List))
			copy(intersection, val.List)
		}
	}

	for _, keyVal := range dict.List[1:] {
		if keyVal, ok := keyVal.(*build.KeyValueExpr); ok {
			if val, ok := keyVal.Value.(*build.ListExpr); ok {
				intersection = ComputeIntersection(intersection, val.List)
				if len(intersection) == 0 {
					return intersection
				}
			} else {
				return nil
			}
		} else {
			return nil
		}
	}

	return intersection
}

// ResolveAttr extracts common elements of the lists inside select dictionaries
// and adds them at attribute level rather than select level, as well as turns
// empty selects into empty lists and concatenates adjacent lists
func ResolveAttr(r *build.Rule, attr, pkg string) {
	var toExtract []build.Expr

	e := r.Attr(attr)
	if e == nil {
		return
	}

	for _, sel := range AllSelects(e) {
		intersection := SelectListsIntersection(sel, pkg)
		if intersection != nil {
			toExtract = append(toExtract, intersection...)
		}
	}

	for _, common := range toExtract {
		e = AddValueToList(e, pkg, common, false) // this will also remove them from selects
	}

	r.SetAttr(attr, RemoveEmptySelectsAndConcatLists(e))
}

// SelectDelete removes the item from all the lists which are values
// in the dictionary of every select
func SelectDelete(e build.Expr, item, pkg string, deleted **build.StringExpr) {
	for _, sel := range AllSelects(e) {
		if len(sel.List) == 0 {
			continue
		}

		if dict, ok := sel.List[0].(*build.DictExpr); ok {
			for _, keyVal := range dict.List {
				if keyVal, ok := keyVal.(*build.KeyValueExpr); ok {
					if val, ok := keyVal.Value.(*build.ListExpr); ok {
						RemoveFromList(val, item, pkg, deleted)
					}
				}
			}
		}
	}
}

// RemoveFromList removes one element from a ListExpr and stores
// the deleted StringExpr at the address pointed by the last parameter
func RemoveFromList(li *build.ListExpr, item, pkg string, deleted **build.StringExpr) {
	var all []build.Expr
	for _, elem := range li.List {
		if str, ok := elem.(*build.StringExpr); ok {
			if LabelsEqual(str.Value, item, pkg) && (DeleteWithComments || !hasComments(str)) {
				if deleted != nil {
					*deleted = str
				}

				continue
			}
		}
		all = append(all, elem)
	}
	li.List = all
}

// ListDelete deletes the item from a list expression in e and returns
// the StringExpr deleted, or nil otherwise.
func ListDelete(e build.Expr, item, pkg string) (deleted *build.StringExpr) {
	if unquoted, _, err := build.Unquote(item); err == nil {
		item = unquoted
	}
	deleted = nil
	item = ShortenLabel(item, pkg)
	for _, li := range AllLists(e) {
		RemoveFromList(li, item, pkg, &deleted)
	}

	SelectDelete(e, item, pkg, &deleted)

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

// ListSubstitute replaces strings matching a regular expression in all lists
// in e and returns a Boolean to indicate whether the replacement was
// successful.
func ListSubstitute(e build.Expr, oldRegexp *regexp.Regexp, newTemplate string) bool {
	substituted := false
	for _, li := range AllLists(e) {
		for k, elem := range li.List {
			str, ok := elem.(*build.StringExpr)
			if !ok {
				continue
			}
			newValue, ok := stringSubstitute(str.Value, oldRegexp, newTemplate)
			if ok {
				li.List[k] = &build.StringExpr{Value: newValue, Comments: *elem.Comment()}
				substituted = true
			}
		}
	}
	return substituted
}

func stringSubstitute(oldValue string, oldRegexp *regexp.Regexp, newTemplate string) (string, bool) {
	match := oldRegexp.FindStringSubmatchIndex(oldValue)
	if match == nil {
		return oldValue, false
	}
	newValue := string(oldRegexp.ExpandString(nil, newTemplate, oldValue, match))
	if match[0] > 0 {
		newValue = oldValue[:match[0]] + newValue
	}
	if match[1] < len(oldValue) {
		newValue = newValue + oldValue[match[1]:]
	}
	return newValue, true
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
func getVariable(expr build.Expr, vars *map[string]*build.AssignExpr) (varAssignment *build.AssignExpr) {
	if vars == nil {
		return nil
	}

	if literal, ok := expr.(*build.Ident); ok {
		if varAssignment = (*vars)[literal.Name]; varAssignment != nil {
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
	if ok {
		if ListFind(oldList, str.Value, pkg) != nil {
			// The value is already in the list.
			return oldList
		}
		SelectDelete(oldList, str.Value, pkg, nil)
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
func AddValueToListAttribute(r *build.Rule, name string, pkg string, item build.Expr, vars *map[string]*build.AssignExpr) {
	old := r.Attr(name)
	sorted := !attributeMustNotBeSorted(r.Kind(), name)
	if varAssignment := getVariable(old, vars); varAssignment != nil {
		varAssignment.RHS = AddValueToList(varAssignment.RHS, pkg, item, sorted)
	} else {
		r.SetAttr(name, AddValueToList(old, pkg, item, sorted))
	}
}

// MoveAllListAttributeValues moves all values from list attribute oldAttr to newAttr,
// and deletes oldAttr.
func MoveAllListAttributeValues(rule *build.Rule, oldAttr, newAttr, pkg string, vars *map[string]*build.AssignExpr) error {
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

// DictionaryGet looks for the key in the dictionary expression, and returns the
// current value. If it is unset, it returns nil.
func DictionaryGet(dict *build.DictExpr, key string) build.Expr {
	for _, e := range dict.List {
		kv, ok := e.(*build.KeyValueExpr)
		if !ok {
			continue
		}
		if k, ok := kv.Key.(*build.StringExpr); ok && k.Value == key {
			return kv.Value
		}
	}
	return nil
}

// DictionaryDelete looks for the key in the dictionary expression. If the key exists,
// it removes the key-value pair and returns it. Otherwise it returns nil.
func DictionaryDelete(dict *build.DictExpr, key string) (deleted build.Expr) {
	if unquoted, _, err := build.Unquote(key); err == nil {
		key = unquoted
	}
	deleted = nil
	var all []build.Expr
	for _, e := range dict.List {
		kv, _ := e.(*build.KeyValueExpr)
		if k, ok := kv.Key.(*build.StringExpr); ok {
			if k.Value == key {
				deleted = kv
			} else {
				all = append(all, e)
			}
		}
	}
	dict.List = all
	return deleted
}

// RenameAttribute renames an attribute in a rule.
func RenameAttribute(r *build.Rule, oldName, newName string) error {
	if r.Attr(newName) != nil {
		return fmt.Errorf("attribute %s already exists in rule %s", newName, r.Name())
	}
	for _, kv := range r.Call.List {
		as, ok := kv.(*build.AssignExpr)
		if !ok {
			continue
		}
		k, ok := as.LHS.(*build.Ident)
		if !ok || k.Name != oldName {
			continue
		}
		k.Name = newName
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
		fct, ok := call.X.(*build.Ident)
		if !ok || fct.Name != name {
			return nil
		}
		return f(call, stk)
	})
}

// UsedSymbols returns the set of symbols used in the BUILD file (variables, function names).
func UsedSymbols(stmt build.Expr) map[string]bool {
	symbols := make(map[string]bool)
	build.Walk(stmt, func(expr build.Expr, stack []build.Expr) {
		// Don't traverse inside load statements
		if len(stack) > 0 {
			if _, ok := stack[len(stack)-1].(*build.LoadStmt); ok {
				return
			}
		}

		literal, ok := expr.(*build.Ident)
		if !ok {
			return
		}
		// Check if we are on the left-side of an assignment
		for _, e := range stack {
			if as, ok := e.(*build.AssignExpr); ok {
				if as.LHS == expr {
					return
				}
			}
		}
		symbols[literal.Name] = true
	})
	return symbols
}

// NewLoad creates a new LoadStmt node
func NewLoad(location string, from, to []string) *build.LoadStmt {
	load := &build.LoadStmt{
		Module: &build.StringExpr{
			Value: location,
		},
		ForceCompact: true,
	}
	for i := range from {
		load.From = append(load.From, &build.Ident{Name: from[i]})
		load.To = append(load.To, &build.Ident{Name: to[i]})
	}
	return load
}

// AppendToLoad appends symbols to an existing load statement
// Returns true if the statement was acually edited (if the required symbols haven't been
// loaded yet)
func AppendToLoad(load *build.LoadStmt, from, to []string) bool {
	symbolsToLoad := make(map[string]string)
	for i, s := range to {
		symbolsToLoad[s] = from[i]
	}
	for _, ident := range load.To {
		delete(symbolsToLoad, ident.Name) // Already loaded.
	}

	if len(symbolsToLoad) == 0 {
		return false
	}

	// Append the remaining loads to the load statement.
	sortedSymbols := []string{}
	for s := range symbolsToLoad {
		sortedSymbols = append(sortedSymbols, s)
	}
	sort.Strings(sortedSymbols)
	for _, s := range sortedSymbols {
		load.From = append(load.From, &build.Ident{Name: symbolsToLoad[s]})
		load.To = append(load.To, &build.Ident{Name: s})
	}
	return true
}

// appendLoad tries to find an existing load location and append symbols to it.
func appendLoad(stmts []build.Expr, location string, from, to []string) bool {
	symbolsToLoad := make(map[string]string)
	for i, s := range to {
		symbolsToLoad[s] = from[i]
	}
	var lastLoad *build.LoadStmt
	for _, s := range stmts {
		load, ok := s.(*build.LoadStmt)
		if !ok {
			continue
		}
		if load.Module.Value != location {
			continue // Loads a different file.
		}
		for _, ident := range load.To {
			delete(symbolsToLoad, ident.Name) // Already loaded.
		}
		// Remember the last insert location, but potentially remove more symbols
		// that are already loaded in other subsequent calls.
		lastLoad = load
	}
	if lastLoad == nil {
		return false
	}

	// Append the remaining loads to the last load location.
	from = []string{}
	to = []string{}
	for t, f := range symbolsToLoad {
		from = append(from, f)
		to = append(to, t)
	}
	AppendToLoad(lastLoad, from, to)
	return true
}

// InsertLoad inserts a load statement at the top of the list of statements.
// The load statement is constructed using a string location and two slices of from- and to-symbols.
// The function panics if the slices aren't of the same lentgh. Symbols that are already loaded
// from the given filepath are ignored. If stmts already contains a load for the
// location in arguments, appends the symbols to load to it.
func InsertLoad(stmts []build.Expr, location string, from, to []string) []build.Expr {
	if len(from) != len(to) {
		panic(fmt.Errorf("length mismatch: %v (from) and %v (to)", len(from), len(to)))
	}

	if appendLoad(stmts, location, from, to) {
		return stmts
	}

	load := NewLoad(location, from, to)

	var all []build.Expr
	added := false
	for i, stmt := range stmts {
		_, isComment := stmt.(*build.CommentBlock)
		_, isString := stmt.(*build.StringExpr)
		isDocString := isString && i == 0
		if isComment || isDocString || added {
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
