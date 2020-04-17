/* Copyright 2018 The Bazel Authors. All rights reserved.

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

// Package rule provides tools for editing Bazel build files. It is intended to
// be a more powerful replacement for
// github.com/bazelbuild/buildtools/build.Rule, adapted for Gazelle's usage. It
// is language agnostic, but it may be used for language-specific rules by
// providing configuration.
//
// File is the primary interface to this package. A File represents an
// individual build file. It comprises a list of Rules and a list of Loads.
// Rules and Loads may be inserted, modified, or deleted. When all changes
// are done, File.Save() may be called to write changes back to a file.
package rule

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"

	bzl "github.com/bazelbuild/buildtools/build"
	bt "github.com/bazelbuild/buildtools/tables"
)

// File provides editing functionality for a build file. You can create a
// new file with EmptyFile or load an existing file with LoadFile. After
// changes have been made, call Save to write changes back to a file.
type File struct {
	// File is the underlying build file syntax tree. Some editing operations
	// may modify this, but editing is not complete until Sync() is called.
	File *bzl.File

	// function is the underlying syntax tree of a bzl file function.
	// This is used for editing the bzl file function specified by the
	// update-repos -to_macro option.
	function *function

	// Pkg is the Bazel package this build file defines.
	Pkg string

	// Path is the file system path to the build file (same as File.Path).
	Path string

	// DefName is the name of the function definition this File refers to
	// if loaded with LoadMacroFile or a similar function. Normally empty.
	DefName string

	// Directives is a list of configuration directives found in top-level
	// comments in the file. This should not be modified after the file is read.
	Directives []Directive

	// Loads is a list of load statements within the file. This should not
	// be modified directly; use Load methods instead.
	Loads []*Load

	// Rules is a list of rules within the file (or function calls that look like
	// rules). This should not be modified directly; use Rule methods instead.
	Rules []*Rule
}

// EmptyFile creates a File wrapped around an empty syntax tree.
func EmptyFile(path, pkg string) *File {
	return &File{
		File: &bzl.File{Path: path, Type: bzl.TypeBuild},
		Path: path,
		Pkg:  pkg,
	}
}

// LoadFile loads a build file from disk, parses it, and scans for rules and
// load statements. The syntax tree within the returned File will be modified
// by editing methods.
//
// This function returns I/O and parse errors without modification. It's safe
// to use os.IsNotExist and similar predicates.
func LoadFile(path, pkg string) (*File, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadData(path, pkg, data)
}

// LoadWorkspaceFile is similar to LoadFile but parses the file as a WORKSPACE
// file.
func LoadWorkspaceFile(path, pkg string) (*File, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadWorkspaceData(path, pkg, data)
}

// LoadMacroFile loads a bzl file from disk, parses it, then scans for the load
// statements and the rules called from the given Starlark function. If there is
// no matching function name, then a new function with that name will be created.
// The function's syntax tree will be returned within File and can be modified by
// Sync and Save calls.
func LoadMacroFile(path, pkg, defName string) (*File, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadMacroData(path, pkg, defName, data)
}

// EmptyMacroFile creates a bzl file at the given path and within the file creates
// a Starlark function with the provided name. The function can then be modified
// by Sync and Save calls.
func EmptyMacroFile(path, pkg, defName string) (*File, error) {
	_, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return LoadMacroData(path, pkg, defName, nil)
}

// LoadData parses a build file from a byte slice and scans it for rules and
// load statements. The syntax tree within the returned File will be modified
// by editing methods.
func LoadData(path, pkg string, data []byte) (*File, error) {
	ast, err := bzl.ParseBuild(path, data)
	if err != nil {
		return nil, err
	}
	return ScanAST(pkg, ast), nil
}

// LoadWorkspaceData is similar to LoadData but parses the data as a
// WORKSPACE file.
func LoadWorkspaceData(path, pkg string, data []byte) (*File, error) {
	ast, err := bzl.ParseWorkspace(path, data)
	if err != nil {
		return nil, err
	}
	return ScanAST(pkg, ast), nil
}

// LoadMacroData parses a bzl file from a byte slice and scans for the load
// statements and the rules called from the given Starlark function. If there is
// no matching function name, then a new function will be created, and added to the
// File the next time Sync is called. The function's syntax tree will be returned
// within File and can be modified by Sync and Save calls.
func LoadMacroData(path, pkg, defName string, data []byte) (*File, error) {
	ast, err := bzl.ParseBzl(path, data)
	if err != nil {
		return nil, err
	}
	return ScanASTBody(pkg, defName, ast), nil
}

// ScanAST creates a File wrapped around the given syntax tree. This tree
// will be modified by editing methods.
func ScanAST(pkg string, bzlFile *bzl.File) *File {
	return ScanASTBody(pkg, "", bzlFile)
}

type function struct {
	stmt              *bzl.DefStmt
	inserted, hasPass bool
}

// ScanASTBody creates a File wrapped around the given syntax tree. It will also
// scan the AST for a function matching the given defName, and if the function
// does not exist it will create a new one and mark it to be added to the File
// the next time Sync is called.
func ScanASTBody(pkg, defName string, bzlFile *bzl.File) *File {
	f := &File{
		File:    bzlFile,
		Pkg:     pkg,
		Path:    bzlFile.Path,
		DefName: defName,
	}
	var defStmt *bzl.DefStmt
	f.Rules, f.Loads, defStmt = scanExprs(defName, bzlFile.Stmt)
	if defStmt != nil {
		f.Rules, _, _ = scanExprs("", defStmt.Body)
		f.function = &function{
			stmt:     defStmt,
			inserted: true,
		}
		if len(defStmt.Body) == 1 {
			if v, ok := defStmt.Body[0].(*bzl.BranchStmt); ok && v.Token == "pass" {
				f.function.hasPass = true
			}
		}
	} else if defName != "" {
		f.function = &function{
			stmt:     &bzl.DefStmt{Name: defName},
			inserted: false,
		}
	}
	if f.function != nil {
		f.Directives = ParseDirectivesFromMacro(f.function.stmt)
	} else {
		f.Directives = ParseDirectives(bzlFile)
	}
	return f
}

func scanExprs(defName string, stmt []bzl.Expr) (rules []*Rule, loads []*Load, fn *bzl.DefStmt) {
	for i, expr := range stmt {
		switch expr := expr.(type) {
		case *bzl.LoadStmt:
			l := loadFromExpr(i, expr)
			loads = append(loads, l)
		case *bzl.CallExpr:
			if r := ruleFromExpr(i, expr); r != nil {
				rules = append(rules, r)
			}
		case *bzl.DefStmt:
			if expr.Name == defName {
				fn = expr
			}
		}
	}
	return rules, loads, fn
}

// MatchBuildFileName looks for a file in files that has a name from names.
// If there is at least one matching file, a path will be returned by joining
// dir and the first matching name. If there are no matching files, the
// empty string is returned.
func MatchBuildFileName(dir string, names []string, files []os.FileInfo) string {
	for _, name := range names {
		for _, fi := range files {
			if fi.Name() == name && !fi.IsDir() {
				return filepath.Join(dir, name)
			}
		}
	}
	return ""
}

// SyncMacroFile syncs the file's syntax tree with another file's. This is
// useful for keeping multiple macro definitions from the same .bzl file in sync.
func (f *File) SyncMacroFile(from *File) {
	fromFunc := *from.function.stmt
	_, _, toFunc := scanExprs(from.function.stmt.Name, f.File.Stmt)
	if toFunc != nil {
		*toFunc = fromFunc
	} else {
		f.File.Stmt = append(f.File.Stmt, &fromFunc)
	}
}

// MacroName returns the name of the macro function that this file is editing,
// or an empty string if a macro function is not being edited.
func (f *File) MacroName() string {
	if f.function != nil && f.function.stmt != nil {
		return f.function.stmt.Name
	}
	return ""
}

// Sync writes all changes back to the wrapped syntax tree. This should be
// called after editing operations, before reading the syntax tree again.
func (f *File) Sync() {
	var loadInserts, loadDeletes, loadStmts []*stmt
	var r, w int
	for r, w = 0, 0; r < len(f.Loads); r++ {
		s := f.Loads[r]
		s.sync()
		if s.deleted {
			loadDeletes = append(loadDeletes, &s.stmt)
			continue
		}
		if s.inserted {
			loadInserts = append(loadInserts, &s.stmt)
			s.inserted = false
		} else {
			loadStmts = append(loadStmts, &s.stmt)
		}
		f.Loads[w] = s
		w++
	}
	f.Loads = f.Loads[:w]
	var ruleInserts, ruleDeletes, ruleStmts []*stmt
	for r, w = 0, 0; r < len(f.Rules); r++ {
		s := f.Rules[r]
		s.sync()
		if s.deleted {
			ruleDeletes = append(ruleDeletes, &s.stmt)
			continue
		}
		if s.inserted {
			ruleInserts = append(ruleInserts, &s.stmt)
			s.inserted = false
		} else {
			ruleStmts = append(ruleStmts, &s.stmt)
		}
		f.Rules[w] = s
		w++
	}
	f.Rules = f.Rules[:w]

	if f.function == nil {
		deletes := append(ruleDeletes, loadDeletes...)
		inserts := append(ruleInserts, loadInserts...)
		stmts := append(ruleStmts, loadStmts...)
		updateStmt(&f.File.Stmt, inserts, deletes, stmts)
	} else {
		updateStmt(&f.File.Stmt, loadInserts, loadDeletes, loadStmts)
		if f.function.hasPass && len(ruleInserts) > 0 {
			f.function.stmt.Body = []bzl.Expr{}
			f.function.hasPass = false
		}
		updateStmt(&f.function.stmt.Body, ruleInserts, ruleDeletes, ruleStmts)
		if len(f.function.stmt.Body) == 0 {
			f.function.stmt.Body = append(f.function.stmt.Body, &bzl.BranchStmt{Token: "pass"})
			f.function.hasPass = true
		}
		if !f.function.inserted {
			f.File.Stmt = append(f.File.Stmt, f.function.stmt)
			f.function.inserted = true
		}
	}
}

func updateStmt(oldStmt *[]bzl.Expr, inserts, deletes, stmts []*stmt) {
	sort.Stable(byIndex(deletes))
	sort.Stable(byIndex(inserts))
	sort.Stable(byIndex(stmts))
	newStmt := make([]bzl.Expr, 0, len(*oldStmt)-len(deletes)+len(inserts))
	var ii, di, si int
	for i, stmt := range *oldStmt {
		for ii < len(inserts) && inserts[ii].index == i {
			inserts[ii].index = len(newStmt)
			newStmt = append(newStmt, inserts[ii].expr)
			ii++
		}
		if di < len(deletes) && deletes[di].index == i {
			di++
			continue
		}
		if si < len(stmts) && stmts[si].expr == stmt {
			stmts[si].index = len(newStmt)
			si++
		}
		newStmt = append(newStmt, stmt)
	}
	for ii < len(inserts) {
		inserts[ii].index = len(newStmt)
		newStmt = append(newStmt, inserts[ii].expr)
		ii++
	}
	*oldStmt = newStmt
}

// Format formats the build file in a form that can be written to disk.
// This method calls Sync internally.
func (f *File) Format() []byte {
	f.Sync()
	return bzl.Format(f.File)
}

// Save writes the build file to disk. This method calls Sync internally.
func (f *File) Save(path string) error {
	f.Sync()
	data := bzl.Format(f.File)
	return ioutil.WriteFile(path, data, 0666)
}

// HasDefaultVisibility returns whether the File contains a "package" rule with
// a "default_visibility" attribute. Rules generated by Gazelle should not
// have their own visibility attributes if this is the case.
func (f *File) HasDefaultVisibility() bool {
	for _, r := range f.Rules {
		if r.Kind() == "package" && r.Attr("default_visibility") != nil {
			return true
		}
	}
	return false
}

type stmt struct {
	index                      int
	deleted, inserted, updated bool
	expr                       bzl.Expr
}

// Index returns the index for this statement within the build file. For
// inserted rules, this is where the rule will be inserted (rules with the
// same index will be inserted in the order Insert was called). For existing
// rules, this is the index of the original statement.
func (s *stmt) Index() int { return s.index }

// Delete marks this statement for deletion. It will be removed from the
// syntax tree when File.Sync is called.
func (s *stmt) Delete() { s.deleted = true }

type byIndex []*stmt

func (s byIndex) Len() int {
	return len(s)
}

func (s byIndex) Less(i, j int) bool {
	return s[i].index < s[j].index
}

func (s byIndex) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// identPair represents one symbol, with or without remapping, in a load
// statement within a build file.
type identPair struct {
	to, from *bzl.Ident
}

// Load represents a load statement within a build file.
type Load struct {
	stmt
	name    string
	symbols map[string]identPair
}

// NewLoad creates a new, empty load statement for the given file name.
func NewLoad(name string) *Load {
	return &Load{
		stmt: stmt{
			expr: &bzl.LoadStmt{
				Module:       &bzl.StringExpr{Value: name},
				ForceCompact: true,
			},
		},
		name:    name,
		symbols: make(map[string]identPair),
	}
}

func loadFromExpr(index int, loadStmt *bzl.LoadStmt) *Load {
	l := &Load{
		stmt:    stmt{index: index, expr: loadStmt},
		name:    loadStmt.Module.Value,
		symbols: make(map[string]identPair),
	}
	for i := range loadStmt.From {
		to, from := loadStmt.To[i], loadStmt.From[i]
		l.symbols[to.Name] = identPair{to: to, from: from}
	}
	return l
}

// Name returns the name of the file this statement loads.
func (l *Load) Name() string {
	return l.name
}

// Symbols returns a list of symbols this statement loads.
func (l *Load) Symbols() []string {
	syms := make([]string, 0, len(l.symbols))
	for sym := range l.symbols {
		syms = append(syms, sym)
	}
	sort.Strings(syms)
	return syms
}

// Has returns true if sym is loaded by this statement.
func (l *Load) Has(sym string) bool {
	_, ok := l.symbols[sym]
	return ok
}

// Add inserts a new symbol into the load statement. This has no effect if
// the symbol is already loaded. Symbols will be sorted, so the order
// doesn't matter.
func (l *Load) Add(sym string) {
	if _, ok := l.symbols[sym]; !ok {
		i := &bzl.Ident{Name: sym}
		l.symbols[sym] = identPair{to: i, from: i}
		l.updated = true
	}
}

// Remove deletes a symbol from the load statement. This has no effect if
// the symbol is not loaded.
func (l *Load) Remove(sym string) {
	if _, ok := l.symbols[sym]; ok {
		delete(l.symbols, sym)
		l.updated = true
	}
}

// IsEmpty returns whether this statement loads any symbols.
func (l *Load) IsEmpty() bool {
	return len(l.symbols) == 0
}

// Insert marks this statement for insertion at the given index. If multiple
// statements are inserted at the same index, they will be inserted in the
// order Insert is called.
func (l *Load) Insert(f *File, index int) {
	l.index = index
	l.inserted = true
	f.Loads = append(f.Loads, l)
}

func (l *Load) sync() {
	if !l.updated {
		return
	}
	l.updated = false

	// args1 and args2 are two different sort groups based on whether a remap of the identifier is present.
	var args1, args2, args []string
	for sym, pair := range l.symbols {
		if pair.from.Name == pair.to.Name {
			args1 = append(args1, sym)
		} else {
			args2 = append(args2, sym)
		}
	}
	sort.Strings(args1)
	sort.Strings(args2)
	args = append(args, args1...)
	args = append(args, args2...)

	loadStmt := l.expr.(*bzl.LoadStmt)
	loadStmt.Module.Value = l.name
	loadStmt.From = make([]*bzl.Ident, 0, len(args))
	loadStmt.To = make([]*bzl.Ident, 0, len(args))
	for _, sym := range args {
		pair := l.symbols[sym]
		loadStmt.From = append(loadStmt.From, pair.from)
		loadStmt.To = append(loadStmt.To, pair.to)
		if pair.from.Name != pair.to.Name {
			loadStmt.ForceCompact = false
		}
	}
}

// Rule represents a rule statement within a build file.
type Rule struct {
	stmt
	kind    string
	args    []bzl.Expr
	attrs   map[string]*bzl.AssignExpr
	private map[string]interface{}
}

// NewRule creates a new, empty rule with the given kind and name.
func NewRule(kind, name string) *Rule {
	nameAttr := &bzl.AssignExpr{
		LHS: &bzl.Ident{Name: "name"},
		RHS: &bzl.StringExpr{Value: name},
		Op:  "=",
	}
	r := &Rule{
		stmt: stmt{
			expr: &bzl.CallExpr{
				X:    &bzl.Ident{Name: kind},
				List: []bzl.Expr{nameAttr},
			},
		},
		kind:    kind,
		attrs:   map[string]*bzl.AssignExpr{"name": nameAttr},
		private: map[string]interface{}{},
	}
	return r
}

func ruleFromExpr(index int, expr bzl.Expr) *Rule {
	call, ok := expr.(*bzl.CallExpr)
	if !ok {
		return nil
	}
	x, ok := call.X.(*bzl.Ident)
	if !ok {
		return nil
	}
	kind := x.Name
	var args []bzl.Expr
	attrs := make(map[string]*bzl.AssignExpr)
	for _, arg := range call.List {
		if attr, ok := arg.(*bzl.AssignExpr); ok {
			key := attr.LHS.(*bzl.Ident) // required by parser
			attrs[key.Name] = attr
		} else {
			args = append(args, arg)
		}
	}
	return &Rule{
		stmt: stmt{
			index: index,
			expr:  call,
		},
		kind:    kind,
		args:    args,
		attrs:   attrs,
		private: map[string]interface{}{},
	}
}

// ShouldKeep returns whether the rule is marked with a "# keep" comment. Rules
// that are kept should not be modified. This does not check whether
// subexpressions within the rule should be kept.
func (r *Rule) ShouldKeep() bool {
	return ShouldKeep(r.expr)
}

// Kind returns the kind of rule this is (for example, "go_library").
func (r *Rule) Kind() string {
	return r.kind
}

// SetKind changes the kind of rule this is.
func (r *Rule) SetKind(kind string) {
	r.kind = kind
	r.updated = true
}

// Name returns the value of the rule's "name" attribute if it is a string
// or "" if the attribute does not exist or is not a string.
func (r *Rule) Name() string {
	return r.AttrString("name")
}

// SetName sets the value of the rule's "name" attribute.
func (r *Rule) SetName(name string) {
	r.SetAttr("name", name)
}

// AttrKeys returns a sorted list of attribute keys used in this rule.
func (r *Rule) AttrKeys() []string {
	keys := make([]string, 0, len(r.attrs))
	for k := range r.attrs {
		keys = append(keys, k)
	}
	sort.SliceStable(keys, func(i, j int) bool {
		if cmp := bt.NamePriority[keys[i]] - bt.NamePriority[keys[j]]; cmp != 0 {
			return cmp < 0
		}
		return keys[i] < keys[j]
	})
	return keys
}

// Attr returns the value of the named attribute. nil is returned when the
// attribute is not set.
func (r *Rule) Attr(key string) bzl.Expr {
	attr, ok := r.attrs[key]
	if !ok {
		return nil
	}
	return attr.RHS
}

// AttrString returns the value of the named attribute if it is a scalar string.
// "" is returned if the attribute is not set or is not a string.
func (r *Rule) AttrString(key string) string {
	attr, ok := r.attrs[key]
	if !ok {
		return ""
	}
	str, ok := attr.RHS.(*bzl.StringExpr)
	if !ok {
		return ""
	}
	return str.Value
}

// AttrStrings returns the string values of an attribute if it is a list.
// nil is returned if the attribute is not set or is not a list. Non-string
// values within the list won't be returned.
func (r *Rule) AttrStrings(key string) []string {
	attr, ok := r.attrs[key]
	if !ok {
		return nil
	}
	list, ok := attr.RHS.(*bzl.ListExpr)
	if !ok {
		return nil
	}
	strs := make([]string, 0, len(list.List))
	for _, e := range list.List {
		if str, ok := e.(*bzl.StringExpr); ok {
			strs = append(strs, str.Value)
		}
	}
	return strs
}

// DelAttr removes the named attribute from the rule.
func (r *Rule) DelAttr(key string) {
	delete(r.attrs, key)
	r.updated = true
}

// SetAttr adds or replaces the named attribute with an expression produced
// by ExprFromValue.
func (r *Rule) SetAttr(key string, value interface{}) {
	rhs := ExprFromValue(value)
	if attr, ok := r.attrs[key]; ok {
		attr.RHS = rhs
	} else {
		r.attrs[key] = &bzl.AssignExpr{
			LHS: &bzl.Ident{Name: key},
			RHS: rhs,
			Op:  "=",
		}
	}
	r.updated = true
}

// PrivateAttrKeys returns a sorted list of private attribute names.
func (r *Rule) PrivateAttrKeys() []string {
	keys := make([]string, 0, len(r.private))
	for k := range r.private {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// PrivateAttr return the private value associated with a key.
func (r *Rule) PrivateAttr(key string) interface{} {
	return r.private[key]
}

// SetPrivateAttr associates a value with a key. Unlike SetAttr, this value
// is not converted to a build syntax tree and will not be written to a build
// file.
func (r *Rule) SetPrivateAttr(key string, value interface{}) {
	r.private[key] = value
}

// Args returns positional arguments passed to a rule.
func (r *Rule) Args() []bzl.Expr {
	return r.args
}

// Insert marks this statement for insertion at the end of the file. Multiple
// statements will be inserted in the order Insert is called.
func (r *Rule) Insert(f *File) {
	// TODO(jayconrod): should rules always be inserted at the end? Should there
	// be some sort order?
	var stmt []bzl.Expr
	if f.function == nil {
		stmt = f.File.Stmt
	} else {
		stmt = f.function.stmt.Body
	}
	r.index = len(stmt)
	r.inserted = true
	f.Rules = append(f.Rules, r)
}

// IsEmpty returns true when the rule contains none of the attributes in attrs
// for its kind. attrs should contain attributes that make the rule buildable
// like srcs or deps and not descriptive attributes like name or visibility.
func (r *Rule) IsEmpty(info KindInfo) bool {
	if info.NonEmptyAttrs == nil {
		return false
	}
	for k := range info.NonEmptyAttrs {
		if _, ok := r.attrs[k]; ok {
			return false
		}
	}
	return true
}

func (r *Rule) sync() {
	if !r.updated {
		return
	}
	r.updated = false

	for _, k := range []string{"srcs", "deps"} {
		if attr, ok := r.attrs[k]; ok {
			bzl.Walk(attr.RHS, sortExprLabels)
		}
	}

	call := r.expr.(*bzl.CallExpr)
	call.X.(*bzl.Ident).Name = r.kind
	if len(r.attrs) > 1 {
		call.ForceMultiLine = true
	}

	list := make([]bzl.Expr, 0, len(r.args)+len(r.attrs))
	list = append(list, r.args...)
	for _, attr := range r.attrs {
		list = append(list, attr)
	}
	sortedAttrs := list[len(r.args):]
	key := func(e bzl.Expr) string { return e.(*bzl.AssignExpr).LHS.(*bzl.Ident).Name }
	sort.SliceStable(sortedAttrs, func(i, j int) bool {
		ki := key(sortedAttrs[i])
		kj := key(sortedAttrs[j])
		if cmp := bt.NamePriority[ki] - bt.NamePriority[kj]; cmp != 0 {
			return cmp < 0
		}
		return ki < kj
	})

	call.List = list
	r.updated = false
}

// ShouldKeep returns whether e is marked with a "# keep" comment. Kept
// expressions should not be removed or modified.
func ShouldKeep(e bzl.Expr) bool {
	for _, c := range append(e.Comment().Before, e.Comment().Suffix...) {
		text := strings.TrimSpace(strings.TrimPrefix(c.Token, "#"))
		if text == "keep" {
			return true
		}
	}
	return false
}

// CheckInternalVisibility overrides the given visibility if the package is
// internal.
func CheckInternalVisibility(rel, visibility string) string {
	if i := strings.LastIndex(rel, "/internal/"); i >= 0 {
		visibility = fmt.Sprintf("//%s:__subpackages__", rel[:i])
	} else if strings.HasPrefix(rel, "internal/") {
		visibility = "//:__subpackages__"
	}
	return visibility
}

type byAttrName []KeyValue

var _ sort.Interface = byAttrName{}

func (s byAttrName) Len() int {
	return len(s)
}

func (s byAttrName) Less(i, j int) bool {
	if cmp := bt.NamePriority[s[i].Key] - bt.NamePriority[s[j].Key]; cmp != 0 {
		return cmp < 0
	}
	return s[i].Key < s[j].Key
}

func (s byAttrName) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
