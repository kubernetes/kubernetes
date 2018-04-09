/* Copyright 2016 The Bazel Authors. All rights reserved.

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

// Package merger provides methods for merging parsed BUILD files.
package merger

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	bf "github.com/bazelbuild/buildtools/build"
)

const keep = "keep" // marker in srcs or deps to tell gazelle to preserve.

// MergableAttrs is the set of attribute names for each kind of rule that
// may be merged. When an attribute is mergeable, a generated value may
// replace or augment an existing value. If an attribute is not mergeable,
// existing values are preserved. Generated non-mergeable attributes may
// still be added to a rule if there is no corresponding existing attribute.
type MergeableAttrs map[string]map[string]bool

var (
	// PreResolveAttrs is the set of attributes that should be merged before
	// dependency resolution, i.e., everything except deps.
	PreResolveAttrs MergeableAttrs

	// PostResolveAttrs is the set of attributes that should be merged after
	// dependency resolution, i.e., deps.
	PostResolveAttrs MergeableAttrs

	// RepoAttrs is the set of attributes that should be merged in repository
	// rules in WORKSPACE.
	RepoAttrs MergeableAttrs

	// nonEmptyAttrs is the set of attributes that disqualify a rule from being
	// deleted after merge.
	nonEmptyAttrs MergeableAttrs
)

func init() {
	PreResolveAttrs = make(MergeableAttrs)
	PostResolveAttrs = make(MergeableAttrs)
	RepoAttrs = make(MergeableAttrs)
	nonEmptyAttrs = make(MergeableAttrs)
	for _, set := range []struct {
		mergeableAttrs MergeableAttrs
		kinds, attrs   []string
	}{
		{
			mergeableAttrs: PreResolveAttrs,
			kinds: []string{
				"go_library",
				"go_binary",
				"go_test",
				"go_proto_library",
				"proto_library",
			},
			attrs: []string{
				"srcs",
			},
		}, {
			mergeableAttrs: PreResolveAttrs,
			kinds: []string{
				"go_library",
				"go_proto_library",
			},
			attrs: []string{
				"importpath",
			},
		}, {
			mergeableAttrs: PreResolveAttrs,
			kinds: []string{
				"go_library",
				"go_binary",
				"go_test",
				"go_proto_library",
			},
			attrs: []string{
				"cgo",
				"clinkopts",
				"copts",
				"embed",
			},
		}, {
			mergeableAttrs: PreResolveAttrs,
			kinds: []string{
				"go_proto_library",
			},
			attrs: []string{
				"proto",
			},
		}, {
			mergeableAttrs: PostResolveAttrs,
			kinds: []string{
				"go_library",
				"go_binary",
				"go_test",
				"go_proto_library",
				"proto_library",
			},
			attrs: []string{
				"deps",
				config.GazelleImportsKey,
			},
		}, {
			mergeableAttrs: RepoAttrs,
			kinds: []string{
				"go_repository",
			},
			attrs: []string{
				"commit",
				"importpath",
				"remote",
				"sha256",
				"strip_prefix",
				"tag",
				"type",
				"urls",
				"vcs",
			},
		}, {
			mergeableAttrs: nonEmptyAttrs,
			kinds: []string{
				"go_binary",
				"go_library",
				"go_test",
				"proto_library",
			},
			attrs: []string{
				"srcs",
				"deps",
			},
		}, {
			mergeableAttrs: nonEmptyAttrs,
			kinds: []string{
				"go_binary",
				"go_library",
				"go_test",
			},
			attrs: []string{
				"embed",
			},
		}, {
			mergeableAttrs: nonEmptyAttrs,
			kinds: []string{
				"go_proto_library",
			},
			attrs: []string{
				"proto",
			},
		},
	} {
		for _, kind := range set.kinds {
			if set.mergeableAttrs[kind] == nil {
				set.mergeableAttrs[kind] = make(map[string]bool)
			}
			for _, attr := range set.attrs {
				set.mergeableAttrs[kind][attr] = true
			}
		}
	}
}

// MergeFile merges the rules in genRules with matching rules in oldFile and
// adds unmatched rules to the end of the merged file. MergeFile also merges
// rules in empty with matching rules in oldFile and deletes rules that
// are empty after merging. attrs is the set of attributes to merge. Attributes
// not in this set will be left alone if they already exist.
func MergeFile(genRules []bf.Expr, empty []bf.Expr, oldFile *bf.File, attrs MergeableAttrs) (mergedFile *bf.File, mergedRules []bf.Expr) {
	// Merge empty rules into the file and delete any rules which become empty.
	mergedFile = new(bf.File)
	*mergedFile = *oldFile
	mergedFile.Stmt = append([]bf.Expr{}, oldFile.Stmt...)
	var deletedIndices []int
	for _, s := range empty {
		emptyCall := s.(*bf.CallExpr)
		if oldCall, i, _ := match(oldFile.Stmt, emptyCall); oldCall != nil {
			mergedRule := mergeRule(emptyCall, oldCall, attrs, oldFile.Path)
			if isRuleEmpty(mergedRule) {
				deletedIndices = append(deletedIndices, i)
			} else {
				mergedFile.Stmt[i] = mergedRule
			}
		}
	}
	if len(deletedIndices) > 0 {
		sort.Ints(deletedIndices)
		mergedFile.Stmt = deleteIndices(mergedFile.Stmt, deletedIndices)
	}

	// Match generated rules with existing rules in the file. Keep track of
	// rules with non-standard names.
	matchIndices := make([]int, len(genRules))
	matchErrors := make([]error, len(genRules))
	substitutions := make(map[string]string)
	for i, s := range genRules {
		genCall := s.(*bf.CallExpr)
		oldCall, oldIndex, err := match(mergedFile.Stmt, genCall)
		if err != nil {
			// TODO(jayconrod): add a verbose mode and log errors. They are too chatty
			// to print by default.
			matchErrors[i] = err
			continue
		}
		matchIndices[i] = oldIndex // < 0 indicates no match
		if oldCall != nil {
			oldRule := bf.Rule{Call: oldCall}
			genRule := bf.Rule{Call: genCall}
			oldName := oldRule.Name()
			genName := genRule.Name()
			if oldName != genName {
				substitutions[genName] = oldName
			}
		}
	}

	// Rename labels in generated rules that refer to other generated rules.
	if len(substitutions) > 0 {
		genRules = append([]bf.Expr{}, genRules...)
		for i, s := range genRules {
			genRules[i] = substituteRule(s.(*bf.CallExpr), substitutions)
		}
	}

	// Merge generated rules with existing rules or append to the end of the file.
	for i := range genRules {
		if matchErrors[i] != nil {
			continue
		}
		if matchIndices[i] < 0 {
			mergedFile.Stmt = append(mergedFile.Stmt, genRules[i])
			mergedRules = append(mergedRules, genRules[i])
		} else {
			mergedRule := mergeRule(genRules[i].(*bf.CallExpr), mergedFile.Stmt[matchIndices[i]].(*bf.CallExpr), attrs, oldFile.Path)
			mergedFile.Stmt[matchIndices[i]] = mergedRule
			mergedRules = append(mergedRules, mergedRule)
		}
	}

	return mergedFile, mergedRules
}

// mergeRule combines information from gen and old and returns an updated rule.
// Both rules must be non-nil and must have the same kind and same name.
// attrs is the set of attributes which may be merged.
// If nil is returned, the rule should be deleted.
func mergeRule(gen, old *bf.CallExpr, attrs MergeableAttrs, filename string) bf.Expr {
	if old != nil && shouldKeep(old) {
		return old
	}

	genRule := bf.Rule{Call: gen}
	oldRule := bf.Rule{Call: old}
	merged := *old
	merged.List = nil
	mergedRule := bf.Rule{Call: &merged}

	// Copy unnamed arguments from the old rule without merging. The only rule
	// generated with unnamed arguments is go_prefix, which we currently
	// leave in place.
	// TODO: maybe gazelle should allow the prefix to be changed.
	for _, a := range old.List {
		if b, ok := a.(*bf.BinaryExpr); ok && b.Op == "=" {
			break
		}
		merged.List = append(merged.List, a)
	}

	// Merge attributes from the old rule. Preserve comments on old attributes.
	// Assume generated attributes have no comments.
	kind := oldRule.Kind()
	for _, k := range oldRule.AttrKeys() {
		oldAttr := oldRule.AttrDefn(k)
		if !attrs[kind][k] || shouldKeep(oldAttr) {
			merged.List = append(merged.List, oldAttr)
			continue
		}

		oldExpr := oldAttr.Y
		genExpr := genRule.Attr(k)
		mergedExpr, err := mergeExpr(genExpr, oldExpr)
		if err != nil {
			start, end := oldExpr.Span()
			log.Printf("%s:%d.%d-%d.%d: could not merge expression", filename, start.Line, start.LineRune, end.Line, end.LineRune)
			mergedExpr = oldExpr
		}
		if mergedExpr != nil {
			mergedAttr := *oldAttr
			mergedAttr.Y = mergedExpr
			merged.List = append(merged.List, &mergedAttr)
		}
	}

	// Merge attributes from genRule that we haven't processed already.
	for _, k := range genRule.AttrKeys() {
		if mergedRule.Attr(k) == nil {
			mergedRule.SetAttr(k, genRule.Attr(k))
		}
	}

	return &merged
}

// mergeExpr combines information from gen and old and returns an updated
// expression. The following kinds of expressions are recognized:
//
//   * nil
//   * strings (can only be merged with strings)
//   * lists of strings
//   * a call to select with a dict argument. The dict keys must be strings,
//     and the values must be lists of strings.
//   * a list of strings combined with a select call using +. The list must
//     be the left operand.
//
// An error is returned if the expressions can't be merged, for example
// because they are not in one of the above formats.
func mergeExpr(gen, old bf.Expr) (bf.Expr, error) {
	if shouldKeep(old) {
		return old, nil
	}
	if gen == nil && (old == nil || isScalar(old)) {
		return nil, nil
	}
	if isScalar(gen) {
		return gen, nil
	}

	genExprs, err := extractPlatformStringsExprs(gen)
	if err != nil {
		return nil, err
	}
	oldExprs, err := extractPlatformStringsExprs(old)
	if err != nil {
		return nil, err
	}
	mergedExprs, err := mergePlatformStringsExprs(genExprs, oldExprs)
	if err != nil {
		return nil, err
	}
	return makePlatformStringsExpr(mergedExprs), nil
}

// platformStringsExprs is a set of sub-expressions that match the structure
// of package.PlatformStrings. rules.Generator produces expressions that
// follow this structure for srcs, deps, and other attributes, so this matches
// all non-scalar expressions generated by Gazelle.
//
// The matched expression has the form:
//
// [] + select({}) + select({}) + select({})
//
// The four collections may appear in any order, and some or all of them may
// be omitted (all fields are nil for a nil expression).
type platformStringsExprs struct {
	generic            *bf.ListExpr
	os, arch, platform *bf.DictExpr
}

// extractPlatformStringsExprs matches an expression and attempts to extract
// sub-expressions in platformStringsExprs. The sub-expressions can then be
// merged with corresponding sub-expressions. Any field in the returned
// structure may be nil. An error is returned if the given expression does
// not follow the pattern described by platformStringsExprs.
func extractPlatformStringsExprs(expr bf.Expr) (platformStringsExprs, error) {
	var ps platformStringsExprs
	if expr == nil {
		return ps, nil
	}

	// Break the expression into a sequence of expressions combined with +.
	var parts []bf.Expr
	for {
		binop, ok := expr.(*bf.BinaryExpr)
		if !ok {
			parts = append(parts, expr)
			break
		}
		parts = append(parts, binop.Y)
		expr = binop.X
	}

	// Process each part. They may be in any order.
	for _, part := range parts {
		switch part := part.(type) {
		case *bf.ListExpr:
			if ps.generic != nil {
				return platformStringsExprs{}, fmt.Errorf("expression could not be matched: multiple list expressions")
			}
			ps.generic = part

		case *bf.CallExpr:
			x, ok := part.X.(*bf.LiteralExpr)
			if !ok || x.Token != "select" || len(part.List) != 1 {
				return platformStringsExprs{}, fmt.Errorf("expression could not be matched: callee other than select or wrong number of args")
			}
			arg, ok := part.List[0].(*bf.DictExpr)
			if !ok {
				return platformStringsExprs{}, fmt.Errorf("expression could not be matched: select argument not dict")
			}
			var dict **bf.DictExpr
			for _, item := range arg.List {
				kv := item.(*bf.KeyValueExpr) // parser guarantees this
				k, ok := kv.Key.(*bf.StringExpr)
				if !ok {
					return platformStringsExprs{}, fmt.Errorf("expression could not be matched: dict keys are not all strings")
				}
				if k.Value == "//conditions:default" {
					continue
				}
				key, err := label.Parse(k.Value)
				if err != nil {
					return platformStringsExprs{}, fmt.Errorf("expression could not be matched: dict key is not label: %q", k.Value)
				}
				if config.KnownOSSet[key.Name] {
					dict = &ps.os
					break
				}
				if config.KnownArchSet[key.Name] {
					dict = &ps.arch
					break
				}
				osArch := strings.Split(key.Name, "_")
				if len(osArch) != 2 || !config.KnownOSSet[osArch[0]] || !config.KnownArchSet[osArch[1]] {
					return platformStringsExprs{}, fmt.Errorf("expression could not be matched: dict key contains unknown platform: %q", k.Value)
				}
				dict = &ps.platform
				break
			}
			if dict == nil {
				// We could not identify the dict because it's empty or only contains
				// //conditions:default. We'll call it the platform dict to avoid
				// dropping it.
				dict = &ps.platform
			}
			if *dict != nil {
				return platformStringsExprs{}, fmt.Errorf("expression could not be matched: multiple selects that are either os-specific, arch-specific, or platform-specific")
			}
			*dict = arg
		}
	}
	return ps, nil
}

// makePlatformStringsExpr constructs a single expression from the
// sub-expressions in ps.
func makePlatformStringsExpr(ps platformStringsExprs) bf.Expr {
	makeSelect := func(dict *bf.DictExpr) bf.Expr {
		return &bf.CallExpr{
			X:    &bf.LiteralExpr{Token: "select"},
			List: []bf.Expr{dict},
		}
	}
	forceMultiline := func(e bf.Expr) {
		switch e := e.(type) {
		case *bf.ListExpr:
			e.ForceMultiLine = true
		case *bf.CallExpr:
			e.List[0].(*bf.DictExpr).ForceMultiLine = true
		}
	}

	var parts []bf.Expr
	if ps.generic != nil {
		parts = append(parts, ps.generic)
	}
	if ps.os != nil {
		parts = append(parts, makeSelect(ps.os))
	}
	if ps.arch != nil {
		parts = append(parts, makeSelect(ps.arch))
	}
	if ps.platform != nil {
		parts = append(parts, makeSelect(ps.platform))
	}

	if len(parts) == 0 {
		return nil
	}
	if len(parts) == 1 {
		return parts[0]
	}
	expr := parts[0]
	forceMultiline(expr)
	for _, part := range parts[1:] {
		forceMultiline(part)
		expr = &bf.BinaryExpr{
			Op: "+",
			X:  expr,
			Y:  part,
		}
	}
	return expr
}

func mergePlatformStringsExprs(gen, old platformStringsExprs) (platformStringsExprs, error) {
	var ps platformStringsExprs
	var err error
	ps.generic = mergeList(gen.generic, old.generic)
	if ps.os, err = mergeDict(gen.os, old.os); err != nil {
		return platformStringsExprs{}, err
	}
	if ps.arch, err = mergeDict(gen.arch, old.arch); err != nil {
		return platformStringsExprs{}, err
	}
	if ps.platform, err = mergeDict(gen.platform, old.platform); err != nil {
		return platformStringsExprs{}, err
	}
	return ps, nil
}

func mergeList(gen, old *bf.ListExpr) *bf.ListExpr {
	if old == nil {
		return gen
	}
	if gen == nil {
		gen = &bf.ListExpr{List: []bf.Expr{}}
	}

	// Build a list of strings from the gen list and keep matching strings
	// in the old list. This preserves comments. Also keep anything with
	// a "# keep" comment, whether or not it's in the gen list.
	genSet := make(map[string]bool)
	for _, v := range gen.List {
		if s := stringValue(v); s != "" {
			genSet[s] = true
		}
	}

	var merged []bf.Expr
	kept := make(map[string]bool)
	keepComment := false
	for _, v := range old.List {
		s := stringValue(v)
		if keep := shouldKeep(v); keep || genSet[s] {
			keepComment = keepComment || keep
			merged = append(merged, v)
			if s != "" {
				kept[s] = true
			}
		}
	}

	// Add anything in the gen list that wasn't kept.
	for _, v := range gen.List {
		if s := stringValue(v); kept[s] {
			continue
		}
		merged = append(merged, v)
	}

	if len(merged) == 0 {
		return nil
	}
	return &bf.ListExpr{
		List:           merged,
		ForceMultiLine: gen.ForceMultiLine || old.ForceMultiLine || keepComment,
	}
}

func mergeDict(gen, old *bf.DictExpr) (*bf.DictExpr, error) {
	if old == nil {
		return gen, nil
	}
	if gen == nil {
		gen = &bf.DictExpr{List: []bf.Expr{}}
	}

	var entries []*dictEntry
	entryMap := make(map[string]*dictEntry)

	for _, kv := range old.List {
		k, v, err := dictEntryKeyValue(kv)
		if err != nil {
			return nil, err
		}
		if _, ok := entryMap[k]; ok {
			return nil, fmt.Errorf("old dict contains more than one case named %q", k)
		}
		e := &dictEntry{key: k, oldValue: v}
		entries = append(entries, e)
		entryMap[k] = e
	}

	for _, kv := range gen.List {
		k, v, err := dictEntryKeyValue(kv)
		if err != nil {
			return nil, err
		}
		e, ok := entryMap[k]
		if !ok {
			e = &dictEntry{key: k}
			entries = append(entries, e)
			entryMap[k] = e
		}
		e.genValue = v
	}

	keys := make([]string, 0, len(entries))
	haveDefault := false
	for _, e := range entries {
		e.mergedValue = mergeList(e.genValue, e.oldValue)
		if e.key == "//conditions:default" {
			// Keep the default case, even if it's empty.
			haveDefault = true
			if e.mergedValue == nil {
				e.mergedValue = &bf.ListExpr{}
			}
		} else if e.mergedValue != nil {
			keys = append(keys, e.key)
		}
	}
	if len(keys) == 0 && (!haveDefault || len(entryMap["//conditions:default"].mergedValue.List) == 0) {
		return nil, nil
	}
	sort.Strings(keys)
	// Always put the default case last.
	if haveDefault {
		keys = append(keys, "//conditions:default")
	}

	mergedEntries := make([]bf.Expr, len(keys))
	for i, k := range keys {
		e := entryMap[k]
		mergedEntries[i] = &bf.KeyValueExpr{
			Key:   &bf.StringExpr{Value: e.key},
			Value: e.mergedValue,
		}
	}

	return &bf.DictExpr{List: mergedEntries, ForceMultiLine: true}, nil
}

type dictEntry struct {
	key                             string
	oldValue, genValue, mergedValue *bf.ListExpr
}

func dictEntryKeyValue(e bf.Expr) (string, *bf.ListExpr, error) {
	kv, ok := e.(*bf.KeyValueExpr)
	if !ok {
		return "", nil, fmt.Errorf("dict entry was not a key-value pair: %#v", e)
	}
	k, ok := kv.Key.(*bf.StringExpr)
	if !ok {
		return "", nil, fmt.Errorf("dict key was not string: %#v", kv.Key)
	}
	v, ok := kv.Value.(*bf.ListExpr)
	if !ok {
		return "", nil, fmt.Errorf("dict value was not list: %#v", kv.Value)
	}
	return k.Value, v, nil
}

// substituteAttrs contains a list of attributes for each kind that should be
// processed by substituteRule and substituteExpr. Note that "name" does not
// need to be substituted since it's not mergeable.
var substituteAttrs = map[string][]string{
	"go_binary":        {"embed"},
	"go_library":       {"embed"},
	"go_test":          {"embed"},
	"go_proto_library": {"proto"},
}

// substituteRule replaces local labels (those beginning with ":", referring to
// targets in the same package) according to a substitution map. This is used
// to update generated rules before merging when the corresponding existing
// rules have different names. If substituteRule replaces a string, it returns
// a new expression; it will not modify the original expression.
func substituteRule(call *bf.CallExpr, substitutions map[string]string) *bf.CallExpr {
	rule := bf.Rule{Call: call}
	attrs, ok := substituteAttrs[rule.Kind()]
	if !ok {
		return call
	}

	didCopy := false
	for i, arg := range call.List {
		kv, ok := arg.(*bf.BinaryExpr)
		if !ok || kv.Op != "=" {
			continue
		}
		key, ok := kv.X.(*bf.LiteralExpr)
		shouldRename := false
		for _, k := range attrs {
			shouldRename = shouldRename || key.Token == k
		}
		if !shouldRename {
			continue
		}

		value := substituteExpr(kv.Y, substitutions)
		if value != kv.Y {
			if !didCopy {
				didCopy = true
				callCopy := *call
				callCopy.List = append([]bf.Expr{}, call.List...)
				call = &callCopy
			}
			kvCopy := *kv
			kvCopy.Y = value
			call.List[i] = &kvCopy
		}
	}
	return call
}

// substituteExpr replaces local labels according to a substitution map.
// It only supports string and list expressions (which should be sufficient
// for generated rules). If it replaces a string, it returns a new expression;
// otherwise, it returns e.
func substituteExpr(e bf.Expr, substitutions map[string]string) bf.Expr {
	switch e := e.(type) {
	case *bf.StringExpr:
		if rename, ok := substitutions[strings.TrimPrefix(e.Value, ":")]; ok {
			return &bf.StringExpr{Value: ":" + rename}
		}
	case *bf.ListExpr:
		var listCopy *bf.ListExpr
		for i, elem := range e.List {
			renamed := substituteExpr(elem, substitutions)
			if renamed != elem {
				if listCopy == nil {
					listCopy = new(bf.ListExpr)
					*listCopy = *e
					listCopy.List = append([]bf.Expr{}, e.List...)
				}
				listCopy.List[i] = renamed
			}
		}
		if listCopy != nil {
			return listCopy
		}
	}
	return e
}

// shouldKeep returns whether an expression from the original file should be
// preserved. This is true if it has a prefix or end-of-line comment "keep".
// Note that bf.Rewrite recognizes "keep sorted" comments which are different,
// so we don't recognize comments that only start with "keep".
func shouldKeep(e bf.Expr) bool {
	for _, c := range append(e.Comment().Before, e.Comment().Suffix...) {
		text := strings.TrimSpace(strings.TrimPrefix(c.Token, "#"))
		if text == keep {
			return true
		}
	}
	return false
}

// matchAttrs contains lists of attributes for each kind that are used in
// matching. For example, importpath attributes can be used to match go_library
// rules, even when the names are different.
var matchAttrs = map[string][]string{
	"go_library":       {"importpath"},
	"go_proto_library": {"importpath"},
	"go_repository":    {"importpath"},
}

// matchAny is a set of kinds which may be matched regardless of attributes.
// For example, if there is only one go_binary in a package, any go_binary
// rule will match.
var matchAny = map[string]bool{"go_binary": true}

// match searches for a rule that can be merged with x in stmts.
//
// A rule is considered a match if its kind is equal to x's kind AND either its
// name is equal OR at least one of the attributes in matchAttrs is equal.
//
// If there are no matches, nil, -1, and nil are returned.
//
// If a rule has the same name but a different kind, nil, -1, and an error
// are returned.
//
// If there is exactly one match, the rule, its index in stmts, and nil
// are returned.
//
// If there are multiple matches, match will attempt to disambiguate, based on
// the quality of the match (name match is best, then attribute match in the
// order that attributes are listed). If disambiguation is successful,
// the rule, its index in stmts, and nil are returned. Otherwise, nil, -1,
// and an error are returned.
func match(stmts []bf.Expr, x *bf.CallExpr) (*bf.CallExpr, int, error) {
	type matchInfo struct {
		rule  bf.Rule
		index int
	}

	xr := bf.Rule{Call: x}
	xname := xr.Name()
	xkind := xr.Kind()
	var nameMatches []matchInfo
	var kindMatches []matchInfo
	for i, s := range stmts {
		y, ok := s.(*bf.CallExpr)
		if !ok {
			continue
		}
		yr := bf.Rule{Call: y}
		if xname == yr.Name() {
			nameMatches = append(nameMatches, matchInfo{yr, i})
		}
		if xkind == yr.Kind() {
			kindMatches = append(kindMatches, matchInfo{yr, i})
		}
	}

	if len(nameMatches) == 1 {
		if ykind := nameMatches[0].rule.Kind(); xkind != ykind {
			return nil, -1, fmt.Errorf("could not merge %s(%s): a rule of the same name has kind %s", xkind, xname, ykind)
		}
		return nameMatches[0].rule.Call, nameMatches[0].index, nil
	}
	if len(nameMatches) > 1 {
		return nil, -1, fmt.Errorf("could not merge %s(%s): multiple rules have the same name", xname)
	}

	attrs := matchAttrs[xr.Kind()]
	for _, key := range attrs {
		var attrMatches []matchInfo
		xvalue := xr.AttrString(key)
		if xvalue == "" {
			continue
		}
		for _, m := range kindMatches {
			if xvalue == m.rule.AttrString(key) {
				attrMatches = append(attrMatches, m)
			}
		}
		if len(attrMatches) == 1 {
			return attrMatches[0].rule.Call, attrMatches[0].index, nil
		} else if len(attrMatches) > 1 {
			return nil, -1, fmt.Errorf("could not merge %s(%s): multiple rules have the same attribute %s = %q", xkind, xname, key, xvalue)
		}
	}

	if matchAny[xkind] {
		if len(kindMatches) == 1 {
			return kindMatches[0].rule.Call, kindMatches[0].index, nil
		} else if len(kindMatches) > 1 {
			return nil, -1, fmt.Errorf("could not merge %s(%s): multiple rules have the same kind but different names", xkind, xname)
		}
	}

	return nil, -1, nil
}

func kind(c *bf.CallExpr) string {
	return (&bf.Rule{Call: c}).Kind()
}

func name(c *bf.CallExpr) string {
	return (&bf.Rule{Call: c}).Name()
}

// isRuleEmpty returns true if a rule cannot be built because it has no sources,
// dependencies, or embeds after merging. This is based on a per-kind whitelist
// of attributes. Other attributes, like "name" and "visibility" don't affect
// emptiness. Always returns false for expressions that aren't in the known
// set of rules.
func isRuleEmpty(e bf.Expr) bool {
	c, ok := e.(*bf.CallExpr)
	if !ok {
		return false
	}
	r := bf.Rule{Call: c}
	kind := r.Kind()
	if nonEmptyAttrs[kind] == nil {
		return false
	}
	for _, attr := range r.AttrKeys() {
		if nonEmptyAttrs[kind][attr] {
			return false
		}
	}
	return true
}

func isScalar(e bf.Expr) bool {
	switch e.(type) {
	case *bf.StringExpr, *bf.LiteralExpr:
		return true
	default:
		return false
	}
}

func stringValue(e bf.Expr) string {
	s, ok := e.(*bf.StringExpr)
	if !ok {
		return ""
	}
	return s.Value
}

// deleteIndices copies a list, dropping elements at deletedIndices.
// deletedIndices must be sorted.
func deleteIndices(stmt []bf.Expr, deletedIndices []int) []bf.Expr {
	kept := make([]bf.Expr, 0, len(stmt)-len(deletedIndices))
	di := 0
	for i, s := range stmt {
		if di < len(deletedIndices) && i == deletedIndices[di] {
			di++
			continue
		}
		kept = append(kept, s)
	}
	return kept
}
