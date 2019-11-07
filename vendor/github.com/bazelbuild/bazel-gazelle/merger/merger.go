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

// Package merger provides functions for merging generated rules into
// existing build files.
//
// Gazelle's normal workflow is roughly as follows:
//
// 1. Read metadata from sources.
//
// 2. Generate new rules.
//
// 3. Merge newly generated rules with rules in the existing build file
// if there is one.
//
// 4. Build an index of merged library rules for dependency resolution.
//
// 5. Resolve dependencies (i.e., convert import strings to deps labels).
//
// 6. Merge the newly resolved dependencies.
//
// 7. Write the merged file back to disk.
//
// This package is used for sets 3 and 6 above.
package merger

import (
	"fmt"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/rule"
)

// Phase indicates which attributes should be merged in matching rules.
type Phase int

const (
	// The pre-resolve merge is performed before rules are indexed for dependency
	// resolution. All attributes not related to dependencies are merged
	// (i.e., rule.KindInfo.MergeableAttrs). This merge must be performed
	// before indexing because attributes related to indexing (e.g.,
	// srcs, importpath) will be affected.
	PreResolve Phase = iota

	// The post-resolve merge is performed after rules are indexed. All attributes
	// related to dependencies are merged (i.e., rule.KindInfo.ResolveAttrs).
	PostResolve
)

// MergeFile combines information from newly generated rules with matching
// rules in an existing build file. MergeFile can also delete rules which
// are empty after merging.
//
// oldFile is the file to merge. It must not be nil.
//
// emptyRules is a list of stub rules (with no attributes other than name)
// which were not generated. These are merged with matching rules. The merged
// rules are deleted if they contain no attributes that make them buildable
// (e.g., srcs, deps, anything in rule.KindInfo.NonEmptyAttrs).
//
// genRules is a list of newly generated rules. These are merged with
// matching rules. A rule matches if it has the same kind and name or if
// some other attribute in rule.KindInfo.MatchAttrs matches (e.g.,
// "importpath" in go_library). Elements of genRules that don't match
// any existing rule are appended to the end of oldFile.
//
// phase indicates whether this is a pre- or post-resolve merge. Different
// attributes (rule.KindInfo.MergeableAttrs or ResolveAttrs) will be merged.
//
// kinds maps rule kinds (e.g., "go_library") to metadata that helps merge
// rules of that kind.
//
// When a generated and existing rule are merged, each attribute is merged
// separately. If an attribute is mergeable (according to KindInfo), values
// from the existing attribute are replaced by values from the generated
// attribute. Comments are preserved on values that are present in both
// versions of the attribute. If at attribute is not mergeable, the generated
// version of the attribute will be added if no existing attribute is present;
// otherwise, the existing attribute will be preserved.
//
// Note that "# keep" comments affect merging. If a value within an existing
// attribute is marked with a "# keep" comment, it will not be removed.
// If an attribute is marked with a "# keep" comment, it will not be merged.
// If a rule is marked with a "# keep" comment, the whole rule will not
// be modified.
func MergeFile(oldFile *rule.File, emptyRules, genRules []*rule.Rule, phase Phase, kinds map[string]rule.KindInfo) {
	getMergeAttrs := func(r *rule.Rule) map[string]bool {
		if phase == PreResolve {
			return kinds[r.Kind()].MergeableAttrs
		} else {
			return kinds[r.Kind()].ResolveAttrs
		}
	}

	// Merge empty rules into the file and delete any rules which become empty.
	for _, emptyRule := range emptyRules {
		if oldRule, _ := Match(oldFile.Rules, emptyRule, kinds[emptyRule.Kind()]); oldRule != nil {
			if oldRule.ShouldKeep() {
				continue
			}
			rule.MergeRules(emptyRule, oldRule, getMergeAttrs(emptyRule), oldFile.Path)
			if oldRule.IsEmpty(kinds[oldRule.Kind()]) {
				oldRule.Delete()
			}
		}
	}
	oldFile.Sync()

	// Match generated rules with existing rules in the file. Keep track of
	// rules with non-standard names.
	matchRules := make([]*rule.Rule, len(genRules))
	matchErrors := make([]error, len(genRules))
	substitutions := make(map[string]string)
	for i, genRule := range genRules {
		oldRule, err := Match(oldFile.Rules, genRule, kinds[genRule.Kind()])
		if err != nil {
			// TODO(jayconrod): add a verbose mode and log errors. They are too chatty
			// to print by default.
			matchErrors[i] = err
			continue
		}
		matchRules[i] = oldRule
		if oldRule != nil {
			if oldRule.Name() != genRule.Name() {
				substitutions[genRule.Name()] = oldRule.Name()
			}
		}
	}

	// Rename labels in generated rules that refer to other generated rules.
	if len(substitutions) > 0 {
		for _, genRule := range genRules {
			substituteRule(genRule, substitutions, kinds[genRule.Kind()])
		}
	}

	// Merge generated rules with existing rules or append to the end of the file.
	for i, genRule := range genRules {
		if matchErrors[i] != nil {
			continue
		}
		if matchRules[i] == nil {
			genRule.Insert(oldFile)
		} else {
			rule.MergeRules(genRule, matchRules[i], getMergeAttrs(genRule), oldFile.Path)
		}
	}
}

// substituteRule replaces local labels (those beginning with ":", referring to
// targets in the same package) according to a substitution map. This is used
// to update generated rules before merging when the corresponding existing
// rules have different names. If substituteRule replaces a string, it returns
// a new expression; it will not modify the original expression.
func substituteRule(r *rule.Rule, substitutions map[string]string, info rule.KindInfo) {
	for attr := range info.SubstituteAttrs {
		if expr := r.Attr(attr); expr != nil {
			expr = rule.MapExprStrings(expr, func(s string) string {
				if rename, ok := substitutions[strings.TrimPrefix(s, ":")]; ok {
					return ":" + rename
				} else {
					return s
				}
			})
			r.SetAttr(attr, expr)
		}
	}
}

// Match searches for a rule that can be merged with x in rules.
//
// A rule is considered a match if its kind is equal to x's kind AND either its
// name is equal OR at least one of the attributes in matchAttrs is equal.
//
// If there are no matches, nil and nil are returned.
//
// If a rule has the same name but a different kind, nill and an error
// are returned.
//
// If there is exactly one match, the rule and nil are returned.
//
// If there are multiple matches, match will attempt to disambiguate, based on
// the quality of the match (name match is best, then attribute match in the
// order that attributes are listed). If disambiguation is successful,
// the rule and nil are returned. Otherwise, nil and an error are returned.
func Match(rules []*rule.Rule, x *rule.Rule, info rule.KindInfo) (*rule.Rule, error) {
	xname := x.Name()
	xkind := x.Kind()
	var nameMatches []*rule.Rule
	var kindMatches []*rule.Rule
	for _, y := range rules {
		if xname == y.Name() {
			nameMatches = append(nameMatches, y)
		}
		if xkind == y.Kind() {
			kindMatches = append(kindMatches, y)
		}
	}

	if len(nameMatches) == 1 {
		y := nameMatches[0]
		if xkind != y.Kind() {
			return nil, fmt.Errorf("could not merge %s(%s): a rule of the same name has kind %s", xkind, xname, y.Kind())
		}
		return y, nil
	}
	if len(nameMatches) > 1 {
		return nil, fmt.Errorf("could not merge %s(%s): multiple rules have the same name", xkind, xname)
	}

	for _, key := range info.MatchAttrs {
		var attrMatches []*rule.Rule
		xvalue := x.AttrString(key)
		if xvalue == "" {
			continue
		}
		for _, y := range kindMatches {
			if xvalue == y.AttrString(key) {
				attrMatches = append(attrMatches, y)
			}
		}
		if len(attrMatches) == 1 {
			return attrMatches[0], nil
		} else if len(attrMatches) > 1 {
			return nil, fmt.Errorf("could not merge %s(%s): multiple rules have the same attribute %s = %q", xkind, xname, key, xvalue)
		}
	}

	if info.MatchAny {
		if len(kindMatches) == 1 {
			return kindMatches[0], nil
		} else if len(kindMatches) > 1 {
			return nil, fmt.Errorf("could not merge %s(%s): multiple rules have the same kind but different names", xkind, xname)
		}
	}

	return nil, nil
}
