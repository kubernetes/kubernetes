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

package rule

// MergableAttrs is the set of attribute names for each kind of rule that
// may be merged. When an attribute is mergeable, a generated value may
// replace or augment an existing value. If an attribute is not mergeable,
// existing values are preserved. Generated non-mergeable attributes may
// still be added to a rule if there is no corresponding existing attribute.
type MergeableAttrs map[string]map[string]bool

// LoadInfo describes a file that Gazelle knows about and the symbols
// it defines.
type LoadInfo struct {
	Name    string
	Symbols []string
	After   []string
}

// KindInfo stores metadata for a kind or fule, for example, "go_library".
type KindInfo struct {
	// MatchAny is true if a rule of this kind may be matched with any rule
	// of the same kind, regardless of attributes, if exactly one rule is
	// present a build file.
	MatchAny bool

	// MatchAttrs is a list of attributes used in matching. For example,
	// for go_library, this list contains "importpath". Attributes are matched
	// in order.
	MatchAttrs []string

	// NonEmptyAttrs is a set of attributes that, if present, disqualify a rule
	// from being deleted after merge.
	NonEmptyAttrs map[string]bool

	// SubstituteAttrs is a set of attributes that should be substituted
	// after matching and before merging. For example, suppose generated rule A
	// references B via an "embed" attribute, and B matches against rule C.
	// The label for B in A's "embed" must be substituted with a label for C.
	// "embed" would need to be in this set.
	SubstituteAttrs map[string]bool

	// MergeableAttrs is a set of attributes that should be merged before
	// dependency resolution. See rule.Merge.
	MergeableAttrs map[string]bool

	// ResolveAttrs is a set of attributes that should be merged after
	// dependency resolution. See rule.Merge.
	ResolveAttrs map[string]bool
}
