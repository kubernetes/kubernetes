/* Copyright 2017 The Bazel Authors. All rights reserved.

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

import (
	"sort"
	"strings"

	bzl "github.com/bazelbuild/buildtools/build"
)

// sortExprLabels sorts lists of strings using the same order as buildifier.
// Buildifier also sorts string lists, but not those involved with "select"
// expressions. This function is intended to be used with bzl.Walk.
func sortExprLabels(e bzl.Expr, _ []bzl.Expr) {
	list, ok := e.(*bzl.ListExpr)
	if !ok || len(list.List) == 0 {
		return
	}

	keys := make([]stringSortKey, len(list.List))
	for i, elem := range list.List {
		s, ok := elem.(*bzl.StringExpr)
		if !ok {
			return // don't sort lists unless all elements are strings
		}
		keys[i] = makeSortKey(i, s)
	}

	before := keys[0].x.Comment().Before
	keys[0].x.Comment().Before = nil
	sort.Sort(byStringExpr(keys))
	keys[0].x.Comment().Before = append(before, keys[0].x.Comment().Before...)
	for i, k := range keys {
		list.List[i] = k.x
	}
}

// Code below this point is adapted from
// github.com/bazelbuild/buildtools/build/rewrite.go

// A stringSortKey records information about a single string literal to be
// sorted. The strings are first grouped into four phases: most strings,
// strings beginning with ":", strings beginning with "//", and strings
// beginning with "@". The next significant part of the comparison is the list
// of elements in the value, where elements are split at `.' and `:'. Finally
// we compare by value and break ties by original index.
type stringSortKey struct {
	phase    int
	split    []string
	value    string
	original int
	x        bzl.Expr
}

func makeSortKey(index int, x *bzl.StringExpr) stringSortKey {
	key := stringSortKey{
		value:    x.Value,
		original: index,
		x:        x,
	}

	switch {
	case strings.HasPrefix(x.Value, ":"):
		key.phase = 1
	case strings.HasPrefix(x.Value, "//"):
		key.phase = 2
	case strings.HasPrefix(x.Value, "@"):
		key.phase = 3
	}

	key.split = strings.Split(strings.Replace(x.Value, ":", ".", -1), ".")
	return key
}

// byStringExpr implements sort.Interface for a list of stringSortKey.
type byStringExpr []stringSortKey

func (x byStringExpr) Len() int      { return len(x) }
func (x byStringExpr) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

func (x byStringExpr) Less(i, j int) bool {
	xi := x[i]
	xj := x[j]

	if xi.phase != xj.phase {
		return xi.phase < xj.phase
	}
	for k := 0; k < len(xi.split) && k < len(xj.split); k++ {
		if xi.split[k] != xj.split[k] {
			return xi.split[k] < xj.split[k]
		}
	}
	if len(xi.split) != len(xj.split) {
		return len(xi.split) < len(xj.split)
	}
	if xi.value != xj.value {
		return xi.value < xj.value
	}
	return xi.original < xj.original
}
