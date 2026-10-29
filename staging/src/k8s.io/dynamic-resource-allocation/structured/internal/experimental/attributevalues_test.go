/*
Copyright The Kubernetes Authors.

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

package experimental

import (
	"reflect"
	"testing"

	"github.com/blang/semver/v4"
)

func TestOverlaps(t *testing.T) {
	for name, tc := range map[string]struct {
		a, b     any
		expectOk bool
	}{
		"equal-strings":            {a: "a", b: "a", expectOk: true},
		"different-strings":        {a: "a", b: "b", expectOk: false},
		"string-in-list":           {a: "b", b: []string{"a", "b"}, expectOk: true},
		"string-not-in-list":       {a: "c", b: []string{"a", "b"}, expectOk: false},
		"list-contains-string":     {a: []string{"a", "b"}, b: "b", expectOk: true},
		"string-lists-overlap":     {a: []string{"a", "b"}, b: []string{"b", "c"}, expectOk: true},
		"string-lists-disjoint":    {a: []string{"a", "b"}, b: []string{"c", "d"}, expectOk: false},
		"empty-list":               {a: []string{}, b: []string{"a"}, expectOk: false},
		"duplicates-in-list":       {a: []string{"a", "a"}, b: []string{"a"}, expectOk: true},
		"equal-ints":               {a: int64(1), b: int64(1), expectOk: true},
		"int-lists-overlap":        {a: []int64{1, 2}, b: []int64{2, 3}, expectOk: true},
		"int-lists-disjoint":       {a: []int64{1, 2}, b: []int64{3, 4}, expectOk: false},
		"equal-bools":              {a: true, b: true, expectOk: true},
		"different-bools":          {a: true, b: false, expectOk: false},
		"bool-lists-overlap":       {a: []bool{false}, b: []bool{false, true}, expectOk: true},
		"type-mismatch":            {a: "1", b: int64(1), expectOk: false},
		"type-mismatch-list":       {a: []string{"1"}, b: []int64{1}, expectOk: false},
		"type-mismatch-list-value": {a: []string{"1"}, b: int64(1), expectOk: false},

		// Equals ignores build metadata, so the comparison here has to as well.
		"versions-differing-only-in-build": {
			a: semver.MustParse("1.0.0+a"), b: semver.MustParse("1.0.0+b"), expectOk: true,
		},
		"different-versions": {
			a: semver.MustParse("1.0.0"), b: semver.MustParse("2.0.0"), expectOk: false,
		},
		"version-in-list": {
			a:        semver.MustParse("1.0.0+a"),
			b:        []semver.Version{semver.MustParse("2.0.0"), semver.MustParse("1.0.0+b")},
			expectOk: true,
		},
		"version-lists-disjoint": {
			a:        []semver.Version{semver.MustParse("1.0.0")},
			b:        []semver.Version{semver.MustParse("2.0.0")},
			expectOk: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if actual := overlaps(tc.a, tc.b); actual != tc.expectOk {
				t.Errorf("overlaps(%v, %v) = %v, expected %v", tc.a, tc.b, actual, tc.expectOk)
			}
			// Overlapping is symmetric.
			if actual := overlaps(tc.b, tc.a); actual != tc.expectOk {
				t.Errorf("overlaps(%v, %v) = %v, expected %v (not symmetric)", tc.b, tc.a, actual, tc.expectOk)
			}
		})
	}
}

func TestIntersect(t *testing.T) {
	for name, tc := range map[string]struct {
		a, b               any
		expectIntersection any
		expectOk           bool
	}{
		"equal-strings":         {a: "a", b: "a", expectIntersection: "a", expectOk: true},
		"different-strings":     {a: "a", b: "b", expectOk: false},
		"string-in-list":        {a: "b", b: []string{"a", "b"}, expectIntersection: "b", expectOk: true},
		"string-not-in-list":    {a: "c", b: []string{"a", "b"}, expectOk: false},
		"list-narrows-to-value": {a: []string{"a", "b"}, b: "b", expectIntersection: "b", expectOk: true},
		"lists-overlap":         {a: []string{"a", "b", "c"}, b: []string{"b", "c", "d"}, expectIntersection: []string{"b", "c"}, expectOk: true},
		"equal-lists":           {a: []string{"a", "b"}, b: []string{"a", "b"}, expectIntersection: []string{"a", "b"}, expectOk: true},
		"lists-disjoint":        {a: []string{"a"}, b: []string{"b"}, expectOk: false},
		"duplicates-in-list":    {a: []string{"a", "b"}, b: []string{"a", "a"}, expectIntersection: []string{"a"}, expectOk: true},
		"equal-ints":            {a: int64(1), b: int64(1), expectIntersection: int64(1), expectOk: true},
		"int-lists-overlap":     {a: []int64{1, 2}, b: []int64{2, 3}, expectIntersection: []int64{2}, expectOk: true},
		"equal-bools":           {a: true, b: true, expectIntersection: true, expectOk: true},
		"bool-list-narrows":     {a: []bool{false, true}, b: []bool{true}, expectIntersection: []bool{true}, expectOk: true},
		"type-mismatch":         {a: "1", b: int64(1), expectOk: false},

		// A CEL expression evaluating to an empty list always yields []string{},
		// whatever its element type. An empty set intersects with nothing.
		"empty-list":            {a: []string{}, b: []string{"a"}, expectOk: false},
		"empty-list-both-sides": {a: []string{}, b: []string{}, expectOk: false},

		"versions-differing-only-in-build": {
			a: semver.MustParse("1.0.0+a"), b: semver.MustParse("1.0.0+b"),
			expectIntersection: semver.MustParse("1.0.0+a"), expectOk: true,
		},
		"version-list-narrows": {
			a:                  []semver.Version{semver.MustParse("1.0.0"), semver.MustParse("2.0.0")},
			b:                  []semver.Version{semver.MustParse("2.0.0")},
			expectIntersection: []semver.Version{semver.MustParse("2.0.0")},
			expectOk:           true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			intersection, ok := intersect(tc.a, tc.b)
			if ok != tc.expectOk {
				t.Fatalf("intersect(%v, %v) ok = %v, expected %v", tc.a, tc.b, ok, tc.expectOk)
			}
			if !ok {
				return
			}
			if !reflect.DeepEqual(intersection, tc.expectIntersection) {
				t.Errorf("intersect(%v, %v) = %v, expected %v", tc.a, tc.b, intersection, tc.expectIntersection)
			}
		})
	}
}

// Intersecting must not allocate as long as the result is one of the values
// that went in: that is what makes a single value cheaper than a one-element
// list.
func TestIntersectDoesNotAllocate(t *testing.T) {
	for name, tc := range map[string]struct{ a, b any }{
		"equal-values":          {a: "a", b: "a"},
		"value-in-list":         {a: "b", b: []string{"a", "b"}},
		"list-narrows-to-value": {a: []string{"a", "b"}, b: "b"},
		"equal-lists":           {a: []string{"a", "b"}, b: []string{"a", "b"}},
		"lists-disjoint":        {a: []string{"a"}, b: []string{"b"}},
	} {
		t.Run(name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(100, func() {
				sinkValue, sinkOk = intersect(tc.a, tc.b)
			})
			if allocs != 0 {
				t.Errorf("intersect(%v, %v) made %v allocations, expected none", tc.a, tc.b, allocs)
			}
		})
	}
}

var (
	sinkValue any
	sinkOk    bool
)

func TestComparableAttributeValue(t *testing.T) {
	for name, tc := range map[string]struct {
		value  any
		expect bool
	}{
		"string":    {value: "a", expect: true},
		"int":       {value: int64(1), expect: true},
		"bool":      {value: true, expect: true},
		"version":   {value: semver.MustParse("1.0.0"), expect: true},
		"strings":   {value: []string{"a"}, expect: true},
		"ints":      {value: []int64{1}, expect: true},
		"bools":     {value: []bool{true}, expect: true},
		"versions":  {value: []semver.Version{semver.MustParse("1.0.0")}, expect: true},
		"nil":       {value: nil, expect: false},
		"unknown":   {value: 1.5, expect: false},
		"unknowns":  {value: []float64{1.5}, expect: false},
		"empty-nil": {value: []string(nil), expect: true},
	} {
		t.Run(name, func(t *testing.T) {
			if actual := comparableAttributeValue(tc.value); actual != tc.expect {
				t.Errorf("comparableAttributeValue(%v) = %v, expected %v", tc.value, actual, tc.expect)
			}
		})
	}
}
