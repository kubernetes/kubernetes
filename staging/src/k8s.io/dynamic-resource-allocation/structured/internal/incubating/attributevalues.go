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

package incubating

import (
	"slices"

	"github.com/blang/semver/v4"
)

// An attribute value is either a single value or a list of values, in the
// types that draapi stores them in. matchAttribute and distinctAttribute both
// treat them as sets, but a single value is never turned into a one-element
// list on the way: that would allocate, and these comparisons sit in the
// allocator's innermost loop. Instead each combination is handled directly.
//
// Lists are short - the API allows a device at most
// ResourceSliceMaxAttributeValuesPerDevice values across all of its attributes
// - so scanning them beats building a set first.
//
// Values of different types never have anything in common, which is also how
// the allocator treats a device whose attribute has an unexpected type.

// overlaps reports whether two attribute values have an element in common.
// It allocates nothing.
//
// This is the same predicate that intersect returns as ok, and the dispatch
// below is the same, but the two are kept apart on purpose: intersect has to
// build the intersection, and distinctAttributeConstraint only ever wants the
// answer. Defining one in terms of the other would allocate on every
// rejection, which is the case that matters there.
func overlaps(a, b any) bool {
	switch a := a.(type) {
	case string:
		switch b := b.(type) {
		case string:
			return a == b
		case []string:
			return slices.Contains(b, a)
		}
	case []string:
		switch b := b.(type) {
		case string:
			return slices.Contains(a, b)
		case []string:
			return overlapsList(a, b)
		}
	case int64:
		switch b := b.(type) {
		case int64:
			return a == b
		case []int64:
			return slices.Contains(b, a)
		}
	case []int64:
		switch b := b.(type) {
		case int64:
			return slices.Contains(a, b)
		case []int64:
			return overlapsList(a, b)
		}
	case bool:
		switch b := b.(type) {
		case bool:
			return a == b
		case []bool:
			return slices.Contains(b, a)
		}
	case []bool:
		switch b := b.(type) {
		case bool:
			return slices.Contains(a, b)
		case []bool:
			return overlapsList(a, b)
		}
	case semver.Version:
		switch b := b.(type) {
		case semver.Version:
			return a.Equals(b)
		case []semver.Version:
			return containsVersion(b, a)
		}
	case []semver.Version:
		switch b := b.(type) {
		case semver.Version:
			return containsVersion(a, b)
		case []semver.Version:
			return overlapsVersions(a, b)
		}
	}
	return false
}

func overlapsList[T comparable](a, b []T) bool {
	return slices.ContainsFunc(a, func(value T) bool { return slices.Contains(b, value) })
}

// containsVersion compares with Equals, which ignores build metadata, the same
// way the scalar comparison in the constraints does.
func containsVersion(versions []semver.Version, version semver.Version) bool {
	return slices.ContainsFunc(versions, version.Equals)
}

func overlapsVersions(a, b []semver.Version) bool {
	return slices.ContainsFunc(a, func(version semver.Version) bool { return containsVersion(b, version) })
}

// intersect returns the values that two attribute values have in common. ok
// is false when they have none, which includes them having different types;
// intersection is only meaningful when ok is true.
//
// The result is one of the two values that went in whenever it can be, which
// covers both common cases: two equal single values, and a device whose list
// contains the single value already shared by everything before it. Only a
// result that is a proper subset of a list has to be built.
func intersect(a, b any) (intersection any, ok bool) {
	switch av := a.(type) {
	case string:
		switch bv := b.(type) {
		case string:
			return a, av == bv
		case []string:
			return a, slices.Contains(bv, av)
		}
	case []string:
		switch bv := b.(type) {
		case string:
			return b, slices.Contains(av, bv)
		case []string:
			return intersectList(a, av, bv)
		}
	case int64:
		switch bv := b.(type) {
		case int64:
			return a, av == bv
		case []int64:
			return a, slices.Contains(bv, av)
		}
	case []int64:
		switch bv := b.(type) {
		case int64:
			return b, slices.Contains(av, bv)
		case []int64:
			return intersectList(a, av, bv)
		}
	case bool:
		switch bv := b.(type) {
		case bool:
			return a, av == bv
		case []bool:
			return a, slices.Contains(bv, av)
		}
	case []bool:
		switch bv := b.(type) {
		case bool:
			return b, slices.Contains(av, bv)
		case []bool:
			return intersectList(a, av, bv)
		}
	case semver.Version:
		switch bv := b.(type) {
		case semver.Version:
			return a, av.Equals(bv)
		case []semver.Version:
			return a, containsVersion(bv, av)
		}
	case []semver.Version:
		switch bv := b.(type) {
		case semver.Version:
			return b, containsVersion(av, bv)
		case []semver.Version:
			return intersectVersions(a, av, bv)
		}
	}
	return nil, false
}

// intersectList takes the first list as both the original interface value and
// the typed slice, so that an unchanged result can be returned without boxing
// the slice again. Both shortcuts matter: they are the common cases, and they
// are the ones that do not have to build anything.
func intersectList[T comparable](original any, a, b []T) (any, bool) {
	inB := func(value T) bool { return slices.Contains(b, value) }
	notInB := func(value T) bool { return !inB(value) }
	if !slices.ContainsFunc(a, inB) {
		// Nothing in common. Also covers an empty list.
		return nil, false
	}
	if !slices.ContainsFunc(a, notInB) {
		// Every value survives, so the intersection is a.
		return original, true
	}
	return slices.DeleteFunc(slices.Clone(a), notInB), true
}

// intersectVersions is intersectList for versions, which need Equals rather
// than == and so cannot use slices.Contains.
func intersectVersions(original any, a, b []semver.Version) (any, bool) {
	inB := func(version semver.Version) bool { return containsVersion(b, version) }
	notInB := func(version semver.Version) bool { return !inB(version) }
	if !slices.ContainsFunc(a, inB) {
		return nil, false
	}
	if !slices.ContainsFunc(a, notInB) {
		return original, true
	}
	return slices.DeleteFunc(slices.Clone(a), notInB), true
}

// comparableAttributeValue reports whether overlaps and intersect understand
// the value. Everything draapi stores for an attribute is understood, so a
// false result means the value came from somewhere else, or that this client
// is older than the API it is reading.
func comparableAttributeValue(value any) bool {
	switch value.(type) {
	case string, int64, bool, semver.Version,
		[]string, []int64, []bool, []semver.Version:
		return true
	}
	return false
}
