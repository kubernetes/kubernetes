// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// Sets returns a cel.EnvOption to configure namespaced set relationship
// functions.
//
// There is no set type within CEL, and while one may be introduced in the
// future, there are cases where a `list` type is known to behave like a set.
// For such cases, this library provides some basic functionality for
// determining set containment, equivalence, and intersection.
//
// # Sets.Contains
//
// Returns whether the first list argument contains all elements in the second
// list argument. The list may contain elements of any type and standard CEL
// equality is used to determine whether a value exists in both lists. If the
// second list is empty, the result will always return true.
//
//	sets.contains(list(T), list(T)) -> bool
//
// Examples:
//
//	sets.contains([], []) // true
//	sets.contains([], [1]) // false
//	sets.contains([1, 2, 3, 4], [2, 3]) // true
//	sets.contains([1, 2.0, 3u], [1.0, 2u, 3]) // true
//
// # Sets.Equivalent
//
// Returns whether the first and second list are set equivalent. Lists are set
// equivalent if for every item in the first list, there is an element in the
// second which is equal. The lists may not be of the same size as they do not
// guarantee the elements within them are unique, so size does not factor into
// the computation.
//
// Examples:
//
//	sets.equivalent([], []) // true
//	sets.equivalent([1], [1, 1]) // true
//	sets.equivalent([1], [1u, 1.0]) // true
//	sets.equivalent([1, 2, 3], [3u, 2.0, 1]) // true
//
// # Sets.Intersects
//
// Returns whether the first list has at least one element whose value is equal
// to an element in the second list. If either list is empty, the result will
// be false.
//
// Examples:
//
//	sets.intersects([1], []) // false
//	sets.intersects([1], [1, 2]) // true
//	sets.intersects([[1], [2, 3]], [[1, 2], [2, 3.0]]) // true
func Sets() cel.EnvOption {
	return cel.Lib(setsLib{})
}

type setsLib struct{}

// LibraryName implements the SingletonLibrary interface method.
func (setsLib) LibraryName() string {
	return "cel.lib.ext.sets"
}

// CompileOptions implements the Library interface method.
func (setsLib) CompileOptions() []cel.EnvOption {
	listType := cel.ListType(cel.TypeParamType("T"))
	return []cel.EnvOption{
		cel.Function("sets.contains",
			cel.Overload("list_sets_contains_list", []*cel.Type{listType, listType}, cel.BoolType,
				cel.BinaryBinding(setsContains))),
		cel.Function("sets.equivalent",
			cel.Overload("list_sets_equivalent_list", []*cel.Type{listType, listType}, cel.BoolType,
				cel.BinaryBinding(setsEquivalent))),
		cel.Function("sets.intersects",
			cel.Overload("list_sets_intersects_list", []*cel.Type{listType, listType}, cel.BoolType,
				cel.BinaryBinding(setsIntersects))),
	}
}

// ProgramOptions implements the Library interface method.
func (setsLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func setsIntersects(listA, listB ref.Val) ref.Val {
	lA := listA.(traits.Lister)
	lB := listB.(traits.Lister)
	it := lA.Iterator()
	for it.HasNext() == types.True {
		exists := lB.Contains(it.Next())
		if exists == types.True {
			return types.True
		}
	}
	return types.False
}

func setsContains(list, sublist ref.Val) ref.Val {
	l := list.(traits.Lister)
	sub := sublist.(traits.Lister)
	it := sub.Iterator()
	for it.HasNext() == types.True {
		exists := l.Contains(it.Next())
		if exists != types.True {
			return exists
		}
	}
	return types.True
}

func setsEquivalent(listA, listB ref.Val) ref.Val {
	aContainsB := setsContains(listA, listB)
	if aContainsB != types.True {
		return aContainsB
	}
	return setsContains(listB, listA)
}
