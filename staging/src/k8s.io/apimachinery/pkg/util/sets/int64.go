/*
Copyright 2022 The Kubernetes Authors.

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

package sets

// Int64 is a set of int64s, implemented via map[int64]struct{} for minimal memory consumption.
//
// Deprecated: use generic OrderedSet instead.
// new ways:
// s1 := OrderedSet[int64]{}
// s2 := NewOrdered[int64]()
type Int64 = OrderedSet[int64]

// NewInt64 creates a Int64 from a list of values.
func NewInt64(items ...int64) Int64 {
	return Int64(New[int64](items...))
}

// Int64KeySet creates a Int64 from a keys of a map[int64](? extends interface{}).
// If the value passed in is not actually a map, this will panic.
func Int64KeySet[T any](theMap map[int64]T) Int64 {
	return Int64(KeySet(theMap))
}
