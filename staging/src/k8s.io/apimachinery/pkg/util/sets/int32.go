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

// Int32 is a set of int32s, implemented via map[int32]struct{} for minimal memory consumption.
//
// Deprecated: use generic OrderedSet instead.
// new ways:
// s1 := OrderedSet[int32]{}
// s2 := NewOrdered[int32]()
type Int32 = OrderedSet[int32]

// NewInt32 creates a Int32 from a list of values.
func NewInt32(items ...int32) Int32 {
	return Int32(New[int32](items...))
}

// Int32KeySet creates a Int32 from a keys of a map[int32](? extends interface{}).
// If the value passed in is not actually a map, this will panic.
func Int32KeySet[T any](theMap map[int32]T) Int32 {
	return Int32(KeySet(theMap))
}
