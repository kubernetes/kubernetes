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

// String is a set of strings, implemented via map[string]struct{} for minimal memory consumption.
//
// Deprecated: use generic OrderedSet instead.
// new ways:
// s1 := OrderedSet[string]{}
// s2 := NewOrdered[string]()
type String = OrderedSet[string]

// NewString creates a String from a list of values.
func NewString(items ...string) String {
	return String(New[string](items...))
}

// StringKeySet creates a String from a keys of a map[string](? extends interface{}).
// If the value passed in is not actually a map, this will panic.
func StringKeySet[T any](theMap map[string]T) String {
	return String(KeySet(theMap))
}
