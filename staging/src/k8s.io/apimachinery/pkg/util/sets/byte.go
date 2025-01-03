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

// Byte is a set of bytes, implemented via map[byte]struct{} for minimal memory consumption.
//
// Deprecated: use generic OrderedSet instead.
// new ways:
// s1 := OrderedSet[byte]{}
// s2 := NewOrdered[byte]()
type Byte = OrderedSet[byte]

// NewByte creates a Byte from a list of values.
func NewByte(items ...byte) Byte {
	return Byte(New[byte](items...))
}

// ByteKeySet creates a Byte from a keys of a map[byte](? extends interface{}).
// If the value passed in is not actually a map, this will panic.
func ByteKeySet[T any](theMap map[byte]T) Byte {
	return Byte(KeySet(theMap))
}
