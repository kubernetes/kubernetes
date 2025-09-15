// Copyright 2018 Google LLC
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

package traits

import "github.com/google/cel-go/common/types/ref"

// Mapper interface which aggregates the traits of a maps.
type Mapper interface {
	ref.Val
	Container
	Indexer
	Iterable
	Sizer

	// Find returns a value, if one exists, for the input key.
	//
	// If the key is not found the function returns (nil, false).
	// If the input key is not valid for the map, or is Err or Unknown the function returns
	// (Unknown|Err, false).
	Find(key ref.Val) (ref.Val, bool)
}

// MutableMapper interface which emits an immutable result after an intermediate computation.
//
// Note, this interface is intended only to be used within Comprehensions where the mutable
// value is not directly observable within the user-authored CEL expression.
type MutableMapper interface {
	Mapper

	// Insert a key, value pair into the map, returning the map if the insert is successful
	// and an error if key already exists in the mutable map.
	Insert(k, v ref.Val) ref.Val

	// ToImmutableMap converts a mutable map into an immutable map.
	ToImmutableMap() Mapper
}
