/*
Copyright 2018 The Kubernetes Authors.

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

package idl_test

// This example shows how to use the listType map attribute and how to
// specify a key to identify elements of the list. The listMapKey
// attribute is used to specify that Name is the key of the map.
func ExampleListType_map() {
	type SomeStruct struct {
		Name  string
		Value string
	}
	type SomeAPI struct {
		// +listType=map
		// +listMapKey=name
		elements []SomeStruct
	}
}

// This example shows how to use the listType set attribute to specify
// that this list should be treated as a set: items in the list can't be
// duplicated.
func ExampleListType_set() {
	type SomeAPI struct {
		// +listType=set
		keys []string
	}
}

// This example shows how to use the listType atomic attribute to
// specify that this list should be treated as a whole.
func ExampleListType_atomic() {
	type SomeStruct struct {
		Name  string
		Value string
	}

	type SomeAPI struct {
		// +listType=atomic
		elements []SomeStruct
	}
}
