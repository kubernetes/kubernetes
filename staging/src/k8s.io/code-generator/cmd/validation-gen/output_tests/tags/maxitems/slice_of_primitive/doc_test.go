/*
Copyright 2024 The Kubernetes Authors.

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

package sliceofprimitive

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectValid()

	st.Value(&Struct{
		Max0Field:         make([]int, 0),
		Max10Field:        make([]int, 0),
		Max0TypedefField:  make([]IntType, 0),
		Max10TypedefField: make([]IntType, 0),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:        make([]int, 1),
		Max10TypedefField: make([]IntType, 1),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:        make([]int, 9),
		Max10TypedefField: make([]IntType, 9),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:        make([]int, 10),
		Max10TypedefField: make([]IntType, 10),
	}).ExpectValid()

	st.Value(&Struct{
		Max0Field:         make([]int, 1),
		Max10Field:        make([]int, 11),
		Max0TypedefField:  make([]IntType, 1),
		Max10TypedefField: make([]IntType, 11),
	}).ExpectRegexpsByPath(map[string][]string{
		"max0Field":         {`Too many:.*must have at most 0 items`},
		"max10Field":        {`Too many:.*must have at most 10 items`},
		"max0TypedefField":  {`Too many:.*must have at most 0 items`},
		"max10TypedefField": {`Too many:.*must have at most 10 items`},
	})
	// Test validation ratcheting
	st.Value(&Struct{
		Max0Field:         make([]int, 1),
		Max10Field:        make([]int, 11),
		Max0TypedefField:  make([]IntType, 1),
		Max10TypedefField: make([]IntType, 11),
	}).OldValue(&Struct{
		Max0Field:         make([]int, 1),
		Max10Field:        make([]int, 11),
		Max0TypedefField:  make([]IntType, 1),
		Max10TypedefField: make([]IntType, 11),
	}).ExpectValid()

}
