/*
Copyright 2025 The Kubernetes Authors.

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

package maxproperties

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectValid()

	st.Value(&Struct{
		Max0Field:            generateMapStringStringWithLength(0),
		Max10Field:           generateMapStringStringWithLength(0),
		Max0TypedefKeyField:  generateMapStringKeyStringWithLength(0),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(0),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:           generateMapStringStringWithLength(1),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(1),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:           generateMapStringStringWithLength(9),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(9),
	}).ExpectValid()

	st.Value(&Struct{
		Max10Field:           generateMapStringStringWithLength(10),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(10),
	}).ExpectValid()

	st.Value(&Struct{
		Max0Field:            generateMapStringStringWithLength(1),
		Max10Field:           generateMapStringStringWithLength(11),
		Max0TypedefKeyField:  generateMapStringKeyStringWithLength(1),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(11),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooMany(field.NewPath("max0Field"), 1, 0),
		field.TooMany(field.NewPath("max10Field"), 11, 10),
		field.TooMany(field.NewPath("max0TypedefKeyField"), 1, 0),
		field.TooMany(field.NewPath("max10TypedefKeyField"), 11, 10),
	})

	// Test validation ratcheting
	st.Value(&Struct{
		Max0Field:            generateMapStringStringWithLength(1),
		Max10Field:           generateMapStringStringWithLength(11),
		Max0TypedefKeyField:  generateMapStringKeyStringWithLength(1),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(11),
	}).OldValue(&Struct{
		Max0Field:            generateMapStringStringWithLength(1),
		Max10Field:           generateMapStringStringWithLength(11),
		Max0TypedefKeyField:  generateMapStringKeyStringWithLength(1),
		Max10TypedefKeyField: generateMapStringKeyStringWithLength(11),
	}).ExpectValid()
}

func generateMapStringStringWithLength(n int) map[string]string {
	out := make(map[string]string)
	for i := range n {
		str := fmt.Sprintf("%d", i)
		out[str] = str
	}
	return out
}

func generateMapStringKeyStringWithLength(n int) map[StringKey]string {
	out := make(map[StringKey]string)
	for i := range n {
		str := fmt.Sprintf("%d", i)
		out[StringKey(str)] = str
	}
	return out
}
