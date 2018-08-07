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

package testvectors

import (
	"testing"
)

// YAMLObject is an object encoded in YAML.
type YAMLObject string

// Schema is an object schema. (TODO: get correct type)
type Schema interface{}

// Vector describes an individual test case. Test cases are exported for ease
// in applying them to various implementations.
type Vector struct {
	Name   string
	Schema Schema

	LastObject YAMLObject
	LiveObject YAMLObject
	NewObject  YAMLObject

	ExpectedObject    YAMLObject
	ExpectedConflicts []interface{} // TODO: get correct type

	// TODO: do we need to indicate whether this supposed to be
	// controller-desired or user-desired behavior? Or is it sufficient for
	// controllers to just not specify the LastObject?
}

// Vectors is a list of all the vectors; each file in this package can add one
// or more vectors to the list.
var Vectors []*Vector

// Implementation is a stand-in for an implementation. It's structured so that
// you can run the same vectors through a system that does the three-way merge
// directly, or through a system that does a separate diff and merge.
type Implementation interface {
	// Test must not modify v.
	Test(t *testing.T, v *Vector)
}

// RunAllVectors runs all vectors through the given implementation.
func RunAllVectors(t *testing.T, impl Implementation) {
	for i := range Vectors {
		v := Vectors[i]
		t.Run(v.Name, func(t *testing.T) {
			impl.Test(t, v)
		})
	}
}
