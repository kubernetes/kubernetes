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
	"fmt"
	"testing"
)

// YAMLObject is an object encoded in YAML.
type YAMLObject string

// SchemaDefinition is an object schema. (TODO: get correct type; for now
// assume this is a yaml-formatted string that can be deserialized.)
type SchemaDefinition string

// Vector describes an individual test case. Test cases are exported for ease
// in applying them to various implementations.
type Vector struct {
	Name string `yaml:"name"`

	// To allow multiple vectors to use the same schema.
	SchemaName string `yaml:"schemaName"`

	LastObject YAMLObject `yaml:"lastObject"`
	LiveObject YAMLObject `yaml:"liveObject"`
	NewObject  YAMLObject `yaml:"newObject"`

	ExpectedObject    YAMLObject    `yaml:"expectedObject"`
	ExpectedConflicts []interface{} `yaml:"expectedConflicts"` // TODO: get correct type

	// TODO: do we need to indicate whether this supposed to be
	// controller-desired or user-desired behavior? Or is it sufficient for
	// controllers to just not specify the LastObject?
}

// Vectors is a list of all the vectors; each file in this package can add one
// or more vectors to the list.
var Vectors []*Vector

// AppendTestVectors adds the given vectors to the global list.
func AppendTestVectors(vectors ...*Vector) {
	for _, v := range vectors {
		// defend against typos, since I'm expecting people to define tests via YAML.
		if v.Name == "" ||
			v.SchemaName == "" ||
			v.LastObject == "" ||
			v.LiveObject == "" ||
			v.NewObject == "" ||
			v.ExpectedObject == "" {
			panic(fmt.Sprintf("Test case %#v is not complete", *v))
		}
		Vectors = append(Vectors, v)
	}
}

// Schemas keeps the schemas that may be referenced by the test vectors.
var Schemas = map[string]SchemaDefinition{}

// Implementation is a stand-in for an implementation. It's structured so that
// you can run the same vectors through a system that does the three-way merge
// directly, or through a system that does a separate diff and merge.
type Implementation interface {
	// Test must not modify v. Test should call t.Parallel().
	Test(t *testing.T, v *Vector, s SchemaDefinition)
}

// RunAllVectors runs all vectors through the given implementation.
func RunAllVectors(t *testing.T, impl Implementation) {
	for i := range Vectors {
		v := Vectors[i]
		s, ok := Schemas[v.SchemaName]
		if !ok {
			t.Fatalf("Test %v references schema %v, but it is not defined", v.Name, v.SchemaName)
		}
		t.Run(v.Name, func(t *testing.T) {
			impl.Test(t, v, s)
		})
	}
}
