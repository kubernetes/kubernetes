/*
Copyright 2019 The Kubernetes Authors.

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

package schema

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	fuzz "github.com/google/gofuzz"
)

func fuzzInterface(i *interface{}, c fuzz.Continue) {
	m := map[string]string{}
	c.Fuzz(&m)
	*i = &m
}

func (Schema) Generate(rand *rand.Rand, size int) reflect.Value {
	s := Schema{}
	f := fuzz.New().RandSource(rand).MaxDepth(4)
	f.Fuzz(&s)
	return reflect.ValueOf(s)
}

func (Map) Generate(rand *rand.Rand, size int) reflect.Value {
	m := Map{}
	f := fuzz.New().RandSource(rand).MaxDepth(4).Funcs(fuzzInterface)
	f.Fuzz(&m)
	return reflect.ValueOf(m)
}

func (TypeDef) Generate(rand *rand.Rand, size int) reflect.Value {
	td := TypeDef{}
	f := fuzz.New().RandSource(rand).MaxDepth(4)
	f.Fuzz(&td)
	return reflect.ValueOf(td)
}

func (Atom) Generate(rand *rand.Rand, size int) reflect.Value {
	a := Atom{}
	f := fuzz.New().RandSource(rand).MaxDepth(4)
	f.Fuzz(&a)
	return reflect.ValueOf(a)
}

func (StructField) Generate(rand *rand.Rand, size int) reflect.Value {
	a := StructField{}
	f := fuzz.New().RandSource(rand).MaxDepth(4).Funcs(fuzzInterface)
	f.Fuzz(&a)
	return reflect.ValueOf(a)
}

func TestEquals(t *testing.T) {
	// In general this test will make sure people update things when they
	// add a field.
	//
	// The "copy known fields" section of these function is to break if folks
	// add new fields without fixing the Equals function and this test.
	funcs := []interface{}{
		func(x Schema) bool {
			if !x.Equals(&x) {
				return false
			}
			var y Schema
			y.Types = x.Types
			return x.Equals(&y) == reflect.DeepEqual(&x, &y)
		},
		func(x TypeDef) bool {
			if !x.Equals(&x) {
				return false
			}
			var y TypeDef
			y.Name = x.Name
			y.Atom = x.Atom
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x TypeRef) bool {
			if !x.Equals(&x) {
				return false
			}
			var y TypeRef
			y.NamedType = x.NamedType
			y.Inlined = x.Inlined
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x Atom) bool {
			if !x.Equals(&x) {
				return false
			}
			var y Atom
			y.Scalar = x.Scalar
			y.List = x.List
			y.Map = x.Map
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x Map) bool {
			if !x.Equals(&x) {
				return false
			}
			var y Map
			y.ElementType = x.ElementType
			y.ElementRelationship = x.ElementRelationship
			y.Fields = x.Fields
			y.Unions = x.Unions
			return x.Equals(&y) == reflect.DeepEqual(&x, &y)
		},
		func(x Union) bool {
			if !x.Equals(&x) {
				return false
			}
			var y Union
			y.Discriminator = x.Discriminator
			y.DeduceInvalidDiscriminator = x.DeduceInvalidDiscriminator
			y.Fields = x.Fields
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x UnionField) bool {
			if !x.Equals(&x) {
				return false
			}
			var y UnionField
			y.DiscriminatorValue = x.DiscriminatorValue
			y.FieldName = x.FieldName
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x StructField) bool {
			if !x.Equals(&x) {
				return false
			}
			var y StructField
			y.Name = x.Name
			y.Type = x.Type
			y.Default = x.Default
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
		func(x List) bool {
			if !x.Equals(&x) {
				return false
			}
			var y List
			y.ElementType = x.ElementType
			y.ElementRelationship = x.ElementRelationship
			y.Keys = x.Keys
			return x.Equals(&y) == reflect.DeepEqual(x, y)
		},
	}
	for i, f := range funcs {
		if err := quick.Check(f, nil); err != nil {
			t.Errorf("%v: %v", i, err)
		}
	}
}
