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

package wholepkg

import (
	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/empty"
)

// Only primitives
type StructPrimitives struct {
	empty.TypeMeta
	BoolField   *bool
	IntField    *int
	StringField *string
	FloatField  *float64
}
type StructPrimitivesAlias StructPrimitives

type StructStructPrimitives struct {
	empty.TypeMeta
	StructField StructPrimitives
}

// Pointer
type StructPointer struct {
	empty.TypeMeta
	PointerStructPrimitivesField             StructPrimitives
	PointerPointerStructPrimitivesField      *StructPrimitives
	PointerStructPrimitivesAliasField        StructPrimitivesAlias
	PointerPointerStructPrimitivesAliasField StructPrimitivesAlias
	PointerStructStructPrimitives            StructStructPrimitives
	PointerPointerStructStructPrimitives     *StructStructPrimitives
}

// Slices
type StructSlices struct {
	empty.TypeMeta
	SliceStructPrimitivesField             []StructPrimitives
	SlicePointerStructPrimitivesField      []*StructPrimitives
	SliceStructPrimitivesAliasField        []StructPrimitivesAlias
	SlicePointerStructPrimitivesAliasField []*StructPrimitivesAlias
	SliceStructStructPrimitives            []StructStructPrimitives
	SlicePointerStructStructPrimitives     []*StructStructPrimitives
}

// Everything
type StructEverything struct {
	empty.TypeMeta
	BoolPtrField       *bool
	IntPtrField        *int
	StringPtrField     *string
	FloatPtrField      *float64
	PointerStructField StructPointer
	SliceBoolField     []bool
	SliceByteField     []byte
	SliceIntField      []int
	SliceStringField   []string
	SliceFloatField    []float64
	SlicesStructField  StructSlices
}
