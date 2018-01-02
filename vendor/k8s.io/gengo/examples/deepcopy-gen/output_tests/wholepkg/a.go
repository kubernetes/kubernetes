/*
Copyright 2016 The Kubernetes Authors.

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

import "k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg"

// Trivial
type Struct_Empty struct{}

// Only primitives
type Struct_Primitives struct {
	BoolField   bool
	IntField    int
	StringField string
	FloatField  float64
}
type Struct_Primitives_Alias Struct_Primitives
type Struct_Embed_Struct_Primitives struct {
	Struct_Primitives
}
type Struct_Embed_Int struct {
	int
}
type Struct_Struct_Primitives struct {
	StructField Struct_Primitives
}

// Manual DeepCopy method
type ManualStruct struct {
	StringField string
}

func (m ManualStruct) DeepCopy() ManualStruct {
	return m
}

type ManualStruct_Alias ManualStruct

type Struct_Embed_ManualStruct struct {
	ManualStruct
}

// Only pointers to primitives
type Struct_PrimitivePointers struct {
	BoolPtrField   *bool
	IntPtrField    *int
	StringPtrField *string
	FloatPtrField  *float64
}
type Struct_PrimitivePointers_Alias Struct_PrimitivePointers
type Struct_Embed_Struct_PrimitivePointers struct {
	Struct_PrimitivePointers
}
type Struct_Embed_Pointer struct {
	*int
}
type Struct_Struct_PrimitivePointers struct {
	StructField Struct_PrimitivePointers
}

// Manual DeepCopy method
type ManualSlice []string

func (m ManualSlice) DeepCopy() ManualSlice {
	r := make(ManualSlice, len(m))
	copy(r, m)
	return r
}

// Slices
type Struct_Slices struct {
	SliceBoolField                         []bool
	SliceByteField                         []byte
	SliceIntField                          []int
	SliceStringField                       []string
	SliceFloatField                        []float64
	SliceStructPrimitivesField             []Struct_Primitives
	SliceStructPrimitivesAliasField        []Struct_Primitives_Alias
	SliceStructPrimitivePointersField      []Struct_PrimitivePointers
	SliceStructPrimitivePointersAliasField []Struct_PrimitivePointers_Alias
	SliceSliceIntField                     [][]int
	SliceManualStructField                 []ManualStruct
	ManualSliceField                       ManualSlice
}
type Struct_Slices_Alias Struct_Slices
type Struct_Embed_Struct_Slices struct {
	Struct_Slices
}
type Struct_Struct_Slices struct {
	StructField Struct_Slices
}

// Everything
type Struct_Everything struct {
	BoolField                 bool
	IntField                  int
	StringField               string
	FloatField                float64
	StructField               Struct_Primitives
	EmptyStructField          Struct_Empty
	ManualStructField         ManualStruct
	ManualStructAliasField    ManualStruct_Alias
	BoolPtrField              *bool
	IntPtrField               *int
	StringPtrField            *string
	FloatPtrField             *float64
	PrimitivePointersField    Struct_PrimitivePointers
	ManualStructPtrField      *ManualStruct
	ManualStructAliasPtrField *ManualStruct_Alias
	SliceBoolField            []bool
	SliceByteField            []byte
	SliceIntField             []int
	SliceStringField          []string
	SliceFloatField           []float64
	SlicesField               Struct_Slices
	SliceManualStructField    []ManualStruct
	ManualSliceField          ManualSlice
}

// An Object
// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
type Struct_ExplicitObject struct {
	x int
}

// An Object which is used a non-pointer
// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type Struct_NonPointerExplicitObject struct {
	x int
}

// +k8s:deepcopy-gen=false
type Struct_TypeMeta struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.List
type Struct_ObjectAndList struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
type Struct_ObjectAndObject struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/wholepkg.Selector
// +k8s:deepcopy-gen:interfaces=k8s.io/gengo/examples/deepcopy-gen/output_tests/otherpkg.Object
type Struct_ExplicitSelectorExplicitObject struct {
	Struct_TypeMeta
}

type Struct_Interfaces struct {
	ObjectField    otherpkg.Object
	NilObjectField otherpkg.Object
	SelectorField  Selector
}
