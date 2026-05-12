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

import "k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg"

// Trivial
type StructEmpty struct{}

// Only primitives
type StructPrimitives struct {
	BoolField   bool
	IntField    int
	StringField string
	FloatField  float64
}
type StructPrimitivesAlias StructPrimitives
type StructEmbedStructPrimitives struct {
	StructPrimitives
}
type StructEmbedInt struct {
	int //nolint:unused
}
type StructStructPrimitives struct {
	StructField StructPrimitives
}

// Manual DeepCopy method
type ManualStruct struct {
	StringField string
}

func (m ManualStruct) DeepCopy() ManualStruct {
	return m
}

type ManualStructAlias ManualStruct

type StructEmbedManualStruct struct {
	ManualStruct
}

// Only pointers to primitives
type StructPrimitivePointers struct {
	BoolPtrField   *bool
	IntPtrField    *int
	StringPtrField *string
	FloatPtrField  *float64
}
type StructPrimitivePointersAlias StructPrimitivePointers
type StructEmbedStructPrimitivePointers struct {
	StructPrimitivePointers
}
type StructEmbedPointer struct {
	*int
}
type StructStructPrimitivePointers struct {
	StructField StructPrimitivePointers
}

// Manual DeepCopy method
type ManualSlice []string

func (m ManualSlice) DeepCopy() ManualSlice {
	r := make(ManualSlice, len(m))
	copy(r, m)
	return r
}

// Slices
type StructSlices struct {
	SliceBoolField                         []bool
	SliceByteField                         []byte
	SliceIntField                          []int
	SliceStringField                       []string
	SliceFloatField                        []float64
	SliceStructPrimitivesField             []StructPrimitives
	SliceStructPrimitivesAliasField        []StructPrimitivesAlias
	SliceStructPrimitivePointersField      []StructPrimitivePointers
	SliceStructPrimitivePointersAliasField []StructPrimitivePointersAlias
	SliceSliceIntField                     [][]int
	SliceManualStructField                 []ManualStruct
	ManualSliceField                       ManualSlice
}
type StructSlicesAlias StructSlices
type StructEmbedStructSlices struct {
	StructSlices
}
type StructStructSlices struct {
	StructField StructSlices
}

// Everything
type StructEverything struct {
	BoolField                 bool
	IntField                  int
	StringField               string
	FloatField                float64
	StructField               StructPrimitives
	EmptyStructField          StructEmpty
	ManualStructField         ManualStruct
	ManualStructAliasField    ManualStructAlias
	BoolPtrField              *bool
	IntPtrField               *int
	StringPtrField            *string
	FloatPtrField             *float64
	PrimitivePointersField    StructPrimitivePointers
	ManualStructPtrField      *ManualStruct
	ManualStructAliasPtrField *ManualStructAlias
	SliceBoolField            []bool
	SliceByteField            []byte
	SliceIntField             []int
	SliceStringField          []string
	SliceFloatField           []float64
	SlicesField               StructSlices
	SliceManualStructField    []ManualStruct
	ManualSliceField          ManualSlice
}

// An Object
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
type StructExplicitObject struct {
	x int //nolint:unused
}

// An Object which is used a non-pointer
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:nonpointer-interfaces=true
type StructNonPointerExplicitObject struct {
	x int //nolint:unused
}

// +k8s:deepcopy-gen=false
type StructTypeMeta struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.List
type StructObjectAndList struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
type StructObjectAndObject struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/wholepkg.Selector
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/otherpkg.Object
type StructExplicitSelectorExplicitObject struct {
	StructTypeMeta
}

type StructInterfaces struct {
	ObjectField    otherpkg.Object
	NilObjectField otherpkg.Object
	SelectorField  Selector
}
