/*
Copyright 2020 The Kubernetes Authors.

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

package marker

import (
	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/empty"
	"k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external3"
)

type Defaulted struct {
	empty.TypeMeta

	// +default="bar"
	StringDefault string

	// Default is forced to empty string
	// Specifying the default is a no-op
	// +default=""
	StringEmptyDefault string

	// Not specifying a default still defaults for non-omitempty
	StringEmpty string

	// +default="default"
	StringPointer *string

	// +default=64
	Int64 *int64

	// +default=32
	Int32 *int32

	// +default=1
	IntDefault int

	// +default=0
	IntEmptyDefault int

	// Default is forced to 0
	IntEmpty int

	// +default=0.5
	FloatDefault float64

	// +default=0.0
	FloatEmptyDefault float64

	FloatEmpty float64

	// +default=["foo", "bar"]
	List []Item
	// +default={"s": "foo", "i": 5}
	Sub *SubStruct

	//+default=[{"s": "foo1", "i": 1}, {"s": "foo2"}]
	StructList []SubStruct

	//+default=[{"s": "foo1", "i": 1}, {"s": "foo2"}]
	PtrStructList []*SubStruct

	//+default=["foo"]
	StringList []string

	// Default is forced to empty struct
	OtherSub SubStruct

	// +default={"foo": "bar"}
	Map map[string]Item

	// +default={"foo": {"S": "string", "I": 1}}
	StructMap map[string]SubStruct

	// +default={"foo": {"S": "string", "I": 1}}
	PtrStructMap map[string]*SubStruct

	// A default specified here overrides the default for the Item type
	// +default="banana"
	AliasPtr Item
}

type DefaultedOmitempty struct {
	empty.TypeMeta `json:",omitempty"`

	// +default="bar"
	StringDefault string `json:",omitempty"`

	// Default is forced to empty string
	// Specifying the default is a no-op
	// +default=""
	StringEmptyDefault string `json:",omitempty"`

	// Not specifying a default still defaults for non-omitempty
	StringEmpty string `json:",omitempty"`

	// +default="default"
	StringPointer *string `json:",omitempty"`

	// +default=64
	Int64 *int64 `json:",omitempty"`

	// +default=32
	Int32 *int32 `json:",omitempty"`

	// +default=1
	IntDefault int `json:",omitempty"`

	// +default=0
	IntEmptyDefault int `json:",omitempty"`

	// Default is forced to 0
	IntEmpty int `json:",omitempty"`

	// +default=0.5
	FloatDefault float64 `json:",omitempty"`

	// +default=0.0
	FloatEmptyDefault float64 `json:",omitempty"`

	FloatEmpty float64 `json:",omitempty"`

	// +default=["foo", "bar"]
	List []Item `json:",omitempty"`
	// +default={"s": "foo", "i": 5}
	Sub *SubStruct `json:",omitempty"`

	//+default=[{"s": "foo1", "i": 1}, {"s": "foo2"}]
	StructList []SubStruct `json:",omitempty"`

	//+default=[{"s": "foo1", "i": 1}, {"s": "foo2"}]
	PtrStructList []*SubStruct `json:",omitempty"`

	//+default=["foo"]
	StringList []string `json:",omitempty"`

	// Default is forced to empty struct
	OtherSub SubStruct `json:",omitempty"`

	// +default={"foo": "bar"}
	Map map[string]Item `json:",omitempty"`

	// +default={"foo": {"S": "string", "I": 1}}
	StructMap map[string]SubStruct `json:",omitempty"`

	// +default={"foo": {"S": "string", "I": 1}}
	PtrStructMap map[string]*SubStruct `json:",omitempty"`

	// A default specified here overrides the default for the Item type
	// +default="banana"
	AliasPtr Item `json:",omitempty"`
}

const SomeDefault = "ACoolConstant"

// +default="apple"
type Item *string

type ValueItem string

// +default=ref(SomeValue)
type DefaultedValueItem ValueItem
type PointerValueItem *DefaultedValueItem

type ItemDefaultWiped Item

const SomeValue ValueItem = "Value"

type SubStruct struct {
	S string
	// +default=1
	I int `json:"I,omitempty"`
}

type DefaultedWithFunction struct {
	empty.TypeMeta
	// +default="default_marker"
	S1 string `json:"S1,omitempty"`
	// +default="default_marker"
	S2 string `json:"S2,omitempty"`
}

type DefaultedWithReference struct {
	empty.TypeMeta

	// Shows that if we have an alias that is a pointer and have a default
	// that is a value convertible to that pointer we can still use it
	// +default=ref(SomeValue)
	AliasConvertDefaultPointer PointerValueItem

	// Shows that default defined on a nested type is not respected through
	// an alias
	AliasWipedDefault ItemDefaultWiped

	// A default defined on a pointer-valued alias is respected
	PointerAliasDefault Item

	// Can have alias that is a pointer to type of constant
	// +default=ref(SomeDefault)
	AliasPointerInside Item

	// Can override default specified on an alias
	// +default=ref(SomeDefault)
	AliasOverride Item

	// Type-level default is not respected unless a pointer
	AliasNonPointerDefault DefaultedValueItem `json:",omitempty"`

	// Type-level default is not respected unless a pointer
	AliasPointerDefault *DefaultedValueItem

	// Can have value typed alias
	// +default=ref(SomeValue)
	AliasNonPointer ValueItem `json:",omitempty"`

	// Can have a pointer to an alias whose default is a non-pointer value
	// +default=ref(SomeValue)
	AliasPointer *ValueItem `json:",omitempty"`

	// Basic ref usage example
	// +default=ref(SomeDefault)
	SymbolReference string `json:",omitempty"`

	// +default=ref(k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external.AConstant)
	SameNamePackageSymbolReference1 string `json:",omitempty"`

	// +default=ref(k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker/external/external.AnotherConstant)
	SameNamePackageSymbolReference2 string `json:",omitempty"`

	// Should convert ValueItem -> string then up to B4 through addressOf and
	// casting
	// +default=ref(SomeValue)
	PointerConversion *B4

	// +default=ref(SomeValue)
	PointerConversionValue B4

	// +default=ref(k8s.io/code-generator/cmd/defaulter-gen/output_tests/marker.SomeValue)
	FullyQualifiedLocalSymbol string

	// Construction of external3.StringPointer requires importing external2
	// Test that generator can handle it
	// +default=ref(SomeValue)
	ImportFromAliasCast external3.StringPointer
}

// Super complicated hierarchy of aliases which includes multiple pointers,
// and sibling types.
type B0 *string
type B1 B0
type B2 *B1
type B3 ****B2
type B4 **B3
