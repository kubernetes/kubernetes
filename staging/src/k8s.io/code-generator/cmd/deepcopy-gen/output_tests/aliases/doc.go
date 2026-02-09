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

// +k8s:deepcopy-gen=package

// This is a test package.
package aliases

// Note: the following AliasInterface and AliasAliasInterface +k8s:deepcopy-gen:interfaces tags
// are necessary because Golang flattens interface alias in the type system. I.e. an alias J of
// an interface I is actually equivalent to I. So support deepcopies of those aliases, we have
// to implement all aliases of that interface.

// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/aliases.Interface
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/aliases.AliasInterface
// +k8s:deepcopy-gen:interfaces=k8s.io/code-generator/cmd/deepcopy-gen/output_tests/aliases.AliasAliasInterface
type Foo struct {
	X int
}

type Interface interface {
	DeepCopyInterface() Interface
	DeepCopyAliasInterface() AliasInterface
	DeepCopyAliasAliasInterface() AliasAliasInterface
}

type Builtin int
type Slice []int
type Pointer *int
type PointerAlias *Builtin
type Struct Foo
type Map map[string]int

type FooAlias Foo
type FooSlice []Foo
type FooPointer *Foo
type FooMap map[string]Foo

type AliasBuiltin Builtin
type AliasSlice Slice
type AliasPointer Pointer
type AliasStruct Struct
type AliasMap Map

type AliasInterface Interface
type AliasAliasInterface AliasInterface
type AliasInterfaceMap map[string]AliasInterface
type AliasInterfaceSlice []AliasInterface

// Aliases
type Ttest struct {
	Builtin      Builtin
	Slice        Slice
	Pointer      Pointer
	PointerAlias PointerAlias
	Struct       Struct
	Map          Map
	SliceSlice   []Slice
	MapSlice     map[string]Slice

	FooAlias   FooAlias
	FooSlice   FooSlice
	FooPointer FooPointer
	FooMap     FooMap

	AliasBuiltin AliasBuiltin
	AliasSlice   AliasSlice
	AliasPointer AliasPointer
	AliasStruct  AliasStruct
	AliasMap     AliasMap

	AliasInterface      AliasInterface
	AliasAliasInterface AliasAliasInterface
	AliasInterfaceMap   AliasInterfaceMap
	AliasInterfaceSlice AliasInterfaceSlice
}
