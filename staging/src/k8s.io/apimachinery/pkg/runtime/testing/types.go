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

package testing

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type EmbeddedTest struct {
	runtime.TypeMeta
	ID          string
	Object      runtime.Object
	EmptyObject runtime.Object
}

type EmbeddedTestExternal struct {
	runtime.TypeMeta `json:",inline"`
	ID               string               `json:"id,omitempty"`
	Object           runtime.RawExtension `json:"object,omitempty"`
	EmptyObject      runtime.RawExtension `json:"emptyObject,omitempty"`
}

type ObjectTest struct {
	runtime.TypeMeta

	ID    string
	Items []runtime.Object
}

type ObjectTestExternal struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`

	ID    string                 `json:"id,omitempty"`
	Items []runtime.RawExtension `json:"items,omitempty"`
}

type InternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExtensionA struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExtensionB struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

type ExternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension"`
}

type InternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension"`
}

type ExternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension,omitempty"`
}

type InternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension,omitempty"`
}

type InternalComplex struct {
	runtime.TypeMeta
	String    string
	Integer   int
	Integer64 int64
	Int64     int64
	Bool      bool
}

type ExternalComplex struct {
	runtime.TypeMeta `json:",inline"`
	String           string `json:"string" description:"testing"`
	Integer          int    `json:"int"`
	Integer64        int64  `json:",omitempty"`
	Int64            int64
	Bool             bool `json:"bool"`
}

// Test a weird version/kind embedding format.
// +k8s:deepcopy-gen=false
type MyWeirdCustomEmbeddedVersionKindField struct {
	ID         string `json:"ID,omitempty"`
	APIVersion string `json:"myVersionKey,omitempty"`
	ObjectKind string `json:"myKindKey,omitempty"`
	Z          string `json:"Z,omitempty"`
	Y          uint64 `json:"Y,omitempty"`
}

type TestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string               `json:"A,omitempty"`
	B                                     int                  `json:"B,omitempty"`
	C                                     int8                 `json:"C,omitempty"`
	D                                     int16                `json:"D,omitempty"`
	E                                     int32                `json:"E,omitempty"`
	F                                     int64                `json:"F,omitempty"`
	G                                     uint                 `json:"G,omitempty"`
	H                                     uint8                `json:"H,omitempty"`
	I                                     uint16               `json:"I,omitempty"`
	J                                     uint32               `json:"J,omitempty"`
	K                                     uint64               `json:"K,omitempty"`
	L                                     bool                 `json:"L,omitempty"`
	M                                     map[string]int       `json:"M,omitempty"`
	N                                     map[string]TestType2 `json:"N,omitempty"`
	O                                     *TestType2           `json:"O,omitempty"`
	P                                     []TestType2          `json:"Q,omitempty"`
}

type TestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

type ExternalTestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

type ExternalTestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string                       `json:"A,omitempty"`
	B                                     int                          `json:"B,omitempty"`
	C                                     int8                         `json:"C,omitempty"`
	D                                     int16                        `json:"D,omitempty"`
	E                                     int32                        `json:"E,omitempty"`
	F                                     int64                        `json:"F,omitempty"`
	G                                     uint                         `json:"G,omitempty"`
	H                                     uint8                        `json:"H,omitempty"`
	I                                     uint16                       `json:"I,omitempty"`
	J                                     uint32                       `json:"J,omitempty"`
	K                                     uint64                       `json:"K,omitempty"`
	L                                     bool                         `json:"L,omitempty"`
	M                                     map[string]int               `json:"M,omitempty"`
	N                                     map[string]ExternalTestType2 `json:"N,omitempty"`
	O                                     *ExternalTestType2           `json:"O,omitempty"`
	P                                     []ExternalTestType2          `json:"Q,omitempty"`
}

type ExternalInternalSame struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     TestType2 `json:"A,omitempty"`
}

type UnversionedType struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string `json:"A,omitempty"`
}

type UnknownType struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string `json:"A,omitempty"`
}

func (obj *MyWeirdCustomEmbeddedVersionKindField) GetObjectKind() schema.ObjectKind { return obj }
func (obj *MyWeirdCustomEmbeddedVersionKindField) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.ObjectKind = gvk.ToAPIVersionAndKind()
}
func (obj *MyWeirdCustomEmbeddedVersionKindField) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.ObjectKind)
}

func (obj *TestType2) GetObjectKind() schema.ObjectKind         { return schema.EmptyObjectKind }
func (obj *ExternalTestType2) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
