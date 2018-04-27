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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type EmbeddedTest struct {
	runtime.TypeMeta
	ID          string
	Object      runtime.Object
	EmptyObject runtime.Object
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type EmbeddedTestExternal struct {
	runtime.TypeMeta `json:",inline"`
	ID               string               `json:"id,omitempty"`
	Object           runtime.RawExtension `json:"object,omitempty"`
	EmptyObject      runtime.RawExtension `json:"emptyObject,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ObjectTest struct {
	runtime.TypeMeta

	ID    string
	Items []runtime.Object
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ObjectTestExternal struct {
	runtime.TypeMeta `yaml:",inline" json:",inline"`

	ID    string                 `json:"id,omitempty"`
	Items []runtime.RawExtension `json:"items,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalSimple struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExtensionA struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExtensionB struct {
	runtime.TypeMeta `json:",inline"`
	TestString       string `json:"testString"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.RawExtension `json:"extension,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalOptionalExtensionType struct {
	runtime.TypeMeta `json:",inline"`
	Extension        runtime.Object `json:"extension,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InternalComplex struct {
	runtime.TypeMeta
	String    string
	Integer   int
	Integer64 int64
	Int64     int64
	Bool      bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type TestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalTestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ExternalInternalSame struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     TestType2 `json:"A,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type UnversionedType struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string `json:"A,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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

// +k8s:deepcopy-gen=false
type Unstructured struct {
	// Object is a JSON compatible map with string, float, int, bool, []interface{}, or
	// map[string]interface{}
	// children.
	Object map[string]interface{}
}

var _ runtime.Unstructured = &Unstructured{}

func (obj *Unstructured) GetObjectKind() schema.ObjectKind { return obj }

func (obj *Unstructured) IsList() bool {
	if obj.Object != nil {
		_, ok := obj.Object["items"]
		return ok
	}
	return false
}

func (obj *Unstructured) EachListItem(fn func(runtime.Object) error) error {
	if obj.Object == nil {
		return fmt.Errorf("content is not a list")
	}
	field, ok := obj.Object["items"]
	if !ok {
		return fmt.Errorf("content is not a list")
	}
	items, ok := field.([]interface{})
	if !ok {
		return nil
	}
	for _, item := range items {
		child, ok := item.(map[string]interface{})
		if !ok {
			return fmt.Errorf("items member is not an object")
		}
		if err := fn(&Unstructured{Object: child}); err != nil {
			return err
		}
	}
	return nil
}

func (obj *Unstructured) UnstructuredContent() map[string]interface{} {
	if obj.Object == nil {
		return make(map[string]interface{})
	}
	return obj.Object
}

func (obj *Unstructured) SetUnstructuredContent(content map[string]interface{}) {
	obj.Object = content
}

// MarshalJSON ensures that the unstructured object produces proper
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) MarshalJSON() ([]byte, error) {
	return json.Marshal(u.Object)
}

// UnmarshalJSON ensures that the unstructured object properly decodes
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) UnmarshalJSON(b []byte) error {
	return json.Unmarshal(b, &u.Object)
}

func (in *Unstructured) DeepCopyObject() runtime.Object {
	return in.DeepCopy()
}

func (in *Unstructured) DeepCopy() *Unstructured {
	if in == nil {
		return nil
	}
	out := new(Unstructured)
	*out = *in
	out.Object = runtime.DeepCopyJSON(in.Object)
	return out
}

func (u *Unstructured) GroupVersionKind() schema.GroupVersionKind {
	apiVersion, ok := u.Object["apiVersion"].(string)
	if !ok {
		return schema.GroupVersionKind{}
	}
	gv, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		return schema.GroupVersionKind{}
	}
	kind, ok := u.Object["kind"].(string)
	if ok {
		return gv.WithKind(kind)
	}
	return schema.GroupVersionKind{}
}

func (u *Unstructured) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	u.Object["apiVersion"] = gvk.GroupVersion().String()
	u.Object["kind"] = gvk.Kind
}
