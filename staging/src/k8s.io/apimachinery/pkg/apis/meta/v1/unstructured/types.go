/*
Copyright 2015 The Kubernetes Authors.

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

package unstructured

import (
	"bytes"
	"encoding/json"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Unstructured allows objects that do not have Golang structs registered to be manipulated
// generically. This can be used to deal with the API objects from a plug-in. Unstructured
// objects still have functioning TypeMeta features-- kind, version, etc.
//
// WARNING: This object has accessors for the v1 standard metadata. You *MUST NOT* use this
// type if you are dealing with objects that are not in the server meta v1 schema.
//
// TODO: make the serialization part of this type distinct from the field accessors.
//
// +k8s:deepcopy-gen=true
type Unstructured struct {
	// Object is a JSON compatible map with string, float, int, bool, []interface{}, or
	// map[string]interface{}
	// children.
	Object map[string]interface{}
}

var _ runtime.Object = &Unstructured{}
var _ runtime.Object = &UnstructuredList{}

var _ runtime.Unstructured = &Unstructured{}
var _ runtime.Unstructured = &UnstructuredList{}

func (obj *Unstructured) GetObjectKind() schema.ObjectKind     { return obj }
func (obj *UnstructuredList) GetObjectKind() schema.ObjectKind { return obj }

func (obj *Unstructured) IsUnstructuredObject()     {}
func (obj *UnstructuredList) IsUnstructuredObject() {}

func (obj *Unstructured) IsList() bool {
	if obj.Object != nil {
		_, ok := obj.Object["items"]
		return ok
	}
	return false
}
func (obj *UnstructuredList) IsList() bool { return true }

func (obj *Unstructured) UnstructuredContent() map[string]interface{} {
	if obj.Object == nil {
		obj.Object = make(map[string]interface{})
	}
	return obj.Object
}

// UnstructuredContent returns a map contain an overlay of the Items field onto
// the Object field. Items always overwrites overlay. Changing "items" in the
// returned object will affect items in the underlying Items field, but changing
// the "items" slice itself will have no effect.
// TODO: expose SetUnstructuredContent on runtime.Unstructured that allows
// items to be changed.
func (obj *UnstructuredList) UnstructuredContent() map[string]interface{} {
	out := obj.Object
	if out == nil {
		out = make(map[string]interface{})
	}
	items := make([]interface{}, len(obj.Items))
	for i, item := range obj.Items {
		items[i] = item.Object
	}
	out["items"] = items
	return out
}

func (u Unstructured) DeepCopy() Unstructured {
	return Unstructured{
		Object: deepCopyJSON(u.Object).(map[string]interface{}),
	}
}

func deepCopyJSON(x interface{}) interface{} {
	switch x := x.(type) {
	case map[string]interface{}:
		clone := make(map[string]interface{}, len(x))
		for k, v := range x {
			clone[k] = deepCopyJSON(v)
		}
		return clone
	case []interface{}:
		clone := make([]interface{}, len(x))
		for i := range x {
			clone[i] = deepCopyJSON(x[i])
		}
		return clone
	default:
		return x
	}
}

// MarshalJSON ensures that the unstructured object produces proper
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(u.Object)
	return buf.Bytes(), err
}

// UnmarshalJSON ensures that the unstructured object properly decodes
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) UnmarshalJSON(b []byte) error {
	m := make(map[string]interface{})
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	u.Object = m
	return nil
}

func (u *Unstructured) GetAPIVersion() string {
	str, _ := u.Object["apiVersion"].(string)
	return str
}

func (u *Unstructured) SetAPIVersion(version string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	u.Object["apiVersion"] = version
}

func (u *Unstructured) GetKind() string {
	str, _ := u.Object["kind"].(string)
	return str
}

func (u *Unstructured) SetKind(kind string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	u.Object["kind"] = kind
}

func (u *Unstructured) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
}

func (u *Unstructured) GroupVersionKind() schema.GroupVersionKind {
	gv, err := schema.ParseGroupVersion(u.GetAPIVersion())
	if err != nil {
		return schema.GroupVersionKind{}
	}
	gvk := gv.WithKind(u.GetKind())
	return gvk
}

// UnstructuredList allows lists that do not have Golang structs
// registered to be manipulated generically. This can be used to deal
// with the API lists from a plug-in.
//
// +k8s:deepcopy-gen=true
type UnstructuredList struct {
	Object map[string]interface{}

	// Items is a list of unstructured objects.
	Items []Unstructured `json:"items"`
}

func (u UnstructuredList) DeepCopy() UnstructuredList {
	clone := UnstructuredList{
		Object: deepCopyJSON(u.Object).(map[string]interface{}),
		Items:  make([]Unstructured, len(u.Items)),
	}
	for i := range u.Items {
		clone.Items[i] = u.Items[i].DeepCopy()
	}
	return clone
}

// MarshalJSON ensures that the unstructured list object produces proper
// JSON when passed to Go's standard JSON library.
func (u *UnstructuredList) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	items := make([]map[string]interface{}, 0, len(u.Items))
	for _, i := range u.Items {
		items = append(items, i.Object)
	}
	u.Object["items"] = items
	defer func() { delete(u.Object, "items") }()
	err := json.NewEncoder(&buf).Encode(u.Object)
	return buf.Bytes(), err
}

// UnmarshalJSON ensures that the unstructured list object properly
// decodes JSON when passed to Go's standard JSON library.
func (u *UnstructuredList) UnmarshalJSON(b []byte) error {
	type decodeList struct {
		Items []json.RawMessage
	}

	var dList decodeList
	if err := json.Unmarshal(b, &dList); err != nil {
		return err
	}

	if err := json.Unmarshal(b, &u.Object); err != nil {
		return err
	}

	// For typed lists, e.g., a PodList, API server doesn't set each item's
	// APIVersion and Kind. We need to set it.
	listAPIVersion := u.GetAPIVersion()
	listKind := u.GetKind()
	itemKind := strings.TrimSuffix(listKind, "List")

	delete(u.Object, "items")
	u.Items = nil
	for _, i := range dList.Items {
		unstruct := &Unstructured{}
		if err := unstruct.UnmarshalJSON([]byte(i)); err != nil {
			return err
		}
		// This is hacky. Set the item's Kind and APIVersion to those inferred
		// from the List.
		if len(unstruct.GetKind()) == 0 && len(unstruct.GetAPIVersion()) == 0 {
			unstruct.SetKind(itemKind)
			unstruct.SetAPIVersion(listAPIVersion)
		}
		u.Items = append(u.Items, *unstruct)
	}
	return nil
}

func (u *UnstructuredList) GetAPIVersion() string {
	str, _ := u.Object["apiVersion"].(string)
	return str
}

func (u *UnstructuredList) SetAPIVersion(version string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	u.Object["apiVersion"] = version
}

func (u *UnstructuredList) GetKind() string {
	str, _ := u.Object["kind"].(string)
	return str
}

func (u *UnstructuredList) SetKind(kind string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	u.Object["kind"] = kind
}

func (u *UnstructuredList) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
}

func (u *UnstructuredList) GroupVersionKind() schema.GroupVersionKind {
	gv, err := schema.ParseGroupVersion(u.GetAPIVersion())
	if err != nil {
		return schema.GroupVersionKind{}
	}
	gvk := gv.WithKind(u.GetKind())
	return gvk
}
