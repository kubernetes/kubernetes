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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var _ runtime.Unstructured = &UnstructuredList{}
var _ metav1.ListInterface = &UnstructuredList{}

// UnstructuredList allows lists that do not have Golang structs
// registered to be manipulated generically. This can be used to deal
// with the API lists from a plug-in.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:deepcopy-gen=true
type UnstructuredList struct {
	Object map[string]interface{}

	// Items is a list of unstructured objects.
	Items []Unstructured `json:"items"`
}

func (u *UnstructuredList) GetObjectKind() schema.ObjectKind { return u }

func (u *UnstructuredList) IsList() bool { return true }

func (u *UnstructuredList) EachListItem(fn func(runtime.Object) error) error {
	for i := range u.Items {
		if err := fn(&u.Items[i]); err != nil {
			return err
		}
	}
	return nil
}

// UnstructuredContent returns a map contain an overlay of the Items field onto
// the Object field. Items always overwrites overlay.
func (u *UnstructuredList) UnstructuredContent() map[string]interface{} {
	out := make(map[string]interface{}, len(u.Object)+1)

	// shallow copy every property
	for k, v := range u.Object {
		out[k] = v
	}

	items := make([]interface{}, len(u.Items))
	for i, item := range u.Items {
		items[i] = item.UnstructuredContent()
	}
	out["items"] = items
	return out
}

// SetUnstructuredContent obeys the conventions of List and keeps Items and the items
// array in sync. If items is not an array of objects in the incoming map, then any
// mismatched item will be removed.
func (obj *UnstructuredList) SetUnstructuredContent(content map[string]interface{}) {
	obj.Object = content
	if content == nil {
		obj.Items = nil
		return
	}
	items, ok := obj.Object["items"].([]interface{})
	if !ok || items == nil {
		items = []interface{}{}
	}
	unstructuredItems := make([]Unstructured, 0, len(items))
	newItems := make([]interface{}, 0, len(items))
	for _, item := range items {
		o, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		unstructuredItems = append(unstructuredItems, Unstructured{Object: o})
		newItems = append(newItems, o)
	}
	obj.Items = unstructuredItems
	obj.Object["items"] = newItems
}

func (u *UnstructuredList) DeepCopy() *UnstructuredList {
	if u == nil {
		return nil
	}
	out := new(UnstructuredList)
	*out = *u
	out.Object = runtime.DeepCopyJSON(u.Object)
	out.Items = make([]Unstructured, len(u.Items))
	for i := range u.Items {
		u.Items[i].DeepCopyInto(&out.Items[i])
	}
	return out
}

// MarshalJSON ensures that the unstructured list object produces proper
// JSON when passed to Go's standard JSON library.
func (u *UnstructuredList) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := UnstructuredJSONScheme.Encode(u, &buf)
	return buf.Bytes(), err
}

// UnmarshalJSON ensures that the unstructured list object properly
// decodes JSON when passed to Go's standard JSON library.
func (u *UnstructuredList) UnmarshalJSON(b []byte) error {
	_, _, err := UnstructuredJSONScheme.Decode(b, nil, u)
	return err
}

func (u *UnstructuredList) GetAPIVersion() string {
	return getNestedString(u.Object, "apiVersion")
}

func (u *UnstructuredList) SetAPIVersion(version string) {
	u.setNestedField(version, "apiVersion")
}

func (u *UnstructuredList) GetKind() string {
	return getNestedString(u.Object, "kind")
}

func (u *UnstructuredList) SetKind(kind string) {
	u.setNestedField(kind, "kind")
}

func (u *UnstructuredList) GetResourceVersion() string {
	return getNestedString(u.Object, "metadata", "resourceVersion")
}

func (u *UnstructuredList) SetResourceVersion(version string) {
	u.setNestedField(version, "metadata", "resourceVersion")
}

func (u *UnstructuredList) GetSelfLink() string {
	return getNestedString(u.Object, "metadata", "selfLink")
}

func (u *UnstructuredList) SetSelfLink(selfLink string) {
	u.setNestedField(selfLink, "metadata", "selfLink")
}

func (u *UnstructuredList) GetContinue() string {
	return getNestedString(u.Object, "metadata", "continue")
}

func (u *UnstructuredList) SetContinue(c string) {
	u.setNestedField(c, "metadata", "continue")
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

func (u *UnstructuredList) setNestedField(value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	SetNestedField(u.Object, value, fields...)
}
