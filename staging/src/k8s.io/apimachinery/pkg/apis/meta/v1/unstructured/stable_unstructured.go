/*
Copyright The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

// StableUnstructured is like Unstructured, but it preserves the order of
// fields from the JSON that it was decoded from and emits them in the same
// order when encoding. Lists are not a separate type: a list is a
// StableUnstructured with an "items" field.
//
// UnstructuredContent and SetUnstructuredContent convert to and from
// map[string]any and therefore lose the order.
type StableUnstructured struct {
	Object Object
}

var _ metav1.Object = &StableUnstructured{}
var _ runtime.Unstructured = &StableUnstructured{}
var _ metav1.ListInterface = &StableUnstructured{}

func (u *StableUnstructured) GetObjectKind() schema.ObjectKind { return u }

// DeepCopyObject implements runtime.Object.
func (u *StableUnstructured) DeepCopyObject() runtime.Object {
	return u.DeepCopy()
}

// DeepCopy returns a deep copy of the object.
func (u *StableUnstructured) DeepCopy() *StableUnstructured {
	if u == nil {
		return nil
	}
	return &StableUnstructured{Object: u.Object.DeepCopy()}
}

func (u *StableUnstructured) IsList() bool {
	items, ok := u.Object.Get("items")
	if !ok {
		return false
	}
	_, ok = items.([]any)
	return ok
}

func (u *StableUnstructured) EachListItem(fn func(runtime.Object) error) error {
	field, ok := u.Object.Get("items")
	if !ok {
		return errors.New("content is not a list")
	}
	items, ok := field.([]any)
	if !ok {
		return fmt.Errorf("content is not a list: %T", field)
	}
	for i, item := range items {
		child, ok := item.(Object)
		if !ok {
			return fmt.Errorf("items member is not an object: %T", item)
		}
		wrapper := &StableUnstructured{Object: child}
		err := fn(wrapper)
		// Set may have reallocated the slice.
		items[i] = wrapper.Object
		if err != nil {
			return err
		}
	}
	return nil
}

func (u *StableUnstructured) EachListItemWithAlloc(fn func(runtime.Object) error) error {
	// EachListItem already allocates a new wrapper for each item.
	return u.EachListItem(fn)
}

func (u *StableUnstructured) NewEmptyInstance() runtime.Unstructured {
	out := &StableUnstructured{}
	if u != nil {
		out.SetGroupVersionKind(u.GroupVersionKind())
	}
	return out
}

// UnstructuredContent returns a converted copy of the content.
func (u *StableUnstructured) UnstructuredContent() map[string]interface{} {
	return u.Object.ToMap()
}

// SetUnstructuredContent replaces the content with a converted copy of the
// map, with fields sorted by name.
func (u *StableUnstructured) SetUnstructuredContent(content map[string]interface{}) {
	u.Object = ObjectFromMap(content)
}

// MarshalJSON emits the fields in order.
func (u *StableUnstructured) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := StableUnstructuredJSONScheme.Encode(u, &buf)
	return buf.Bytes(), err
}

// UnmarshalJSON decodes the fields in order.
func (u *StableUnstructured) UnmarshalJSON(b []byte) error {
	_, _, err := StableUnstructuredJSONScheme.Decode(b, nil, u)
	return err
}

// nestedGet returns the value found by following the path.
func (u *StableUnstructured) nestedGet(fields ...string) (any, bool) {
	var cur any = u.Object
	for _, f := range fields {
		obj, ok := cur.(Object)
		if !ok {
			return nil, false
		}
		if cur, ok = obj.Get(f); !ok {
			return nil, false
		}
	}
	return cur, true
}

// nestedSet sets the value found by following the path, replacing
// anything that is not an object along the way.
func (u *StableUnstructured) nestedSet(value any, fields ...string) {
	setNestedStable(&u.Object, value, fields)
}

func setNestedStable(o *Object, value any, fields []string) {
	if len(fields) == 1 {
		o.Set(fields[0], value)
		return
	}
	child, _ := o.getObject(fields[0])
	setNestedStable(&child, value, fields[1:])
	o.Set(fields[0], child)
}

func (o Object) getObject(name string) (Object, bool) {
	v, _ := o.Get(name)
	obj, ok := v.(Object)
	return obj, ok
}

// nestedDelete removes the value found by following the path, if there is one.
func (u *StableUnstructured) nestedDelete(fields ...string) {
	deleteNestedStable(&u.Object, fields)
}

func deleteNestedStable(o *Object, fields []string) {
	if len(fields) == 1 {
		o.Delete(fields[0])
		return
	}
	child, ok := o.getObject(fields[0])
	if !ok {
		return
	}
	deleteNestedStable(&child, fields[1:])
	o.Set(fields[0], child)
}

// legacyView returns an Unstructured containing a map-based copy of the given
// metadata fields so that the accessor semantics of Unstructured are reused.
func (u *StableUnstructured) legacyView(keys ...string) *Unstructured {
	metadata := map[string]any{}
	for _, k := range keys {
		if v, ok := u.nestedGet("metadata", k); ok {
			metadata[k] = stableToPlain(v)
		}
	}
	return &Unstructured{Object: map[string]any{"metadata": metadata}}
}

// legacyUpdate applies fn to a legacyView of one metadata field and copies the result back.
func (u *StableUnstructured) legacyUpdate(key string, fn func(*Unstructured)) {
	view := u.legacyView(key)
	fn(view)
	if v, ok := view.Object["metadata"].(map[string]any)[key]; ok {
		u.nestedSet(stableFromPlain(v), "metadata", key)
	} else {
		u.nestedDelete("metadata", key)
	}
}

func (u *StableUnstructured) getString(fields ...string) string {
	v, _ := u.nestedGet(fields...)
	s, _ := v.(string)
	return s
}

func (u *StableUnstructured) GetOwnerReferences() []metav1.OwnerReference {
	return u.legacyView("ownerReferences").GetOwnerReferences()
}

func (u *StableUnstructured) SetOwnerReferences(references []metav1.OwnerReference) {
	u.legacyUpdate("ownerReferences", func(v *Unstructured) { v.SetOwnerReferences(references) })
}

func (u *StableUnstructured) GetAPIVersion() string {
	return u.getString("apiVersion")
}

func (u *StableUnstructured) SetAPIVersion(version string) {
	u.nestedSet(version, "apiVersion")
}

func (u *StableUnstructured) GetKind() string {
	return u.getString("kind")
}

func (u *StableUnstructured) SetKind(kind string) {
	u.nestedSet(kind, "kind")
}

func (u *StableUnstructured) GetNamespace() string {
	return u.getString("metadata", "namespace")
}

func (u *StableUnstructured) SetNamespace(namespace string) {
	u.legacyUpdate("namespace", func(v *Unstructured) { v.SetNamespace(namespace) })
}

func (u *StableUnstructured) GetName() string {
	return u.getString("metadata", "name")
}

func (u *StableUnstructured) SetName(name string) {
	u.legacyUpdate("name", func(v *Unstructured) { v.SetName(name) })
}

func (u *StableUnstructured) GetGenerateName() string {
	return u.getString("metadata", "generateName")
}

func (u *StableUnstructured) SetGenerateName(generateName string) {
	u.legacyUpdate("generateName", func(v *Unstructured) { v.SetGenerateName(generateName) })
}

func (u *StableUnstructured) GetUID() types.UID {
	return types.UID(u.getString("metadata", "uid"))
}

func (u *StableUnstructured) SetUID(uid types.UID) {
	u.legacyUpdate("uid", func(v *Unstructured) { v.SetUID(uid) })
}

func (u *StableUnstructured) GetResourceVersion() string {
	return u.getString("metadata", "resourceVersion")
}

func (u *StableUnstructured) SetResourceVersion(resourceVersion string) {
	u.legacyUpdate("resourceVersion", func(v *Unstructured) { v.SetResourceVersion(resourceVersion) })
}

func (u *StableUnstructured) GetGeneration() int64 {
	return u.legacyView("generation").GetGeneration()
}

func (u *StableUnstructured) SetGeneration(generation int64) {
	u.legacyUpdate("generation", func(v *Unstructured) { v.SetGeneration(generation) })
}

func (u *StableUnstructured) GetSelfLink() string {
	return u.getString("metadata", "selfLink")
}

func (u *StableUnstructured) SetSelfLink(selfLink string) {
	u.legacyUpdate("selfLink", func(v *Unstructured) { v.SetSelfLink(selfLink) })
}

func (u *StableUnstructured) GetContinue() string {
	return u.getString("metadata", "continue")
}

func (u *StableUnstructured) SetContinue(c string) {
	u.legacyUpdate("continue", func(v *Unstructured) { v.SetContinue(c) })
}

func (u *StableUnstructured) GetRemainingItemCount() *int64 {
	return u.legacyView("remainingItemCount").GetRemainingItemCount()
}

func (u *StableUnstructured) SetRemainingItemCount(c *int64) {
	u.legacyUpdate("remainingItemCount", func(v *Unstructured) { v.SetRemainingItemCount(c) })
}

func (u *StableUnstructured) GetCreationTimestamp() metav1.Time {
	return u.legacyView("creationTimestamp").GetCreationTimestamp()
}

func (u *StableUnstructured) SetCreationTimestamp(timestamp metav1.Time) {
	u.legacyUpdate("creationTimestamp", func(v *Unstructured) { v.SetCreationTimestamp(timestamp) })
}

func (u *StableUnstructured) GetDeletionTimestamp() *metav1.Time {
	return u.legacyView("deletionTimestamp").GetDeletionTimestamp()
}

func (u *StableUnstructured) SetDeletionTimestamp(timestamp *metav1.Time) {
	u.legacyUpdate("deletionTimestamp", func(v *Unstructured) { v.SetDeletionTimestamp(timestamp) })
}

func (u *StableUnstructured) GetDeletionGracePeriodSeconds() *int64 {
	return u.legacyView("deletionGracePeriodSeconds").GetDeletionGracePeriodSeconds()
}

func (u *StableUnstructured) SetDeletionGracePeriodSeconds(deletionGracePeriodSeconds *int64) {
	u.legacyUpdate("deletionGracePeriodSeconds", func(v *Unstructured) {
		v.SetDeletionGracePeriodSeconds(deletionGracePeriodSeconds)
	})
}

func (u *StableUnstructured) GetLabels() map[string]string {
	return u.legacyView("labels").GetLabels()
}

func (u *StableUnstructured) SetLabels(labels map[string]string) {
	u.legacyUpdate("labels", func(v *Unstructured) { v.SetLabels(labels) })
}

func (u *StableUnstructured) GetAnnotations() map[string]string {
	return u.legacyView("annotations").GetAnnotations()
}

func (u *StableUnstructured) SetAnnotations(annotations map[string]string) {
	u.legacyUpdate("annotations", func(v *Unstructured) { v.SetAnnotations(annotations) })
}

func (u *StableUnstructured) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
}

func (u *StableUnstructured) GroupVersionKind() schema.GroupVersionKind {
	gv, err := schema.ParseGroupVersion(u.GetAPIVersion())
	if err != nil {
		return schema.GroupVersionKind{}
	}
	return gv.WithKind(u.GetKind())
}

func (u *StableUnstructured) GetFinalizers() []string {
	return u.legacyView("finalizers").GetFinalizers()
}

func (u *StableUnstructured) SetFinalizers(finalizers []string) {
	u.legacyUpdate("finalizers", func(v *Unstructured) { v.SetFinalizers(finalizers) })
}

func (u *StableUnstructured) GetManagedFields() []metav1.ManagedFieldsEntry {
	return u.legacyView("managedFields").GetManagedFields()
}

func (u *StableUnstructured) SetManagedFields(managedFields []metav1.ManagedFieldsEntry) {
	u.legacyUpdate("managedFields", func(v *Unstructured) { v.SetManagedFields(managedFields) })
}

// StableUnstructuredJSONScheme is the counterpart of UnstructuredJSONScheme
// for StableUnstructured.
var StableUnstructuredJSONScheme runtime.Codec = stableUnstructuredJSONScheme{}

type stableUnstructuredJSONScheme struct{}

const stableUnstructuredJSONSchemeIdentifier runtime.Identifier = "stableUnstructuredJSON"

func (s stableUnstructuredJSONScheme) Decode(data []byte, _ *schema.GroupVersionKind, obj runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	switch t := obj.(type) {
	case nil:
		obj = &StableUnstructured{}
		if err := s.decodeInto(data, obj.(*StableUnstructured)); err != nil {
			return nil, nil, err
		}
	case *StableUnstructured:
		if err := s.decodeInto(data, t); err != nil {
			return nil, nil, err
		}
	default:
		if err := UnstructuredJSONScheme.(unstructuredJSONScheme).decodeInto(data, obj); err != nil {
			return nil, nil, err
		}
	}

	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, &gvk, runtime.NewMissingKindErr(string(data))
	}
	return obj, &gvk, nil
}

func (s stableUnstructuredJSONScheme) decodeInto(data []byte, u *StableUnstructured) error {
	var o Object
	if err := o.UnmarshalJSON(data); err != nil {
		return err
	}
	u.Object = o

	// For typed lists, e.g., a PodList, API server doesn't set each item's
	// APIVersion and Kind. We need to set it.
	if items, ok := o.Get("items"); ok {
		// A null list is an empty list, as in UnstructuredJSONScheme.
		if items == nil {
			items = []any{}
			o.Set("items", items)
		}
		if list, ok := items.([]any); ok {
			listAPIVersion := u.GetAPIVersion()
			itemKind := strings.TrimSuffix(u.GetKind(), "List")
			for i, item := range list {
				child, ok := item.(Object)
				if !ok {
					continue
				}
				_, hasKind := child.Get("kind")
				_, hasAPIVersion := child.Get("apiVersion")
				if !hasKind && !hasAPIVersion {
					// Put them first, where they usually are.
					list[i] = append(Object{{Name: "kind", Value: itemKind}, {Name: "apiVersion", Value: listAPIVersion}}, child...)
				}
			}
		}
	}
	return nil
}

func (s stableUnstructuredJSONScheme) Encode(obj runtime.Object, w io.Writer) error {
	if co, ok := obj.(runtime.CacheableObject); ok {
		return co.CacheEncode(s.Identifier(), s.doEncode, w)
	}
	return s.doEncode(obj, w)
}

func (stableUnstructuredJSONScheme) doEncode(obj runtime.Object, w io.Writer) error {
	t, ok := obj.(*StableUnstructured)
	if !ok {
		return UnstructuredJSONScheme.(unstructuredJSONScheme).doEncode(obj, w)
	}
	data, err := t.Object.MarshalJSON()
	if err != nil {
		return err
	}
	// Same trailing newline as json.Encoder.
	_, err = w.Write(append(data, '\n'))
	return err
}

// Identifier implements runtime.Encoder interface.
func (stableUnstructuredJSONScheme) Identifier() runtime.Identifier {
	return stableUnstructuredJSONSchemeIdentifier
}
