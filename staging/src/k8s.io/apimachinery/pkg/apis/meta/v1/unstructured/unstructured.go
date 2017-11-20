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
	"errors"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/jsonlike"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// Unstructured allows objects that do not have Golang structs registered to be manipulated
// generically. This can be used to deal with the API objects from a plug-in. Unstructured
// objects still have functioning TypeMeta features-- kind, version, etc.
//
// WARNING: This object has accessors for the v1 standard metadata. You *MUST NOT* use this
// type if you are dealing with objects that are not in the server meta v1 schema.
//
// TODO: make the serialization part of this type distinct from the field accessors.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:deepcopy-gen=true
type Unstructured struct {
	// Object is a JSON compatible map with string, float, int, bool, []interface{}, or
	// map[string]interface{}
	// children.
	Object jsonlike.Object
}

var _ metav1.Object = &Unstructured{}
var _ runtime.Unstructured = &Unstructured{}

func (obj *Unstructured) GetObjectKind() schema.ObjectKind { return obj }

func (obj *Unstructured) IsList() bool {
	field, ok := obj.Object["items"]
	if !ok {
		return false
	}
	_, ok = field.([]interface{})
	return ok
}

func (obj *Unstructured) EachListItem(fn func(runtime.Object) error) error {
	field, ok := obj.Object["items"]
	if !ok {
		return errors.New("content is not a list")
	}
	items, ok := field.([]interface{})
	if !ok {
		return fmt.Errorf("content is not a list: %T", field)
	}
	for _, item := range items {
		child, ok := item.(map[string]interface{})
		if !ok {
			return fmt.Errorf("items member is not an object: %T", child)
		}
		if err := fn(&Unstructured{Object: child}); err != nil {
			return err
		}
	}
	return nil
}

func (obj *Unstructured) UnstructuredContent() jsonlike.Object {
	if obj.Object == nil {
		obj.Object = make(map[string]interface{})
	}
	return obj.Object
}

func (obj *Unstructured) SetUnstructuredContent(content map[string]interface{}) {
	obj.Object = content
}

// MarshalJSON ensures that the unstructured object produces proper
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := UnstructuredJSONScheme.Encode(u, &buf)
	return buf.Bytes(), err
}

// UnmarshalJSON ensures that the unstructured object properly decodes
// JSON when passed to Go's standard JSON library.
func (u *Unstructured) UnmarshalJSON(b []byte) error {
	_, _, err := UnstructuredJSONScheme.Decode(b, nil, u)
	return err
}

func (in *Unstructured) DeepCopy() *Unstructured {
	if in == nil {
		return nil
	}
	out := new(Unstructured)
	*out = *in
	out.Object = in.Object.DeepCopy()
	return out
}

func (u *Unstructured) GetOwnerReferences() []metav1.OwnerReference {
	field, ok := u.Object.Field("metadata", "ownerReferences")
	if !ok {
		return nil
	}
	original, ok := field.([]interface{})
	if !ok {
		return nil
	}
	ret := make([]metav1.OwnerReference, 0, len(original))
	for _, obj := range original {
		o, ok := obj.(map[string]interface{})
		if !ok {
			// expected map[string]interface{}, got something else
			return nil
		}
		ret = append(ret, extractOwnerReference(o))
	}
	return ret
}

func (u *Unstructured) SetOwnerReferences(references []metav1.OwnerReference) {
	newReferences := make([]interface{}, 0, len(references))
	for _, reference := range references {
		out, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&reference)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to convert Owner Reference: %v", err))
			continue
		}
		newReferences = append(newReferences, out)
	}
	u.Object.SetField(newReferences, "metadata", "ownerReferences")
}

func (u *Unstructured) GetAPIVersion() string {
	return u.Object.GetString("apiVersion")
}

func (u *Unstructured) SetAPIVersion(version string) {
	u.UnstructuredContent().SetField(version, "apiVersion")
}

func (u *Unstructured) GetKind() string {
	return u.Object.GetString("kind")
}

func (u *Unstructured) SetKind(kind string) {
	u.UnstructuredContent().SetField(kind, "kind")
}

func (u *Unstructured) GetNamespace() string {
	return u.Object.GetString("metadata", "namespace")
}

func (u *Unstructured) SetNamespace(namespace string) {
	u.UnstructuredContent().SetField(namespace, "metadata", "namespace")
}

func (u *Unstructured) GetName() string {
	return u.Object.GetString("metadata", "name")
}

func (u *Unstructured) SetName(name string) {
	u.UnstructuredContent().SetField(name, "metadata", "name")
}

func (u *Unstructured) GetGenerateName() string {
	return u.Object.GetString("metadata", "generateName")
}

func (u *Unstructured) SetGenerateName(name string) {
	u.UnstructuredContent().SetField(name, "metadata", "generateName")
}

func (u *Unstructured) GetUID() types.UID {
	return types.UID(u.Object.GetString("metadata", "uid"))
}

func (u *Unstructured) SetUID(uid types.UID) {
	u.UnstructuredContent().SetField(string(uid), "metadata", "uid")
}

func (u *Unstructured) GetResourceVersion() string {
	return u.Object.GetString("metadata", "resourceVersion")
}

func (u *Unstructured) SetResourceVersion(version string) {
	u.UnstructuredContent().SetField(version, "metadata", "resourceVersion")
}

func (u *Unstructured) GetGeneration() int64 {
	val, ok := u.Object.Int64("metadata", "generation")
	if !ok {
		return 0
	}
	return val
}

func (u *Unstructured) SetGeneration(generation int64) {
	u.UnstructuredContent().SetField(generation, "metadata", "generation")
}

func (u *Unstructured) GetSelfLink() string {
	return u.Object.GetString("metadata", "selfLink")
}

func (u *Unstructured) SetSelfLink(selfLink string) {
	u.UnstructuredContent().SetField(selfLink, "metadata", "selfLink")
}

func (u *Unstructured) GetContinue() string {
	return u.Object.GetString("metadata", "continue")
}

func (u *Unstructured) SetContinue(c string) {
	u.UnstructuredContent().SetField(c, "metadata", "continue")
}

func (u *Unstructured) GetCreationTimestamp() metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(u.Object.GetString("metadata", "creationTimestamp"))
	return timestamp
}

func (u *Unstructured) SetCreationTimestamp(timestamp metav1.Time) {
	ts, _ := timestamp.MarshalQueryParameter()
	if len(ts) == 0 || timestamp.Time.IsZero() {
		u.Object.RemoveField("metadata", "creationTimestamp")
		return
	}
	u.UnstructuredContent().SetField(ts, "metadata", "creationTimestamp")
}

func (u *Unstructured) GetDeletionTimestamp() *metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(u.Object.GetString("metadata", "deletionTimestamp"))
	if timestamp.IsZero() {
		return nil
	}
	return &timestamp
}

func (u *Unstructured) SetDeletionTimestamp(timestamp *metav1.Time) {
	if timestamp == nil {
		u.Object.RemoveField("metadata", "deletionTimestamp")
		return
	}
	ts, _ := timestamp.MarshalQueryParameter()
	u.UnstructuredContent().SetField(ts, "metadata", "deletionTimestamp")
}

func (u *Unstructured) GetDeletionGracePeriodSeconds() *int64 {
	val, ok := u.Object.Int64("metadata", "deletionGracePeriodSeconds")
	if !ok {
		return nil
	}
	return &val
}

func (u *Unstructured) SetDeletionGracePeriodSeconds(deletionGracePeriodSeconds *int64) {
	if deletionGracePeriodSeconds == nil {
		u.Object.RemoveField("metadata", "deletionGracePeriodSeconds")
		return
	}
	u.UnstructuredContent().SetField(*deletionGracePeriodSeconds, "metadata", "deletionGracePeriodSeconds")
}

func (u *Unstructured) GetLabels() map[string]string {
	m, _ := u.Object.StringMap("metadata", "labels")
	return m
}

func (u *Unstructured) SetLabels(labels map[string]string) {
	u.UnstructuredContent().SetStringMap(labels, "metadata", "labels")
}

func (u *Unstructured) GetAnnotations() map[string]string {
	m, _ := u.Object.StringMap("metadata", "annotations")
	return m
}

func (u *Unstructured) SetAnnotations(annotations map[string]string) {
	u.UnstructuredContent().SetStringMap(annotations, "metadata", "annotations")
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

func (u *Unstructured) GetInitializers() *metav1.Initializers {
	field, ok := u.Object.Field("metadata", "initializers")
	if !ok {
		return nil
	}
	obj, ok := field.(map[string]interface{})
	if !ok {
		return nil
	}
	out := &metav1.Initializers{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj, out); err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to retrieve initializers for object: %v", err))
		return nil
	}
	return out
}

func (u *Unstructured) SetInitializers(initializers *metav1.Initializers) {
	if initializers == nil {
		u.Object.RemoveField("metadata", "initializers")
		return
	}
	out, err := runtime.DefaultUnstructuredConverter.ToUnstructured(initializers)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to retrieve initializers for object: %v", err))
	}
	u.UnstructuredContent().SetField(out, "metadata", "initializers")
}

func (u *Unstructured) GetFinalizers() []string {
	val, _ := u.Object.StringSlice("metadata", "finalizers")
	return val
}

func (u *Unstructured) SetFinalizers(finalizers []string) {
	u.UnstructuredContent().SetStringSlice(finalizers, "metadata", "finalizers")
}

func (u *Unstructured) GetClusterName() string {
	return u.Object.GetString("metadata", "clusterName")
}

func (u *Unstructured) SetClusterName(clusterName string) {
	u.UnstructuredContent().SetField(clusterName, "metadata", "clusterName")
}

func extractOwnerReference(v jsonlike.Object) metav1.OwnerReference {
	// though this field is a *bool, but when decoded from JSON, it's
	// unmarshalled as bool.
	var controllerPtr *bool
	if controller, ok := v.Bool("controller"); ok {
		controllerPtr = &controller
	}
	var blockOwnerDeletionPtr *bool
	if blockOwnerDeletion, ok := v.Bool("blockOwnerDeletion"); ok {
		blockOwnerDeletionPtr = &blockOwnerDeletion
	}
	return metav1.OwnerReference{
		Kind:               v.GetString("kind"),
		Name:               v.GetString("name"),
		APIVersion:         v.GetString("apiVersion"),
		UID:                types.UID(v.GetString("uid")),
		Controller:         controllerPtr,
		BlockOwnerDeletion: blockOwnerDeletionPtr,
	}
}

func setOwnerReference(src metav1.OwnerReference) map[string]interface{} {
	ret := map[string]interface{}{
		"kind":       src.Kind,
		"name":       src.Name,
		"apiVersion": src.APIVersion,
		"uid":        string(src.UID),
	}
	// json.Unmarshal() extracts boolean json fields as bool, not as *bool and hence extractOwnerReference()
	// expects bool or a missing field, not *bool. So if pointer is nil, fields are omitted from the ret object.
	// If pointer is non-nil, they are set to the referenced value.
	if src.Controller != nil {
		ret["controller"] = *src.Controller
	}
	if src.BlockOwnerDeletion != nil {
		ret["blockOwnerDeletion"] = *src.BlockOwnerDeletion
	}
	return ret
}
