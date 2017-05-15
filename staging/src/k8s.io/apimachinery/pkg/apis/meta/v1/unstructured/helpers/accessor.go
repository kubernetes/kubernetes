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

package helpers

import (
	"fmt"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
)

type Accessor unstructured.Unstructured
type ListAccessor unstructured.UnstructuredList

var _ metav1.Object = &Accessor{}

func getNestedField(obj map[string]interface{}, fields ...string) interface{} {
	var val interface{} = obj
	for _, field := range fields {
		if _, ok := val.(map[string]interface{}); !ok {
			return nil
		}
		val = val.(map[string]interface{})[field]
	}
	return val
}

func getNestedString(obj map[string]interface{}, fields ...string) string {
	if str, ok := getNestedField(obj, fields...).(string); ok {
		return str
	}
	return ""
}

func getNestedInt64(obj map[string]interface{}, fields ...string) int64 {
	if str, ok := getNestedField(obj, fields...).(int64); ok {
		return str
	}
	return 0
}

func getNestedInt64Pointer(obj map[string]interface{}, fields ...string) *int64 {
	if str, ok := getNestedField(obj, fields...).(*int64); ok {
		return str
	}
	return nil
}

func getNestedSlice(obj map[string]interface{}, fields ...string) []string {
	if m, ok := getNestedField(obj, fields...).([]interface{}); ok {
		strSlice := make([]string, 0, len(m))
		for _, v := range m {
			if str, ok := v.(string); ok {
				strSlice = append(strSlice, str)
			}
		}
		return strSlice
	}
	return nil
}

func getNestedMap(obj map[string]interface{}, fields ...string) map[string]string {
	if m, ok := getNestedField(obj, fields...).(map[string]interface{}); ok {
		strMap := make(map[string]string, len(m))
		for k, v := range m {
			if str, ok := v.(string); ok {
				strMap[k] = str
			}
		}
		return strMap
	}
	return nil
}

func setNestedField(obj map[string]interface{}, value interface{}, fields ...string) {
	m := obj
	if len(fields) > 1 {
		for _, field := range fields[0 : len(fields)-1] {
			if _, ok := m[field].(map[string]interface{}); !ok {
				m[field] = make(map[string]interface{})
			}
			m = m[field].(map[string]interface{})
		}
	}
	m[fields[len(fields)-1]] = value
}

func setNestedSlice(obj map[string]interface{}, value []string, fields ...string) {
	m := make([]interface{}, 0, len(value))
	for _, v := range value {
		m = append(m, v)
	}
	setNestedField(obj, m, fields...)
}

func setNestedMap(obj map[string]interface{}, value map[string]string, fields ...string) {
	m := make(map[string]interface{}, len(value))
	for k, v := range value {
		m[k] = v
	}
	setNestedField(obj, m, fields...)
}

func (u *Accessor) setNestedField(value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedField(u.Object, value, fields...)
}

func (u *Accessor) setNestedSlice(value []string, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedSlice(u.Object, value, fields...)
}

func (u *Accessor) setNestedMap(value map[string]string, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedMap(u.Object, value, fields...)
}

func extractOwnerReference(src interface{}) metav1.OwnerReference {
	v := src.(map[string]interface{})
	// though this field is a *bool, but when decoded from JSON, it's
	// unmarshalled as bool.
	var controllerPtr *bool
	controller, ok := (getNestedField(v, "controller")).(bool)
	if !ok {
		controllerPtr = nil
	} else {
		controllerCopy := controller
		controllerPtr = &controllerCopy
	}
	var blockOwnerDeletionPtr *bool
	blockOwnerDeletion, ok := (getNestedField(v, "blockOwnerDeletion")).(bool)
	if !ok {
		blockOwnerDeletionPtr = nil
	} else {
		blockOwnerDeletionCopy := blockOwnerDeletion
		blockOwnerDeletionPtr = &blockOwnerDeletionCopy
	}
	return metav1.OwnerReference{
		Kind:               getNestedString(v, "kind"),
		Name:               getNestedString(v, "name"),
		APIVersion:         getNestedString(v, "apiVersion"),
		UID:                (types.UID)(getNestedString(v, "uid")),
		Controller:         controllerPtr,
		BlockOwnerDeletion: blockOwnerDeletionPtr,
	}
}

func setOwnerReference(src metav1.OwnerReference) map[string]interface{} {
	ret := make(map[string]interface{})
	controllerPtr := src.Controller
	if controllerPtr != nil {
		controller := *controllerPtr
		controllerPtr = &controller
	}
	blockOwnerDeletionPtr := src.BlockOwnerDeletion
	if blockOwnerDeletionPtr != nil {
		blockOwnerDeletion := *blockOwnerDeletionPtr
		blockOwnerDeletionPtr = &blockOwnerDeletion
	}
	setNestedField(ret, src.Kind, "kind")
	setNestedField(ret, src.Name, "name")
	setNestedField(ret, src.APIVersion, "apiVersion")
	setNestedField(ret, string(src.UID), "uid")
	setNestedField(ret, controllerPtr, "controller")
	setNestedField(ret, blockOwnerDeletionPtr, "blockOwnerDeletion")
	return ret
}

func getOwnerReferences(object map[string]interface{}) ([]map[string]interface{}, error) {
	field := getNestedField(object, "metadata", "ownerReferences")
	if field == nil {
		return nil, fmt.Errorf("cannot find field metadata.ownerReferences in %v", object)
	}
	ownerReferences, ok := field.([]map[string]interface{})
	if ok {
		return ownerReferences, nil
	}
	// TODO: This is hacky...
	interfaces, ok := field.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expect metadata.ownerReferences to be a slice in %#v", object)
	}
	ownerReferences = make([]map[string]interface{}, 0, len(interfaces))
	for i := 0; i < len(interfaces); i++ {
		r, ok := interfaces[i].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("expect element metadata.ownerReferences to be a map[string]interface{} in %#v", object)
		}
		ownerReferences = append(ownerReferences, r)
	}
	return ownerReferences, nil
}

func (u *Accessor) GetOwnerReferences() []metav1.OwnerReference {
	original, err := getOwnerReferences(u.Object)
	if err != nil {
		glog.V(6).Info(err)
		return nil
	}
	ret := make([]metav1.OwnerReference, 0, len(original))
	for i := 0; i < len(original); i++ {
		ret = append(ret, extractOwnerReference(original[i]))
	}
	return ret
}

func (u *Accessor) SetOwnerReferences(references []metav1.OwnerReference) {
	var newReferences = make([]map[string]interface{}, 0, len(references))
	for i := 0; i < len(references); i++ {
		newReferences = append(newReferences, setOwnerReference(references[i]))
	}
	u.setNestedField(newReferences, "metadata", "ownerReferences")
}

func (u *Accessor) GetNamespace() string {
	return getNestedString(u.Object, "metadata", "namespace")
}

func (u *Accessor) SetNamespace(namespace string) {
	u.setNestedField(namespace, "metadata", "namespace")
}

func (u *Accessor) GetName() string {
	return getNestedString(u.Object, "metadata", "name")
}

func (u *Accessor) SetName(name string) {
	u.setNestedField(name, "metadata", "name")
}

func (u *Accessor) GetGenerateName() string {
	return getNestedString(u.Object, "metadata", "generateName")
}

func (u *Accessor) SetGenerateName(name string) {
	u.setNestedField(name, "metadata", "generateName")
}

func (u *Accessor) GetUID() types.UID {
	return types.UID(getNestedString(u.Object, "metadata", "uid"))
}

func (u *Accessor) SetUID(uid types.UID) {
	u.setNestedField(string(uid), "metadata", "uid")
}

func (u *Accessor) GetResourceVersion() string {
	return getNestedString(u.Object, "metadata", "resourceVersion")
}

func (u *Accessor) SetResourceVersion(version string) {
	u.setNestedField(version, "metadata", "resourceVersion")
}

func (u *Accessor) GetGeneration() int64 {
	return getNestedInt64(u.Object, "metadata", "generation")
}

func (u *Accessor) SetGeneration(generation int64) {
	u.setNestedField(generation, "metadata", "generation")
}

func (u *Accessor) GetSelfLink() string {
	return getNestedString(u.Object, "metadata", "selfLink")
}

func (u *Accessor) SetSelfLink(selfLink string) {
	u.setNestedField(selfLink, "metadata", "selfLink")
}

func (u *Accessor) GetCreationTimestamp() metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "creationTimestamp"))
	return timestamp
}

func (u *Accessor) SetCreationTimestamp(timestamp metav1.Time) {
	ts, _ := timestamp.MarshalQueryParameter()
	u.setNestedField(ts, "metadata", "creationTimestamp")
}

func (u *Accessor) GetDeletionTimestamp() *metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "deletionTimestamp"))
	if timestamp.IsZero() {
		return nil
	}
	return &timestamp
}

func (u *Accessor) SetDeletionTimestamp(timestamp *metav1.Time) {
	if timestamp == nil {
		u.setNestedField(nil, "metadata", "deletionTimestamp")
		return
	}
	ts, _ := timestamp.MarshalQueryParameter()
	u.setNestedField(ts, "metadata", "deletionTimestamp")
}

func (u *Accessor) GetDeletionGracePeriodSeconds() *int64 {
	return getNestedInt64Pointer(u.Object, "metadata", "deletionGracePeriodSeconds")
}

func (u *Accessor) SetDeletionGracePeriodSeconds(deletionGracePeriodSeconds *int64) {
	u.setNestedField(deletionGracePeriodSeconds, "metadata", "deletionGracePeriodSeconds")
}

func (u *Accessor) GetLabels() map[string]string {
	return getNestedMap(u.Object, "metadata", "labels")
}

func (u *Accessor) SetLabels(labels map[string]string) {
	u.setNestedMap(labels, "metadata", "labels")
}

func (u *Accessor) GetAnnotations() map[string]string {
	return getNestedMap(u.Object, "metadata", "annotations")
}

func (u *Accessor) SetAnnotations(annotations map[string]string) {
	u.setNestedMap(annotations, "metadata", "annotations")
}

func (u *Accessor) GetFinalizers() []string {
	return getNestedSlice(u.Object, "metadata", "finalizers")
}

func (u *Accessor) SetFinalizers(finalizers []string) {
	u.setNestedSlice(finalizers, "metadata", "finalizers")
}

func (u *Accessor) GetClusterName() string {
	return getNestedString(u.Object, "metadata", "clusterName")
}

func (u *Accessor) SetClusterName(clusterName string) {
	u.setNestedField(clusterName, "metadata", "clusterName")
}

func (u *ListAccessor) setNestedField(value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedField(u.Object, value, fields...)
}

func (u *ListAccessor) GetResourceVersion() string {
	return getNestedString(u.Object, "metadata", "resourceVersion")
}

func (u *ListAccessor) SetResourceVersion(version string) {
	u.setNestedField(version, "metadata", "resourceVersion")
}

func (u *ListAccessor) GetSelfLink() string {
	return getNestedString(u.Object, "metadata", "selfLink")
}

func (u *ListAccessor) SetSelfLink(selfLink string) {
	u.setNestedField(selfLink, "metadata", "selfLink")
}
