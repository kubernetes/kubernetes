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
	gojson "encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"

	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/runtime/schema"
	"k8s.io/client-go/pkg/types"
	"k8s.io/client-go/pkg/util/json"
)

// Unstructured allows objects that do not have Golang structs registered to be manipulated
// generically. This can be used to deal with the API objects from a plug-in. Unstructured
// objects still have functioning TypeMeta features-- kind, version, etc.
//
// WARNING: This object has accessors for the v1 standard metadata. You *MUST NOT* use this
// type if you are dealing with objects that are not in the server meta v1 schema.
//
// TODO: make the serialization part of this type distinct from the field accessors.
type Unstructured struct {
	// Object is a JSON compatible map with string, float, int, bool, []interface{}, or
	// map[string]interface{}
	// children.
	Object map[string]interface{}
}

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
func (obj *UnstructuredList) UnstructuredContent() map[string]interface{} {
	if obj.Object == nil {
		obj.Object = make(map[string]interface{})
	}
	return obj.Object
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

func (u *Unstructured) setNestedField(value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedField(u.Object, value, fields...)
}

func (u *Unstructured) setNestedSlice(value []string, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedSlice(u.Object, value, fields...)
}

func (u *Unstructured) setNestedMap(value map[string]string, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedMap(u.Object, value, fields...)
}

func extractOwnerReference(src interface{}) metav1.OwnerReference {
	v := src.(map[string]interface{})
	controllerPtr, ok := (getNestedField(v, "controller")).(*bool)
	if !ok {
		controllerPtr = nil
	} else {
		if controllerPtr != nil {
			controller := *controllerPtr
			controllerPtr = &controller
		}
	}
	return metav1.OwnerReference{
		Kind:       getNestedString(v, "kind"),
		Name:       getNestedString(v, "name"),
		APIVersion: getNestedString(v, "apiVersion"),
		UID:        (types.UID)(getNestedString(v, "uid")),
		Controller: controllerPtr,
	}
}

func setOwnerReference(src metav1.OwnerReference) map[string]interface{} {
	ret := make(map[string]interface{})
	controllerPtr := src.Controller
	if controllerPtr != nil {
		controller := *controllerPtr
		controllerPtr = &controller
	}
	setNestedField(ret, src.Kind, "kind")
	setNestedField(ret, src.Name, "name")
	setNestedField(ret, src.APIVersion, "apiVersion")
	setNestedField(ret, string(src.UID), "uid")
	setNestedField(ret, controllerPtr, "controller")
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

func (u *Unstructured) GetOwnerReferences() []metav1.OwnerReference {
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

func (u *Unstructured) SetOwnerReferences(references []metav1.OwnerReference) {
	var newReferences = make([]map[string]interface{}, 0, len(references))
	for i := 0; i < len(references); i++ {
		newReferences = append(newReferences, setOwnerReference(references[i]))
	}
	u.setNestedField(newReferences, "metadata", "ownerReferences")
}

func (u *Unstructured) GetAPIVersion() string {
	return getNestedString(u.Object, "apiVersion")
}

func (u *Unstructured) SetAPIVersion(version string) {
	u.setNestedField(version, "apiVersion")
}

func (u *Unstructured) GetKind() string {
	return getNestedString(u.Object, "kind")
}

func (u *Unstructured) SetKind(kind string) {
	u.setNestedField(kind, "kind")
}

func (u *Unstructured) GetNamespace() string {
	return getNestedString(u.Object, "metadata", "namespace")
}

func (u *Unstructured) SetNamespace(namespace string) {
	u.setNestedField(namespace, "metadata", "namespace")
}

func (u *Unstructured) GetName() string {
	return getNestedString(u.Object, "metadata", "name")
}

func (u *Unstructured) SetName(name string) {
	u.setNestedField(name, "metadata", "name")
}

func (u *Unstructured) GetGenerateName() string {
	return getNestedString(u.Object, "metadata", "generateName")
}

func (u *Unstructured) SetGenerateName(name string) {
	u.setNestedField(name, "metadata", "generateName")
}

func (u *Unstructured) GetUID() types.UID {
	return types.UID(getNestedString(u.Object, "metadata", "uid"))
}

func (u *Unstructured) SetUID(uid types.UID) {
	u.setNestedField(string(uid), "metadata", "uid")
}

func (u *Unstructured) GetResourceVersion() string {
	return getNestedString(u.Object, "metadata", "resourceVersion")
}

func (u *Unstructured) SetResourceVersion(version string) {
	u.setNestedField(version, "metadata", "resourceVersion")
}

func (u *Unstructured) GetSelfLink() string {
	return getNestedString(u.Object, "metadata", "selfLink")
}

func (u *Unstructured) SetSelfLink(selfLink string) {
	u.setNestedField(selfLink, "metadata", "selfLink")
}

func (u *Unstructured) GetCreationTimestamp() metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "creationTimestamp"))
	return timestamp
}

func (u *Unstructured) SetCreationTimestamp(timestamp metav1.Time) {
	ts, _ := timestamp.MarshalQueryParameter()
	u.setNestedField(ts, "metadata", "creationTimestamp")
}

func (u *Unstructured) GetDeletionTimestamp() *metav1.Time {
	var timestamp metav1.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "deletionTimestamp"))
	if timestamp.IsZero() {
		return nil
	}
	return &timestamp
}

func (u *Unstructured) SetDeletionTimestamp(timestamp *metav1.Time) {
	ts, _ := timestamp.MarshalQueryParameter()
	u.setNestedField(ts, "metadata", "deletionTimestamp")
}

func (u *Unstructured) GetLabels() map[string]string {
	return getNestedMap(u.Object, "metadata", "labels")
}

func (u *Unstructured) SetLabels(labels map[string]string) {
	u.setNestedMap(labels, "metadata", "labels")
}

func (u *Unstructured) GetAnnotations() map[string]string {
	return getNestedMap(u.Object, "metadata", "annotations")
}

func (u *Unstructured) SetAnnotations(annotations map[string]string) {
	u.setNestedMap(annotations, "metadata", "annotations")
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

func (u *Unstructured) GetFinalizers() []string {
	return getNestedSlice(u.Object, "metadata", "finalizers")
}

func (u *Unstructured) SetFinalizers(finalizers []string) {
	u.setNestedSlice(finalizers, "metadata", "finalizers")
}

func (u *Unstructured) GetClusterName() string {
	return getNestedString(u.Object, "metadata", "clusterName")
}

func (u *Unstructured) SetClusterName(clusterName string) {
	u.setNestedField(clusterName, "metadata", "clusterName")
}

// UnstructuredList allows lists that do not have Golang structs
// registered to be manipulated generically. This can be used to deal
// with the API lists from a plug-in.
type UnstructuredList struct {
	Object map[string]interface{}

	// Items is a list of unstructured objects.
	Items []*Unstructured `json:"items"`
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

func (u *UnstructuredList) setNestedField(value interface{}, fields ...string) {
	if u.Object == nil {
		u.Object = make(map[string]interface{})
	}
	setNestedField(u.Object, value, fields...)
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

// UnstructuredJSONScheme is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
// TODO: move into serializer/json.
var UnstructuredJSONScheme runtime.Codec = unstructuredJSONScheme{}

type unstructuredJSONScheme struct{}

func (s unstructuredJSONScheme) Decode(data []byte, _ *schema.GroupVersionKind, obj runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	var err error
	if obj != nil {
		err = s.decodeInto(data, obj)
	} else {
		obj, err = s.decode(data)
	}

	if err != nil {
		return nil, nil, err
	}

	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, &gvk, runtime.NewMissingKindErr(string(data))
	}

	return obj, &gvk, nil
}

func (unstructuredJSONScheme) Encode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case *Unstructured:
		return json.NewEncoder(w).Encode(t.Object)
	case *UnstructuredList:
		items := make([]map[string]interface{}, 0, len(t.Items))
		for _, i := range t.Items {
			items = append(items, i.Object)
		}
		t.Object["items"] = items
		defer func() { delete(t.Object, "items") }()
		return json.NewEncoder(w).Encode(t.Object)
	case *runtime.Unknown:
		// TODO: Unstructured needs to deal with ContentType.
		_, err := w.Write(t.Raw)
		return err
	default:
		return json.NewEncoder(w).Encode(t)
	}
}

func (s unstructuredJSONScheme) decode(data []byte) (runtime.Object, error) {
	type detector struct {
		Items gojson.RawMessage
	}
	var det detector
	if err := json.Unmarshal(data, &det); err != nil {
		return nil, err
	}

	if det.Items != nil {
		list := &UnstructuredList{}
		err := s.decodeToList(data, list)
		return list, err
	}

	// No Items field, so it wasn't a list.
	unstruct := &Unstructured{}
	err := s.decodeToUnstructured(data, unstruct)
	return unstruct, err
}

func (s unstructuredJSONScheme) decodeInto(data []byte, obj runtime.Object) error {
	switch x := obj.(type) {
	case *Unstructured:
		return s.decodeToUnstructured(data, x)
	case *UnstructuredList:
		return s.decodeToList(data, x)
	case *runtime.VersionedObjects:
		o, err := s.decode(data)
		if err == nil {
			x.Objects = []runtime.Object{o}
		}
		return err
	default:
		return json.Unmarshal(data, x)
	}
}

func (unstructuredJSONScheme) decodeToUnstructured(data []byte, unstruct *Unstructured) error {
	m := make(map[string]interface{})
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	unstruct.Object = m

	return nil
}

func (s unstructuredJSONScheme) decodeToList(data []byte, list *UnstructuredList) error {
	type decodeList struct {
		Items []gojson.RawMessage
	}

	var dList decodeList
	if err := json.Unmarshal(data, &dList); err != nil {
		return err
	}

	if err := json.Unmarshal(data, &list.Object); err != nil {
		return err
	}

	// For typed lists, e.g., a PodList, API server doesn't set each item's
	// APIVersion and Kind. We need to set it.
	listAPIVersion := list.GetAPIVersion()
	listKind := list.GetKind()
	itemKind := strings.TrimSuffix(listKind, "List")

	delete(list.Object, "items")
	list.Items = nil
	for _, i := range dList.Items {
		unstruct := &Unstructured{}
		if err := s.decodeToUnstructured([]byte(i), unstruct); err != nil {
			return err
		}
		// This is hacky. Set the item's Kind and APIVersion to those inferred
		// from the List.
		if len(unstruct.GetKind()) == 0 && len(unstruct.GetAPIVersion()) == 0 {
			unstruct.SetKind(itemKind)
			unstruct.SetAPIVersion(listAPIVersion)
		}
		list.Items = append(list.Items, unstruct)
	}
	return nil
}

// UnstructuredObjectConverter is an ObjectConverter for use with
// Unstructured objects. Since it has no schema or type information,
// it will only succeed for no-op conversions. This is provided as a
// sane implementation for APIs that require an object converter.
type UnstructuredObjectConverter struct{}

func (UnstructuredObjectConverter) Convert(in, out, context interface{}) error {
	unstructIn, ok := in.(*Unstructured)
	if !ok {
		return fmt.Errorf("input type %T in not valid for unstructured conversion", in)
	}

	unstructOut, ok := out.(*Unstructured)
	if !ok {
		return fmt.Errorf("output type %T in not valid for unstructured conversion", out)
	}

	// maybe deep copy the map? It is documented in the
	// ObjectConverter interface that this function is not
	// guaranteeed to not mutate the input. Or maybe set the input
	// object to nil.
	unstructOut.Object = unstructIn.Object
	return nil
}

func (UnstructuredObjectConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	if kind := in.GetObjectKind().GroupVersionKind(); !kind.Empty() {
		gvk, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{kind})
		if !ok {
			// TODO: should this be a typed error?
			return nil, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", kind, target)
		}
		in.GetObjectKind().SetGroupVersionKind(gvk)
	}
	return in, nil
}

func (UnstructuredObjectConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return "", "", errors.New("unstructured cannot convert field labels")
}
