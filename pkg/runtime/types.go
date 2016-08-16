/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"bytes"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/types"
)

// Note that the types provided in this file are not versioned and are intended to be
// safe to use from within all versions of every API object.

// TypeMeta is shared by all top level objects. The proper way to use it is to inline it in your type,
// like this:
// type MyAwesomeAPIObject struct {
//      runtime.TypeMeta    `json:",inline"`
//      ... // other fields
// }
// func (obj *MyAwesomeAPIObject) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) { unversioned.UpdateTypeMeta(obj,gvk) }; GroupVersionKind() *GroupVersionKind
//
// TypeMeta is provided here for convenience. You may use it directly from this package or define
// your own with the same fields.
//
// +k8s:deepcopy-gen=true
// +protobuf=true
type TypeMeta struct {
	APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty" protobuf:"bytes,1,opt,name=apiVersion"`
	Kind       string `json:"kind,omitempty" yaml:"kind,omitempty" protobuf:"bytes,2,opt,name=kind"`
}

const (
	ContentTypeJSON string = "application/json"
)

// RawExtension is used to hold extensions in external versions.
//
// To use this, make a field which has RawExtension as its type in your external, versioned
// struct, and Object in your internal struct. You also need to register your
// various plugin types.
//
// // Internal package:
// type MyAPIObject struct {
// 	runtime.TypeMeta `json:",inline"`
//	MyPlugin runtime.Object `json:"myPlugin"`
// }
// type PluginA struct {
//	AOption string `json:"aOption"`
// }
//
// // External package:
// type MyAPIObject struct {
// 	runtime.TypeMeta `json:",inline"`
//	MyPlugin runtime.RawExtension `json:"myPlugin"`
// }
// type PluginA struct {
//	AOption string `json:"aOption"`
// }
//
// // On the wire, the JSON will look something like this:
// {
//	"kind":"MyAPIObject",
//	"apiVersion":"v1",
//	"myPlugin": {
//		"kind":"PluginA",
//		"aOption":"foo",
//	},
// }
//
// So what happens? Decode first uses json or yaml to unmarshal the serialized data into
// your external MyAPIObject. That causes the raw JSON to be stored, but not unpacked.
// The next step is to copy (using pkg/conversion) into the internal struct. The runtime
// package's DefaultScheme has conversion functions installed which will unpack the
// JSON stored in RawExtension, turning it into the correct object type, and storing it
// in the Object. (TODO: In the case where the object is of an unknown type, a
// runtime.Unknown object will be created and stored.)
//
// +k8s:deepcopy-gen=true
// +protobuf=true
type RawExtension struct {
	// Raw is the underlying serialization of this object.
	//
	// TODO: Determine how to detect ContentType and ContentEncoding of 'Raw' data.
	Raw []byte `protobuf:"bytes,1,opt,name=raw"`
	// Object can hold a representation of this extension - useful for working with versioned
	// structs.
	Object Object `json:"-"`
}

// Unknown allows api objects with unknown types to be passed-through. This can be used
// to deal with the API objects from a plug-in. Unknown objects still have functioning
// TypeMeta features-- kind, version, etc.
// TODO: Make this object have easy access to field based accessors and settors for
// metadata and field mutatation.
//
// +k8s:deepcopy-gen=true
// +protobuf=true
type Unknown struct {
	TypeMeta `json:",inline" protobuf:"bytes,1,opt,name=typeMeta"`
	// Raw will hold the complete serialized object which couldn't be matched
	// with a registered type. Most likely, nothing should be done with this
	// except for passing it through the system.
	Raw []byte `protobuf:"bytes,2,opt,name=raw"`
	// ContentEncoding is encoding used to encode 'Raw' data.
	// Unspecified means no encoding.
	ContentEncoding string `protobuf:"bytes,3,opt,name=contentEncoding"`
	// ContentType  is serialization method used to serialize 'Raw'.
	// Unspecified means ContentTypeJSON.
	ContentType string `protobuf:"bytes,4,opt,name=contentType"`
}

// Unstructured allows objects that do not have Golang structs registered to be manipulated
// generically. This can be used to deal with the API objects from a plug-in. Unstructured
// objects still have functioning TypeMeta features-- kind, version, etc.
// TODO: Make this object have easy access to field based accessors and settors for
// metadata and field mutatation.
type Unstructured struct {
	// Object is a JSON compatible map with string, float, int, []interface{}, or map[string]interface{}
	// children.
	Object map[string]interface{}
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

func extractOwnerReference(src interface{}) metatypes.OwnerReference {
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
	return metatypes.OwnerReference{
		Kind:       getNestedString(v, "kind"),
		Name:       getNestedString(v, "name"),
		APIVersion: getNestedString(v, "apiVersion"),
		UID:        (types.UID)(getNestedString(v, "uid")),
		Controller: controllerPtr,
	}
}

func setOwnerReference(src metatypes.OwnerReference) map[string]interface{} {
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

func (u *Unstructured) GetOwnerReferences() []metatypes.OwnerReference {
	original, err := getOwnerReferences(u.Object)
	if err != nil {
		glog.V(6).Info(err)
		return nil
	}
	ret := make([]metatypes.OwnerReference, 0, len(original))
	for i := 0; i < len(original); i++ {
		ret = append(ret, extractOwnerReference(original[i]))
	}
	return ret
}

func (u *Unstructured) SetOwnerReferences(references []metatypes.OwnerReference) {
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

func (u *Unstructured) GetCreationTimestamp() unversioned.Time {
	var timestamp unversioned.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "creationTimestamp"))
	return timestamp
}

func (u *Unstructured) SetCreationTimestamp(timestamp unversioned.Time) {
	ts, _ := timestamp.MarshalQueryParameter()
	u.setNestedField(ts, "metadata", "creationTimestamp")
}

func (u *Unstructured) GetDeletionTimestamp() *unversioned.Time {
	var timestamp unversioned.Time
	timestamp.UnmarshalQueryParameter(getNestedString(u.Object, "metadata", "deletionTimestamp"))
	if timestamp.IsZero() {
		return nil
	}
	return &timestamp
}

func (u *Unstructured) SetDeletionTimestamp(timestamp *unversioned.Time) {
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

func (u *Unstructured) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
}

func (u *Unstructured) GroupVersionKind() unversioned.GroupVersionKind {
	gv, err := unversioned.ParseGroupVersion(u.GetAPIVersion())
	if err != nil {
		return unversioned.GroupVersionKind{}
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

func (u *UnstructuredList) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
}

func (u *UnstructuredList) GroupVersionKind() unversioned.GroupVersionKind {
	gv, err := unversioned.ParseGroupVersion(u.GetAPIVersion())
	if err != nil {
		return unversioned.GroupVersionKind{}
	}
	gvk := gv.WithKind(u.GetKind())
	return gvk
}

// VersionedObjects is used by Decoders to give callers a way to access all versions
// of an object during the decoding process.
type VersionedObjects struct {
	// Objects is the set of objects retrieved during decoding, in order of conversion.
	// The 0 index is the object as serialized on the wire. If conversion has occurred,
	// other objects may be present. The right most object is the same as would be returned
	// by a normal Decode call.
	Objects []Object
}
