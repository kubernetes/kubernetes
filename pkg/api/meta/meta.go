/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package meta

import (
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

// Accessor takes an arbitrary object pointer and returns meta.Interface.
// obj must be a pointer to an API type. An error is returned if the minimum
// required fields are missing. Fields that are not required return the default
// value and are a no-op if set.
func Accessor(obj interface{}) (Object, error) {
	if oi, ok := obj.(ObjectMetaAccessor); ok {
		if om := oi.GetObjectMeta(); om != nil {
			return om, nil
		}
	}
	// we may get passed an object that is directly portable to Object
	if oi, ok := obj.(Object); ok {
		return oi, nil
	}
	// legacy path for objects that do not implement Object and ObjectMetaAccessor via
	// reflection - very slow code path.
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), t, v.Interface())
	}

	typeMeta := v.FieldByName("TypeMeta")
	if !typeMeta.IsValid() {
		return nil, fmt.Errorf("struct %v lacks embedded TypeMeta type", t)
	}

	a := &genericAccessor{}
	if err := extractFromTypeMeta(typeMeta, a); err != nil {
		return nil, fmt.Errorf("unable to find type fields on %#v: %v", typeMeta, err)
	}

	objectMeta := v.FieldByName("ObjectMeta")
	if objectMeta.IsValid() {
		// look for the ObjectMeta fields
		if err := extractFromObjectMeta(objectMeta, a); err != nil {
			return nil, fmt.Errorf("unable to find object fields on %#v: %v", objectMeta, err)
		}
	} else {
		listMeta := v.FieldByName("ListMeta")
		if listMeta.IsValid() {
			// look for the ListMeta fields
			if err := extractFromListMeta(listMeta, a); err != nil {
				return nil, fmt.Errorf("unable to find list fields on %#v: %v", listMeta, err)
			}
		} else {
			// look for the older TypeMeta with all metadata
			if err := extractFromObjectMeta(typeMeta, a); err != nil {
				return nil, fmt.Errorf("unable to find object fields on %#v: %v", typeMeta, err)
			}
		}
	}

	return a, nil
}

// TypeAccessor returns an interface that allows retrieving and modifying the APIVersion
// and Kind of an in-memory internal object.
// TODO: this interface is used to test code that does not have ObjectMeta or ListMeta
// in round tripping (objects which can use apiVersion/kind, but do not fit the Kube
// api conventions).
func TypeAccessor(obj interface{}) (Type, error) {
	if typed, ok := obj.(runtime.Object); ok {
		return objectAccessor{typed}, nil
	}
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	t := v.Type()
	if v.Kind() != reflect.Struct {
		return nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), t, v.Interface())
	}

	typeMeta := v.FieldByName("TypeMeta")
	if !typeMeta.IsValid() {
		return nil, fmt.Errorf("struct %v lacks embedded TypeMeta type", t)
	}
	a := &genericAccessor{}
	if err := extractFromTypeMeta(typeMeta, a); err != nil {
		return nil, fmt.Errorf("unable to find type fields on %#v: %v", typeMeta, err)
	}
	return a, nil
}

type objectAccessor struct {
	runtime.Object
}

func (obj objectAccessor) GetKind() string {
	if gvk := obj.GetObjectKind().GroupVersionKind(); gvk != nil {
		return gvk.Kind
	}
	return ""
}

func (obj objectAccessor) SetKind(kind string) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	if gvk == nil {
		gvk = &unversioned.GroupVersionKind{}
	}
	gvk.Kind = kind
	obj.GetObjectKind().SetGroupVersionKind(gvk)
}

func (obj objectAccessor) GetAPIVersion() string {
	if gvk := obj.GetObjectKind().GroupVersionKind(); gvk != nil {
		return gvk.GroupVersion().String()
	}
	return ""
}

func (obj objectAccessor) SetAPIVersion(version string) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	if gvk == nil {
		gvk = &unversioned.GroupVersionKind{}
	}
	gv, err := unversioned.ParseGroupVersion(version)
	if err != nil {
		gv = unversioned.GroupVersion{Version: version}
	}
	gvk.Group, gvk.Version = gv.Group, gv.Version
	obj.GetObjectKind().SetGroupVersionKind(gvk)
}

// NewAccessor returns a MetadataAccessor that can retrieve
// or manipulate resource version on objects derived from core API
// metadata concepts.
func NewAccessor() MetadataAccessor {
	return resourceAccessor{}
}

// resourceAccessor implements ResourceVersioner and SelfLinker.
type resourceAccessor struct{}

func (resourceAccessor) Kind(obj runtime.Object) (string, error) {
	return objectAccessor{obj}.GetKind(), nil
}

func (resourceAccessor) SetKind(obj runtime.Object, kind string) error {
	objectAccessor{obj}.SetKind(kind)
	return nil
}

func (resourceAccessor) APIVersion(obj runtime.Object) (string, error) {
	return objectAccessor{obj}.GetAPIVersion(), nil
}

func (resourceAccessor) SetAPIVersion(obj runtime.Object, version string) error {
	objectAccessor{obj}.SetAPIVersion(version)
	return nil
}

func (resourceAccessor) Namespace(obj runtime.Object) (string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetNamespace(), nil
}

func (resourceAccessor) SetNamespace(obj runtime.Object, namespace string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetNamespace(namespace)
	return nil
}

func (resourceAccessor) Name(obj runtime.Object) (string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetName(), nil
}

func (resourceAccessor) SetName(obj runtime.Object, name string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetName(name)
	return nil
}

func (resourceAccessor) GenerateName(obj runtime.Object) (string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetGenerateName(), nil
}

func (resourceAccessor) SetGenerateName(obj runtime.Object, name string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetGenerateName(name)
	return nil
}

func (resourceAccessor) UID(obj runtime.Object) (types.UID, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetUID(), nil
}

func (resourceAccessor) SetUID(obj runtime.Object, uid types.UID) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetUID(uid)
	return nil
}

func (resourceAccessor) SelfLink(obj runtime.Object) (string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetSelfLink(), nil
}

func (resourceAccessor) SetSelfLink(obj runtime.Object, selfLink string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetSelfLink(selfLink)
	return nil
}

func (resourceAccessor) Labels(obj runtime.Object) (map[string]string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return nil, err
	}
	return accessor.GetLabels(), nil
}

func (resourceAccessor) SetLabels(obj runtime.Object, labels map[string]string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetLabels(labels)
	return nil
}

func (resourceAccessor) Annotations(obj runtime.Object) (map[string]string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return nil, err
	}
	return accessor.GetAnnotations(), nil
}

func (resourceAccessor) SetAnnotations(obj runtime.Object, annotations map[string]string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetAnnotations(annotations)
	return nil
}

func (resourceAccessor) ResourceVersion(obj runtime.Object) (string, error) {
	accessor, err := Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetResourceVersion(), nil
}

func (resourceAccessor) SetResourceVersion(obj runtime.Object, version string) error {
	accessor, err := Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetResourceVersion(version)
	return nil
}

// genericAccessor contains pointers to strings that can modify an arbitrary
// struct and implements the Accessor interface.
type genericAccessor struct {
	namespace       *string
	name            *string
	generateName    *string
	uid             *types.UID
	apiVersion      *string
	kind            *string
	resourceVersion *string
	selfLink        *string
	labels          *map[string]string
	annotations     *map[string]string
}

func (a genericAccessor) GetNamespace() string {
	if a.namespace == nil {
		return ""
	}
	return *a.namespace
}

func (a genericAccessor) SetNamespace(namespace string) {
	if a.namespace == nil {
		return
	}
	*a.namespace = namespace
}

func (a genericAccessor) GetName() string {
	if a.name == nil {
		return ""
	}
	return *a.name
}

func (a genericAccessor) SetName(name string) {
	if a.name == nil {
		return
	}
	*a.name = name
}

func (a genericAccessor) GetGenerateName() string {
	if a.generateName == nil {
		return ""
	}
	return *a.generateName
}

func (a genericAccessor) SetGenerateName(generateName string) {
	if a.generateName == nil {
		return
	}
	*a.generateName = generateName
}

func (a genericAccessor) GetUID() types.UID {
	if a.uid == nil {
		return ""
	}
	return *a.uid
}

func (a genericAccessor) SetUID(uid types.UID) {
	if a.uid == nil {
		return
	}
	*a.uid = uid
}

func (a genericAccessor) GetAPIVersion() string {
	return *a.apiVersion
}

func (a genericAccessor) SetAPIVersion(version string) {
	*a.apiVersion = version
}

func (a genericAccessor) GetKind() string {
	return *a.kind
}

func (a genericAccessor) SetKind(kind string) {
	*a.kind = kind
}

func (a genericAccessor) GetResourceVersion() string {
	return *a.resourceVersion
}

func (a genericAccessor) SetResourceVersion(version string) {
	*a.resourceVersion = version
}

func (a genericAccessor) GetSelfLink() string {
	return *a.selfLink
}

func (a genericAccessor) SetSelfLink(selfLink string) {
	*a.selfLink = selfLink
}

func (a genericAccessor) GetLabels() map[string]string {
	if a.labels == nil {
		return nil
	}
	return *a.labels
}

func (a genericAccessor) SetLabels(labels map[string]string) {
	*a.labels = labels
}

func (a genericAccessor) GetAnnotations() map[string]string {
	if a.annotations == nil {
		return nil
	}
	return *a.annotations
}

func (a genericAccessor) SetAnnotations(annotations map[string]string) {
	if a.annotations == nil {
		emptyAnnotations := make(map[string]string)
		a.annotations = &emptyAnnotations
	}
	*a.annotations = annotations
}

// extractFromTypeMeta extracts pointers to version and kind fields from an object
func extractFromTypeMeta(v reflect.Value, a *genericAccessor) error {
	if err := runtime.FieldPtr(v, "APIVersion", &a.apiVersion); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "Kind", &a.kind); err != nil {
		return err
	}
	return nil
}

// extractFromObjectMeta extracts pointers to metadata fields from an object
func extractFromObjectMeta(v reflect.Value, a *genericAccessor) error {
	if err := runtime.FieldPtr(v, "Namespace", &a.namespace); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "Name", &a.name); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "GenerateName", &a.generateName); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "UID", &a.uid); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "ResourceVersion", &a.resourceVersion); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "SelfLink", &a.selfLink); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "Labels", &a.labels); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "Annotations", &a.annotations); err != nil {
		return err
	}
	return nil
}

// extractFromObjectMeta extracts pointers to metadata fields from a list object
func extractFromListMeta(v reflect.Value, a *genericAccessor) error {
	if err := runtime.FieldPtr(v, "ResourceVersion", &a.resourceVersion); err != nil {
		return err
	}
	if err := runtime.FieldPtr(v, "SelfLink", &a.selfLink); err != nil {
		return err
	}
	return nil
}
