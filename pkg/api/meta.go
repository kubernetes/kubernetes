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

package api

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// FillObjectMetaSystemFields populates fields that are managed by the system on ObjectMeta.
func FillObjectMetaSystemFields(ctx Context, meta *ObjectMeta) {
	meta.CreationTimestamp = unversioned.Now()
	meta.UID = util.NewUUID()
	meta.SelfLink = ""
}

// HasObjectMetaSystemFieldValues returns true if fields that are managed by the system on ObjectMeta have values.
func HasObjectMetaSystemFieldValues(meta *ObjectMeta) bool {
	return !meta.CreationTimestamp.Time.IsZero() ||
		len(meta.UID) != 0
}

// ObjectMetaFor returns a pointer to a provided object's ObjectMeta.
// TODO: allow runtime.Unknown to extract this object
func ObjectMetaFor(obj runtime.Object) (*ObjectMeta, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	var meta *ObjectMeta
	err = runtime.FieldPtr(v, "ObjectMeta", &meta)
	return meta, err
}

// ListMetaFor returns a pointer to a provided object's ListMeta,
// or an error if the object does not have that pointer.
// TODO: allow runtime.Unknown to extract this object
func ListMetaFor(obj runtime.Object) (*unversioned.ListMeta, error) {
	v, err := conversion.EnforcePtr(obj)
	if err != nil {
		return nil, err
	}
	var meta *unversioned.ListMeta
	err = runtime.FieldPtr(v, "ListMeta", &meta)
	return meta, err
}

// Namespace implements meta.Object for any object with an ObjectMeta typed field. Allows
// fast, direct access to metadata fields for API objects.
func (meta *ObjectMeta) GetNamespace() string                         { return meta.Namespace }
func (meta *ObjectMeta) SetNamespace(namespace string)                { meta.Namespace = namespace }
func (meta *ObjectMeta) GetName() string                              { return meta.Name }
func (meta *ObjectMeta) SetName(name string)                          { meta.Name = name }
func (meta *ObjectMeta) GetGenerateName() string                      { return meta.GenerateName }
func (meta *ObjectMeta) SetGenerateName(generateName string)          { meta.GenerateName = generateName }
func (meta *ObjectMeta) GetUID() types.UID                            { return meta.UID }
func (meta *ObjectMeta) SetUID(uid types.UID)                         { meta.UID = uid }
func (meta *ObjectMeta) GetResourceVersion() string                   { return meta.ResourceVersion }
func (meta *ObjectMeta) SetResourceVersion(version string)            { meta.ResourceVersion = version }
func (meta *ObjectMeta) GetSelfLink() string                          { return meta.SelfLink }
func (meta *ObjectMeta) SetSelfLink(selfLink string)                  { meta.SelfLink = selfLink }
func (meta *ObjectMeta) GetLabels() map[string]string                 { return meta.Labels }
func (meta *ObjectMeta) SetLabels(labels map[string]string)           { meta.Labels = labels }
func (meta *ObjectMeta) GetAnnotations() map[string]string            { return meta.Annotations }
func (meta *ObjectMeta) SetAnnotations(annotations map[string]string) { meta.Annotations = annotations }
