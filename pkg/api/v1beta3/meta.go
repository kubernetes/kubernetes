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

package v1beta3

import (
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/types"
)

func (obj *ObjectMeta) GetObjectMeta() meta.Object { return obj }

// Namespace implements meta.Object for any object with an ObjectMeta typed field. Allows
// fast, direct access to metadata fields for API objects.
func (meta *ObjectMeta) GetNamespace() string                   { return meta.Namespace }
func (meta *ObjectMeta) SetNamespace(namespace string)          { meta.Namespace = namespace }
func (meta *ObjectMeta) GetName() string                        { return meta.Name }
func (meta *ObjectMeta) SetName(name string)                    { meta.Name = name }
func (meta *ObjectMeta) GetGenerateName() string                { return meta.GenerateName }
func (meta *ObjectMeta) SetGenerateName(generateName string)    { meta.GenerateName = generateName }
func (meta *ObjectMeta) GetUID() types.UID                      { return meta.UID }
func (meta *ObjectMeta) SetUID(uid types.UID)                   { meta.UID = uid }
func (meta *ObjectMeta) GetResourceVersion() string             { return meta.ResourceVersion }
func (meta *ObjectMeta) SetResourceVersion(version string)      { meta.ResourceVersion = version }
func (meta *ObjectMeta) GetSelfLink() string                    { return meta.SelfLink }
func (meta *ObjectMeta) SetSelfLink(selfLink string)            { meta.SelfLink = selfLink }
func (meta *ObjectMeta) GetCreationTimestamp() unversioned.Time { return meta.CreationTimestamp }
func (meta *ObjectMeta) SetCreationTimestamp(creationTimestamp unversioned.Time) {
	meta.CreationTimestamp = creationTimestamp
}
func (meta *ObjectMeta) GetDeletionTimestamp() *unversioned.Time { return meta.DeletionTimestamp }
func (meta *ObjectMeta) SetDeletionTimestamp(deletionTimestamp *unversioned.Time) {
	meta.DeletionTimestamp = deletionTimestamp
}
func (meta *ObjectMeta) GetLabels() map[string]string                 { return meta.Labels }
func (meta *ObjectMeta) SetLabels(labels map[string]string)           { meta.Labels = labels }
func (meta *ObjectMeta) GetAnnotations() map[string]string            { return meta.Annotations }
func (meta *ObjectMeta) SetAnnotations(annotations map[string]string) { meta.Annotations = annotations }
func (meta *ObjectMeta) GetFinalizers() []string                      { return nil }
func (meta *ObjectMeta) SetFinalizers(finalizers []string)            {}

func (meta *ObjectMeta) GetOwnerReferences() []metatypes.OwnerReference { return nil }

func (meta *ObjectMeta) SetOwnerReferences(references []metatypes.OwnerReference) {}
