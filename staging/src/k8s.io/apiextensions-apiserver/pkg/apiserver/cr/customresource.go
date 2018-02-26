/*
Copyright 2017 The Kubernetes Authors.

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

package cr

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

type CustomResource struct {
	Obj *unstructured.Unstructured
}

var _ v1.Object = &CustomResource{}
var _ runtime.Object = &CustomResource{}

func (c *CustomResource) DeepCopy() *CustomResource {
	return &CustomResource{Obj: c.Obj.DeepCopy()}
}
func (c *CustomResource) DeepCopyObject() runtime.Object {
	return c.DeepCopy()
}
func (c *CustomResource) GetObjectKind() schema.ObjectKind { return c.Obj }

// MarshalJSON ensures that the unstructured object produces proper
// JSON when passed to Go's standard JSON library.
func (c *CustomResource) MarshalJSON() ([]byte, error) {
	return c.Obj.MarshalJSON()
}

// UnmarshalJSON ensures that the unstructured object properly decodes
// JSON when passed to Go's standard JSON library.
func (c *CustomResource) UnmarshalJSON(b []byte) error {
	if c.Obj == nil {
		c.Obj = &unstructured.Unstructured{}
	}
	if c.Obj.Object == nil {
		c.Obj.Object = map[string]interface{}{}
	}
	return c.Obj.UnmarshalJSON(b)
}

func (c *CustomResource) GetNamespace() string              { return c.Obj.GetNamespace() }
func (c *CustomResource) SetNamespace(namespace string)     { c.Obj.SetNamespace(namespace) }
func (c *CustomResource) GetName() string                   { return c.Obj.GetName() }
func (c *CustomResource) SetName(name string)               { c.Obj.SetName(name) }
func (c *CustomResource) GetGenerateName() string           { return c.Obj.GetGenerateName() }
func (c *CustomResource) SetGenerateName(name string)       { c.Obj.SetGenerateName(name) }
func (c *CustomResource) GetUID() types.UID                 { return c.Obj.GetUID() }
func (c *CustomResource) SetUID(uid types.UID)              { c.Obj.SetUID(uid) }
func (c *CustomResource) GetResourceVersion() string        { return c.Obj.GetResourceVersion() }
func (c *CustomResource) SetResourceVersion(version string) { c.Obj.SetResourceVersion(version) }
func (c *CustomResource) GetGeneration() int64              { return c.Obj.GetGeneration() }
func (c *CustomResource) SetGeneration(generation int64)    { c.Obj.SetGeneration(generation) }
func (c *CustomResource) GetSelfLink() string               { return c.Obj.GetSelfLink() }
func (c *CustomResource) SetSelfLink(selfLink string)       { c.Obj.SetSelfLink(selfLink) }
func (c *CustomResource) GetCreationTimestamp() v1.Time     { return c.Obj.GetCreationTimestamp() }
func (c *CustomResource) SetCreationTimestamp(timestamp v1.Time) {
	c.Obj.SetCreationTimestamp(timestamp)
}
func (c *CustomResource) GetDeletionTimestamp() *v1.Time { return c.Obj.GetDeletionTimestamp() }
func (c *CustomResource) SetDeletionTimestamp(timestamp *v1.Time) {
	c.Obj.SetDeletionTimestamp(timestamp)
}
func (c *CustomResource) GetDeletionGracePeriodSeconds() *int64 {
	return c.Obj.GetDeletionGracePeriodSeconds()
}
func (c *CustomResource) SetDeletionGracePeriodSeconds(t *int64) {
	c.Obj.SetDeletionGracePeriodSeconds(t)
}
func (c *CustomResource) GetLabels() map[string]string       { return c.Obj.GetLabels() }
func (c *CustomResource) SetLabels(labels map[string]string) { c.Obj.SetLabels(labels) }
func (c *CustomResource) GetAnnotations() map[string]string  { return c.Obj.GetAnnotations() }
func (c *CustomResource) SetAnnotations(annotations map[string]string) {
	c.Obj.SetAnnotations(annotations)
}
func (c *CustomResource) GetInitializers() *v1.Initializers { return c.Obj.GetInitializers() }
func (c *CustomResource) SetInitializers(initializers *v1.Initializers) {
	c.Obj.SetInitializers(initializers)
}
func (c *CustomResource) GetFinalizers() []string           { return c.Obj.GetFinalizers() }
func (c *CustomResource) SetFinalizers(finalizers []string) { c.Obj.SetFinalizers(finalizers) }
func (c *CustomResource) GetOwnerReferences() []v1.OwnerReference {
	return c.Obj.GetOwnerReferences()
}
func (c *CustomResource) SetOwnerReferences(p []v1.OwnerReference) { c.Obj.SetOwnerReferences(p) }
func (c *CustomResource) GetClusterName() string                   { return c.Obj.GetClusterName() }
func (c *CustomResource) SetClusterName(clusterName string)        { c.Obj.SetClusterName(clusterName) }
