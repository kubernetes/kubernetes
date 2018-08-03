/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
)

// cluster scoped resources don't have namespaces.  Default to the item's namespace, but clear it for cluster scoped resources
func resourceDefaultNamespace(namespaced bool, defaultNamespace string) string {
	if namespaced {
		return defaultNamespace
	}
	return ""
}

// apiResource consults the REST mapper to translate an <apiVersion, kind,
// namespace> tuple to a unversioned.APIResource struct.
func (gc *GarbageCollector) apiResource(apiVersion, kind string) (schema.GroupVersionResource, bool, error) {
	fqKind := schema.FromAPIVersionAndKind(apiVersion, kind)
	mapping, err := gc.restMapper.RESTMapping(fqKind.GroupKind(), fqKind.Version)
	if err != nil {
		return schema.GroupVersionResource{}, false, newRESTMappingError(kind, apiVersion)
	}
	return mapping.Resource, mapping.Scope == meta.RESTScopeNamespace, nil
}

func (gc *GarbageCollector) deleteObject(item objectReference, policy *metav1.DeletionPropagation) error {
	resource, namespaced, err := gc.apiResource(item.APIVersion, item.Kind)
	if err != nil {
		return err
	}
	uid := item.UID
	preconditions := metav1.Preconditions{UID: &uid}
	deleteOptions := metav1.DeleteOptions{Preconditions: &preconditions, PropagationPolicy: policy}
	return gc.dynamicClient.Resource(resource).Namespace(resourceDefaultNamespace(namespaced, item.Namespace)).Delete(item.Name, &deleteOptions)
}

func (gc *GarbageCollector) getObject(item objectReference) (*unstructured.Unstructured, error) {
	resource, namespaced, err := gc.apiResource(item.APIVersion, item.Kind)
	if err != nil {
		return nil, err
	}
	return gc.dynamicClient.Resource(resource).Namespace(resourceDefaultNamespace(namespaced, item.Namespace)).Get(item.Name, metav1.GetOptions{})
}

func (gc *GarbageCollector) updateObject(item objectReference, obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	resource, namespaced, err := gc.apiResource(item.APIVersion, item.Kind)
	if err != nil {
		return nil, err
	}
	return gc.dynamicClient.Resource(resource).Namespace(resourceDefaultNamespace(namespaced, item.Namespace)).Update(obj, metav1.UpdateOptions{})
}

func (gc *GarbageCollector) patchObject(item objectReference, patch []byte, pt types.PatchType) (*unstructured.Unstructured, error) {
	resource, namespaced, err := gc.apiResource(item.APIVersion, item.Kind)
	if err != nil {
		return nil, err
	}
	return gc.dynamicClient.Resource(resource).Namespace(resourceDefaultNamespace(namespaced, item.Namespace)).Patch(item.Name, pt, patch, metav1.UpdateOptions{})
}

// TODO: Using Patch when strategicmerge supports deleting an entry from a
// slice of a base type.
func (gc *GarbageCollector) removeFinalizer(owner *node, targetFinalizer string) error {
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		ownerObject, err := gc.getObject(owner.identity)
		if errors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("cannot finalize owner %s, because cannot get it: %v. The garbage collector will retry later.", owner.identity, err)
		}
		accessor, err := meta.Accessor(ownerObject)
		if err != nil {
			return fmt.Errorf("cannot access the owner object %v: %v. The garbage collector will retry later.", ownerObject, err)
		}
		finalizers := accessor.GetFinalizers()
		var newFinalizers []string
		found := false
		for _, f := range finalizers {
			if f == targetFinalizer {
				found = true
				continue
			}
			newFinalizers = append(newFinalizers, f)
		}
		if !found {
			glog.V(5).Infof("the %s finalizer is already removed from object %s", targetFinalizer, owner.identity)
			return nil
		}
		// remove the owner from dependent's OwnerReferences
		ownerObject.SetFinalizers(newFinalizers)
		_, err = gc.updateObject(owner.identity, ownerObject)
		return err
	})
	if errors.IsConflict(err) {
		return fmt.Errorf("updateMaxRetries(%d) has reached. The garbage collector will retry later for owner %v.", retry.DefaultBackoff.Steps, owner.identity)
	}
	return err
}
