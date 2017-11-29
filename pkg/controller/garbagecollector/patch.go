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
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
)

func deleteOwnerRefStrategicMergePatch(dependentUID types.UID, ownerUIDs ...types.UID) []byte {
	var pieces []string
	for _, ownerUID := range ownerUIDs {
		pieces = append(pieces, fmt.Sprintf(`{"$patch":"delete","uid":"%s"}`, ownerUID))
	}
	patch := fmt.Sprintf(`{"metadata":{"ownerReferences":[%s],"uid":"%s"}}`, strings.Join(pieces, ","), dependentUID)
	return []byte(patch)
}

// TODO: remove this function when we can use strategic merge patch
func (gc *GarbageCollector) getCachedMetadata(apiVersion, kind, namespace, name string) (metav1.Object, error) {
	fqKind := schema.FromAPIVersionAndKind(apiVersion, kind)
	mapping, err := gc.restMapper.RESTMapping(fqKind.GroupKind(), fqKind.Version)
	if err != nil {
		return nil, newRESTMappingError(kind, apiVersion)
	}
	gvr := schema.GroupVersionResource{Group: fqKind.Group, Version: fqKind.Version, Resource: mapping.Resource}
	gc.dependencyGraphBuilder.monitorLock.RLock()
	defer gc.dependencyGraphBuilder.monitorLock.RUnlock()
	m, ok := gc.dependencyGraphBuilder.monitors[gvr]
	if !ok || m == nil {
		// TODO: consider to make a call to apiserver in case the local cache is
		// unavailable.
		return nil, fmt.Errorf("doesn't have a local cache for %s", gvr)
	}
	var raw interface{}
	var exist bool
	if len(namespace) != 0 {
		raw, exist, err = m.store.GetByKey(namespace + "/" + name)
	} else {
		raw, exist, err = m.store.GetByKey(name)
	}
	if err != nil {
		return nil, err
	}
	if !exist {
		return nil, errors.NewNotFound(gvr.GroupResource(), name)
	}
	obj, ok := raw.(runtime.Object)
	if !ok {
		return nil, fmt.Errorf("expect a runtime.Object, got %v", raw)
	}
	return meta.Accessor(obj)
}

type objectForPatch struct {
	ObjectMetaForPatch `json:"metadata"`
}

type ObjectMetaForPatch struct {
	ResourceVersion string                  `json:"resourceVersion"`
	OwnerReferences []metav1.OwnerReference `json:"ownerReferences"`
}

// TODO: Switch back to use strategic merge patch once it's supported on Custom
// Resource.
// Returns JSON merge patch that removes the ownerReferences matching ownerUIDs.
func (gc *GarbageCollector) deleteOwnerRefJSONMergePatch(item *node, ownerUIDs ...types.UID) ([]byte, error) {
	accessor, err := gc.getCachedMetadata(item.identity.APIVersion, item.identity.Kind, item.identity.Namespace, item.identity.Name)
	if err != nil {
		return nil, err
	}
	expectedObjectMeta := ObjectMetaForPatch{}
	expectedObjectMeta.ResourceVersion = accessor.GetResourceVersion()
	refs := accessor.GetOwnerReferences()
	for _, ref := range refs {
		var skip bool
		for _, ownerUID := range ownerUIDs {
			if ref.UID == ownerUID {
				skip = true
				break
			}
		}
		if !skip {
			expectedObjectMeta.OwnerReferences = append(expectedObjectMeta.OwnerReferences, ref)
		}
	}
	return json.Marshal(objectForPatch{expectedObjectMeta})
}

// Generate a patch that unsets the BlockOwnerDeletion field of all
// ownerReferences of node.
func (n *node) unblockOwnerReferencesStrategicMergePatch() ([]byte, error) {
	var dummy metaonly.MetadataOnlyObject
	var blockingRefs []metav1.OwnerReference
	falseVar := false
	for _, owner := range n.owners {
		if owner.BlockOwnerDeletion != nil && *owner.BlockOwnerDeletion {
			ref := owner
			ref.BlockOwnerDeletion = &falseVar
			blockingRefs = append(blockingRefs, ref)
		}
	}
	dummy.ObjectMeta.SetOwnerReferences(blockingRefs)
	dummy.ObjectMeta.UID = n.identity.UID
	return json.Marshal(dummy)
}

// TODO: Switch back to use strategic merge patch once it's supported on Custom
// Resource.
// Generate a JSON merge patch that unsets the BlockOwnerDeletion field of all
// ownerReferences of node.
func (gc *GarbageCollector) unblockOwnerReferencesJSONMergePatch(n *node) ([]byte, error) {
	accessor, err := gc.getCachedMetadata(n.identity.APIVersion, n.identity.Kind, n.identity.Namespace, n.identity.Name)
	if err != nil {
		return nil, err
	}
	expectedObjectMeta := ObjectMetaForPatch{}
	expectedObjectMeta.ResourceVersion = accessor.GetResourceVersion()
	var expectedOwners []metav1.OwnerReference
	falseVar := false
	for _, owner := range n.owners {
		owner.BlockOwnerDeletion = &falseVar
		expectedOwners = append(expectedOwners, owner)
	}
	expectedObjectMeta.OwnerReferences = expectedOwners
	return json.Marshal(objectForPatch{expectedObjectMeta})
}
