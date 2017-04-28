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

package app

import (
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server/storage"
)

// Function to get a map of resources and the corresponding storages.
type getResourcesStorageFunc func() map[string]rest.Storage

// Filters the resources from the given resources storage map to those that are enabled in the given apiResourceConfigSource.
// resourcesStorageMap is expected to contain all resources in a group version.
// Returns false if none of the resources are enabled and hence the whole group version should be disabled.
func enabledResources(groupVersion schema.GroupVersion, resourcesStorageMap map[string]getResourcesStorageFunc, apiResourceConfigSource storage.APIResourceConfigSource) (bool, map[string]rest.Storage) {
	enabledResources := map[string]rest.Storage{}
	groupName := groupVersion.Group
	if !apiResourceConfigSource.AnyResourcesForGroupEnabled(groupName) {
		glog.V(1).Infof("Skipping disabled API group %q", groupName)
		return false, enabledResources
	}
	for resource, fn := range resourcesStorageMap {
		if apiResourceConfigSource.ResourceEnabled(groupVersion.WithResource(resource)) {
			resources := fn()
			for k, v := range resources {
				enabledResources[k] = v
			}
		} else {
			glog.V(1).Infof("Skipping disabled resource %s in API group %q", resource, groupName)
		}
	}
	if len(enabledResources) == 0 {
		glog.V(1).Infof("Skipping API group %q since there is no enabled resource", groupName)
		return false, enabledResources
	}
	return true, enabledResources
}
