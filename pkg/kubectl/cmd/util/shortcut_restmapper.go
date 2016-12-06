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

package util

import (
	"strings"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

// ShortcutExpander is a RESTMapper that can be used for Kubernetes resources.   It expands the resource first, then invokes the wrapped
type ShortcutExpander struct {
	RESTMapper meta.RESTMapper

	All []schema.GroupResource

	discoveryClient discovery.DiscoveryInterface
}

var _ meta.RESTMapper = &ShortcutExpander{}

func NewShortcutExpander(delegate meta.RESTMapper, client discovery.DiscoveryInterface) ShortcutExpander {
	return ShortcutExpander{All: userResources, RESTMapper: delegate, discoveryClient: client}
}

func (e ShortcutExpander) getAll() []schema.GroupResource {
	if e.discoveryClient == nil {
		return e.All
	}

	// Check if we have access to server resources
	apiResources, err := e.discoveryClient.ServerResources()
	if err != nil {
		return e.All
	}

	availableResources, err := discovery.GroupVersionResources(apiResources)
	if err != nil {
		return e.All
	}

	availableAll := []schema.GroupResource{}
	for _, requestedResource := range e.All {
		for availableResource := range availableResources {
			if requestedResource.Group == availableResource.Group &&
				requestedResource.Resource == availableResource.Resource {
				availableAll = append(availableAll, requestedResource)
				break
			}
		}
	}

	return availableAll
}

func (e ShortcutExpander) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	return e.RESTMapper.KindFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	return e.RESTMapper.KindsFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourcesFor(resource schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return e.RESTMapper.ResourcesFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	return e.RESTMapper.ResourceFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceSingularizer(resource string) (string, error) {
	return e.RESTMapper.ResourceSingularizer(expandResourceShortcut(schema.GroupVersionResource{Resource: resource}).Resource)
}

func (e ShortcutExpander) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMapping(gk, versions...)
}

func (e ShortcutExpander) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMappings(gk, versions...)
}

// userResources are the resource names that apply to the primary, user facing resources used by
// client tools. They are in deletion-first order - dependent resources should be last.
var userResources = []schema.GroupResource{
	{Group: "", Resource: "pods"},
	{Group: "", Resource: "replicationcontrollers"},
	{Group: "", Resource: "services"},
	{Group: "apps", Resource: "statefulsets"},
	{Group: "autoscaling", Resource: "horizontalpodautoscalers"},
	{Group: "extensions", Resource: "jobs"},
	{Group: "extensions", Resource: "deployments"},
	{Group: "extensions", Resource: "replicasets"},
}

// AliasesForResource returns whether a resource has an alias or not
func (e ShortcutExpander) AliasesForResource(resource string) ([]string, bool) {
	if strings.ToLower(resource) == "all" {
		var resources []schema.GroupResource
		if resources = e.getAll(); len(resources) == 0 {
			resources = userResources
		}
		aliases := []string{}
		for _, r := range resources {
			aliases = append(aliases, r.Resource)
		}
		return aliases, true
	}
	expanded := expandResourceShortcut(schema.GroupVersionResource{Resource: resource}).Resource
	return []string{expanded}, (expanded != resource)
}

// expandResourceShortcut will return the expanded version of resource
// (something that a pkg/api/meta.RESTMapper can understand), if it is
// indeed a shortcut. Otherwise, will return resource unmodified.
func expandResourceShortcut(resource schema.GroupVersionResource) schema.GroupVersionResource {
	if expanded, ok := kubectl.ShortForms[resource.Resource]; ok {
		resource.Resource = expanded
		return resource
	}
	return resource
}
