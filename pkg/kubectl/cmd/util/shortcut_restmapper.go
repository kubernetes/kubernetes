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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/kubectl"
)

// ShortcutExpander is a RESTMapper that can be used for Kubernetes resources.   It expands the resource first, then invokes the wrapped
type ShortcutExpander struct {
	RESTMapper meta.RESTMapper

	All []schema.GroupResource

	discoveryClient discovery.DiscoveryInterface
}

var _ meta.RESTMapper = &ShortcutExpander{}

func NewShortcutExpander(delegate meta.RESTMapper, client discovery.DiscoveryInterface) ShortcutExpander {
	return ShortcutExpander{All: UserResources, RESTMapper: delegate, discoveryClient: client}
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
	return e.RESTMapper.KindFor(e.expandResourceShortcut(resource))
}

func (e ShortcutExpander) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	return e.RESTMapper.KindsFor(e.expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourcesFor(resource schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return e.RESTMapper.ResourcesFor(e.expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	return e.RESTMapper.ResourceFor(e.expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceSingularizer(resource string) (string, error) {
	return e.RESTMapper.ResourceSingularizer(e.expandResourceShortcut(schema.GroupVersionResource{Resource: resource}).Resource)
}

func (e ShortcutExpander) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMapping(gk, versions...)
}

func (e ShortcutExpander) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMappings(gk, versions...)
}

// UserResources are the resource names that apply to the primary, user facing resources used by
// client tools. They are in deletion-first order - dependent resources should be last.
// Should remain exported in order to expose a current list of resources to downstream
// composition that wants to build on the concept of 'all' for their CLIs.
var UserResources = []schema.GroupResource{
	{Group: "", Resource: "pods"},
	{Group: "", Resource: "replicationcontrollers"},
	{Group: "", Resource: "services"},
	{Group: "apps", Resource: "statefulsets"},
	{Group: "autoscaling", Resource: "horizontalpodautoscalers"},
	{Group: "batch", Resource: "jobs"},
	{Group: "extensions", Resource: "deployments"},
	{Group: "extensions", Resource: "replicasets"},
}

// AliasesForResource returns whether a resource has an alias or not
func (e ShortcutExpander) AliasesForResource(resource string) ([]string, bool) {
	if strings.ToLower(resource) == "all" {
		var resources []schema.GroupResource
		if resources = e.getAll(); len(resources) == 0 {
			resources = UserResources
		}
		aliases := []string{}
		for _, r := range resources {
			aliases = append(aliases, r.Resource)
		}
		return aliases, true
	}
	expanded := e.expandResourceShortcut(schema.GroupVersionResource{Resource: resource}).Resource
	return []string{expanded}, (expanded != resource)
}

// getShortcutMappings returns a hardcoded set of tuples.
// First the list of potential resources will be taken from the instance variable
// which holds the anticipated result of the discovery API.
// Next we will fall back to the hardcoded list of resources.
// Note that the list is ordered by group priority.
// TODO: Wire this to discovery API.
func (e ShortcutExpander) getShortcutMappings() ([]kubectl.ResourceShortcuts, error) {
	res := []kubectl.ResourceShortcuts{
		{
			ShortForm: schema.GroupResource{Group: "storage.k8s.io", Resource: "sc"},
			LongForm:  schema.GroupResource{Group: "storage.k8s.io", Resource: "storageclasses"},
		},
	}

	// append hardcoded short forms at the end of the list
	res = append(res, kubectl.ResourcesShortcutStatic...)
	return res, nil
}

// expandResourceShortcut will return the expanded version of resource
// (something that a pkg/api/meta.RESTMapper can understand), if it is
// indeed a shortcut. If no match has been found, we will match on group prefixing.
// Lastly we will return resource unmodified.
func (e ShortcutExpander) expandResourceShortcut(resource schema.GroupVersionResource) schema.GroupVersionResource {
	// get the shortcut mappings and return on first match.
	if resources, err := e.getShortcutMappings(); err == nil {
		for _, item := range resources {
			if len(resource.Group) != 0 && resource.Group != item.ShortForm.Group {
				continue
			}
			if resource.Resource == item.ShortForm.Resource {
				resource.Resource = item.LongForm.Resource
				return resource
			}
		}

		// we didn't find exact match so match on group prefixing. This allows autoscal to match autoscaling
		if len(resource.Group) == 0 {
			return resource
		}
		for _, item := range resources {
			if !strings.HasPrefix(item.ShortForm.Group, resource.Group) {
				continue
			}
			if resource.Resource == item.ShortForm.Resource {
				resource.Resource = item.LongForm.Resource
				return resource
			}
		}
	}

	return resource
}
