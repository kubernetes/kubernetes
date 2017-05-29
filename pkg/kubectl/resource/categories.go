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

package resource

import (
	"errors"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
)

type CategoryExpander interface {
	Expand(category string) ([]schema.GroupResource, bool)
}

type SimpleCategoryExpander struct {
	Expansions map[string][]schema.GroupResource
}

func (e SimpleCategoryExpander) Expand(category string) ([]schema.GroupResource, bool) {
	ret, ok := e.Expansions[category]
	return ret, ok
}

type discoveryFilteredExpander struct {
	delegate CategoryExpander

	discoveryClient discovery.DiscoveryInterface
}

// NewDiscoveryFilteredExpander returns a category expander that filters the returned groupresources by
// what the server has available
func NewDiscoveryFilteredExpander(delegate CategoryExpander, client discovery.DiscoveryInterface) (discoveryFilteredExpander, error) {
	if client == nil {
		return discoveryFilteredExpander{}, errors.New("Please provide discovery client to shortcut expander")
	}
	return discoveryFilteredExpander{delegate: delegate, discoveryClient: client}, nil
}

func (e discoveryFilteredExpander) Expand(category string) ([]schema.GroupResource, bool) {
	delegateExpansion, ok := e.delegate.Expand(category)

	// Check if we have access to server resources
	apiResources, err := e.discoveryClient.ServerResources()
	if err != nil {
		return delegateExpansion, ok
	}

	availableResources, err := discovery.GroupVersionResources(apiResources)
	if err != nil {
		return delegateExpansion, ok
	}

	available := []schema.GroupResource{}
	for _, requestedResource := range delegateExpansion {
		for availableResource := range availableResources {
			if requestedResource.Group == availableResource.Group &&
				requestedResource.Resource == availableResource.Resource {
				available = append(available, requestedResource)
				break
			}
		}
	}

	return available, ok
}

// legacyUserResources are the resource names that apply to the primary, user facing resources used by
// client tools. They are in deletion-first order - dependent resources should be last.
// Should remain exported in order to expose a current list of resources to downstream
// composition that wants to build on the concept of 'all' for their CLIs.
var legacyUserResources = []schema.GroupResource{
	{Group: "", Resource: "pods"},
	{Group: "", Resource: "replicationcontrollers"},
	{Group: "", Resource: "services"},
	{Group: "apps", Resource: "statefulsets"},
	{Group: "autoscaling", Resource: "horizontalpodautoscalers"},
	{Group: "batch", Resource: "jobs"},
	{Group: "extensions", Resource: "deployments"},
	{Group: "extensions", Resource: "replicasets"},
}

// LegacyCategoryExpander is the old hardcoded expansion
var LegacyCategoryExpander CategoryExpander = SimpleCategoryExpander{
	Expansions: map[string][]schema.GroupResource{
		"all": legacyUserResources,
	},
}

// LegacyFederationCategoryExpander is the old hardcoded expansion for federation
var LegacyFederationCategoryExpander CategoryExpander = SimpleCategoryExpander{
	Expansions: map[string][]schema.GroupResource{
		"all": {{Group: "", Resource: "services"}},
	},
}
