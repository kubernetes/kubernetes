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

package restmapper

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
)

// CategoryExpander maps category strings to GroupResources.
// Categories are classification or 'tag' of a group of resources.
type CategoryExpander interface {
	Expand(category string) ([]schema.GroupResource, bool)
}

// SimpleCategoryExpander implements CategoryExpander interface
// using a static mapping of categories to GroupResource mapping.
type SimpleCategoryExpander struct {
	Expansions map[string][]schema.GroupResource
}

// Expand fulfills CategoryExpander
func (e SimpleCategoryExpander) Expand(category string) ([]schema.GroupResource, bool) {
	ret, ok := e.Expansions[category]
	return ret, ok
}

// discoveryCategoryExpander struct lets a REST Client wrapper (discoveryClient) to retrieve list of APIResourceList,
// and then convert to fallbackExpander
type discoveryCategoryExpander struct {
	discoveryClient discovery.DiscoveryInterface
}

// NewDiscoveryCategoryExpander returns a category expander that makes use of the "categories" fields from
// the API, found through the discovery client. In case of any error or no category found (which likely
// means we're at a cluster prior to categories support, fallback to the expander provided.
func NewDiscoveryCategoryExpander(client discovery.DiscoveryInterface) CategoryExpander {
	if client == nil {
		panic("Please provide discovery client to shortcut expander")
	}
	return discoveryCategoryExpander{discoveryClient: client}
}

// Expand fulfills CategoryExpander
func (e discoveryCategoryExpander) Expand(category string) ([]schema.GroupResource, bool) {
	// Get all supported resources for groups and versions from server, if no resource found, fallback anyway.
	_, apiResourceLists, _ := e.discoveryClient.ServerGroupsAndResources()
	if len(apiResourceLists) == 0 {
		return nil, false
	}

	discoveredExpansions := map[string][]schema.GroupResource{}
	for _, apiResourceList := range apiResourceLists {
		gv, err := schema.ParseGroupVersion(apiResourceList.GroupVersion)
		if err != nil {
			continue
		}
		// Collect GroupVersions by categories
		for _, apiResource := range apiResourceList.APIResources {
			if categories := apiResource.Categories; len(categories) > 0 {
				for _, category := range categories {
					groupResource := schema.GroupResource{
						Group:    gv.Group,
						Resource: apiResource.Name,
					}
					discoveredExpansions[category] = append(discoveredExpansions[category], groupResource)
				}
			}
		}
	}

	ret, ok := discoveredExpansions[category]
	return ret, ok
}

// UnionCategoryExpander implements CategoryExpander interface.
// It maps given category string to union of expansions returned by all the CategoryExpanders in the list.
type UnionCategoryExpander []CategoryExpander

// Expand fulfills CategoryExpander
func (u UnionCategoryExpander) Expand(category string) ([]schema.GroupResource, bool) {
	ret := []schema.GroupResource{}
	ok := false

	// Expand the category for each CategoryExpander in the list and merge/combine the results.
	for _, expansion := range u {
		curr, currOk := expansion.Expand(category)

		for _, currGR := range curr {
			found := false
			for _, existing := range ret {
				if existing == currGR {
					found = true
					break
				}
			}
			if !found {
				ret = append(ret, currGR)
			}
		}
		ok = ok || currOk
	}

	return ret, ok
}
