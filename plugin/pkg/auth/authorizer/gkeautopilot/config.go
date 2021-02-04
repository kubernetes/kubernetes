/*
Copyright 2020 The Kubernetes Authors.

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

package gkeautopilot

import (
	"k8s.io/apimachinery/pkg/util/sets"
)

// configValidationError is an error type used for validation issues
type configValidationError struct {
	Msg string
}

func (e *configValidationError) Error() string {
	return e.Msg
}

func newConfigErr(msg string) *configValidationError {
	return &configValidationError{msg}
}

// errors that might be returned for when validating
var (
	errManagedNamespaceEmptyVerbs          = newConfigErr("managedNamespace.deniedVerbs.[*] cannot be empty")
	errManagedNamespaceEmptyResource       = newConfigErr("managedNamespace.deniedResources.[*].resource cannot be empty")
	errManagedResourceEmptyResource        = newConfigErr("managedResource.resource cannot be empty")
	errManagedResourceEmptyName            = newConfigErr("managedResource.name cannot be empty")
	errManagedResourceEmptySubresourceName = newConfigErr("managedResource.subresources.[*].name cannot be empty")
)

// Identities is a struct with lists of users and groups
type Identities struct {
	Users  []string
	Groups []string
}

// ManagedNamespace is consisted of a name, and a slice of RequestPatterns
type ManagedNamespace struct {
	// Name of the namespace
	Name string
	// A request will be denied if its namespace matches Name and its verb is in
	// DeniedVerbs set
	DeniedVerbs []string
	// A request will be denied if its namespace matches Name and its
	// resource/subresource is in DeniedResources set
	DeniedResources []ResourceSubresource
}

// ResourceSubresource is a pair of Resource and Subresource
type ResourceSubresource struct {
	Resource    string
	Subresource string
}

func resSubToString(resource, subresource string) string {
	return resource + "/" + subresource
}

// ManagedResources has a slice of paths,for which only the given verbs
// are allowed.
type ManagedResources struct {
	Resources    []ManagedResource
	AllowedVerbs []string
}

// ManagedResource specifies a resource that is being managed by GKEAutopilot
type ManagedResource struct {
	APIGroup     string
	Resource     string
	Subresources []Subresource

	Namespace string
	Name      string
}

// Subresource specifies the name and allowed verbs for a subresource
type Subresource struct {
	Name         string
	AllowedVerbs []string
}

// config contains configuration data for gkeautopilot authorizer
type config struct {
	// IgnoredIdentities are the identities that the authorizer does not have an
	// opinion on their requests
	IgnoredIdentities         Identities
	ManagedNamespaces         []ManagedNamespace
	ManagedResources          []ManagedResources
	PolicyEnforcerWebhookName string
}

// treeNode is a datastructure that is used to create a tree for managed
// resources
type treeNode struct {
	allowedVerbs sets.String
	children     map[string]*treeNode
}

func newTreeNode() *treeNode {
	return &treeNode{
		children: map[string]*treeNode{},
	}
}

func (t *treeNode) addPath(path []string, allowedVerbs sets.String) {
	if len(path) == 0 {
		t.allowedVerbs = allowedVerbs
		return
	}

	if t.children[path[0]] == nil {
		t.children[path[0]] = newTreeNode()
	}

	t.children[path[0]].addPath(path[1:], allowedVerbs)
}

// namespaceDeniedSets is used by configHelper and specifies the denied verbs
// and subresources as sets, for fast lookup
type namespaceDeniedSets struct {
	deniedVerbs               sets.String
	deniedResourceSubresource sets.String
}

// configHelper is used by the authz client to pre-calculate a few
// datastructures, so the config data could be queried faster
type configHelper struct {
	ignoredUsersSet      sets.String
	ignoredGroupsSet     sets.String
	managedNamespacesMap map[string]namespaceDeniedSets

	// every path from the root of the tree to a leaf, traverses a resource
	// path in the following order: [APIGroup, Namespace, Resource, Name,
	// Subresource]. This datastructure is used to find a subresource node and
	// exit the tree early, if a resource is not managed (which happens when a
	// nil encountered before getting to Name).
	// This yields in a faster lookup than calculating the entire path of the
	// resource and then looking it up in a map.
	// As a visualization example, the resources with the following paths:
	// - ""/ns1/pod/pod1/exec
	// - ""/ns1/pod/pod1/log
	// - ""/ns1/pod/pod2
	// - ""/ns1/deployment/dep1
	// - rbac.authorization.k8s.io/""/clusterroles/role1  #cluster-scoped resource
	// the tree will look like following:
	//                                       root
	//                                  /            \
	// [APIGroup]                      ""       rbac.authorization.k8s.io
	//                               /                   \
	// [Namespace]                 ns1                    ""
	//                           /      \                  \
	// [Resource]             pods      deployments        clusterroles
	//                      /     \          \               \
	// [Name]             pod1    pod2        dep1           role1
	//                   /   \
	// [Subresource]   exec  log
	managedResourcesTree *treeNode
}

// validates the config data
func (c *config) validate() error {
	// check for invalid empty fields in managed namespaces
	for _, mns := range c.ManagedNamespaces {
		for _, v := range mns.DeniedVerbs {
			if v == "" {
				return errManagedNamespaceEmptyVerbs
			}
		}

		for _, resSubres := range mns.DeniedResources {
			if resSubres.Resource == "" {
				return errManagedNamespaceEmptyResource
			}
		}
	}

	// check for invalid empty fields in managed resources
	for _, mr := range c.ManagedResources {
		for _, res := range mr.Resources {
			if res.Resource == "" {
				return errManagedResourceEmptyResource
			}
			if res.Name == "" {
				return errManagedResourceEmptyName
			}

			for _, sub := range res.Subresources {
				if sub.Name == "" {
					return errManagedResourceEmptySubresourceName
				}
			}

		}
	}

	return nil
}

// builds a type which pre-calculates maps required for efficient query of
// config lists
func buildConfigHelper(c *config) *configHelper {
	// nsMap is a map from a namespace to a set of verbs
	nsMap := map[string]namespaceDeniedSets{}
	for _, mns := range c.ManagedNamespaces {
		resSubList := []string{}
		for _, rs := range mns.DeniedResources {
			resSubList = append(resSubList, resSubToString(rs.Resource, rs.Subresource))
		}

		nsMap[mns.Name] = namespaceDeniedSets{
			deniedVerbs:               sets.NewString(mns.DeniedVerbs...),
			deniedResourceSubresource: sets.NewString(resSubList...),
		}
	}

	resTree := newTreeNode()
	for _, mres := range c.ManagedResources {
		verbsSet := sets.NewString(mres.AllowedVerbs...)
		for _, res := range mres.Resources {
			resPath := []string{res.APIGroup, res.Namespace, res.Resource, res.Name}
			resTree.addPath(resPath, verbsSet)
			// add the path for subresources of the resource, and their allowed
			// verbs
			for _, sub := range res.Subresources {
				subresPath := append(resPath, sub.Name)
				subresVerbs := sets.NewString(sub.AllowedVerbs...)
				resTree.addPath(subresPath, subresVerbs)
			}
		}
	}

	return &configHelper{
		ignoredUsersSet:      sets.NewString(c.IgnoredIdentities.Users...),
		ignoredGroupsSet:     sets.NewString(c.IgnoredIdentities.Groups...),
		managedNamespacesMap: nsMap,
		managedResourcesTree: resTree,
	}
}
