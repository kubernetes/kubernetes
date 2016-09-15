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
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
)

// ShortcutExpander is a RESTMapper that can be used for OpenShift resources.   It expands the resource first, then invokes the wrapped
type ShortcutExpander struct {
	RESTMapper meta.RESTMapper

	All []string
}

var _ meta.RESTMapper = &ShortcutExpander{}

func NewShortcutExpander(delegate meta.RESTMapper) ShortcutExpander {
	return ShortcutExpander{All: userResources, RESTMapper: delegate}
}

func (e ShortcutExpander) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	return e.RESTMapper.KindFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error) {
	return e.RESTMapper.KindsFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourcesFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	return e.RESTMapper.ResourcesFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	return e.RESTMapper.ResourceFor(expandResourceShortcut(resource))
}

func (e ShortcutExpander) ResourceSingularizer(resource string) (string, error) {
	return e.RESTMapper.ResourceSingularizer(expandResourceShortcut(unversioned.GroupVersionResource{Resource: resource}).Resource)
}

func (e ShortcutExpander) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMapping(gk, versions...)
}

func (e ShortcutExpander) RESTMappings(gk unversioned.GroupKind) ([]*meta.RESTMapping, error) {
	return e.RESTMapper.RESTMappings(gk)
}

// userResources are the resource names that apply to the primary, user facing resources used by
// client tools. They are in deletion-first order - dependent resources should be last.
var userResources = []string{"rc", "svc", "pods", "pvc"}

// AliasesForResource returns whether a resource has an alias or not
func (e ShortcutExpander) AliasesForResource(resource string) ([]string, bool) {
	if resource == "all" {
		return e.All, true
	}

	expanded := expandResourceShortcut(unversioned.GroupVersionResource{Resource: resource}).Resource
	return []string{expanded}, (expanded != resource)
}

// expandResourceShortcut will return the expanded version of resource
// (something that a pkg/api/meta.RESTMapper can understand), if it is
// indeed a shortcut. Otherwise, will return resource unmodified.
func expandResourceShortcut(resource unversioned.GroupVersionResource) unversioned.GroupVersionResource {
	if expanded, ok := kubectl.ShortForms[resource.Resource]; ok {
		resource.Resource = expanded
		return resource
	}
	return resource
}
