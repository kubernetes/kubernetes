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

package meta

import (
	"fmt"

	"k8s.io/client-go/pkg/runtime/schema"
)

const (
	AnyGroup    = "*"
	AnyVersion  = "*"
	AnyResource = "*"
	AnyKind     = "*"
)

// PriorityRESTMapper is a wrapper for automatically choosing a particular Resource or Kind
// when multiple matches are possible
type PriorityRESTMapper struct {
	// Delegate is the RESTMapper to use to locate all the Kind and Resource matches
	Delegate RESTMapper

	// ResourcePriority is a list of priority patterns to apply to matching resources.
	// The list of all matching resources is narrowed based on the patterns until only one remains.
	// A pattern with no matches is skipped.  A pattern with more than one match uses its
	// matches as the list to continue matching against.
	ResourcePriority []schema.GroupVersionResource

	// KindPriority is a list of priority patterns to apply to matching kinds.
	// The list of all matching kinds is narrowed based on the patterns until only one remains.
	// A pattern with no matches is skipped.  A pattern with more than one match uses its
	// matches as the list to continue matching against.
	KindPriority []schema.GroupVersionKind
}

func (m PriorityRESTMapper) String() string {
	return fmt.Sprintf("PriorityRESTMapper{\n\t%v\n\t%v\n\t%v\n}", m.ResourcePriority, m.KindPriority, m.Delegate)
}

// ResourceFor finds all resources, then passes them through the ResourcePriority patterns to find a single matching hit.
func (m PriorityRESTMapper) ResourceFor(partiallySpecifiedResource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	originalGVRs, err := m.Delegate.ResourcesFor(partiallySpecifiedResource)
	if err != nil {
		return schema.GroupVersionResource{}, err
	}
	if len(originalGVRs) == 1 {
		return originalGVRs[0], nil
	}

	remainingGVRs := append([]schema.GroupVersionResource{}, originalGVRs...)
	for _, pattern := range m.ResourcePriority {
		matchedGVRs := []schema.GroupVersionResource{}
		for _, gvr := range remainingGVRs {
			if resourceMatches(pattern, gvr) {
				matchedGVRs = append(matchedGVRs, gvr)
			}
		}

		switch len(matchedGVRs) {
		case 0:
			// if you have no matches, then nothing matched this pattern just move to the next
			continue
		case 1:
			// one match, return
			return matchedGVRs[0], nil
		default:
			// more than one match, use the matched hits as the list moving to the next pattern.
			// this way you can have a series of selection criteria
			remainingGVRs = matchedGVRs
		}
	}

	return schema.GroupVersionResource{}, &AmbiguousResourceError{PartialResource: partiallySpecifiedResource, MatchingResources: originalGVRs}
}

// KindFor finds all kinds, then passes them through the KindPriority patterns to find a single matching hit.
func (m PriorityRESTMapper) KindFor(partiallySpecifiedResource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	originalGVKs, err := m.Delegate.KindsFor(partiallySpecifiedResource)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	if len(originalGVKs) == 1 {
		return originalGVKs[0], nil
	}

	remainingGVKs := append([]schema.GroupVersionKind{}, originalGVKs...)
	for _, pattern := range m.KindPriority {
		matchedGVKs := []schema.GroupVersionKind{}
		for _, gvr := range remainingGVKs {
			if kindMatches(pattern, gvr) {
				matchedGVKs = append(matchedGVKs, gvr)
			}
		}

		switch len(matchedGVKs) {
		case 0:
			// if you have no matches, then nothing matched this pattern just move to the next
			continue
		case 1:
			// one match, return
			return matchedGVKs[0], nil
		default:
			// more than one match, use the matched hits as the list moving to the next pattern.
			// this way you can have a series of selection criteria
			remainingGVKs = matchedGVKs
		}
	}

	return schema.GroupVersionKind{}, &AmbiguousResourceError{PartialResource: partiallySpecifiedResource, MatchingKinds: originalGVKs}
}

func resourceMatches(pattern schema.GroupVersionResource, resource schema.GroupVersionResource) bool {
	if pattern.Group != AnyGroup && pattern.Group != resource.Group {
		return false
	}
	if pattern.Version != AnyVersion && pattern.Version != resource.Version {
		return false
	}
	if pattern.Resource != AnyResource && pattern.Resource != resource.Resource {
		return false
	}

	return true
}

func kindMatches(pattern schema.GroupVersionKind, kind schema.GroupVersionKind) bool {
	if pattern.Group != AnyGroup && pattern.Group != kind.Group {
		return false
	}
	if pattern.Version != AnyVersion && pattern.Version != kind.Version {
		return false
	}
	if pattern.Kind != AnyKind && pattern.Kind != kind.Kind {
		return false
	}

	return true
}

func (m PriorityRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	mappings, err := m.Delegate.RESTMappings(gk)
	if err != nil {
		return nil, err
	}

	// any versions the user provides take priority
	priorities := m.KindPriority
	if len(versions) > 0 {
		priorities = make([]schema.GroupVersionKind, 0, len(m.KindPriority)+len(versions))
		for _, version := range versions {
			gv, err := schema.ParseGroupVersion(version)
			if err != nil {
				return nil, err
			}
			priorities = append(priorities, gv.WithKind(AnyKind))
		}
		priorities = append(priorities, m.KindPriority...)
	}

	remaining := append([]*RESTMapping{}, mappings...)
	for _, pattern := range priorities {
		var matching []*RESTMapping
		for _, m := range remaining {
			if kindMatches(pattern, m.GroupVersionKind) {
				matching = append(matching, m)
			}
		}

		switch len(matching) {
		case 0:
			// if you have no matches, then nothing matched this pattern just move to the next
			continue
		case 1:
			// one match, return
			return matching[0], nil
		default:
			// more than one match, use the matched hits as the list moving to the next pattern.
			// this way you can have a series of selection criteria
			remaining = matching
		}
	}
	if len(remaining) == 1 {
		return remaining[0], nil
	}

	var kinds []schema.GroupVersionKind
	for _, m := range mappings {
		kinds = append(kinds, m.GroupVersionKind)
	}
	return nil, &AmbiguousKindError{PartialKind: gk.WithVersion(""), MatchingKinds: kinds}
}

func (m PriorityRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*RESTMapping, error) {
	return m.Delegate.RESTMappings(gk, versions...)
}

func (m PriorityRESTMapper) AliasesForResource(alias string) (aliases []string, ok bool) {
	return m.Delegate.AliasesForResource(alias)
}

func (m PriorityRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return m.Delegate.ResourceSingularizer(resource)
}

func (m PriorityRESTMapper) ResourcesFor(partiallySpecifiedResource schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return m.Delegate.ResourcesFor(partiallySpecifiedResource)
}

func (m PriorityRESTMapper) KindsFor(partiallySpecifiedResource schema.GroupVersionResource) (gvk []schema.GroupVersionKind, err error) {
	return m.Delegate.KindsFor(partiallySpecifiedResource)
}
