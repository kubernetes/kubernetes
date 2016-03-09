/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
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
	ResourcePriority []unversioned.GroupVersionResource

	// KindPriority is a list of priority patterns to apply to matching kinds.
	// The list of all matching kinds is narrowed based on the patterns until only one remains.
	// A pattern with no matches is skipped.  A pattern with more than one match uses its
	// matches as the list to continue matching against.
	KindPriority []unversioned.GroupVersionKind
}

func (m PriorityRESTMapper) String() string {
	return fmt.Sprintf("PriorityRESTMapper{\n\t%v\n\t%v\n\t%v\n}", m.ResourcePriority, m.KindPriority, m.Delegate)
}

// ResourceFor finds all resources, then passes them through the ResourcePriority patterns to find a single matching hit.
func (m PriorityRESTMapper) ResourceFor(partiallySpecifiedResource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	originalGVRs, err := m.Delegate.ResourcesFor(partiallySpecifiedResource)
	if err != nil {
		return unversioned.GroupVersionResource{}, err
	}
	if len(originalGVRs) == 1 {
		return originalGVRs[0], nil
	}

	remainingGVRs := append([]unversioned.GroupVersionResource{}, originalGVRs...)
	for _, pattern := range m.ResourcePriority {
		matchedGVRs := []unversioned.GroupVersionResource{}
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

	return unversioned.GroupVersionResource{}, &AmbiguousResourceError{PartialResource: partiallySpecifiedResource, MatchingResources: originalGVRs}
}

// KindFor finds all kinds, then passes them through the KindPriority patterns to find a single matching hit.
func (m PriorityRESTMapper) KindFor(partiallySpecifiedResource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	originalGVKs, err := m.Delegate.KindsFor(partiallySpecifiedResource)
	if err != nil {
		return unversioned.GroupVersionKind{}, err
	}
	if len(originalGVKs) == 1 {
		return originalGVKs[0], nil
	}

	remainingGVKs := append([]unversioned.GroupVersionKind{}, originalGVKs...)
	for _, pattern := range m.KindPriority {
		matchedGVKs := []unversioned.GroupVersionKind{}
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

	return unversioned.GroupVersionKind{}, &AmbiguousResourceError{PartialResource: partiallySpecifiedResource, MatchingKinds: originalGVKs}
}

func resourceMatches(pattern unversioned.GroupVersionResource, resource unversioned.GroupVersionResource) bool {
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

func kindMatches(pattern unversioned.GroupVersionKind, kind unversioned.GroupVersionKind) bool {
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

func (m PriorityRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	return m.Delegate.RESTMapping(gk, versions...)
}

func (m PriorityRESTMapper) AliasesForResource(alias string) (aliases []string, ok bool) {
	return m.Delegate.AliasesForResource(alias)
}

func (m PriorityRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return m.Delegate.ResourceSingularizer(resource)
}

func (m PriorityRESTMapper) ResourcesFor(partiallySpecifiedResource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	return m.Delegate.ResourcesFor(partiallySpecifiedResource)
}

func (m PriorityRESTMapper) KindsFor(partiallySpecifiedResource unversioned.GroupVersionResource) (gvk []unversioned.GroupVersionKind, err error) {
	return m.Delegate.KindsFor(partiallySpecifiedResource)
}
