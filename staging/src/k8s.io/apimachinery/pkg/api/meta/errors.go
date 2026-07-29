/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// AmbiguousResourceError is returned if the RESTMapper finds multiple matches for a resource
type AmbiguousResourceError struct {
	PartialResource schema.GroupVersionResource

	MatchingResources []schema.GroupVersionResource
	MatchingKinds     []schema.GroupVersionKind
}

func (e *AmbiguousResourceError) Error() string {
	switch {
	case len(e.MatchingKinds) > 0 && len(e.MatchingResources) > 0:
		return fmt.Sprintf("%v matches multiple resources %v and kinds %v", e.PartialResource, e.MatchingResources, e.MatchingKinds)
	case len(e.MatchingKinds) > 0:
		return fmt.Sprintf("%v matches multiple kinds %v", e.PartialResource, e.MatchingKinds)
	case len(e.MatchingResources) > 0:
		return fmt.Sprintf("%v matches multiple resources %v", e.PartialResource, e.MatchingResources)
	}
	return fmt.Sprintf("%v matches multiple resources or kinds", e.PartialResource)
}

func (*AmbiguousResourceError) Is(target error) bool {
	_, ok := target.(*AmbiguousResourceError)
	return ok
}

// AmbiguousKindError is returned if the RESTMapper finds multiple matches for a kind
type AmbiguousKindError struct {
	PartialKind schema.GroupVersionKind

	MatchingResources []schema.GroupVersionResource
	MatchingKinds     []schema.GroupVersionKind
}

func (e *AmbiguousKindError) Error() string {
	switch {
	case len(e.MatchingKinds) > 0 && len(e.MatchingResources) > 0:
		return fmt.Sprintf("%v matches multiple resources %v and kinds %v", e.PartialKind, e.MatchingResources, e.MatchingKinds)
	case len(e.MatchingKinds) > 0:
		return fmt.Sprintf("%v matches multiple kinds %v", e.PartialKind, e.MatchingKinds)
	case len(e.MatchingResources) > 0:
		return fmt.Sprintf("%v matches multiple resources %v", e.PartialKind, e.MatchingResources)
	}
	return fmt.Sprintf("%v matches multiple resources or kinds", e.PartialKind)
}

func (*AmbiguousKindError) Is(target error) bool {
	_, ok := target.(*AmbiguousKindError)
	return ok
}

func IsAmbiguousError(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, &AmbiguousResourceError{}) || errors.Is(err, &AmbiguousKindError{})
}

// NoResourceMatchError is returned if the RESTMapper can't find any match for a resource
type NoResourceMatchError struct {
	PartialResource schema.GroupVersionResource
}

func (e *NoResourceMatchError) Error() string {
	return fmt.Sprintf("no matches for %v", e.PartialResource)
}

func (*NoResourceMatchError) Is(target error) bool {
	_, ok := target.(*NoResourceMatchError)
	return ok
}

// NoKindMatchError is returned if the RESTMapper can't find any match for a kind
type NoKindMatchError struct {
	// GroupKind is the API group and kind that was searched
	GroupKind schema.GroupKind
	// SearchedVersions is the optional list of versions the search was restricted to
	SearchedVersions []string
}

func (e *NoKindMatchError) Error() string {
	searchedVersions := sets.NewString()
	for _, v := range e.SearchedVersions {
		searchedVersions.Insert(schema.GroupVersion{Group: e.GroupKind.Group, Version: v}.String())
	}

	switch len(searchedVersions) {
	case 0:
		return fmt.Sprintf("no matches for kind %q in group %q", e.GroupKind.Kind, e.GroupKind.Group)
	case 1:
		return fmt.Sprintf("no matches for kind %q in version %q", e.GroupKind.Kind, searchedVersions.List()[0])
	default:
		return fmt.Sprintf("no matches for kind %q in versions %q", e.GroupKind.Kind, searchedVersions.List())
	}
}

func (*NoKindMatchError) Is(target error) bool {
	_, ok := target.(*NoKindMatchError)
	return ok
}

func IsNoMatchError(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, &NoResourceMatchError{}) || errors.Is(err, &NoKindMatchError{})
}
