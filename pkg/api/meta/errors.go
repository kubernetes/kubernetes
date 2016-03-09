/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// AmbiguousResourceError is returned if the RESTMapper finds multiple matches for a resource
type AmbiguousResourceError struct {
	PartialResource unversioned.GroupVersionResource

	MatchingResources []unversioned.GroupVersionResource
	MatchingKinds     []unversioned.GroupVersionKind
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

func IsAmbiguousResourceError(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(*AmbiguousResourceError)
	return ok
}

// NoResourceMatchError is returned if the RESTMapper can't find any match for a resource
type NoResourceMatchError struct {
	PartialResource unversioned.GroupVersionResource
}

func (e *NoResourceMatchError) Error() string {
	return fmt.Sprintf("no matches for %v", e.PartialResource)
}

func IsNoResourceMatchError(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(*NoResourceMatchError)
	return ok
}
