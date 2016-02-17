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
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// MultiRESTMapper is a wrapper for multiple RESTMappers.
type MultiRESTMapper []RESTMapper

func (m MultiRESTMapper) String() string {
	nested := []string{}
	for _, t := range m {
		currString := fmt.Sprintf("%v", t)
		splitStrings := strings.Split(currString, "\n")
		nested = append(nested, strings.Join(splitStrings, "\n\t"))
	}

	return fmt.Sprintf("MultiRESTMapper{\n\t%s\n}", strings.Join(nested, "\n\t"))
}

// ResourceSingularizer converts a REST resource name from plural to singular (e.g., from pods to pod)
// This implementation supports multiple REST schemas and return the first match.
func (m MultiRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	for _, t := range m {
		singular, err = t.ResourceSingularizer(resource)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiRESTMapper) ResourcesFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	for _, t := range m {
		gvrs, err := t.ResourcesFor(resource)
		// ignore "no match" errors, but any other error percolates back up
		if !IsNoResourceMatchError(err) {
			return gvrs, err
		}
	}
	return nil, &NoResourceMatchError{PartialResource: resource}
}

// KindsFor provides the Kind mappings for the REST resources. This implementation supports multiple REST schemas and returns
// the first match.
func (m MultiRESTMapper) KindsFor(resource unversioned.GroupVersionResource) (gvk []unversioned.GroupVersionKind, err error) {
	for _, t := range m {
		gvks, err := t.KindsFor(resource)
		// ignore "no match" errors, but any other error percolates back up
		if !IsNoResourceMatchError(err) {
			return gvks, err
		}
	}
	return nil, &NoResourceMatchError{PartialResource: resource}
}

func (m MultiRESTMapper) ResourceFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	for _, t := range m {
		gvr, err := t.ResourceFor(resource)
		// ignore "no match" errors, but any other error percolates back up
		if !IsNoResourceMatchError(err) {
			return gvr, err
		}
	}
	return unversioned.GroupVersionResource{}, &NoResourceMatchError{PartialResource: resource}
}

// KindsFor provides the Kind mapping for the REST resources. This implementation supports multiple REST schemas and returns
// the first match.
func (m MultiRESTMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	for _, t := range m {
		gvk, err := t.KindFor(resource)
		// ignore "no match" errors, but any other error percolates back up
		if !IsNoResourceMatchError(err) {
			return gvk, err
		}
	}
	return unversioned.GroupVersionKind{}, &NoResourceMatchError{PartialResource: resource}
}

// RESTMapping provides the REST mapping for the resource based on the
// kind and version. This implementation supports multiple REST schemas and
// return the first match.
func (m MultiRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	for _, t := range m {
		mapping, err = t.RESTMapping(gk, versions...)
		if err == nil {
			return
		}
	}
	return
}

// AliasesForResource finds the first alias response for the provided mappers.
func (m MultiRESTMapper) AliasesForResource(alias string) (aliases []string, ok bool) {
	for _, t := range m {
		if aliases, ok = t.AliasesForResource(alias); ok {
			return
		}
	}
	return nil, false
}
