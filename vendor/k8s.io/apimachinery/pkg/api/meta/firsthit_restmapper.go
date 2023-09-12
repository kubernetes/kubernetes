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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

var (
	_ ResettableRESTMapper = &FirstHitRESTMapper{}
)

// FirstHitRESTMapper is a wrapper for multiple RESTMappers which returns the
// first successful result for the singular requests
type FirstHitRESTMapper struct {
	MultiRESTMapper
}

func (m FirstHitRESTMapper) String() string {
	return fmt.Sprintf("FirstHitRESTMapper{\n\t%v\n}", m.MultiRESTMapper)
}

func (m FirstHitRESTMapper) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	errors := []error{}
	for _, t := range m.MultiRESTMapper {
		ret, err := t.ResourceFor(resource)
		if err == nil {
			return ret, nil
		}
		errors = append(errors, err)
	}

	return schema.GroupVersionResource{}, collapseAggregateErrors(errors)
}

func (m FirstHitRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	errors := []error{}
	for _, t := range m.MultiRESTMapper {
		ret, err := t.KindFor(resource)
		if err == nil {
			return ret, nil
		}
		errors = append(errors, err)
	}

	return schema.GroupVersionKind{}, collapseAggregateErrors(errors)
}

// RESTMapping provides the REST mapping for the resource based on the
// kind and version. This implementation supports multiple REST schemas and
// return the first match.
func (m FirstHitRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*RESTMapping, error) {
	errors := []error{}
	for _, t := range m.MultiRESTMapper {
		ret, err := t.RESTMapping(gk, versions...)
		if err == nil {
			return ret, nil
		}
		errors = append(errors, err)
	}

	return nil, collapseAggregateErrors(errors)
}

func (m FirstHitRESTMapper) Reset() {
	m.MultiRESTMapper.Reset()
}

// collapseAggregateErrors returns the minimal errors.  it handles empty as nil, handles one item in a list
// by returning the item, and collapses all NoMatchErrors to a single one (since they should all be the same)
func collapseAggregateErrors(errors []error) error {
	if len(errors) == 0 {
		return nil
	}
	if len(errors) == 1 {
		return errors[0]
	}

	allNoMatchErrors := true
	for _, err := range errors {
		allNoMatchErrors = allNoMatchErrors && IsNoMatchError(err)
	}
	if allNoMatchErrors {
		return errors[0]
	}

	return utilerrors.NewAggregate(errors)
}
