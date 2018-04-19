/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/kubectl"
)

type NoResourceWithSuggestionError struct {
	*meta.NoResourceMatchError

	Suggestions []string
}

var _ error = &NoResourceWithSuggestionError{}

// suggestionRESTMapper is a decorator of meta.RESTMapper. If the KindFor,
// KindsFor, ResourcesFor, ResourceFor returns NoResourceMatchError, it will try
// to guess resource names based on shorted edit distance.
type suggestionRESTMapper struct {
	*resourceDiscover

	delegate     meta.RESTMapper
	excludeShort bool // exclude short form if set as true
}

var _ meta.RESTMapper = &suggestionRESTMapper{}

func (m *suggestionRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	result, err := m.delegate.KindFor(resource)
	return result, m.wrapError(err, resource)
}

func (m *suggestionRESTMapper) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	result, err := m.delegate.KindsFor(resource)
	return result, m.wrapError(err, resource)
}

func (m *suggestionRESTMapper) ResourcesFor(resource schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	result, err := m.delegate.ResourcesFor(resource)
	return result, m.wrapError(err, resource)
}

func (m *suggestionRESTMapper) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	result, err := m.delegate.ResourceFor(resource)
	return result, m.wrapError(err, resource)
}

func (m *suggestionRESTMapper) ResourceSingularizer(resource string) (string, error) {
	return m.delegate.ResourceSingularizer(schema.GroupVersionResource{Resource: resource}.Resource)
}

func (m *suggestionRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return m.delegate.RESTMapping(gk, versions...)
}

func (m *suggestionRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	return m.delegate.RESTMappings(gk, versions...)
}

func (m *suggestionRESTMapper) similarResources(resource schema.GroupVersionResource) []string {
	var shortcuts []kubectl.ResourceShortcuts
	var resources []*metav1.APIResourceList
	var err error

	if resources, shortcuts, err = m.getResources(); err != nil || len(shortcuts) == 0 {
		return []string{}
	}

	allNames := map[string]bool{}
	for _, resourceList := range resources {
		for _, resource := range resourceList.APIResources {
			allNames[resource.Name] = true
			allNames[resource.SingularName] = true
		}
	}
	for _, shortcut := range shortcuts {
		allNames[shortcut.LongForm.String()] = true
		allNames[shortcut.LongForm.Resource] = true
		if !m.excludeShort {
			allNames[shortcut.ShortForm.String()] = true
			allNames[shortcut.ShortForm.Resource] = true
		}
	}

	results := map[int][]string{}
	gr := resource.GroupResource()
	res := (&gr).String()
	max := len(res) / 2

	for name := range allNames {
		// TODO: parallel it
		dist := editDistance(res, name)
		if dist <= max {
			results[dist] = append(results[dist], name)
		}
	}

	for i := 1; i <= max; i++ {
		if result, ok := results[i]; ok {
			return result
		}
	}

	return []string{}
}

func (m *suggestionRESTMapper) wrapError(err error, resource schema.GroupVersionResource) error {
	switch er := err.(type) {
	case *meta.NoResourceMatchError:
		return &NoResourceWithSuggestionError{
			NoResourceMatchError: er,
			Suggestions:          m.similarResources(resource),
		}
	}

	return err
}
