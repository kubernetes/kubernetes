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

package openapi

import (
	openapi_v2 "github.com/google/gnostic/openapiv2"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

// Resources interface describe a resources provider, that can give you
// resource based on group-version-kind.
type Resources interface {
	LookupResource(gvk schema.GroupVersionKind) proto.Schema
}

// groupVersionKindExtensionKey is the key used to lookup the
// GroupVersionKind value for an object definition from the
// definition's "extensions" map.
const groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"

// document is an implementation of `Resources`. It looks for
// resources in an openapi Schema.
type document struct {
	// Maps gvk to model name
	resources map[schema.GroupVersionKind]string
	models    proto.Models
}

var _ Resources = &document{}

// NewOpenAPIData creates a new `Resources` out of the openapi document
func NewOpenAPIData(doc *openapi_v2.Document) (Resources, error) {
	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return nil, err
	}

	resources := map[schema.GroupVersionKind]string{}
	for _, modelName := range models.ListModels() {
		model := models.LookupModel(modelName)
		if model == nil {
			panic("ListModels returns a model that can't be looked-up.")
		}
		gvkList := parseGroupVersionKind(model)
		for _, gvk := range gvkList {
			if len(gvk.Kind) > 0 {
				resources[gvk] = modelName
			}
		}
	}

	return &document{
		resources: resources,
		models:    models,
	}, nil
}

func (d *document) LookupResource(gvk schema.GroupVersionKind) proto.Schema {
	modelName, found := d.resources[gvk]
	if !found {
		return nil
	}
	return d.models.LookupModel(modelName)
}

// Get and parse GroupVersionKind from the extension. Returns empty if it doesn't have one.
func parseGroupVersionKind(s proto.Schema) []schema.GroupVersionKind {
	extensions := s.GetExtensions()

	gvkListResult := []schema.GroupVersionKind{}

	// Get the extensions
	gvkExtension, ok := extensions[groupVersionKindExtensionKey]
	if !ok {
		return []schema.GroupVersionKind{}
	}

	// gvk extension must be a list of at least 1 element.
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return []schema.GroupVersionKind{}
	}

	for _, gvk := range gvkList {
		// gvk extension list must be a map with group, version, and
		// kind fields
		gvkMap, ok := gvk.(map[interface{}]interface{})
		if !ok {
			continue
		}
		group, ok := gvkMap["group"].(string)
		if !ok {
			continue
		}
		version, ok := gvkMap["version"].(string)
		if !ok {
			continue
		}
		kind, ok := gvkMap["kind"].(string)
		if !ok {
			continue
		}

		gvkListResult = append(gvkListResult, schema.GroupVersionKind{
			Group:   group,
			Version: version,
			Kind:    kind,
		})
	}

	return gvkListResult
}
