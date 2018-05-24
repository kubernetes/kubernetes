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

package openapi

import (
	"encoding/json"
	"fmt"

	"github.com/go-openapi/spec"
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

const (
	// groupVersionKindExtensionKey is the key used to lookup the
	// GroupVersionKind value for an object definition from the
	// definition's "extensions" map.
	groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"
)

// ToProtoSchema builds the proto formatted schema from an OpenAPI spec
func ToProtoSchema(openAPIDefinitions *spec.Definitions, gvk schema.GroupVersionKind) (proto.Schema, error) {
	openAPISpec := newMinimalValidOpenAPISpec()
	openAPISpec.Definitions = *openAPIDefinitions

	specBytes, err := json.MarshalIndent(openAPISpec, " ", " ")
	if err != nil {
		return nil, err
	}

	var info yaml.MapSlice
	err = yaml.Unmarshal(specBytes, &info)
	if err != nil {
		return nil, err
	}

	doc, err := openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	if err != nil {
		return nil, err
	}

	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return nil, err
	}

	for _, modelName := range models.ListModels() {
		model := models.LookupModel(modelName)
		if model == nil {
			return nil, fmt.Errorf("the ListModels function returned a model that can't be looked-up")
		}
		gvkList := parseGroupVersionKind(model)
		for _, modelGVK := range gvkList {
			if modelGVK == gvk {
				return model, nil
			}
		}
	}

	return nil, fmt.Errorf("no model found with a %v tag matching %v", groupVersionKindExtensionKey, gvk)
}

// newMinimalValidOpenAPISpec creates a minimal openapi spec with only the required fields filled in
func newMinimalValidOpenAPISpec() *spec.Swagger {
	return &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger: "2.0",
			Info: &spec.Info{
				InfoProps: spec.InfoProps{
					Title:   "Kubernetes",
					Version: "0.0.0",
				},
			},
		},
	}
}

// parseGroupVersionKind gets and parses GroupVersionKind from the extension. Returns empty if it doesn't have one.
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
