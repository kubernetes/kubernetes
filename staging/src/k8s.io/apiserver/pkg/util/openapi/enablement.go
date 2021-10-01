/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"

	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// enumTypeDescriptionHeader is the header of enum section in schema description.
const enumTypeDescriptionHeader = "Possible enum values:"

// WrapGetOpenAPIDefinitions wraps a GetOpenAPIDefinitions to revert
// any change to the schema that was made by a disabled feature.
func WrapGetOpenAPIDefinitions(GetOpenAPIDefinitions common.GetOpenAPIDefinitions) common.GetOpenAPIDefinitions {
	return func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
		defs := GetOpenAPIDefinitions(ref)
		amendDefinitions(defs)
		return defs
	}
}

func amendDefinitions(defs map[string]common.OpenAPIDefinition) {
	// revert changes from OpenAPIEnums
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.OpenAPIEnums) {
		for gvk, def := range defs {
			if pruneEnums(&def.Schema) {
				defs[gvk] = def
			}
		}
	}
}

func pruneEnums(schema *spec.Schema) (changed bool) {
	if len(schema.Enum) != 0 {
		schema.Enum = nil
		changed = true
	}
	if headerIndex := strings.Index(schema.Description, enumTypeDescriptionHeader); headerIndex != -1 {
		// remove the enum section from description.
		// note that the new lines before the header should be removed too,
		// thus the slice range.
		schema.Description = schema.Description[:headerIndex]
		changed = true
	}
	for k, v := range schema.Definitions {
		if pruneEnums(&v) {
			changed = true
			schema.Definitions[k] = v
		}
	}
	for k, v := range schema.Properties {
		if pruneEnums(&v) {
			changed = true
			schema.Properties[k] = v
		}
	}
	for k, v := range schema.PatternProperties {
		if pruneEnums(&v) {
			changed = true
			schema.Properties[k] = v
		}
	}
	for k, v := range schema.Dependencies {
		if v.Schema != nil {
			if pruneEnums(v.Schema) {
				changed = true
				schema.Dependencies[k] = v
			}
		}
	}
	for i, v := range schema.AllOf {
		if pruneEnums(&v) {
			changed = true
			schema.AllOf[i] = v
		}
	}
	for i, v := range schema.AnyOf {
		if pruneEnums(&v) {
			changed = true
			schema.AnyOf[i] = v
		}
	}
	for i, v := range schema.OneOf {
		if pruneEnums(&v) {
			changed = true
			schema.OneOf[i] = v
		}
	}
	if schema.Not != nil {
		if pruneEnums(schema.Not) {
			changed = true
		}
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		if pruneEnums(schema.AdditionalProperties.Schema) {
			changed = true
		}
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		if pruneEnums(schema.AdditionalItems.Schema) {
			changed = true
		}
	}
	if schema.Items != nil {
		if schema.Items.Schema != nil {
			if pruneEnums(schema.Items.Schema) {
				changed = true
			}
		} else {
			for i, v := range schema.Items.Schemas {
				if pruneEnums(&v) {
					changed = true
					schema.Items.Schemas[i] = v
				}
			}
		}
	}
	return
}
