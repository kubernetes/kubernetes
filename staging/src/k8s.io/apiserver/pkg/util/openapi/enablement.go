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
			pruneEnums(&def.Schema)
			defs[gvk] = def
		}
	}
}

func pruneEnums(schema *spec.Schema) {
	if headerIndex := strings.Index(schema.Description, enumTypeDescriptionHeader); headerIndex != -1 {
		// remove the enum section from description.
		// note that the new lines before the header should be removed too,
		// thus the slice range.
		schema.Description = schema.Description[:headerIndex]
	}
	schema.Enum = nil
	for name, prop := range schema.Properties {
		pruneEnums(&prop)
		schema.Properties[name] = prop
	}
}
