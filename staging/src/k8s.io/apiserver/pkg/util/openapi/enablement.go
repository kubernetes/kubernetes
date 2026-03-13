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
	"k8s.io/kube-openapi/pkg/schemamutation"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// enumTypeDescriptionHeader is the header of enum section in schema description.
const enumTypeDescriptionHeader = "Possible enum values:"

// GetOpenAPIDefinitionsWithoutDisabledFeatures wraps a GetOpenAPIDefinitions to revert
// any change to the schema that was made by disabled features.
func GetOpenAPIDefinitionsWithoutDisabledFeatures(GetOpenAPIDefinitions common.GetOpenAPIDefinitions) common.GetOpenAPIDefinitions {
	return func(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
		defs := GetOpenAPIDefinitions(ref)
		restoreDefinitions(defs)
		return defs
	}
}

// restoreDefinitions restores any changes by disabled features from definition map.
func restoreDefinitions(defs map[string]common.OpenAPIDefinition) {
	// revert changes from OpenAPIEnums
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.OpenAPIEnums) {
		for gvk, def := range defs {
			orig := &def.Schema
			if ret := pruneEnums(orig); ret != orig {
				def.Schema = *ret
				defs[gvk] = def
			}
		}
	}
}

func pruneEnums(schema *spec.Schema) *spec.Schema {
	walker := schemamutation.Walker{
		SchemaCallback: func(schema *spec.Schema) *spec.Schema {
			orig := schema
			clone := func() {
				if orig == schema { // if schema has not been mutated yet
					schema = new(spec.Schema)
					*schema = *orig // make a clone from orig to schema
				}
			}
			if headerIndex := strings.Index(schema.Description, enumTypeDescriptionHeader); headerIndex != -1 {
				// remove the enum section from description.
				// note that the new lines before the header should be removed too,
				// thus the slice range.
				clone()
				schema.Description = strings.TrimSpace(schema.Description[:headerIndex])
			}
			if len(schema.Enum) != 0 {
				// remove the enum field
				clone()
				schema.Enum = nil
			}
			return schema
		},
		RefCallback: schemamutation.RefCallbackNoop,
	}
	return walker.WalkSchema(schema)
}
