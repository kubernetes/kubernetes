/*
Copyright 2020 The Kubernetes Authors.

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

package handler

import "github.com/go-openapi/spec"

// PruneDefaults remove all the defaults recursively from all the
// schemas in the definitions, and does not modify the definitions in
// place.
func PruneDefaults(definitions spec.Definitions) spec.Definitions {
	definitionsCloned := false
	for k, v := range definitions {
		if s := PruneDefaultsSchema(&v); s != &v {
			if !definitionsCloned {
				definitionsCloned = true
				orig := definitions
				definitions = make(spec.Definitions, len(orig))
				for k2, v2 := range orig {
					definitions[k2] = v2
				}
			}
			definitions[k] = *s
		}
	}
	return definitions
}

// PruneDefaultsSchema remove all the defaults recursively from the
// schema in place.
func PruneDefaultsSchema(schema *spec.Schema) *spec.Schema {
	if schema == nil {
		return nil
	}

	orig := schema
	clone := func() {
		if orig == schema {
			schema = &spec.Schema{}
			*schema = *orig
		}
	}

	if schema.Default != nil {
		clone()
		schema.Default = nil
	}

	definitionsCloned := false
	for k, v := range schema.Definitions {
		if s := PruneDefaultsSchema(&v); s != &v {
			if !definitionsCloned {
				definitionsCloned = true
				clone()
				schema.Definitions = make(spec.Definitions, len(orig.Definitions))
				for k2, v2 := range orig.Definitions {
					schema.Definitions[k2] = v2
				}
			}
			schema.Definitions[k] = *s
		}
	}

	propertiesCloned := false
	for k, v := range schema.Properties {
		if s := PruneDefaultsSchema(&v); s != &v {
			if !propertiesCloned {
				propertiesCloned = true
				clone()
				schema.Properties = make(map[string]spec.Schema, len(orig.Properties))
				for k2, v2 := range orig.Properties {
					schema.Properties[k2] = v2
				}
			}
			schema.Properties[k] = *s
		}
	}

	patternPropertiesCloned := false
	for k, v := range schema.PatternProperties {
		if s := PruneDefaultsSchema(&v); s != &v {
			if !patternPropertiesCloned {
				patternPropertiesCloned = true
				clone()
				schema.PatternProperties = make(map[string]spec.Schema, len(orig.PatternProperties))
				for k2, v2 := range orig.PatternProperties {
					schema.PatternProperties[k2] = v2
				}
			}
			schema.PatternProperties[k] = *s
		}
	}

	dependenciesCloned := false
	for k, v := range schema.Dependencies {
		if s := PruneDefaultsSchema(v.Schema); s != v.Schema {
			if !dependenciesCloned {
				dependenciesCloned = true
				clone()
				schema.Dependencies = make(spec.Dependencies, len(orig.Dependencies))
				for k2, v2 := range orig.Dependencies {
					schema.Dependencies[k2] = v2
				}
			}
			v.Schema = s
			schema.Dependencies[k] = v
		}
	}

	allOfCloned := false
	for i := range schema.AllOf {
		if s := PruneDefaultsSchema(&schema.AllOf[i]); s != &schema.AllOf[i] {
			if !allOfCloned {
				allOfCloned = true
				clone()
				schema.AllOf = make([]spec.Schema, len(orig.AllOf))
				copy(schema.AllOf, orig.AllOf)
			}
			schema.AllOf[i] = *s
		}
	}

	anyOfCloned := false
	for i := range schema.AnyOf {
		if s := PruneDefaultsSchema(&schema.AnyOf[i]); s != &schema.AnyOf[i] {
			if !anyOfCloned {
				anyOfCloned = true
				clone()
				schema.AnyOf = make([]spec.Schema, len(orig.AnyOf))
				copy(schema.AnyOf, orig.AnyOf)
			}
			schema.AnyOf[i] = *s
		}
	}

	oneOfCloned := false
	for i := range schema.OneOf {
		if s := PruneDefaultsSchema(&schema.OneOf[i]); s != &schema.OneOf[i] {
			if !oneOfCloned {
				oneOfCloned = true
				clone()
				schema.OneOf = make([]spec.Schema, len(orig.OneOf))
				copy(schema.OneOf, orig.OneOf)
			}
			schema.OneOf[i] = *s
		}
	}

	if schema.Not != nil {
		if s := PruneDefaultsSchema(schema.Not); s != schema.Not {
			clone()
			schema.Not = s
		}
	}

	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		if s := PruneDefaultsSchema(schema.AdditionalProperties.Schema); s != schema.AdditionalProperties.Schema {
			clone()
			schema.AdditionalProperties = &spec.SchemaOrBool{Schema: s, Allows: schema.AdditionalProperties.Allows}
		}
	}

	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		if s := PruneDefaultsSchema(schema.AdditionalItems.Schema); s != schema.AdditionalItems.Schema {
			clone()
			schema.AdditionalItems = &spec.SchemaOrBool{Schema: s, Allows: schema.AdditionalItems.Allows}
		}
	}

	if schema.Items != nil {
		if schema.Items.Schema != nil {
			if s := PruneDefaultsSchema(schema.Items.Schema); s != schema.Items.Schema {
				clone()
				schema.Items = &spec.SchemaOrArray{Schema: s}
			}
		} else {
			itemsCloned := false
			for i := range schema.Items.Schemas {
				if s := PruneDefaultsSchema(&schema.Items.Schemas[i]); s != &schema.Items.Schemas[i] {
					if !itemsCloned {
						clone()
						schema.Items = &spec.SchemaOrArray{
							Schemas: make([]spec.Schema, len(orig.Items.Schemas)),
						}
						itemsCloned = true
						copy(schema.Items.Schemas, orig.Items.Schemas)
					}
					schema.Items.Schemas[i] = *s
				}
			}
		}
	}

	return schema
}
