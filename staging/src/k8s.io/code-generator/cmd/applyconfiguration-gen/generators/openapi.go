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

package generators

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	openapiv2 "github.com/google/gnostic/openapiv2"
	"k8s.io/gengo/types"
	utilproto "k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type typeModels struct {
	models           utilproto.Models
	gvkToOpenAPIType map[gvk]string
}

type gvk struct {
	group, version, kind string
}

func newTypeModels(openAPISchemaFilePath string, pkgTypes map[string]*types.Package) (*typeModels, error) {
	if len(openAPISchemaFilePath) == 0 {
		return emptyModels, nil // No Extract<type>() functions will be generated.
	}

	rawOpenAPISchema, err := os.ReadFile(openAPISchemaFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read openapi-schema file: %w", err)
	}

	// Read in the provided openAPI schema.
	openAPISchema := &spec.Swagger{}
	err = json.Unmarshal(rawOpenAPISchema, openAPISchema)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal typeModels JSON: %w", err)
	}

	// Build a mapping from openAPI type name to GVK.
	// Find the root types needed by by client-go for apply.
	gvkToOpenAPIType := map[gvk]string{}
	rootDefs := map[string]spec.Schema{}
	for _, p := range pkgTypes {
		gv := groupVersion(p)
		for _, t := range p.Types {
			tags := genclientTags(t)
			hasApply := tags.HasVerb("apply") || tags.HasVerb("applyStatus")
			if tags.GenerateClient && hasApply {
				openAPIType := friendlyName(typeName(t))
				gvk := gvk{
					group:   gv.Group.String(),
					version: gv.Version.String(),
					kind:    t.Name.Name,
				}
				rootDefs[openAPIType] = openAPISchema.Definitions[openAPIType]
				gvkToOpenAPIType[gvk] = openAPIType
			}
		}
	}

	// Trim the schema down to just the types needed by client-go for apply.
	requiredDefs := make(map[string]spec.Schema)
	for name, def := range rootDefs {
		requiredDefs[name] = def
		findReferenced(&def, openAPISchema.Definitions, requiredDefs)
	}
	openAPISchema.Definitions = requiredDefs

	// Convert the openAPI schema to the models format and validate it.
	models, err := toValidatedModels(openAPISchema)
	if err != nil {
		return nil, err
	}
	return &typeModels{models: models, gvkToOpenAPIType: gvkToOpenAPIType}, nil
}

var emptyModels = &typeModels{
	models:           &utilproto.Definitions{},
	gvkToOpenAPIType: map[gvk]string{},
}

func toValidatedModels(openAPISchema *spec.Swagger) (utilproto.Models, error) {
	// openapi_v2.ParseDocument only accepts a []byte of the JSON or YAML file to be parsed.
	// so we do an inefficient marshal back to json and then read it back in as yaml
	// but get the benefit of running the models through utilproto.NewOpenAPIData to
	// validate all the references between types
	rawMinimalOpenAPISchema, err := json.Marshal(openAPISchema)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal openAPI as JSON: %w", err)
	}

	document, err := openapiv2.ParseDocument(rawMinimalOpenAPISchema)
	if err != nil {
		return nil, fmt.Errorf("failed to parse OpenAPI document for file: %w", err)
	}
	// Construct the models and validate all references are valid.
	models, err := utilproto.NewOpenAPIData(document)
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAPI models for file: %w", err)
	}
	return models, nil
}

// findReferenced recursively finds all schemas referenced from the given def.
// toValidatedModels makes sure no references get missed.
func findReferenced(def *spec.Schema, allSchemas, referencedOut map[string]spec.Schema) {
	// follow $ref, if any
	refPtr := def.Ref.GetPointer()
	if refPtr != nil && !refPtr.IsEmpty() {
		name := refPtr.String()
		if !strings.HasPrefix(name, "/definitions/") {
			return
		}
		name = strings.TrimPrefix(name, "/definitions/")
		schema, ok := allSchemas[name]
		if !ok {
			panic(fmt.Sprintf("allSchemas schema is missing referenced type: %s", name))
		}
		if _, ok := referencedOut[name]; !ok {
			referencedOut[name] = schema
			findReferenced(&schema, allSchemas, referencedOut)
		}
	}

	// follow any nested schemas
	if def.Items != nil {
		if def.Items.Schema != nil {
			findReferenced(def.Items.Schema, allSchemas, referencedOut)
		}
		for _, item := range def.Items.Schemas {
			findReferenced(&item, allSchemas, referencedOut)
		}
	}
	if def.AllOf != nil {
		for _, s := range def.AllOf {
			findReferenced(&s, allSchemas, referencedOut)
		}
	}
	if def.AnyOf != nil {
		for _, s := range def.AnyOf {
			findReferenced(&s, allSchemas, referencedOut)
		}
	}
	if def.OneOf != nil {
		for _, s := range def.OneOf {
			findReferenced(&s, allSchemas, referencedOut)
		}
	}
	if def.Not != nil {
		findReferenced(def.Not, allSchemas, referencedOut)
	}
	if def.Properties != nil {
		for _, prop := range def.Properties {
			findReferenced(&prop, allSchemas, referencedOut)
		}
	}
	if def.AdditionalProperties != nil && def.AdditionalProperties.Schema != nil {
		findReferenced(def.AdditionalProperties.Schema, allSchemas, referencedOut)
	}
	if def.PatternProperties != nil {
		for _, s := range def.PatternProperties {
			findReferenced(&s, allSchemas, referencedOut)
		}
	}
	if def.Dependencies != nil {
		for _, d := range def.Dependencies {
			if d.Schema != nil {
				findReferenced(d.Schema, allSchemas, referencedOut)
			}
		}
	}
	if def.AdditionalItems != nil && def.AdditionalItems.Schema != nil {
		findReferenced(def.AdditionalItems.Schema, allSchemas, referencedOut)
	}
	if def.Definitions != nil {
		for _, s := range def.Definitions {
			findReferenced(&s, allSchemas, referencedOut)
		}
	}
}
