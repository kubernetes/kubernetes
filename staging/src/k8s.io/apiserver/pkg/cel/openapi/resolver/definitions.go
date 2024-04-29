/*
Copyright 2023 The Kubernetes Authors.

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

package resolver

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/openapi"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// DefinitionsSchemaResolver resolves the schema of a built-in type
// by looking up the OpenAPI definitions.
type DefinitionsSchemaResolver struct {
	defs     map[string]common.OpenAPIDefinition
	gvkToRef map[schema.GroupVersionKind]string
}

// NewDefinitionsSchemaResolver creates a new DefinitionsSchemaResolver.
// An example working setup:
// getDefinitions = "k8s.io/kubernetes/pkg/generated/openapi".GetOpenAPIDefinitions
// scheme         = "k8s.io/client-go/kubernetes/scheme".Scheme
func NewDefinitionsSchemaResolver(getDefinitions common.GetOpenAPIDefinitions, schemes ...*runtime.Scheme) *DefinitionsSchemaResolver {
	gvkToRef := make(map[schema.GroupVersionKind]string)
	namer := openapi.NewDefinitionNamer(schemes...)
	defs := getDefinitions(func(path string) spec.Ref {
		return spec.MustCreateRef(path)
	})
	for name := range defs {
		_, e := namer.GetDefinitionName(name)
		gvks := extensionsToGVKs(e)
		for _, gvk := range gvks {
			gvkToRef[gvk] = name
		}
	}
	return &DefinitionsSchemaResolver{
		gvkToRef: gvkToRef,
		defs:     defs,
	}
}

func (d *DefinitionsSchemaResolver) LookupSchema(ref string) (*spec.Schema, error) {
	s, err := PopulateRefs(func(ref string) (*spec.Schema, bool) {
		// find the schema by the ref string, and return a deep copy
		def, ok := d.defs[ref]
		if !ok {
			return nil, false
		}
		s := def.Schema

		//!TODO: Fix codegen to do this
		//!TODO: I think this bug also affects VAP? int-or-string fields
		// have no type and are ignored by CEL decltype construction
		if len(s.Type) == 0 && len(s.OneOf) == 2 && len(s.OneOf[0].Type) > 0 && len(s.OneOf[1].Type) > 0 {
			oneOfTypes := sets.New[string](s.OneOf[0].Type[0], s.OneOf[1].Type[0])
			if oneOfTypes.Has("string") && (oneOfTypes.Has("number") || oneOfTypes.Has("integer")) {
				extCopy := make(spec.Extensions, len(s.Extensions))
				for k, v := range s.Extensions {
					extCopy[k] = v
				}
				s.Extensions = extCopy
				s.AddExtension("x-kubernetes-int-or-string", true)

				// OneOf is not valid in structural schema, so better to avoid it
				// in favor of the x-kubernetes extension
				if len(oneOfTypes) == 2 {
					s.OneOf = nil
				}
			}
		}

		// Native type schemas for now may use unsupported formats that
		// should be strippe such as int-or-string
		//!TODO: move this somewhere else like some sort of schema visitor
		validation.StripUnsupportedFormatsPostProcess(&s)
		return &s, true
	}, ref)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (d *DefinitionsSchemaResolver) ResolveSchema(gvk schema.GroupVersionKind) (*spec.Schema, error) {
	ref, ok := d.gvkToRef[gvk]
	if !ok {
		return nil, fmt.Errorf("cannot resolve %v: %w", gvk, ErrSchemaNotFound)
	}
	return d.LookupSchema(ref)
}

func extensionsToGVKs(extensions spec.Extensions) []schema.GroupVersionKind {
	gvksAny, ok := extensions[extGVK]
	if !ok {
		return nil
	}
	gvks, ok := gvksAny.([]any)
	if !ok {
		return nil
	}
	result := make([]schema.GroupVersionKind, 0, len(gvks))
	for _, gvkAny := range gvks {
		// type check the map and all fields
		gvkMap, ok := gvkAny.(map[string]any)
		if !ok {
			return nil
		}
		g, ok := gvkMap["group"].(string)
		if !ok {
			return nil
		}
		v, ok := gvkMap["version"].(string)
		if !ok {
			return nil
		}
		k, ok := gvkMap["kind"].(string)
		if !ok {
			return nil
		}
		result = append(result, schema.GroupVersionKind{
			Group:   g,
			Version: v,
			Kind:    k,
		})
	}
	return result
}
