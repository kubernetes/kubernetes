/*
Copyright 2022 The Kubernetes Authors.

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

package apiserver

import (
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

func ScopeWithFieldManager(typeConverter managedfields.TypeConverter, reqScope handlers.RequestScope, resetFields map[fieldpath.APIVersion]*fieldpath.Set, subresource string) (handlers.RequestScope, error) {
	return scopeWithFieldManager(typeConverter, reqScope, resetFields, subresource)
}

func NewUnstructuredNegotiatedSerializer(
	typer runtime.ObjectTyper,
	creator runtime.ObjectCreater,
	converter runtime.ObjectConvertor,
	structuralSchemas map[string]*structuralschema.Structural, // by version
	structuralSchemaGK schema.GroupKind,
	preserveUnknownFields bool,
) unstructuredNegotiatedSerializer {
	return unstructuredNegotiatedSerializer{
		typer:                 typer,
		creator:               creator,
		converter:             converter,
		structuralSchemas:     structuralSchemas,
		structuralSchemaGK:    structuralSchemaGK,
		preserveUnknownFields: preserveUnknownFields,

		supportedMediaTypes: []runtime.SerializerInfo{
			{
				MediaType:        "application/json",
				MediaTypeType:    "application",
				MediaTypeSubType: "json",
				EncodesAsText:    true,
				Serializer:       json.NewSerializer(json.DefaultMetaFactory, creator, typer, false),
				PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, creator, typer, true),
				StrictSerializer: json.NewSerializerWithOptions(json.DefaultMetaFactory, creator, typer, json.SerializerOptions{
					Strict: true,
				}),
				StreamSerializer: &runtime.StreamSerializerInfo{
					EncodesAsText: true,
					Serializer:    json.NewSerializer(json.DefaultMetaFactory, creator, typer, false),
					Framer:        json.Framer,
				},
			},
			{
				MediaType:        "application/yaml",
				MediaTypeType:    "application",
				MediaTypeSubType: "yaml",
				EncodesAsText:    true,
				Serializer:       json.NewYAMLSerializer(json.DefaultMetaFactory, creator, typer),
				StrictSerializer: json.NewSerializerWithOptions(json.DefaultMetaFactory, creator, typer, json.SerializerOptions{
					Yaml:   true,
					Strict: true,
				}),
			},
		},
	}
}

func NewUnstructuredDefaulter(
	delegate runtime.ObjectDefaulter,
	structuralSchemas map[string]*structuralschema.Structural, // by version
	structuralSchemaGK schema.GroupKind,
) unstructuredDefaulter {
	return unstructuredDefaulter{
		delegate:           delegate,
		structuralSchemas:  structuralSchemas,
		structuralSchemaGK: structuralSchemaGK,
	}
}
