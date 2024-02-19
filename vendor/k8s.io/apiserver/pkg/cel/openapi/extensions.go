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

package openapi

import (
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

var intOrStringFormat = intstr.IntOrString{}.OpenAPISchemaFormat()

func isExtension(schema *spec.Schema, key string) bool {
	v, ok := schema.Extensions.GetBool(key)
	return v && ok
}

func isXIntOrString(schema *spec.Schema) bool {
	// built-in types have the Format while CRDs use extension
	// both are valid, checking both
	return schema.Format == intOrStringFormat || isExtension(schema, extIntOrString)
}

func isXEmbeddedResource(schema *spec.Schema) bool {
	return isExtension(schema, extEmbeddedResource)
}

func isXPreserveUnknownFields(schema *spec.Schema) bool {
	return isExtension(schema, extPreserveUnknownFields)
}

func getXListType(schema *spec.Schema) string {
	s, _ := schema.Extensions.GetString(extListType)
	return s
}

func getXListMapKeys(schema *spec.Schema) []string {
	mapKeys, ok := schema.Extensions.GetStringSlice(extListMapKeys)
	if !ok {
		return nil
	}
	return mapKeys
}

const extIntOrString = "x-kubernetes-int-or-string"
const extEmbeddedResource = "x-kubernetes-embedded-resource"
const extPreserveUnknownFields = "x-kubernetes-preserve-unknown-fields"
const extListType = "x-kubernetes-list-type"
const extListMapKeys = "x-kubernetes-list-map-keys"
