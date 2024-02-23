/*
Copyright 2024 The Kubernetes Authors.

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

package schemawatcher

import (
	"crypto/sha512"
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const gvkExtensionName = "x-kubernetes-group-version-kind"

type schemaWithHash struct {
	Schema *spec.Schema
	Hash   [32]byte
}

// parseOpenAPIv3Doc parses the OpenAPI v3 doc, and returns all parsed schemas and their corresponding hashes.
//
// The hash is calculated based on the raw binary JSON form of the schema.
// Even if the schema is schematically unchanged, for example, when only the order of fields changes or whitespace-only
// changes, the hash will change. However, the API server always returns "minimized" JSON, so this is not a problem.
func parseOpenAPIv3Doc(doc []byte) []schemaWithHash {
	var rawDoc struct {
		Components struct {
			Schemas map[string]json.RawMessage `json:"schemas"`
		} `json:"components"`
	}
	err := json.Unmarshal(doc, &rawDoc)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("fail to parse OpenAPI v3 document: %w", err))
		return nil
	}
	var result []schemaWithHash
	for _, rawSchema := range rawDoc.Components.Schemas {
		s := new(spec.Schema)
		err := json.Unmarshal(rawSchema, &s)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("fail to parse OpenAPI v3 schema: %w", err))
		}
		hash := sha512.Sum512_256(rawSchema)
		result = append(result, schemaWithHash{Hash: hash, Schema: s})
	}
	return result
}

func parseOpenAPIPathItem(path string, gv openapi.GroupVersion) (groupVersion schema.GroupVersion, hash string, err error) {
	groupVersion, err = openapi3.PathToGroupVersion(path)
	if err != nil {
		return
	}
	hash, err = gv.Hash()
	if err != nil {
		return schema.GroupVersion{}, "", fmt.Errorf("%w: %w", ErrNoHash, err)
	}
	if hash == "" {
		return schema.GroupVersion{}, "", fmt.Errorf("%w: empty hash", ErrNoHash)
	}
	return
}
