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

	openapi_v2 "github.com/google/gnostic-models/openapiv2"

	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// ToProtoModels builds the proto formatted models from OpenAPI spec
func ToProtoModels(openAPISpec *spec.Swagger) (proto.Models, error) {
	specBytes, err := json.MarshalIndent(openAPISpec, " ", " ")
	if err != nil {
		return nil, err
	}

	doc, err := openapi_v2.ParseDocument(specBytes)
	if err != nil {
		return nil, err
	}

	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return nil, err
	}

	return models, nil
}
