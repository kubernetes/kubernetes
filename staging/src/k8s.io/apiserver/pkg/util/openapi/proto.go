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

	"github.com/go-openapi/spec"
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/kube-openapi/pkg/util/proto"
)

// ToProtoModels builds the proto formatted models from OpenAPI spec
func ToProtoModels(openAPISpec *spec.Swagger) (proto.Models, error) {
	specBytes, err := json.MarshalIndent(openAPISpec, " ", " ")
	if err != nil {
		return nil, err
	}

	var info yaml.MapSlice
	err = yaml.Unmarshal(specBytes, &info)
	if err != nil {
		return nil, err
	}

	doc, err := openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	if err != nil {
		return nil, err
	}

	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return nil, err
	}

	return models, nil
}
