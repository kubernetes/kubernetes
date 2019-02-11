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
	"reflect"
	"testing"

	"github.com/go-openapi/spec"

	"k8s.io/kube-openapi/pkg/util/proto"
)

// TestOpenAPIDefinitionsToProtoSchema tests the openapi parser
func TestOpenAPIDefinitionsToProtoModels(t *testing.T) {
	openAPISpec := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger: "2.0",
			Info: &spec.Info{
				InfoProps: spec.InfoProps{
					Title:   "Kubernetes",
					Version: "0.0.0",
				},
			},
			Definitions: spec.Definitions{
				"io.k8s.api.testgroup.v1.Foo": spec.Schema{
					SchemaProps: spec.SchemaProps{
						Description: "Description of Foos",
						Properties:  map[string]spec.Schema{},
					},
					VendorExtensible: spec.VendorExtensible{
						Extensions: spec.Extensions{
							"x-kubernetes-group-version-kind": []map[string]string{
								{
									"group":   "testgroup.k8s.io",
									"version": "v1",
									"kind":    "Foo",
								},
							},
						},
					},
				},
			},
		},
	}
	expectedSchema := &proto.Arbitrary{
		BaseSchema: proto.BaseSchema{
			Description: "Description of Foos",
			Extensions: map[string]interface{}{
				"x-kubernetes-group-version-kind": []interface{}{
					map[interface{}]interface{}{
						"group":   "testgroup.k8s.io",
						"version": "v1",
						"kind":    "Foo",
					},
				},
			},
			Path: proto.NewPath("io.k8s.api.testgroup.v1.Foo"),
		},
	}
	protoModels, err := ToProtoModels(openAPISpec)
	if err != nil {
		t.Fatalf("expected ToProtoModels not to return an error")
	}
	actualSchema := protoModels.LookupModel("io.k8s.api.testgroup.v1.Foo")
	if !reflect.DeepEqual(expectedSchema, actualSchema) {
		t.Fatalf("expected schema:\n%v\nbut got:\n%v", expectedSchema, actualSchema)
	}
}
