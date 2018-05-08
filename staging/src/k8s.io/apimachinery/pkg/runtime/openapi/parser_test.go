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
	"fmt"
	"testing"

	"github.com/go-openapi/spec"
	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	fooSwaggerJSON = []byte(`{
  "swagger": "2.0",
  "info": {
   "title": "Kubernetes",
   "version": "v1.11.0"
  },
  "definitions": {
   "io.k8s.api.testgroup.v1.Kind": {
    "x-kubernetes-group-version-kind": [
     {
      "group": "testgroup.k8s.io",
      "kind": "Foo",
      "version": "v1"
     }
    ]
   }
  }}`)
	fooGVK = schema.GroupVersionKind{
		Group:   "testgroup.k8s.io",
		Version: "v1",
		Kind:    "Foo",
	}
)

// TestOpenAPIParserPBv2 creates a new SpecParser and makes sure it can parse an example proto formatted spec
// into a Resources objects and tests that the GVK lookup works.
func TestOpenAPIParserPBv2(t *testing.T) {
	parser := NewParserPBv2()
	fooSwaggerPB, err := jsonToPB(fooSwaggerJSON)
	if err != nil {
		t.Errorf("Error generating protobuf: %v\n", err)
	}
	resources, err := parser.Parse(fooSwaggerPB)
	if err != nil {
		t.Errorf("Unexpected error parsing spec: %v", err)
	}
	_, err = resources.LookupResource(fooGVK)
	if err != nil {
		t.Errorf("Error calling LookupResource: %v", err)
	}
}

func jsonToPB(swaggerJSON []byte) ([]byte, error) {
	var s spec.Swagger
	err := s.UnmarshalJSON(swaggerJSON)
	if err != nil {
		return nil, fmt.Errorf("Unexpected error in unmarshalling swaggerJSON: %v", err)
	}
	returnedJSON, err := json.MarshalIndent(s, " ", " ")
	if err != nil {
		return nil, fmt.Errorf("Unexpected error in preparing returnedJSON: %v", err)
	}
	var info yaml.MapSlice
	err = yaml.Unmarshal(returnedJSON, &info)
	if err != nil {
		return nil, err
	}
	document, err := openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	if err != nil {
		return nil, err
	}
	returnedPB, err := proto.Marshal(document)
	if err != nil {
		return nil, err
	}
	return returnedPB, nil
}
