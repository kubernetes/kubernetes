/*
Copyright 2017 The Kubernetes Authors.

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

package aggregator

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/go-openapi/spec"
	jsoniter "github.com/json-iterator/go"
	"github.com/stretchr/testify/assert"
	"k8s.io/kube-openapi/pkg/handler"
	"sigs.k8s.io/yaml"
)

type DebugSpec struct {
	*spec.Swagger
}

func (d DebugSpec) String() string {
	bytes, err := json.MarshalIndent(d.Swagger, "", " ")
	if err != nil {
		return fmt.Sprintf("DebugSpec.String failed: %s", err)
	}
	return string(bytes)
}
func TestFilterSpecs(t *testing.T) {
	var spec1, spec1_filtered *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec1_filtered)

	ast := assert.New(t)
	orig_spec1, _ := cloneSpec(spec1)
	new_spec1 := FilterSpecByPathsWithoutSideEffects(spec1, []string{"/test"})
	ast.Equal(DebugSpec{spec1_filtered}, DebugSpec{new_spec1})
	ast.Equal(DebugSpec{orig_spec1}, DebugSpec{spec1}, "unexpected mutation of input")
}

func TestFilterSpecsWithUnusedDefinitions(t *testing.T) {
	var spec1, spec1Filtered *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
  Unused:
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
  Unused:
    type: "object"
`), &spec1Filtered)

	ast := assert.New(t)
	orig_spec1, _ := cloneSpec(spec1)
	new_spec1 := FilterSpecByPathsWithoutSideEffects(spec1, []string{"/test"})
	ast.Equal(DebugSpec{spec1Filtered}, DebugSpec{new_spec1})
	ast.Equal(DebugSpec{orig_spec1}, DebugSpec{spec1}, "unexpected mutation of input")
}

func TestMergeSpecsSimple(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsEmptyDefinitions(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
      responses:
        405:
          description: "Invalid input"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
      responses:
        405:
          description: "Invalid input"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsEmptyPaths(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test2"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
  Test2:
    type: "object"
    properties:
      other:
        $ref: "#/definitions/Other"
  Other:
    type: "string"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsReuseModel(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsRenameModel(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "This Test has a description"
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
  InvalidInput:
    type: "string"
    format: "string"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      tags:
      - "test"
      summary: "Test API"
      operationId: "addTest"
      parameters:
      - in: "body"
        name: "body"
        description: "test object"
        required: true
        schema:
          $ref: "#/definitions/Test"
      responses:
        405:
          description: "Invalid input"
          $ref: "#/definitions/InvalidInput"
  /othertest:
    post:
      tags:
      - "test2"
      summary: "Test2 API"
      operationId: "addTest2"
      consumes:
      - "application/json"
      produces:
      - "application/xml"
      parameters:
      - in: "body"
        name: "body"
        description: "test2 object"
        required: true
        schema:
          $ref: "#/definitions/Test_v2"
definitions:
  Test:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "Status"
  Test_v2:
    description: "This Test has a description"
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
  InvalidInput:
    type: "string"
    format: "string"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1}, DebugSpec{spec1}.String())
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsRenameModelWithExistingV2InDestination(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /testv2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
definitions:
  Test:
    type: "object"
  Test_v2:
    description: "This is an existing Test_v2 in destination schema"
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "This Test has a description"
    type: "object"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /testv2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v3"
definitions:
  Test:
    type: "object"
  Test_v2:
    description: "This is an existing Test_v2 in destination schema"
    type: "object"
  Test_v3:
    description: "This Test has a description"
    type: "object"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func TestMergeSpecsRenameModelWithExistingV2InSource(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /testv2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
definitions:
  Test:
    description: "This Test has a description"
    type: "object"
  Test_v2:
    description: "This is an existing Test_v2 in source schema"
    type: "object"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /testv2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v3"
definitions:
  Test:
    type: "object"
  Test_v2:
    description: "This is an existing Test_v2 in source schema"
    type: "object"
  Test_v3:
    description: "This Test has a description"
    type: "object"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

// This tests if there are three specs, where the first two use the same object definition,
// while the third one uses its own.
// We expect the merged schema to contain two versions of the object, not three
func TestTwoMergeSpecsFirstTwoSchemasHaveSameDefinition(t *testing.T) {
	var spec1, spec2, spec3, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "spec1 and spec2 use the same object definition, while spec3 doesn't"
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "spec1 and spec2 use the same object definition, while spec3 doesn't"
    type: "object"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test3:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "spec3 has its own definition (the description doesn't match)"
    type: "object"
`), &spec3)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /test2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /test3:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
definitions:
  Test:
    description: "spec1 and spec2 use the same object definition, while spec3 doesn't"
    type: "object"
  Test_v2:
    description: "spec3 has its own definition (the description doesn't match)"
    type: "object"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	orig_spec3, _ := cloneSpec(spec3)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	if !ast.NoError(MergeSpecs(spec1, spec3)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of spec2 input")
	ast.Equal(DebugSpec{orig_spec3}, DebugSpec{spec3}, "unexpected mutation of spec3 input")
}

// This tests if there are three specs, where the last two use the same object definition,
// while the first one uses its own.
// We expect the merged schema to contain two versions of the object, not three
func TestTwoMergeSpecsLastTwoSchemasHaveSameDefinition(t *testing.T) {
	var spec1, spec2, spec3, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "spec2 and spec3 use the same object definition, while spec1 doesn't"
    type: "object"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /othertest2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    description: "spec2 and spec3 use the same object definition, while spec1 doesn't"
    type: "object"
`), &spec3)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /othertest:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
  /othertest2:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test_v2"
definitions:
  Test:
    type: "object"
  Test_v2:
    description: "spec2 and spec3 use the same object definition, while spec1 doesn't"
    type: "object"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	orig_spec3, _ := cloneSpec(spec3)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	if !ast.NoError(MergeSpecs(spec1, spec3)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of spec2 input")
	ast.Equal(DebugSpec{orig_spec3}, DebugSpec{spec3}, "unexpected mutation of spec3 input")

}

func TestSafeMergeSpecsSimple(t *testing.T) {
	var fooSpec, barSpec, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &fooSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /bar:
    post:
      summary: "Bar API"
      operationId: "barTest"
      parameters:
      - in: "body"
        name: "body"
        description: "bar object"
        required: true
        schema:
          $ref: "#/definitions/Bar"
      responses:
        200:
          description: "OK"
definitions:
  Bar:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &barSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
  /bar:
    post:
      summary: "Bar API"
      operationId: "barTest"
      parameters:
      - in: "body"
        name: "body"
        description: "bar object"
        required: true
        schema:
          $ref: "#/definitions/Bar"
      responses:
        200:
          description: "OK"
definitions:
    Foo:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
    Bar:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
  `), &expected)

	ast := assert.New(t)
	orig_barSpec, err := cloneSpec(barSpec)
	if !ast.NoError(err) {
		return
	}
	if !ast.NoError(MergeSpecsFailOnDefinitionConflict(fooSpec, barSpec)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{fooSpec})
	ast.Equal(DebugSpec{orig_barSpec}, DebugSpec{barSpec}, "unexpected mutation of input")
}

func TestSafeMergeSpecsReuseModel(t *testing.T) {
	var fooSpec, barSpec, expected *spec.Swagger
	if err := yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
    x-kubernetes-group-version-kind:
    - group: group1
      version: v1
      kind: Foo
    - group: group3
      version: v1
      kind: Foo
`), &fooSpec); err != nil {
		t.Fatal(err)
	}

	if err := yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /refoo:
    post:
      summary: "Refoo API"
      operationId: "refooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
    x-kubernetes-group-version-kind:
    - group: group2
      version: v1
      kind: Foo
`), &barSpec); err != nil {
		t.Fatal(err)
	}

	if err := yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
  /refoo:
    post:
      summary: "Refoo API"
      operationId: "refooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
    Foo:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
      x-kubernetes-group-version-kind:
      - group: group1
        version: v1
        kind: Foo
      - group: group2
        version: v1
        kind: Foo
      - group: group3
        version: v1
        kind: Foo
  `), &expected); err != nil {
		t.Fatal(err)
	}

	ast := assert.New(t)
	orig_barSpec, err := cloneSpec(barSpec)
	if !ast.NoError(err) {
		return
	}
	if !ast.NoError(MergeSpecsFailOnDefinitionConflict(fooSpec, barSpec)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{fooSpec})
	ast.Equal(DebugSpec{orig_barSpec}, DebugSpec{barSpec}, "unexpected mutation of input")
}

func TestSafeMergeSpecsReuseModelFails(t *testing.T) {
	var fooSpec, barSpec, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &fooSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /refoo:
    post:
      summary: "Refoo API"
      operationId: "refooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      new_field:
        type: "string"
`), &barSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
  /refoo:
    post:
      summary: "Refoo API"
      operationId: "refooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
    Foo:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
  `), &expected)

	ast := assert.New(t)
	ast.Error(MergeSpecsFailOnDefinitionConflict(fooSpec, barSpec))
}

func TestMergeSpecsIgnorePathConflicts(t *testing.T) {
	var fooSpec, barSpec, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &fooSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Should be ignored"
  /bar:
    post:
      summary: "Bar API"
      operationId: "barTest"
      parameters:
      - in: "body"
        name: "body"
        description: "bar object"
        required: true
        schema:
          $ref: "#/definitions/Bar"
      responses:
        200:
          description: "OK"
definitions:
  Bar:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &barSpec)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
  /bar:
    post:
      summary: "Bar API"
      operationId: "barTest"
      parameters:
      - in: "body"
        name: "body"
        description: "bar object"
        required: true
        schema:
          $ref: "#/definitions/Bar"
      responses:
        200:
          description: "OK"
definitions:
    Foo:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
    Bar:
      type: "object"
      properties:
        id:
          type: "integer"
          format: "int64"
  `), &expected)

	ast := assert.New(t)
	actual, _ := cloneSpec(fooSpec)
	orig_barSpec, _ := cloneSpec(barSpec)
	if !ast.Error(MergeSpecs(actual, barSpec)) {
		return
	}
	ast.Equal(DebugSpec{orig_barSpec}, DebugSpec{barSpec}, "unexpected mutation of input")

	actual, _ = cloneSpec(fooSpec)
	if !ast.NoError(MergeSpecsIgnorePathConflict(actual, barSpec)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{actual})
	ast.Equal(DebugSpec{orig_barSpec}, DebugSpec{barSpec}, "unexpected mutation of input")
}

func TestMergeSpecsIgnorePathConflictsAllConflicting(t *testing.T) {
	var fooSpec *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /foo:
    post:
      summary: "Foo API"
      operationId: "fooTest"
      parameters:
      - in: "body"
        name: "body"
        description: "foo object"
        required: true
        schema:
          $ref: "#/definitions/Foo"
      responses:
        200:
          description: "OK"
definitions:
  Foo:
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
`), &fooSpec)

	ast := assert.New(t)
	foo2Spec, _ := cloneSpec(fooSpec)
	actual, _ := cloneSpec(fooSpec)
	if !ast.NoError(MergeSpecsIgnorePathConflict(actual, foo2Spec)) {
		return
	}
	ast.Equal(DebugSpec{fooSpec}, DebugSpec{actual})
	ast.Equal(DebugSpec{fooSpec}, DebugSpec{foo2Spec}, "unexpected mutation of input")
}

func TestMergeSpecsIgnorePathConflictsWithKubeSpec(t *testing.T) {
	ast := assert.New(t)

	specs, expected := loadTestData()
	sp, specs := specs[0], specs[1:]

	origSpecs := make([]*spec.Swagger, len(specs))
	for i := range specs {
		cpy, err := cloneSpec(specs[i])
		if err != nil {
			t.Fatal(err)
		}
		ast.NoError(err)
		origSpecs[i] = cpy
	}

	for i := range specs {
		if err := MergeSpecsIgnorePathConflict(sp, specs[i]); err != nil {
			t.Fatalf("merging spec %d failed: %v", i, err)
		}
	}

	ast.Equal(DebugSpec{expected}, DebugSpec{sp})

	for i := range specs {
		ast.Equal(DebugSpec{origSpecs[i]}, DebugSpec{specs[i]}, "unexpected mutation of specs[%d]", i)
	}
}

func BenchmarkMergeSpecsIgnorePathConflictsWithKubeSpec(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()
	b.ResetTimer()

	specs, _ := loadTestData()
	start, specs := specs[0], specs[1:]

	for n := 0; n < b.N; n++ {
		sp, err := cloneSpec(start)
		if err != nil {
			b.Fatal(err)
		}

		b.StartTimer()
		for i := range specs {
			if err := MergeSpecsIgnorePathConflict(sp, specs[i]); err != nil {
				panic(err)
			}
		}

		specBytes, _ := jsoniter.Marshal(sp)
		var json map[string]interface{}
		if err := jsoniter.Unmarshal(specBytes, &json); err != nil {
			b.Fatal(err)
		}
		handler.ToProtoBinary(json)

		b.StopTimer()
	}
}

func TestMergeSpecReplacesAllPossibleRefs(t *testing.T) {
	var spec1, spec2, expected *spec.Swagger
	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
definitions:
  Test:
    type: "object"
    properties:
      foo:
        $ref: "#/definitions/TestProperty"
  TestProperty:
    type: "object"
`), &spec1)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test2:
    post:
      parameters:
      - name: "test2"
        schema:
          $ref: "#/definitions/Test2"
      - name: "test3"
        schema:
          $ref: "#/definitions/Test3"
      - name: "test4"
        schema:
          $ref: "#/definitions/Test4"
      - name: "test5"
        schema:
          $ref: "#/definitions/Test5"
definitions:
  Test2:
    $ref: "#/definitions/TestProperty"
  Test3:
    type: "object"
    properties:
      withRef:
        $ref: "#/definitions/TestProperty"
      withAllOf:
        type: "object"
        allOf:
        - $ref: "#/definitions/TestProperty"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty"
      withAnyOf:
        type: "object"
        anyOf:
        - $ref: "#/definitions/TestProperty"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty"
      withOneOf:
        type: "object"
        oneOf:
        - $ref: "#/definitions/TestProperty"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty"
      withNot:
        type: "object"
        not:
          $ref: "#/definitions/TestProperty"
    patternProperties:
      "prefix.*":
        $ref: "#/definitions/TestProperty"
    additionalProperties:
      $ref: "#/definitions/TestProperty"
    definitions:
      SomeDefinition:
        $ref: "#/definitions/TestProperty"
  Test4:
    type: "array"
    items:
      $ref: "#/definitions/TestProperty"
    additionalItems:
      $ref: "#/definitions/TestProperty"
  Test5:
    type: "array"
    items:
    - $ref: "#/definitions/TestProperty"
    - $ref: "#/definitions/TestProperty"
  TestProperty:
    description: "This TestProperty is different from the one in spec1"
    type: "object"
`), &spec2)

	yaml.Unmarshal([]byte(`
swagger: "2.0"
paths:
  /test:
    post:
      parameters:
      - name: "body"
        schema:
          $ref: "#/definitions/Test"
  /test2:
    post:
      parameters:
      - name: "test2"
        schema:
          $ref: "#/definitions/Test2"
      - name: "test3"
        schema:
          $ref: "#/definitions/Test3"
      - name: "test4"
        schema:
          $ref: "#/definitions/Test4"
      - name: "test5"
        schema:
          $ref: "#/definitions/Test5"
definitions:
  Test:
    type: "object"
    properties:
      foo:
        $ref: "#/definitions/TestProperty"
  TestProperty:
    type: "object"
  Test2:
    $ref: "#/definitions/TestProperty_v2"
  Test3:
    type: "object"
    properties:
      withRef:
        $ref: "#/definitions/TestProperty_v2"
      withAllOf:
        type: "object"
        allOf:
        - $ref: "#/definitions/TestProperty_v2"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty_v2"
      withAnyOf:
        type: "object"
        anyOf:
        - $ref: "#/definitions/TestProperty_v2"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty_v2"
      withOneOf:
        type: "object"
        oneOf:
        - $ref: "#/definitions/TestProperty_v2"
        - type: object
          properties:
            test:
              $ref: "#/definitions/TestProperty_v2"
      withNot:
        type: "object"
        not:
          $ref: "#/definitions/TestProperty_v2"
    patternProperties:
      "prefix.*":
        $ref: "#/definitions/TestProperty_v2"
    additionalProperties:
      $ref: "#/definitions/TestProperty_v2"
    definitions:
      SomeDefinition:
        $ref: "#/definitions/TestProperty_v2"
  Test4:
    type: "array"
    items:
      $ref: "#/definitions/TestProperty_v2"
    additionalItems:
      $ref: "#/definitions/TestProperty_v2"
  Test5:
    type: "array"
    items:
    - $ref: "#/definitions/TestProperty_v2"
    - $ref: "#/definitions/TestProperty_v2"
  TestProperty_v2:
    description: "This TestProperty is different from the one in spec1"
    type: "object"
`), &expected)

	ast := assert.New(t)
	orig_spec2, _ := cloneSpec(spec2)
	if !ast.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	ast.Equal(DebugSpec{expected}, DebugSpec{spec1})
	ast.Equal(DebugSpec{orig_spec2}, DebugSpec{spec2}, "unexpected mutation of input")
}

func loadTestData() ([]*spec.Swagger, *spec.Swagger) {
	loadSpec := func(fileName string) *spec.Swagger {
		bs, err := ioutil.ReadFile(filepath.Join("../../test/integration/testdata/aggregator", fileName))
		if err != nil {
			panic(err)
		}
		sp := spec.Swagger{}

		if err := json.Unmarshal(bs, &sp); err != nil {
			panic(err)
		}
		return &sp
	}

	specs := []*spec.Swagger{
		loadSpec("openapi-0.json"),
		loadSpec("openapi-1.json"),
		loadSpec("openapi-2.json"),
	}
	expected := loadSpec("openapi.json")

	return specs, expected
}

func TestCloneSpec(t *testing.T) {
	_, sp := loadTestData()
	clone, err := cloneSpec(sp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ast := assert.New(t)
	ast.Equal(DebugSpec{sp}, DebugSpec{clone})
}

func cloneSpec(source *spec.Swagger) (*spec.Swagger, error) {
	bytes, err := json.Marshal(source)
	if err != nil {
		return nil, err
	}
	var ret spec.Swagger
	err = json.Unmarshal(bytes, &ret)
	if err != nil {
		return nil, err
	}
	return &ret, nil
}

func TestMergedGVKs(t *testing.T) {
	gvk1 := map[string]interface{}{"group": "group1", "version": "v1", "kind": "Foo"}
	gvk2 := map[string]interface{}{"group": "group2", "version": "v1", "kind": "Bar"}
	gvk3 := map[string]interface{}{"group": "group3", "version": "v1", "kind": "Abc"}
	gvk4 := map[string]interface{}{"group": "group4", "version": "v1", "kind": "Abc"}

	tests := []struct {
		name        string
		gvks1       interface{}
		gvks2       interface{}
		want        interface{}
		wantChanged bool
		wantErr     bool
	}{
		{"nil", nil, nil, nil, false, false},
		{"first only", []interface{}{gvk1, gvk2}, nil, []interface{}{gvk1, gvk2}, false, false},
		{"second only", nil, []interface{}{gvk1, gvk2}, []interface{}{gvk1, gvk2}, true, false},
		{"both", []interface{}{gvk1, gvk2}, []interface{}{gvk3}, []interface{}{gvk1, gvk2, gvk3}, true, false},
		{"equal, different order", []interface{}{gvk1, gvk2, gvk3}, []interface{}{gvk3, gvk2, gvk1}, []interface{}{gvk1, gvk2, gvk3}, false, false},
		{"ordered", []interface{}{gvk3, gvk1, gvk4}, []interface{}{gvk2}, []interface{}{gvk1, gvk2, gvk3, gvk4}, true, false},
		{"not ordered when not changed", []interface{}{gvk3, gvk1, gvk4}, []interface{}{}, []interface{}{gvk3, gvk1, gvk4}, false, false},
		{"empty", []interface{}{}, []interface{}{}, []interface{}{}, false, false},
		{"overlapping", []interface{}{gvk1, gvk2}, []interface{}{gvk2, gvk3}, []interface{}{gvk1, gvk2, gvk3}, true, false},
		{"first no slice", 42, []interface{}{gvk1}, nil, false, true},
		{"second no slice", []interface{}{gvk1}, 42, nil, false, true},
		{"no map in slice", []interface{}{42}, []interface{}{gvk1}, nil, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ext1, ext2 map[string]interface{}
			if tt.gvks1 != nil {
				ext1 = map[string]interface{}{"x-kubernetes-group-version-kind": tt.gvks1}
			}
			if tt.gvks2 != nil {
				ext2 = map[string]interface{}{"x-kubernetes-group-version-kind": tt.gvks2}
			}

			got, gotChanged, gotErr := mergedGVKs(
				&spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: ext1}},
				&spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: ext2}},
			)
			if (gotErr != nil) != tt.wantErr {
				t.Errorf("mergedGVKs() error = %v, wantErr %v", gotErr, tt.wantErr)
				return
			}
			if gotChanged != tt.wantChanged {
				t.Errorf("mergedGVKs() changed = %v, want %v", gotChanged, tt.wantChanged)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mergedGVKs() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDeepEqualDefinitionsModuloGVKs(t *testing.T) {
	tests := []struct {
		name  string
		s1    *spec.Schema
		s2    *spec.Schema
		equal bool
	}{
		{name: "nil", equal: true},
		{name: "nil, non-nil", s1: nil, s2: &spec.Schema{}},
		{name: "equal", s1: &spec.Schema{}, s2: &spec.Schema{}, equal: true},
		{name: "different", s1: &spec.Schema{SchemaProps: spec.SchemaProps{ID: "abc"}}, s2: &spec.Schema{}},
		{name: "equal modulo: nil, empty",
			s1:    &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: nil}},
			s2:    &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{}}},
			equal: true,
		},
		{name: "equal modulo: nil, gvk",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: nil}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
			}}},
			equal: true,
		},
		{name: "equal modulo: empty, gvk",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{}}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
			}}},
			equal: true,
		},
		{name: "equal modulo: non-empty, gvk",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{"foo": "bar"}}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
				"foo":  "bar",
			}}},
			equal: true,
		},
		{name: "equal modulo: gvk, gvk",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: false,
				"foo":  "bar",
			}}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
				"foo":  "bar",
			}}},
			equal: true,
		},
		{name: "different values",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: false,
				"foo":  "bar",
			}}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
				"foo":  "abc",
			}}},
		},
		{name: "different sizes",
			s1: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: false,
				"foo":  "bar",
				"xyz":  "123",
			}}},
			s2: &spec.Schema{VendorExtensible: spec.VendorExtensible{Extensions: spec.Extensions{
				gvkKey: true,
				"foo":  "abc",
			}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := deepEqualDefinitionsModuloGVKs(tt.s1, tt.s2); got != tt.equal {
				t.Errorf("deepEqualDefinitionsModuloGVKs(s1, v2) = %v, want %v", got, tt.equal)
			}

			if got := deepEqualDefinitionsModuloGVKs(tt.s2, tt.s1); got != tt.equal {
				t.Errorf("deepEqualDefinitionsModuloGVKs(s2, s1) = %v, want %v", got, tt.equal)
			}
		})
	}
}
