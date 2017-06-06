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

package openapi

import (
	"testing"

	"github.com/ghodss/yaml"
	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
)

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
	assert := assert.New(t)
	FilterSpecByPaths(spec1, []string{"/test"})
	assert.Equal(spec1_filtered, spec1)
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
	assert := assert.New(t)
	if !assert.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	assert.Equal(expected, spec1)
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
    description: "This Test has a description"
    type: "object"
    properties:
      id:
        type: "integer"
        format: "int64"
      status:
        type: "string"
        description: "This status has another description"
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
	assert := assert.New(t)
	if !assert.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	assert.Equal(expected, spec1)
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
	assert := assert.New(t)
	if !assert.NoError(MergeSpecs(spec1, spec2)) {
		return
	}

	expected_yaml, _ := yaml.Marshal(expected)
	spec1_yaml, _ := yaml.Marshal(spec1)

	assert.Equal(string(expected_yaml), string(spec1_yaml))
}
