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
	"testing"

	"github.com/ghodss/yaml"
	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
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
	assert := assert.New(t)
	FilterSpecByPaths(spec1, []string{"/test"})
	assert.Equal(DebugSpec{spec1_filtered}, DebugSpec{spec1})
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
	assert := assert.New(t)
	FilterSpecByPaths(spec1, []string{"/test"})
	assert.Equal(DebugSpec{spec1Filtered}, DebugSpec{spec1})
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
	assert.Equal(DebugSpec{expected}, DebugSpec{spec1})
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
	assert := assert.New(t)
	if !assert.NoError(MergeSpecs(spec1, spec2)) {
		return
	}
	assert.Equal(DebugSpec{expected}, DebugSpec{spec1})
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
	assert.Equal(DebugSpec{expected}, DebugSpec{spec1})
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
	assert := assert.New(t)
	actual, err := CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	if !assert.NoError(MergeSpecsFailOnDefinitionConflict(actual, barSpec)) {
		return
	}
	assert.Equal(DebugSpec{expected}, DebugSpec{actual})
}

func TestSafeMergeSpecsReuseModel(t *testing.T) {
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
	assert := assert.New(t)
	actual, err := CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	if !assert.NoError(MergeSpecsFailOnDefinitionConflict(actual, barSpec)) {
		return
	}
	assert.Equal(DebugSpec{expected}, DebugSpec{actual})
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
	assert := assert.New(t)
	actual, err := CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	assert.Error(MergeSpecsFailOnDefinitionConflict(actual, barSpec))
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
	assert := assert.New(t)
	actual, err := CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	if !assert.Error(MergeSpecs(actual, barSpec)) {
		return
	}
	actual, err = CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	if !assert.NoError(MergeSpecsIgnorePathConflict(actual, barSpec)) {
		return
	}
	assert.Equal(DebugSpec{expected}, DebugSpec{actual})
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
	assert := assert.New(t)
	foo2Spec, err := CloneSpec(fooSpec)
	actual, err := CloneSpec(fooSpec)
	if !assert.NoError(err) {
		return
	}
	if !assert.NoError(MergeSpecsIgnorePathConflict(actual, foo2Spec)) {
		return
	}
	assert.Equal(DebugSpec{fooSpec}, DebugSpec{actual})
}
