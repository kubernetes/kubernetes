// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
Package validate provides methods to validate a swagger specification,
as well as tools to validate data against their schema.

This package follows Swagger 2.0. specification (aka OpenAPI 2.0). Reference
can be found here: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md.

Validating a specification

Validates a spec document (from JSON or YAML) against the JSON schema for swagger,
then checks a number of extra rules that can't be expressed in JSON schema.

Entry points:
  - Spec()
  - NewSpecValidator()
  - SpecValidator.Validate()

Reported as errors:
  [x] definition can't declare a property that's already defined by one of its ancestors
  [x] definition's ancestor can't be a descendant of the same model
  [x] path uniqueness: each api path should be non-verbatim (account for path param names) unique per method
  [x] each security reference should contain only unique scopes
  [x] each security scope in a security definition should be unique
  [x] parameters in path must be unique
  [x] each path parameter must correspond to a parameter placeholder and vice versa
  [x] each referenceable definition must have references
  [x] each definition property listed in the required array must be defined in the properties of the model
  [x] each parameter should have a unique `name` and `type` combination
  [x] each operation should have only 1 parameter of type body
  [x] each reference must point to a valid object
  [x] every default value that is specified must validate against the schema for that property
  [x] items property is required for all schemas/definitions of type `array`
  [x] path parameters must be declared a required
  [x] headers must not contain $ref
  [x] schema and property examples provided must validate against their respective object's schema
  [x] examples provided must validate their schema

Reported as warnings:
  [x] path parameters should not contain any of [{,},\w]
  [x] empty path
  [x] unused definitions
  [x] unsupported validation of examples on non-JSON media types
  [x] examples in response without schema
  [x] readOnly properties should not be required

Validating a schema

The schema validation toolkit validates data against JSON-schema-draft 04 schema.

It is tested against the full json-schema-testing-suite (https://github.com/json-schema-org/JSON-Schema-Test-Suite),
except for the optional part (bignum, ECMA regexp, ...).

It supports the complete JSON-schema vocabulary, including keywords not supported by Swagger (e.g. additionalItems, ...)

Entry points:
  - AgainstSchema()
  - ...

Known limitations

With the current version of this package, the following aspects of swagger are not yet supported:
  [ ] errors and warnings are not reported with key/line number in spec
  [ ] default values and examples on responses only support application/json producer type
  [ ] invalid numeric constraints (such as Minimum, etc..) are not checked except for default and example values
  [ ] rules for collectionFormat are not implemented
  [ ] no validation rule for polymorphism support (discriminator) [not done here]
  [ ] valid js ECMA regexp not supported by Go regexp engine are considered invalid
  [ ] arbitrary large numbers are not supported: max is math.MaxFloat64

*/
package validate
