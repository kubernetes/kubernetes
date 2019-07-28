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
Package analysis provides methods to work with a Swagger specification document from
package go-openapi/spec.

Analyzing a specification

An analysed specification object (type Spec) provides methods to work with swagger definition.

Flattening or expanding a specification

Flattening a specification bundles all remote $ref in the main spec document.
Depending on flattening options, additional preprocessing may take place:
  - full flattening: replacing all inline complex constructs by a named entry in #/definitions
  - expand: replace all $ref's in the document by their expanded content

Merging several specifications

Mixin several specifications merges all Swagger constructs, and warns about found conflicts.

Fixing a specification

Unmarshalling a specification with golang json unmarshalling may lead to
some unwanted result on present but empty fields.

Analyzing a Swagger schema

Swagger schemas are analyzed to determine their complexity and qualify their content.
*/
package analysis
