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

package schema

// SchemaSchemaYAML is a schema against which you can validate other schemas.
// It will validate itself. It can be unmarshalled into a Schema type.
var SchemaSchemaYAML = `types:
- name: schema
  struct:
    fields:
      - name: types
        type:
          list:
            elementRelationship: associative
            elementType:
              namedType: typeDef
            keys:
            - name
- name: typeDef
  struct:
    fields:
    - name: name
      type:
        scalar: string
    - name: scalar
      type:
        scalar: string
    - name: struct
      type:
        namedType: struct
    - name: list
      type:
        namedType: list
    - name: map
      type:
        namedType: map
    - name: untyped
      type:
        namedType: untyped
- name: typeRef
  struct:
    fields:
    - name: namedType
      type:
        scalar: string
    - name: scalar
      type:
        scalar: string
    - name: struct
      type:
        namedType: struct
    - name: list
      type:
        namedType: list
    - name: map
      type:
        namedType: map
    - name: untyped
      type:
        namedType: untyped
- name: scalar
  scalar: string
- name: struct
  struct:
    fields:
    - name: fields
      type:
        list:
          elementType:
            namedType: structField
          elementRelationship: associative
          keys: [ "name" ]
    - name: elementRelationship
      type:
        scalar: string
- name: structField
  struct:
    fields:
    - name: name
      type:
        scalar: string
    - name: type
      type:
        namedType: typeRef
- name: list
  struct:
    fields:
    - name: elementType
      type:
        namedType: typeRef
    - name: elementRelationship
      type:
        scalar: string
    - name: keys
      type:
        list:
          elementType:
            scalar: string
- name: map
  struct:
    fields:
    - name: elementType
      type:
        namedType: typeRef
    - name: elementRelationship
      type:
        scalar: string
- name: untyped
  struct:
    fields:
    - name: elementRelationship
      type:
        scalar: string
`
