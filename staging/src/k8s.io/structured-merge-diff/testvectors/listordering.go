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

package testvectors

import (
	"fmt"

	"gopkg.in/yaml.v2"
)

func init() {
	Schemas["listordering-primitiveInlineList"] = SchemaDefinition(`
type:
  inlineList:
    elementType:
      inlineScalar: string
  primitiveAssociative: true
`)

	var tests []*Vector
	err := yaml.Unmarshal([]byte(`
- name: list insert 1
  schemaName: listordering-primitiveInlineList
  lastObject: |
      items:
      - a
      - b
      - c
  liveObject: |
      items:
      - a
      - b
      - c
      - d
  newObject: |
      items:
      - a
      - b
      - e
      - c
  expectedObject: |
      items:
      - a
      - b
      - e
      - c
      - d
`), &tests)
	if err != nil {
		panic(fmt.Sprintf("test cases defined wrong: %v", err))
	}

	AppendTestVectors(tests...)
}
