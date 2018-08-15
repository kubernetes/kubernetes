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

// TODO: Use our actual schema type.
var primitiveInlineListSchema = Schema(`
type:
  inlineList:
    elementType:
      inlineScalar: string
  primitiveAssociative: true`)

func init() {
	Vectors = append(
		Vectors,
		&Vector{
			Name:   "list insert 1",
			Schema: primitiveInlineListSchema,
			LiveObject: YAMLObject(`
items:
- a
- b
- c
- d`),
			LastObject: YAMLObject(`
items:
- a
- b
- c
`),
			NewObject: YAMLObject(`
items:
- a
- b
- e
- c
`),
			ExpectedObject: YAMLObject(`
items:
- a
- b
- e
- c
- d`),
		},
	)
}
