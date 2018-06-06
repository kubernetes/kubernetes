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

package apply

// This package is used for creating and applying patches generated
// from a last recorded config, local config, remote object.
// Example usage for a test:
//
//fakeSchema := tst.Fake{Path: swaggerPath}
//s, err := fakeSchema.OpenAPISchema()
//Expect(err).To(BeNil())
//resources, err := openapi.NewOpenAPIData(s)
//Expect(err).To(BeNil())
//elementParser := parse.Factory{resources}
//
//obj, err := parser.CreateElement(recorded, local, remote)
//Expect(err).Should(Not(HaveOccurred()))
//
//merged, err := obj.Merge(strategy.Create(strategy.Options{}))
//
