/*
Copyright 2023 The Kubernetes Authors.

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

package testing

import (
	"os"
	"sync"

	"k8s.io/kube-openapi/pkg/spec3"
)

type OpenAPIV3Getter struct {
	Path      string
	once      sync.Once
	bytes     []byte
	openapiv3 spec3.OpenAPI
}

func (f *OpenAPIV3Getter) SchemaBytesOrDie() []byte {
	f.once.Do(func() {
		_, err := os.Stat(f.Path)
		if err != nil {
			panic(err)
		}
		spec, err := os.ReadFile(f.Path)
		if err != nil {
			panic(err)
		}
		f.bytes = spec
	})
	return f.bytes
}

func (f *OpenAPIV3Getter) SchemaOrDie() *spec3.OpenAPI {
	f.once.Do(func() {
		_, err := os.Stat(f.Path)
		if err != nil {
			panic(err)
		}
		spec, err := os.ReadFile(f.Path)
		if err != nil {
			panic(err)
		}

		err = f.openapiv3.UnmarshalJSON(spec)
		if err != nil {
			panic(err)
		}
	})
	return &f.openapiv3
}
