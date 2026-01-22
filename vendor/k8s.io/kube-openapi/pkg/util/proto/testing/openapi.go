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

package testing

import (
	"os"
	"sync"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
)

// Fake opens and returns a openapi swagger from a file Path. It will
// parse only once and then return the same copy everytime.
type Fake struct {
	Path string

	once     sync.Once
	document *openapi_v2.Document
	err      error
}

// OpenAPISchema returns the openapi document and a potential error.
func (f *Fake) OpenAPISchema() (*openapi_v2.Document, error) {
	f.once.Do(func() {
		_, err := os.Stat(f.Path)
		if err != nil {
			f.err = err
			return
		}
		spec, err := os.ReadFile(f.Path)
		if err != nil {
			f.err = err
			return
		}
		f.document, f.err = openapi_v2.ParseDocument(spec)
	})
	return f.document, f.err
}

type Empty struct{}

func (Empty) OpenAPISchema() (*openapi_v2.Document, error) {
	return nil, nil
}
