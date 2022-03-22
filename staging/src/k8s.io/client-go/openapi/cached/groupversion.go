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

package cached

import (
	"sync"

	openapi_v3 "github.com/google/gnostic/openapiv3"
	"k8s.io/client-go/openapi"
)

type groupversion struct {
	delegate openapi.GroupVersion
	once     sync.Once
	doc      *openapi_v3.Document
	err      error
}

func newGroupVersion(delegate openapi.GroupVersion) *groupversion {
	return &groupversion{
		delegate: delegate,
	}
}

func (g *groupversion) Schema() (*openapi_v3.Document, error) {
	g.once.Do(func() {
		g.doc, g.err = g.delegate.Schema()
	})

	return g.doc, g.err
}
