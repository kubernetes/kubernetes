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

package openapi

import (
	"sync"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
	"k8s.io/client-go/discovery"
)

// CachedOpenAPIGetter fetches the openapi schema once and then caches it in memory
type CachedOpenAPIGetter struct {
	openAPIClient discovery.OpenAPISchemaInterface

	// Cached results
	sync.Once
	openAPISchema *openapi_v2.Document
	err           error
}

var _ discovery.OpenAPISchemaInterface = &CachedOpenAPIGetter{}

// NewOpenAPIGetter returns an object to return OpenAPIDatas which reads
// from a server, and then stores in memory for subsequent invocations
func NewOpenAPIGetter(openAPIClient discovery.OpenAPISchemaInterface) *CachedOpenAPIGetter {
	return &CachedOpenAPIGetter{
		openAPIClient: openAPIClient,
	}
}

// OpenAPISchema implements OpenAPISchemaInterface.
func (g *CachedOpenAPIGetter) OpenAPISchema() (*openapi_v2.Document, error) {
	g.Do(func() {
		g.openAPISchema, g.err = g.openAPIClient.OpenAPISchema()
	})

	// Return the saved result.
	return g.openAPISchema, g.err
}

type CachedOpenAPIParser struct {
	openAPIClient discovery.OpenAPISchemaInterface

	// Cached results
	sync.Once
	openAPIResources Resources
	err              error
}

func NewOpenAPIParser(openAPIClient discovery.OpenAPISchemaInterface) *CachedOpenAPIParser {
	return &CachedOpenAPIParser{
		openAPIClient: openAPIClient,
	}
}

func (p *CachedOpenAPIParser) Parse() (Resources, error) {
	p.Do(func() {
		oapi, err := p.openAPIClient.OpenAPISchema()
		if err != nil {
			p.err = err
			return
		}
		p.openAPIResources, p.err = NewOpenAPIData(oapi)
	})

	return p.openAPIResources, p.err
}
