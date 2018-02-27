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

	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	"k8s.io/client-go/discovery"
)

// synchronizedOpenAPIGetter fetches the openapi schema once and then caches it in memory
type synchronizedOpenAPIGetter struct {
	// Cached results
	sync.Once
	openAPISchema Resources
	err           error

	openAPIClient discovery.DiscoveryInterface
}

var _ Getter = &synchronizedOpenAPIGetter{}

// Getter is an interface for fetching openapi specs and parsing them into an Resources struct
type Getter interface {
	// OpenAPIData returns the parsed OpenAPIData
	Get() (Resources, error)
}

// NewOpenAPIGetter returns an object to return OpenAPIDatas which reads
// from a server, and then stores in memory for subsequent invocations
func NewOpenAPIGetter(openAPIClient discovery.DiscoveryInterface) Getter {
	return &synchronizedOpenAPIGetter{
		openAPIClient: openAPIClient,
	}
}

func getOpenAPISchema(d discovery.DiscoveryInterface) (*openapi_v2.Document, error) {
	data, err := d.RESTClient().Get().AbsPath(discovery.OpenAPIV2SchemaPath).Do().Raw()
	if err != nil {
		return nil, err
	}
	document := &openapi_v2.Document{}
	err = proto.Unmarshal(data, document)
	if err != nil {
		return nil, err
	}
	return document, nil
}

// Resources implements Getter
func (g *synchronizedOpenAPIGetter) Get() (Resources, error) {
	g.Do(func() {
		s, err := getOpenAPISchema(g.openAPIClient)
		if err != nil {
			g.err = err
			return
		}

		g.openAPISchema, g.err = NewOpenAPIData(s)
	})

	// Return the save result
	return g.openAPISchema, g.err
}
