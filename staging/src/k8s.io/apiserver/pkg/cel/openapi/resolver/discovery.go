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

package resolver

import (
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// ClientDiscoveryResolver uses client-go discovery to resolve schemas at run time.
type ClientDiscoveryResolver struct {
	Discovery discovery.DiscoveryInterface
}

var _ SchemaResolver = (*ClientDiscoveryResolver)(nil)

func (r *ClientDiscoveryResolver) ResolveSchema(gvk schema.GroupVersionKind) (*spec.Schema, error) {
	p, err := r.Discovery.OpenAPIV3().Paths()
	if err != nil {
		return nil, err
	}
	resourcePath := resourcePathFromGV(gvk.GroupVersion())
	c, ok := p[resourcePath]
	if !ok {
		return nil, fmt.Errorf("cannot resolve group version %q: %w", gvk.GroupVersion(), ErrSchemaNotFound)
	}
	b, err := c.Schema(runtime.ContentTypeJSON)
	if err != nil {
		return nil, err
	}
	resp := new(schemaResponse)
	err = json.Unmarshal(b, resp)
	if err != nil {
		return nil, err
	}
	s, err := resolveType(resp, gvk)
	if err != nil {
		return nil, err
	}
	s, err = populateRefs(func(ref string) (*spec.Schema, bool) {
		s, ok := resp.Components.Schemas[strings.TrimPrefix(ref, refPrefix)]
		return s, ok
	}, s)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func resolveType(resp *schemaResponse, gvk schema.GroupVersionKind) (*spec.Schema, error) {
	for _, s := range resp.Components.Schemas {
		var gvks []schema.GroupVersionKind
		err := s.Extensions.GetObject(extGVK, &gvks)
		if err != nil {
			return nil, err
		}
		for _, g := range gvks {
			if g == gvk {
				return s, nil
			}
		}
	}
	return nil, fmt.Errorf("cannot resolve group version kind %q: %w", gvk, ErrSchemaNotFound)
}

func resourcePathFromGV(gv schema.GroupVersion) string {
	var resourcePath string
	if len(gv.Group) == 0 {
		resourcePath = fmt.Sprintf("api/%s", gv.Version)
	} else {
		resourcePath = fmt.Sprintf("apis/%s/%s", gv.Group, gv.Version)
	}
	return resourcePath
}

type schemaResponse struct {
	Components struct {
		Schemas map[string]*spec.Schema `json:"schemas"`
	} `json:"components"`
}

const refPrefix = "#/components/schemas/"

const extGVK = "x-kubernetes-group-version-kind"
