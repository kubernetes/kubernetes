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

package openapi

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
)

type resourceSource struct {
	specSource SpecSource
	gvk        schema.GroupVersionKind
}

var _ ResourceSource = &resourceSource{}

// NewResourceSource creates a new ResourceSource for a specSource and gvk
func NewResourceSource(specSource SpecSource, gvk schema.GroupVersionKind) ResourceSource {
	return &resourceSource{
		specSource: specSource,
		gvk:        gvk,
	}
}

// Get implements ResourceSource
func (s *resourceSource) Get() (proto.Schema, error) {
	if s.specSource == nil {
		return nil, fmt.Errorf("unable to get OpenAPI spec: SpecSource is nil")
	}
	spec, err := s.specSource.Get()
	if err != nil {
		return nil, err
	}
	return spec.LookupResource(s.gvk)
}
