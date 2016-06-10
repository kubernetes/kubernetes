/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package template

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

type Registry interface {
	GetTemplate(ctx api.Context, templateName string) (*extensions.Template, error)
}

var _ Registry = &registry{}

// storage puts strong typing around storage calls
type registry struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &registry{s}
}

func (r *registry) GetTemplate(ctx api.Context, templateName string) (*extensions.Template, error) {
	obj, err := r.Get(ctx, templateName)
	if err != nil {
		return nil, err
	}
	return obj.(*extensions.Template), nil
}
