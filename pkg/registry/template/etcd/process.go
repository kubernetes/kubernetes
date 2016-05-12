/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/template"
	"k8s.io/kubernetes/pkg/runtime"
	//"gopkg.in/yaml.v2"
)

type ProcessREST struct {
	processor template.TemplateProcessor
	registry  template.Registry
}

// Constructor
func NewProcessRest(store *registry.Store) *ProcessREST {
	r := template.NewRegistry(store)
	p := template.NewTemplateProcessor(r)
	return &ProcessREST{processor: p, registry: r}
}

var _ rest.Creater = &ProcessREST{}

// Override=rest.Creater
func (r *ProcessREST) New() runtime.Object {
	return &extensions.TemplateParameters{}
}

// Override=rest.Creater
func (r *ProcessREST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	params, ok := obj.(*extensions.TemplateParameters)
	if !ok {
		return nil, fmt.Errorf("expected input object type to be TemplateParameters, but was %T", obj)
	}
	if errs := extvalidation.ValidateTemplateParams(params); len(errs) > 0 {
		return nil, errors.NewInvalid(extensions.Kind("TemplateParams"), params.Name, errs)
	}
	obj, err := r.processor.Process(ctx, params)
	//j, _ := yaml.Marshal(obj)
	//fmt.Printf("\n\n\n	R1: %s\n\n\n", j)
	return obj, err
}

var _ rest.Updater = &ProcessREST{}

// Override=rest.Updater
func (r *ProcessREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	obj, err := r.Create(ctx, obj)
	if err != nil {
		return obj, false, err
	}
	//j, _ := yaml.Marshal(obj)
	//fmt.Printf("\n\n\n	R2: %s\n\n\n", j)
	return obj, true, err
}
