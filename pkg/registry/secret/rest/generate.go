/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rest

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/validation"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	secretplugins "k8s.io/kubernetes/pkg/secret"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

// GenerateREST implements the generate endpoint for a Secret.
type GenerateREST struct {
	Store *etcdgeneric.Etcd
}

// GenerateREST implements Creater
var _ = rest.NamedCreater(&GenerateREST{})

// New creates a new generate secret request object.
func (r *GenerateREST) New() runtime.Object {
	return &api.GenerateSecretRequest{}
}

// Get generates a secret & returns it.
func (r *GenerateREST) Create(ctx api.Context, name string, obj runtime.Object) (runtime.Object, error) {
	req, ok := obj.(*api.GenerateSecretRequest)
	if !ok {
		return nil, fmt.Errorf("Invalid request object: %#v", obj)
	}
	if errs := validation.ValidateGenerateSecretRequest(req); len(errs) > 0 {
		return nil, errors.NewInvalid("generatesecretrequest", name, errs)
	}
	generator := secretplugins.GetPlugin(string(req.Type))
	if generator == nil {
		generators := secretplugins.GetPlugins()

		msg := fmt.Sprintf("unknown generator '%s'. Known generators are %v", req.Type, generators)
		err := fielderrors.NewFieldInvalid("type", req.Type, msg)
		errs := fielderrors.ValidationErrorList{err}
		return nil, errors.NewInvalid("generatesecretrequest", name, errs)
	}
	vals, err := generator.GenerateValues(req)
	if err != nil {
		return nil, err
	}
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:        name,
			Namespace:   api.NamespaceValue(ctx),
			Annotations: req.Annotations,
		},
		Type: req.Type,
		Data: vals,
	}
	// Store the secret
	s, err := r.Store.Create(ctx, secret)
	if err != nil {
		return nil, err
	}
	return s, nil
}
