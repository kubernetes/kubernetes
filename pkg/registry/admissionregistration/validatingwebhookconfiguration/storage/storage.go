/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/validatingwebhookconfiguration"
)

// REST implements a RESTStorage for validatingWebhookConfiguration against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against validatingWebhookConfiguration.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, error) {
	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &admissionregistration.ValidatingWebhookConfiguration{} },
		NewListFunc: func() runtime.Object { return &admissionregistration.ValidatingWebhookConfigurationList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*admissionregistration.ValidatingWebhookConfiguration).Name, nil
		},
		DefaultQualifiedResource:  admissionregistration.Resource("validatingwebhookconfigurations"),
		SingularQualifiedResource: admissionregistration.Resource("validatingwebhookconfiguration"),

		CreateStrategy: validatingwebhookconfiguration.Strategy,
		UpdateStrategy: validatingwebhookconfiguration.Strategy,
		DeleteStrategy: validatingwebhookconfiguration.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}
	return &REST{store}, nil
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"api-extensions"}
}
