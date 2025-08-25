/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/resource/resourceclaimtemplate"
)

// REST implements a RESTStorage for ResourceClaimTemplate.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ResourceClass.
func NewREST(optsGetter generic.RESTOptionsGetter, nsClient v1.NamespaceInterface) (*REST, error) {
	if nsClient == nil {
		return nil, fmt.Errorf("namespace client is required")
	}
	strategy := resourceclaimtemplate.NewStrategy(nsClient)
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &resource.ResourceClaimTemplate{} },
		NewListFunc:               func() runtime.Object { return &resource.ResourceClaimTemplateList{} },
		DefaultQualifiedResource:  resource.Resource("resourceclaimtemplates"),
		SingularQualifiedResource: resource.Resource("resourceclaimtemplate"),

		CreateStrategy:      strategy,
		UpdateStrategy:      strategy,
		DeleteStrategy:      strategy,
		ReturnDeletedObject: true,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: resourceclaimtemplate.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}

	return &REST{store}, nil
}
