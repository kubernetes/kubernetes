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

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/certificates/clustertrustbundle"
)

// REST is a RESTStorage for ClusterTrustBundle.
type REST struct {
	*genericregistry.Store
}

var _ rest.StandardStorage = &REST{}
var _ rest.TableConvertor = &REST{}
var _ genericregistry.GenericStore = &REST{}

// NewREST returns a RESTStorage object for ClusterTrustBundle objects.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &api.ClusterTrustBundle{} },
		NewListFunc:               func() runtime.Object { return &api.ClusterTrustBundleList{} },
		DefaultQualifiedResource:  api.Resource("clustertrustbundles"),
		SingularQualifiedResource: api.Resource("clustertrustbundle"),

		CreateStrategy: clustertrustbundle.Strategy,
		UpdateStrategy: clustertrustbundle.Strategy,
		DeleteStrategy: clustertrustbundle.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{
		RESTOptions: optsGetter,
		AttrFunc:    getAttrs,
	}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}
	return &REST{store}, nil
}

func getAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	bundle, ok := obj.(*api.ClusterTrustBundle)
	if !ok {
		return nil, nil, fmt.Errorf("not a clustertrustbundle")
	}

	selectableFields := generic.MergeFieldsSets(generic.ObjectMetaFieldsSet(&bundle.ObjectMeta, false), fields.Set{
		"spec.signerName": bundle.Spec.SignerName,
	})

	return labels.Set(bundle.Labels), selectableFields, nil
}
