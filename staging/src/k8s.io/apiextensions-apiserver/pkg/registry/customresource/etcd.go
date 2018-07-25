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

package customresource

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
)

// REST implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against API services.
func NewREST(resource schema.GroupResource, listKind schema.GroupVersionKind, strategy customResourceDefinitionStorageStrategy, optsGetter generic.RESTOptionsGetter) *REST {
	store := &genericregistry.Store{
		NewFunc: func() runtime.Object { return &unstructured.Unstructured{} },
		NewListFunc: func() runtime.Object {
			// lists are never stored, only manufactured, so stomp in the right kind
			ret := &unstructured.UnstructuredList{}
			ret.SetGroupVersionKind(listKind)
			return ret
		},
		PredicateFunc:            strategy.MatchCustomResourceDefinitionStorage,
		DefaultQualifiedResource: resource,

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: strategy.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

// List returns a list of items matching labels and field according to the store's PredicateFunc.
func (e *REST) List(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	l, err := e.Store.List(ctx, options)
	if err != nil {
		return nil, err
	}

	// Shallow copy ObjectMeta in returned list for each item. Native types have `Items []Item` fields and therefore
	// implicitly shallow copy ObjectMeta. The generic store sets the self-link for each item. So this is necessary
	// to avoid mutation of the objects from the cache.
	if ul, ok := l.(*unstructured.UnstructuredList); ok {
		for i := range ul.Items {
			shallowCopyObjectMeta(&ul.Items[i])
		}
	}

	return l, nil
}

func (r *REST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	o, err := r.Store.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	if u, ok := o.(*unstructured.Unstructured); ok {
		shallowCopyObjectMeta(u)
	}
	return o, nil
}

func shallowCopyObjectMeta(u runtime.Unstructured) {
	obj := shallowMapDeepCopy(u.UnstructuredContent())
	if metadata, ok := obj["metadata"]; ok {
		if metadata, ok := metadata.(map[string]interface{}); ok {
			obj["metadata"] = shallowMapDeepCopy(metadata)
			u.SetUnstructuredContent(obj)
		}
	}
}

func shallowMapDeepCopy(in map[string]interface{}) map[string]interface{} {
	if in == nil {
		return nil
	}

	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = v
	}

	return out
}
