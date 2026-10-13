/*
Copyright 2025 The Kubernetes Authors.

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

package proxy

import (
	"fmt"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/labels"
	discoveryv1informer "k8s.io/client-go/informers/discovery/v1"
	discoveryv1lister "k8s.io/client-go/listers/discovery/v1"
	"k8s.io/client-go/tools/cache"
)

// EndpointSliceGetter is an interface for a helper that lets you easily get all
// EndpointSlices for a Service.
type EndpointSliceGetter interface {
	// GetEndpointSlices returns all of the known slices associated with the given
	// service. If there are no slices associated with the service, it will return an
	// empty list, not an error.
	GetEndpointSlices(namespaceName, serviceName string) ([]*discoveryv1.EndpointSlice, error)
}

const indexKey = "namespaceName_serviceName"

// ensureServiceNameIndexer ensures that indexer has a namespace/serviceName indexer
func ensureServiceNameIndexer(indexer cache.Indexer) error {
	if _, exists := indexer.GetIndexers()[indexKey]; exists {
		return nil
	}
	err := indexer.AddIndexers(map[string]cache.IndexFunc{indexKey: func(obj any) ([]string, error) {
		ep, ok := obj.(*discoveryv1.EndpointSlice)
		if !ok {
			return nil, fmt.Errorf("expected *discoveryv1.EndpointSlice, got %T", obj)
		}
		serviceName, labelExists := ep.Labels[discoveryv1.LabelServiceName]
		if !labelExists {
			// Not associated with a service; don't add to this index.
			return nil, nil
		}
		return []string{ep.Namespace + "/" + serviceName}, nil
	}})
	if err != nil {
		// Check if the indexer exists now; if so, that means we were racing with
		// another thread, and they successfully installed the indexer, so we can
		// ignore the error.
		if _, exists := indexer.GetIndexers()[indexKey]; exists {
			err = nil
		}
	}
	return err
}

// NewEndpointSliceIndexerGetter returns an EndpointSliceGetter that wraps an informer and
// updates its indexes so that you can efficiently find the EndpointSlices associated with
// a Service later. (Note that sliceInformer will continue the additional indexing for as
// long as it runs, even if if the EndpointSliceGetter is destroyed. Use
// NewEndpointSliceListerGetter if you want want to fetch EndpointSlices without changing
// the underlying cache.)
func NewEndpointSliceIndexerGetter(sliceInformer discoveryv1informer.EndpointSliceInformer) (EndpointSliceGetter, error) {
	indexer := sliceInformer.Informer().GetIndexer()
	if err := ensureServiceNameIndexer(indexer); err != nil {
		return nil, err
	}
	return &endpointSliceIndexerGetter{indexer: indexer}, nil
}

type endpointSliceIndexerGetter struct {
	indexer cache.Indexer
}

func (e *endpointSliceIndexerGetter) GetEndpointSlices(namespaceName, serviceName string) ([]*discoveryv1.EndpointSlice, error) {
	objs, err := e.indexer.ByIndex(indexKey, namespaceName+"/"+serviceName)
	if err != nil {
		return nil, err
	}
	eps := make([]*discoveryv1.EndpointSlice, 0, len(objs))
	for _, obj := range objs {
		ep, ok := obj.(*discoveryv1.EndpointSlice)
		if !ok {
			return nil, fmt.Errorf("expected *discoveryv1.EndpointSlice, got %T", obj)
		}
		eps = append(eps, ep)
	}
	return eps, nil
}

// NewEndpointSliceListerGetter returns an EndpointSliceGetter that uses a lister to do a
// full selection on every lookup.
func NewEndpointSliceListerGetter(sliceLister discoveryv1lister.EndpointSliceLister) (EndpointSliceGetter, error) {
	return &endpointSliceListerGetter{lister: sliceLister}, nil
}

type endpointSliceListerGetter struct {
	lister discoveryv1lister.EndpointSliceLister
}

func (e *endpointSliceListerGetter) GetEndpointSlices(namespaceName, serviceName string) ([]*discoveryv1.EndpointSlice, error) {
	return e.lister.EndpointSlices(namespaceName).List(labels.SelectorFromSet(labels.Set{discoveryv1.LabelServiceName: serviceName}))
}
