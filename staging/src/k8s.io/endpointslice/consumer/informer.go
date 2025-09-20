/*
Copyright 2024 The Kubernetes Authors.

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

package consumer

import (
	"context"
	"time"

	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	discoveryinformers "k8s.io/client-go/informers/discovery/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// EndpointSliceInformer provides an informer-like interface for EndpointSlices
// that handles merging multiple slices for the same service.
type EndpointSliceInformer struct {
	// consumer is the underlying EndpointSliceConsumer.
	consumer *EndpointSliceConsumer

	// informer is the underlying EndpointSlice informer.
	informer discoveryinformers.EndpointSliceInformer
}

// NewEndpointSliceInformer creates a new EndpointSliceInformer.
func NewEndpointSliceInformer(
	informerFactory informers.SharedInformerFactory,
	nodeName string,
) *EndpointSliceInformer {
	consumer := NewEndpointSliceConsumer(nodeName)
	informer := informerFactory.Discovery().V1().EndpointSlices()

	return &EndpointSliceInformer{
		consumer: consumer,
		informer: informer,
	}
}

// Run starts the informer and syncs the cache.
func (i *EndpointSliceInformer) Run(ctx context.Context) error {
	// Add event handlers to the informer
	_, err := i.informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if endpointSlice, ok := obj.(*discovery.EndpointSlice); ok {
				i.consumer.OnEndpointSliceAdd(endpointSlice)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			if oldEndpointSlice, ok := oldObj.(*discovery.EndpointSlice); ok {
				if newEndpointSlice, ok := newObj.(*discovery.EndpointSlice); ok {
					i.consumer.OnEndpointSliceUpdate(oldEndpointSlice, newEndpointSlice)
				}
			}
		},
		DeleteFunc: func(obj interface{}) {
			if endpointSlice, ok := obj.(*discovery.EndpointSlice); ok {
				i.consumer.OnEndpointSliceDelete(endpointSlice)
			} else {
				// Handle the case where the object is a tombstone
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					klog.ErrorS(nil, "Couldn't get object from tombstone", "object", obj)
					return
				}
				endpointSlice, ok := tombstone.Obj.(*discovery.EndpointSlice)
				if !ok {
					klog.ErrorS(nil, "Tombstone contained object that is not an EndpointSlice", "object", tombstone.Obj)
					return
				}
				i.consumer.OnEndpointSliceDelete(endpointSlice)
			}
		},
	})

	if err != nil {
		return err
	}

	// Wait for the cache to sync
	if !cache.WaitForCacheSync(ctx.Done(), i.informer.Informer().HasSynced) {
		return ctx.Err()
	}

	return nil
}

// AddEventHandler adds a handler for endpoint changes.
func (i *EndpointSliceInformer) AddEventHandler(handler EndpointChangeHandler) {
	i.consumer.AddEventHandler(handler)
}

// GetEndpointSlices returns all EndpointSlices for a service.
func (i *EndpointSliceInformer) GetEndpointSlices(serviceNN types.NamespacedName) []*discovery.EndpointSlice {
	return i.consumer.GetEndpointSlices(serviceNN)
}

// GetEndpoints returns all endpoints for a service, merging and deduplicating
// endpoints from all EndpointSlices for the service.
func (i *EndpointSliceInformer) GetEndpoints(serviceNN types.NamespacedName) []discovery.Endpoint {
	return i.consumer.GetEndpoints(serviceNN)
}

// HasSynced returns true if the underlying informer has synced.
func (i *EndpointSliceInformer) HasSynced() bool {
	return i.informer.Informer().HasSynced()
}

// LastSyncResourceVersion returns the resource version the underlying
// informer has synced to.
func (i *EndpointSliceInformer) LastSyncResourceVersion() string {
	return i.informer.Informer().LastSyncResourceVersion()
}

// SetTransform sets a transform function for the underlying informer.
func (i *EndpointSliceInformer) SetTransform(transform cache.TransformFunc) error {
	return i.informer.Informer().SetTransform(transform)
}

// SetWatchErrorHandler sets a watch error handler for the underlying informer.
func (i *EndpointSliceInformer) SetWatchErrorHandler(handler cache.WatchErrorHandler) error {
	return i.informer.Informer().SetWatchErrorHandler(handler)
}

// SetRelistDuration sets the relist duration for the underlying informer.
func (i *EndpointSliceInformer) SetRelistDuration(duration time.Duration) {
	i.informer.Informer().SetRelistDuration(duration)
}
