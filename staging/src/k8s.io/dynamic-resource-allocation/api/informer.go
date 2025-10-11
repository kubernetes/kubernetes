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

package api

import (
	"context"
	"fmt"
	"sync"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	kubernetes "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/listers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

type ResourceSliceInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() ResourceSliceLister
}

type resourceSliceInformer struct {
	factory informers.SharedInformerFactory
}

func (r *resourceSliceInformer) Informer() cache.SharedIndexInformer {
	return r.factory.InformerFor(&ResourceSlice{}, newResourceSliceInformer)
}

func (r *resourceSliceInformer) Lister() ResourceSliceLister {
	return NewResourceSliceLister(r.Informer().GetIndexer())
}

// ResourceSliceLister helps list ResourceSlices.
// All objects returned here must be treated as read-only.
type ResourceSliceLister interface {
	// List lists all ResourceSlices in the indexer.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []*ResourceSlice, err error)
	// Get retrieves the ResourceSlice from the index for a given name.
	// Objects returned here must be treated as read-only.
	Get(name string) (*ResourceSlice, error)
}

// resourceSliceLister implements the ResourceSliceLister interface.
type resourceSliceLister struct {
	listers.ResourceIndexer[*ResourceSlice]
}

// NewResourceSliceLister returns a new ResourceSliceLister.
func NewResourceSliceLister(indexer cache.Indexer) ResourceSliceLister {
	return &resourceSliceLister{listers.New[*ResourceSlice](indexer, resourceapi.Resource("resourceslice"))}
}

// NewResourceSliceInformer returns the ResourceSlice informer in the factory
// if there is one, otherwise it creates and registers a new one.
func NewResourceSliceInformer(informerFactory informers.SharedInformerFactory) ResourceSliceInformer {
	return &resourceSliceInformer{factory: informerFactory}
}

// newResourceSliceInformer constructs a new informer for ResourceSlice type.
func newResourceSliceInformer(client kubernetes.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListWithContextFunc: func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				in, err := client.ResourceV1().ResourceSlices().List(ctx, options)
				if err != nil {
					return nil, err
				}
				var out ResourceSliceList
				if err := Convert_v1_ResourceSliceList_To_api_ResourceSliceList(in, &out, nil); err != nil {
					return nil, err
				}
				klog.FromContext(ctx).V(6).Info("Converted ResourceSlice list", "numSlices", len(in.Items))
				return &out, nil
			},
			WatchFuncWithContext: func(ctx context.Context, options metav1.ListOptions) (watch.Interface, error) {
				in, err := client.ResourceV1().ResourceSlices().Watch(ctx, options)
				if err != nil {
					return nil, err
				}
				out := &watchResourceSlice{
					upstream:   in,
					resultChan: make(chan watch.Event),
					stopChan:   make(chan struct{}),
				}
				go out.run(klog.FromContext(ctx))
				return out, nil
			},
		},
		&ResourceSlice{},
		resyncPeriod,
		cache.Indexers{},
	)
}

type watchResourceSlice struct {
	upstream   watch.Interface
	resultChan chan watch.Event
	stopChan   chan struct{}
	stopOnce   sync.Once
}

func (w *watchResourceSlice) Stop() {
	w.upstream.Stop()
	w.stopOnce.Do(func() {
		close(w.stopChan)
	})
}

func (w *watchResourceSlice) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func (w *watchResourceSlice) run(logger klog.Logger) {
	logger.V(4).Info("Starting to process ResourceSlice watch stream")
	defer func() {
		logger.V(4).Info("Done processing ResourceSlice watch stream")
		close(w.resultChan)
	}()

	resultChan := w.upstream.ResultChan()
	for {
		select {
		case e, ok := <-resultChan:
			if !ok {
				// The producer stopped first.
				return
			}
			switch in := e.Object.(type) {
			case *resourceapi.ResourceSlice:
				var out ResourceSlice
				if err := Convert_v1_ResourceSlice_To_api_ResourceSlice(in, &out, nil); err != nil {
					logger.Error(err, "Converting ResourceSlice in watch event failed")
				} else {
					logger.V(6).Info("Converted ResourceSlice in watch event", "slice", klog.KObj(in), "event", e.Type)
				}
				e = watch.Event{
					Type:   e.Type,
					Object: &out,
				}
			case *resourceapi.ResourceSliceList:
				// Not needed?
				var out ResourceSliceList
				if err := Convert_v1_ResourceSliceList_To_api_ResourceSliceList(in, &out, nil); err != nil {
					logger.Error(err, "Converting ResourceSlice list in watch event failed")
				} else {
					logger.V(6).Info("Converted ResourceSlice list in watch event", "numSlices", len(in.Items), "event", e.Type)
				}
				e = watch.Event{
					Type:   e.Type,
					Object: &out,
				}
			default:
				logger.V(6).Info("Not converting other ResourceSlice watch stream object", "type", fmt.Sprintf("%T", in))
			}
			// This must not get blocked when the consumer stops reading,
			// hence the stopChan.
			select {
			case w.resultChan <- e:
			case <-w.stopChan:
				return
			}
		case <-w.stopChan:
			return
		}
	}
}
