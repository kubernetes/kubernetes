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

package main

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
)

// Watcher is an example of watching on resource create/update/delete events
type Watcher struct {
	clientset     *kubernetes.Clientset
	exampleClient *rest.RESTClient
	exampleScheme *runtime.Scheme
}

// Run starts an Example resource watcher
func (w *Watcher) Run(ctx context.Context) error {
	fmt.Printf("Watch Example objects\n")

	// Watch Example objects
	handler := ExampleEventHandler{}
	_, err := watchExamples(ctx, w.exampleClient, w.exampleScheme, &handler)
	if err != nil {
		fmt.Printf("Failed to register watch for Example resource: %v\n", err)
		return err
	}

	<-ctx.Done()
	return ctx.Err()
}

func watchExamples(ctx context.Context, exampleClient cache.Getter, exampleScheme *runtime.Scheme, handler cache.ResourceEventHandler) (cache.Controller, error) {
	parameterCodec := runtime.NewParameterCodec(exampleScheme)

	source := newListWatchFromClient(
		exampleClient,
		ExampleResourcePath,
		api.NamespaceAll,
		fields.Everything(),
		parameterCodec)

	store, controller := cache.NewInformer(
		source,

		// The object type.
		&Example{},

		// resyncPeriod
		// Every resyncPeriod, all resources in the cache will retrigger events.
		// Set to 0 to disable the resync.
		0,

		// Your custom resource event handlers.
		handler)

	// store can be used to List and Get
	for _, obj := range store.List() {
		example := obj.(*Example)
		fmt.Printf("Existing example: %#v\n", example)
	}

	go controller.Run(ctx.Done())

	return controller, nil
}

// See the issue comment: https://github.com/kubernetes/kubernetes/issues/16376#issuecomment-272167794
// newListWatchFromClient is a copy of cache.NewListWatchFromClient() method with custom codec
// Cannot use cache.NewListWatchFromClient() because it uses global api.ParameterCodec which uses global
// api.Scheme which does not know about custom types (Example in our case) group/version.
func newListWatchFromClient(c cache.Getter, resource string, namespace string, fieldSelector fields.Selector, paramCodec runtime.ParameterCodec) *cache.ListWatch {
	listFunc := func(options metav1.ListOptions) (runtime.Object, error) {
		return c.Get().
			Namespace(namespace).
			Resource(resource).
			VersionedParams(&options, paramCodec).
			FieldsSelectorParam(fieldSelector).
			Do().
			Get()
	}
	watchFunc := func(options metav1.ListOptions) (watch.Interface, error) {
		return c.Get().
			Prefix("watch").
			Namespace(namespace).
			Resource(resource).
			VersionedParams(&options, paramCodec).
			FieldsSelectorParam(fieldSelector).
			Watch()
	}
	return &cache.ListWatch{ListFunc: listFunc, WatchFunc: watchFunc}
}

// ExampleEventHandler can handle events for Example resource
type ExampleEventHandler struct {
}

func (h *ExampleEventHandler) OnAdd(obj interface{}) {
	example := obj.(*Example)
	fmt.Printf("[WATCH] OnAdd %s\n", example.Metadata.SelfLink)
}

func (h *ExampleEventHandler) OnUpdate(oldObj, newObj interface{}) {
	oldExample := oldObj.(*Example)
	newExample := newObj.(*Example)
	fmt.Printf("[WATCH] OnUpdate oldObj: %s\n", oldExample.Metadata.SelfLink)
	fmt.Printf("[WATCH] OnUpdate newObj: %s\n", newExample.Metadata.SelfLink)
}

func (h *ExampleEventHandler) OnDelete(obj interface{}) {
	example := obj.(*Example)
	fmt.Printf("[WATCH] OnDelete %s\n", example.Metadata.SelfLink)
}
