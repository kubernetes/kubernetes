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

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/kubernetes"
	apiv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	tprv1 "k8s.io/client-go/examples/third-party-resources/apis/tpr/v1"
)

// Watcher is an example of watching on resource create/update/delete events
type ExampleController struct {
	clientset     *kubernetes.Clientset
	exampleClient *rest.RESTClient
}

// Run starts an Example resource watcher
func (w *ExampleController) Run(ctx context.Context) error {
	fmt.Printf("Watch Example objects\n")

	// Watch Example objects
	_, err := watchExamples(ctx, w.exampleClient)
	if err != nil {
		fmt.Printf("Failed to register watch for Example resource: %v\n", err)
		return err
	}

	<-ctx.Done()
	return ctx.Err()
}

func watchExamples(ctx context.Context, exampleClient cache.Getter) (cache.Controller, error) {
	source := cache.NewListWatchFromClient(
		exampleClient,
		tprv1.ExampleResourcePlural,
		apiv1.NamespaceAll,
		fields.Everything())

	store, controller := cache.NewInformer(
		source,

		// The object type.
		&tprv1.Example{},

		// resyncPeriod
		// Every resyncPeriod, all resources in the cache will retrigger events.
		// Set to 0 to disable the resync.
		0,

		// Your custom resource event handlers.
		cache.ResourceEventHandlerFuncs{
			AddFunc:    onAdd,
			UpdateFunc: onUpdate,
			DeleteFunc: onDelete,
		})

	// store can be used to List and Get
	for _, obj := range store.List() {
		example := obj.(*tprv1.Example)
		fmt.Printf("Existing example: %#v\n", example)
	}

	go controller.Run(ctx.Done())

	return controller, nil
}

func onAdd(obj interface{}) {
	example := obj.(*tprv1.Example)
	fmt.Printf("[WATCH] OnAdd %s\n", example.Metadata.SelfLink)
}

func onUpdate(oldObj, newObj interface{}) {
	oldExample := oldObj.(*tprv1.Example)
	newExample := newObj.(*tprv1.Example)
	fmt.Printf("[WATCH] OnUpdate oldObj: %s\n", oldExample.Metadata.SelfLink)
	fmt.Printf("[WATCH] OnUpdate newObj: %s\n", newExample.Metadata.SelfLink)
}

func onDelete(obj interface{}) {
	example := obj.(*tprv1.Example)
	fmt.Printf("[WATCH] OnDelete %s\n", example.Metadata.SelfLink)
}
