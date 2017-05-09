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
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
)

func Watch(client *rest.RESTClient) {

	stop := make(chan struct{}, 1)
	source := cache.NewListWatchFromClient(
		client,
		"examples",
		v1.NamespaceAll,
		fields.Everything())

	store, controller := cache.NewInformer(
		source,

		// The object type.
		&Example{},

		// resyncPeriod
		// Every resyncPeriod, all resources in the cache will retrigger events.
		// Set to 0 to disable the resync.
		time.Second*60,

		// Your custom resource event handlers.
		cache.ResourceEventHandlerFuncs{
			// Takes a single argument of type interface{}.
			// Called on controller startup and when new resources are created.
			AddFunc: create,

			// Takes two arguments of type interface{}.
			// Called on resource update and every resyncPeriod on existing resources.
			UpdateFunc: update,

			// Takes a single argument of type interface{}.
			// Called on resource deletion.
			DeleteFunc: delete,
		})

	// store can be used to List and Get
	// NEVER modify objects from the store. It's a read-only, local cache.
	fmt.Println("listing examples from store:")
	for _, obj := range store.List() {
		example := obj.(*Example)

		// This will likely be empty the first run, but may not
		fmt.Printf("%#v\n", example)
	}

	// the controller run starts the event processing loop
	go controller.Run(stop)

	// and now we block on a signal
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	s := <-signals
	fmt.Printf("received signal %#v, exiting...\n", s)
	close(stop)
	os.Exit(0)
}

// Handler functions as per the controller above.
// Note the coercion of the interface{} into a pointer of the expected type.

func create(obj interface{}) {
	example := obj.(*Example)

	fmt.Println("CREATED:", printExample(example))
}

func update(old, new interface{}) {
	oldExample := old.(*Example)
	newExample := new.(*Example)

	fmt.Printf("UPDATED:\n  old: %s\n  new: %s\n", printExample(oldExample), printExample(newExample))
}

func delete(obj interface{}) {
	example := obj.(*Example)

	fmt.Println("DELETED:", printExample(example))
}

// convenience functions
func printExample(example *Example) string {
	return fmt.Sprintf("%s/%s, APIVersion: %s, Kind: %s, Value: %#v", example.Metadata.Namespace, example.Metadata.Name, example.APIVersion, example.Kind, example)
}
