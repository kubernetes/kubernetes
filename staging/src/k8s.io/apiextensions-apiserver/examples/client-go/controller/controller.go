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

package controller

import (
	"context"
	"fmt"
	"sync"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"

	crv1 "k8s.io/apiextensions-apiserver/examples/client-go/apis/cr/v1"
)

// Watcher is an example of watching on resource create/update/delete events
type ExampleController struct {
	ExampleClient *rest.RESTClient
	ExampleScheme *runtime.Scheme
}

// Run starts an Example resource controller
func (c *ExampleController) Run(ctx context.Context) error {
	fmt.Print("Watch Example objects\n")

	// Watch Example objects
	_, err := c.watchExamples(ctx)
	if err != nil {
		fmt.Printf("Failed to register watch for Example resource: %v\n", err)
		return err
	}

	<-ctx.Done()
	return ctx.Err()
}

type debugWatch struct {
	w          watch.Interface
	once       sync.Once
	resultChan chan watch.Event
}

func (d *debugWatch) Stop() {
	fmt.Println("DEBUG: Stop()")
	d.w.Stop()
}

func (d *debugWatch) ResultChan() <-chan watch.Event {
	d.once.Do(func() {
		// create our result channel
		d.resultChan = make(chan watch.Event)
		// copy from the underlying channel into it
		go func() {
			for result := range d.w.ResultChan() {
				fmt.Printf("DEBUG: %#v\n", result)
				d.resultChan <- result
			}
			fmt.Println("DEBUG: closing")
			close(d.resultChan)
		}()
	})
	return d.resultChan
}

func (c *ExampleController) watchExamples(ctx context.Context) (cache.Controller, error) {
	listFunc := func(options metav1.ListOptions) (runtime.Object, error) {
		options.FieldSelector = fields.Everything().String()
		obj, err := c.ExampleClient.Get().
			Namespace(apiv1.NamespaceAll).
			Resource(crv1.ExampleResourcePlural).
			VersionedParams(&options, metav1.ParameterCodec).
			Do().
			Get()
		if err != nil {
			fmt.Println("DEBUG:", err)
		} else {
			fmt.Printf("DEBUG: %#v\n", obj)
		}
		return obj, err
	}
	watchFunc := func(options metav1.ListOptions) (watch.Interface, error) {
		options.Watch = true
		options.FieldSelector = fields.Everything().String()
		watch, err := c.ExampleClient.Get().
			Namespace(apiv1.NamespaceAll).
			Resource(crv1.ExampleResourcePlural).
			VersionedParams(&options, metav1.ParameterCodec).
			Watch()
		if err != nil {
			fmt.Println("DEBUG:", err)
		} else {
			watch = &debugWatch{w: watch}
		}
		return watch, err
	}
	source := &cache.ListWatch{ListFunc: listFunc, WatchFunc: watchFunc}

	_, controller := cache.NewInformer(
		source,

		// The object type.
		&crv1.Example{},

		// resyncPeriod
		// Every resyncPeriod, all resources in the cache will retrigger events.
		// Set to 0 to disable the resync.
		0,

		// Your custom resource event handlers.
		cache.ResourceEventHandlerFuncs{
			AddFunc:    c.onAdd,
			UpdateFunc: c.onUpdate,
			DeleteFunc: c.onDelete,
		})

	go controller.Run(ctx.Done())
	return controller, nil
}

func (c *ExampleController) onAdd(obj interface{}) {
	example, ok := obj.(*crv1.Example)
	if !ok {
		fmt.Printf("[CONTROLLER] onAdd unknown type %T\n", obj)
		return
	}
	fmt.Printf("[CONTROLLER] OnAdd %s\n", example.ObjectMeta.SelfLink)

	// NEVER modify objects from the store. It's a read-only, local cache.
	// You can use exampleScheme.Copy() to make a deep copy of original object and modify this copy
	// Or create a copy manually for better performance
	copyObj, err := c.ExampleScheme.Copy(example)
	if err != nil {
		fmt.Printf("ERROR creating a deep copy of example object: %v\n", err)
		return
	}

	exampleCopy := copyObj.(*crv1.Example)
	exampleCopy.Status = crv1.ExampleStatus{
		State:   crv1.ExampleStateProcessed,
		Message: "Successfully processed by controller",
	}

	err = c.ExampleClient.Put().
		Name(example.ObjectMeta.Name).
		Namespace(example.ObjectMeta.Namespace).
		Resource(crv1.ExampleResourcePlural).
		Body(exampleCopy).
		Do().
		Error()

	if err != nil {
		fmt.Printf("ERROR updating status: %v\n", err)
	} else {
		fmt.Printf("UPDATED status: %#v\n", exampleCopy)
	}
}

func (c *ExampleController) onUpdate(oldObj, newObj interface{}) {
	oldExample, ok := oldObj.(*crv1.Example)
	if !ok {
		fmt.Printf("[CONTROLLER] onUpdate unknown type %T\n", oldObj)
		return
	}
	newExample, ok := newObj.(*crv1.Example)
	if !ok {
		fmt.Printf("[CONTROLLER] onUpdate unknown type %T\n", newObj)
		return
	}
	fmt.Printf("[CONTROLLER] OnUpdate oldObj: %s\n", oldExample.ObjectMeta.SelfLink)
	fmt.Printf("[CONTROLLER] OnUpdate newObj: %s\n", newExample.ObjectMeta.SelfLink)
}

func (c *ExampleController) onDelete(obj interface{}) {
	example, ok := obj.(*crv1.Example)
	if !ok {
		fmt.Printf("[CONTROLLER] onDelete unknown type %T\n", obj)
		return
	}
	fmt.Printf("[CONTROLLER] OnDelete %s\n", example.ObjectMeta.SelfLink)
}
