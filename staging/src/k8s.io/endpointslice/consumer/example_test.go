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

package consumer_test

import (
	"context"
	"fmt"
	"time"

	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/endpointslice/consumer"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// This example demonstrates how to use the EndpointSliceConsumer directly.
func Example_directUsage() {
	// Create a new consumer
	c := consumer.NewEndpointSliceConsumer("node1")

	// Add an event handler
	c.AddEventHandler(consumer.EndpointChangeHandlerFunc(func(serviceNN types.NamespacedName, slices []*discovery.EndpointSlice) {
		fmt.Printf("Service %s/%s has %d slices\n", serviceNN.Namespace, serviceNN.Name, len(slices))
	}))

	// Create some test EndpointSlices
	slice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "slice1",
			Namespace: "default",
			Labels: map[string]string{
				discovery.LabelServiceName: "my-service",
			},
		},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.0.1"},
				NodeName:  ptr.To("node1"),
				Conditions: discovery.EndpointConditions{
					Ready: ptr.To(true),
				},
			},
			{
				Addresses: []string{"10.0.0.2"},
				NodeName:  ptr.To("node2"),
				Conditions: discovery.EndpointConditions{
					Ready: ptr.To(true),
				},
			},
		},
	}

	slice2 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "slice2",
			Namespace: "default",
			Labels: map[string]string{
				discovery.LabelServiceName: "my-service",
			},
		},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{
			{
				Addresses: []string{"10.0.0.3"},
				NodeName:  ptr.To("node3"),
				Conditions: discovery.EndpointConditions{
					Ready: ptr.To(true),
				},
			},
		},
	}

	// Add the slices to the consumer
	c.OnEndpointSliceAdd(slice1)
	c.OnEndpointSliceAdd(slice2)

	// Get all slices for the service
	serviceNN := types.NamespacedName{Namespace: "default", Name: "my-service"}
	slices := c.GetEndpointSlices(serviceNN)
	fmt.Printf("Found %d slices for service %s/%s\n", len(slices), serviceNN.Namespace, serviceNN.Name)

	// Get all endpoints for the service
	endpoints := c.GetEndpoints(serviceNN)
	fmt.Printf("Found %d endpoints for service %s/%s\n", len(endpoints), serviceNN.Namespace, serviceNN.Name)

	// Output:
	// Service default/my-service has 1 slices
	// Service default/my-service has 2 slices
	// Found 2 slices for service default/my-service
	// Found 3 endpoints for service default/my-service
}

// This example demonstrates how to use the EndpointSliceInformer.
func Example_informerUsage() {
	// In a real application, you would get a client from a clientset
	var client kubernetes.Interface

	// Create a shared informer factory
	informerFactory := informers.NewSharedInformerFactory(client, 10*time.Minute)

	// Create a new EndpointSliceInformer
	informer := consumer.NewEndpointSliceInformer(informerFactory, "node1")

	// Add an event handler
	informer.AddEventHandler(consumer.EndpointChangeHandlerFunc(func(serviceNN types.NamespacedName, slices []*discovery.EndpointSlice) {
		klog.InfoS("Service endpoints changed", "namespace", serviceNN.Namespace, "name", serviceNN.Name, "slices", len(slices))
	}))

	// Start the informer
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start all informers
	informerFactory.Start(ctx.Done())

	// Wait for the informer to sync
	if err := informer.Run(ctx); err != nil {
		klog.ErrorS(err, "Failed to run EndpointSliceInformer")
		return
	}

	// Get endpoints for a service
	serviceNN := types.NamespacedName{Namespace: "default", Name: "my-service"}
	endpoints := informer.GetEndpoints(serviceNN)
	klog.InfoS("Found endpoints", "namespace", serviceNN.Namespace, "name", serviceNN.Name, "count", len(endpoints))

	// Wait for events
	<-ctx.Done()
}

// This example demonstrates how to use the EndpointSliceLister.
func Example_listerUsage() {
	// In a real application, you would get a lister from an informer
	var endpointSliceLister consumer.EndpointSliceNamespaceLister

	// Get all EndpointSlices for a service
	slices, err := endpointSliceLister.Get("my-service")
	if err != nil {
		klog.ErrorS(err, "Failed to get EndpointSlices")
		return
	}

	fmt.Printf("Found %d slices for service my-service\n", len(slices))

	// Get all endpoints for a service
	endpoints, err := endpointSliceLister.GetEndpoints("my-service")
	if err != nil {
		klog.ErrorS(err, "Failed to get endpoints")
		return
	}

	fmt.Printf("Found %d endpoints for service my-service\n", len(endpoints))
}
