# EndpointSlice Consumer Helpers

This package provides helper functions for consuming EndpointSlices in Kubernetes.

## Overview

The EndpointSlice API is the mechanism that Kubernetes uses to let Services scale to handle large numbers of backends. However, working with EndpointSlices directly can be challenging because a single Service may have multiple EndpointSlice objects, and endpoints can move between slices.

This package provides helper functions that make it easier to consume EndpointSlices by:

1. Providing a unified view of all EndpointSlices for a service across multiple EndpointSlice objects
2. Handling the complexity of tracking EndpointSlice additions, updates, and deletions
3. Offering informer-like and lister-like interfaces that are familiar to Kubernetes developers

## Components

### EndpointSliceConsumer

The `EndpointSliceConsumer` is the core component that tracks EndpointSlices and provides a unified view for a service. It is kept up-to-date via `OnEndpointSliceAdd`, `OnEndpointSliceUpdate`, and `OnEndpointSliceDelete` callbacks (typically wired up via an informer).

```go
// Create a new consumer
c := consumer.NewEndpointSliceConsumer()

// Add an event handler
c.AddEventHandler(consumer.EndpointChangeHandlerFunc(func(serviceNN types.NamespacedName, slices []*discovery.EndpointSlice) {
    fmt.Printf("Service %s/%s has %d slices\n", serviceNN.Namespace, serviceNN.Name, len(slices))
}))

// Get all EndpointSlices for a service
slices := c.GetEndpointSlices(types.NamespacedName{Namespace: "default", Name: "my-service"})
```

### EndpointSliceInformer

The `EndpointSliceInformer` provides an informer-like interface for EndpointSlices that handles tracking multiple slices for the same service.

```go
// Create a new informer
informer := consumer.NewEndpointSliceInformer(informerFactory)

// Add an event handler
informer.AddEventHandler(consumer.EndpointChangeHandlerFunc(func(serviceNN types.NamespacedName, slices []*discovery.EndpointSlice) {
    klog.InfoS("Service endpoints changed", "namespace", serviceNN.Namespace, "name", serviceNN.Name, "slices", len(slices))
}))

// Start the informer
if err := informer.Run(ctx); err != nil {
    klog.ErrorS(err, "Failed to run EndpointSliceInformer")
    return
}

// Get all EndpointSlices for a service
slices := informer.GetEndpointSlices(types.NamespacedName{Namespace: "default", Name: "my-service"})
```

### EndpointSliceLister

The `EndpointSliceLister` provides a lister-like interface for EndpointSlices that allows listing slices for a given service by label.

```go
// Create a new lister
lister := consumer.NewEndpointSliceLister(endpointSliceLister)

// Get all EndpointSlices for a service
slices, err := lister.EndpointSlices("default").Get("my-service")
if err != nil {
    klog.ErrorS(err, "Failed to get EndpointSlices")
    return
}
```

## Migrating from Endpoints to EndpointSlices

This package is designed to help applications migrate from using the Endpoints API to using the EndpointSlices API. The interfaces provided by this package are similar to those used with Endpoints, making the migration easier.

When migrating from Endpoints to EndpointSlices, consider the following:

1. EndpointSlices are more scalable than Endpoints, especially for services with a large number of backends
2. EndpointSlices provide more information about endpoints, such as topology hints and node names
3. Each EndpointSlice carries its own `Ports` field; consumers that need to associate ports with endpoints should iterate over `GetEndpointSlices` rather than flattening all endpoints into a single list
4. EndpointSlices are the preferred way to access endpoint information in Kubernetes
