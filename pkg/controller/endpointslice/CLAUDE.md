# Package: endpointslice

EndpointSlice controller implementation for scalable service endpoint management.

## Key Types

- `Controller`: Watches Services and Pods to create/update EndpointSlice resources

## Key Constants

- `ControllerName`: "endpointslice-controller.k8s.io"
- `maxRetries`: 15 - maximum sync retries before dropping from queue

## Key Functions

- `NewController()`: Creates the controller with pod, service, node, and endpointslice informers
- `Run()`: Starts the reconciliation loop
- `syncService()`: Main reconciliation logic for a single Service

## Purpose

Populates EndpointSlice resources with endpoints from pods matching a Service's selector. EndpointSlices provide better scalability than Endpoints by splitting large endpoint lists into multiple smaller resources.

## Key Features

- Configurable max endpoints per slice (default 100)
- Topology-aware hints support for traffic routing
- Batched updates to reduce API calls
- Node-based topology cache for hint calculation

## Design Notes

- Uses workqueue with exponential backoff (1s to 1000s)
- Processes service queue, pod queue, and topology queue
- EndpointSlices are labeled with `kubernetes.io/service-name` and managed-by label
- Supports dual-stack services with multiple address types
