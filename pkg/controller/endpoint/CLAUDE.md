# Package: endpoint

Endpoints controller implementation for managing Service endpoints.

## Key Types

- `Controller`: Watches Services and Pods to create/update Endpoints resources

## Key Constants

- `maxCapacity`: 1000 - maximum addresses per Endpoints resource
- `ControllerName`: "endpoint-controller"
- `LabelManagedBy`: "endpoints.kubernetes.io/managed-by"

## Key Functions

- `NewEndpointController()`: Creates the controller with pod, service, and endpoints informers
- `Run()`: Starts the reconciliation loop
- `syncService()`: Main reconciliation logic for a single Service

## Purpose

Populates Endpoints resources with IP addresses of pods matching a Service's selector. This is the legacy endpoints implementation; the newer EndpointSlice controller provides similar functionality with better scalability.

## Key Features

- Watches pod and service changes
- Batches endpoint updates to reduce API calls
- Truncates endpoints exceeding maxCapacity with annotation
- Filters pods based on readiness gates and conditions

## Design Notes

- Uses workqueue with rate limiting (max 15 retries)
- Supports endpoint update batching via `endpointUpdatesBatchPeriod`
- Tolerant IP families handling for dual-stack services
