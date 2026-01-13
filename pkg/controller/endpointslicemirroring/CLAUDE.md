# Package: endpointslicemirroring

Controller that mirrors Endpoints resources to EndpointSlices.

## Key Types

- `Controller`: Watches Endpoints resources and creates corresponding EndpointSlices

## Key Constants

- `ControllerName`: "endpointslicemirroring-controller.k8s.io"
- `maxRetries`: 15

## Key Functions

- `NewController()`: Creates the controller with endpoints, endpointslice, and service informers
- `Run()`: Starts the reconciliation loop
- `syncEndpoints()`: Main reconciliation logic

## Purpose

Mirrors Endpoints to EndpointSlices for services without selectors (e.g., ExternalName services or manually managed endpoints). This enables features that depend on EndpointSlices to work with legacy Endpoints-based services.

## Key Features

- Only mirrors Endpoints for services without selectors
- Respects `endpoints.kubernetes.io/skip-mirror` annotation to opt-out
- Configurable max endpoints per subset
- Batched updates support

## Design Notes

- Uses workqueue with exponential backoff (1s to 100s max)
- Creates EndpointSlices with managed-by label for ownership
- Skips Endpoints managed by the main endpoint controller
