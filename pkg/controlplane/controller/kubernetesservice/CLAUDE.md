# Package: kubernetesservice

## Purpose
This controller manages the "kubernetes" Service in the default namespace. It creates and maintains the core API server service that clients use to communicate with the Kubernetes API, and reconciles endpoints to ensure the service points to healthy API servers.

## Key Types

- **Controller**: Manages the kubernetes service and its endpoints
- **Config**: Configuration including PublicIP, ServiceIP, ports, EndpointReconciler, and reconciliation interval

## Key Functions

- **New()**: Creates a controller with given config and service informer
- **Start()**: Removes stale endpoints from previous instances, starts the reconciliation loop
- **Stop()**: Gracefully shuts down, removing this API server's endpoint from the service
- **Run()**: Waits for server readiness, then periodically calls UpdateKubernetesService
- **UpdateKubernetesService()**: Updates service definition and reconciles endpoints
- **CreateOrUpdateMasterServiceIfNeeded()**: Creates the kubernetes service if it doesn't exist

## Service Configuration

- Service name: "kubernetes" in default namespace
- Labels: provider=kubernetes, component=apiserver
- IPFamilyPolicy: SingleStack
- Type: ClusterIP (or NodePort if KubernetesServiceNodePort > 0)
- Selector: nil (endpoints managed manually, not by pod selector)

## Design Notes

- Polls /readyz every 100ms before starting reconciliation to ensure server is ready
- Uses EndpointReconciler interface for endpoint management (supports master count-based or lease-based)
- Removes its own endpoints on shutdown to allow faster failover
- Service definition is only reconciled on first run, not continuously updated
- Has 2x EndpointInterval timeout for graceful shutdown cleanup
