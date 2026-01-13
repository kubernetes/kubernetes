# Package: controlplane

## Purpose
Defines the kube-apiserver control plane instance configuration and initialization. This is the main package for setting up the Kubernetes API server with all its API groups and controllers.

## Key Types

- **Config**: Main configuration containing ControlPlane config and Extra settings.
- **Extra**: Kube-apiserver specific configuration including:
  - Service IP ranges (primary and secondary for dual-stack)
  - API server service IP and port
  - NodePort ranges
  - Endpoint reconciler settings
  - Master count and endpoint TTL
- **CompletedConfig**: Validated and completed configuration ready for use.
- **EndpointReconcilerConfig**: Configuration for the kubernetes service endpoint reconciler.
- **Instance**: The running kube-apiserver instance.

## Key Constants

- **DefaultEndpointReconcilerInterval**: 10 seconds between endpoint reconciliations.
- **DefaultEndpointReconcilerTTL**: 15 seconds TTL for endpoint records.
- **repairLoopInterval**: 3 minutes between ClusterIP/NodePort repair loops.

## Key Subpackages

- **apiserver**: Generic API server configuration and setup.
- **controller**: Built-in controllers (apiserverleasegc, kubernetesservice, systemnamespaces, etc.).
- **reconcilers**: Endpoint reconciliation strategies (lease, instancecount, none).
- **storageversionhashdata**: Storage version hash data for API discovery.

## API Groups Installed

Installs REST storage for: admissionregistration, apiserverinternal, apps, authentication, authorization, autoscaling, batch, certificates, coordination, core, discovery, events, flowcontrol, networking, node, policy, rbac, resource, scheduling, storage, storagemigration.

## Design Notes

- Supports dual-stack service IP ranges.
- Endpoint reconciler ensures kubernetes service endpoints stay current.
- Repair loops fix ClusterIP and NodePort allocations.
- Built on top of generic API server infrastructure.
