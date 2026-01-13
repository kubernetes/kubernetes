# Package: rootcacertpublisher

## Purpose
Publishes the kube-apiserver root CA certificate to a ConfigMap in every namespace, enabling pods to verify API server connections.

## Key Types/Structs
- `Publisher`: Controller that maintains the kube-root-ca.crt ConfigMap across all namespaces

## Key Constants
- `RootCACertConfigMapName`: "kube-root-ca.crt" - the ConfigMap name
- `DescriptionAnnotation`: "kubernetes.io/description" - annotation key for documentation
- `Description`: Explains the ConfigMap's purpose (verifying kube-apiserver internal endpoints)

## Key Functions
- `NewPublisher(cmInformer, nsInformer, client, rootCA)`: Creates the publisher controller
- `Run(ctx, workers)`: Starts the controller with specified worker count
- `syncNamespace(ctx, ns)`: Ensures ConfigMap exists with correct data in namespace

## Behavior
- Creates ConfigMap with ca.crt key containing root CA certificate
- Updates ConfigMap if data or description annotation changes
- Re-creates ConfigMap if deleted
- Skips terminating namespaces
- Watches both ConfigMaps and Namespaces for changes

## Design Notes
- Uses workqueue with rate limiting
- Queues namespace names, not ConfigMap keys
- Deep copies ConfigMap before modification to avoid cache corruption
- No other usage is guaranteed across Kubernetes distributions
