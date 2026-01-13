# Package: clustertrustbundlepublisher

## Purpose
Publishes ClusterTrustBundle resources containing CA certificates for signers, making trust bundles available cluster-wide.

## Key Types/Structs
- `ClusterTrustBundlePublisher[T]`: Generic controller supporting both v1alpha1 and v1beta1 API versions
- `clusterTrustBundle`: Type constraint for v1alpha1 and v1beta1 ClusterTrustBundle types
- `alphaHandlers` / `betaHandlers`: API-version specific handlers for CRUD operations

## Key Interfaces
- `PublisherRunner`: Interface with `Run(context.Context)` method
- `clusterTrustBundlesClient[T]`: API-version independent CRUD interface
- `clusterTrustBundlesLister[T]`: API-version independent lister

## Key Functions
- `NewClusterTrustBundlePublisher(signerName, ca, client, informer)`: Creates publisher for specific signer
- Bundle naming convention: `{signerName}:{sha256-hash}` for unique identification

## Behavior
- Watches CA content provider for certificate changes
- Creates/updates ClusterTrustBundle resources with current CA certificates
- Uses content hash in bundle name for versioning
- Cleans up old bundles when CA changes
- Supports both alpha and beta API versions via generics

## Design Notes
- Uses Go generics to support multiple API versions cleanly
- Registers Prometheus metrics for monitoring
- CAContentProvider enables dynamic certificate reloading
