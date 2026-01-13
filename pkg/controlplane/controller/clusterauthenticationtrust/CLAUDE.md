# Package: clusterauthenticationtrust

## Purpose
This controller maintains the `extension-apiserver-authentication` ConfigMap in the `kube-system` namespace. This ConfigMap contains authentication information that aggregated API servers use to configure themselves, including client CA certificates and request header configuration.

## Key Types

- **Controller**: Watches and updates the authentication ConfigMap, combining existing data with required authentication data
- **ClusterAuthenticationInfo**: Holds authentication configuration including ClientCA, RequestHeaderCA, and various request header settings (username, UID, group headers, etc.)

## Key Functions

- **NewClusterAuthenticationTrustController()**: Creates a controller watching the specific ConfigMap
- **Run()**: Starts the controller with informer and worker, also polls every minute for safety
- **syncConfigMap()**: Reads existing ConfigMap, combines with required auth data, and updates if changed
- **combinedClusterAuthenticationInfo()**: Merges two ClusterAuthenticationInfo structs, deduplicating certificates and header values
- **filterExpiredCerts()**: Removes expired certificates (with 5-minute slack) from CA bundles
- **Enqueue()**: Allows external callers to trigger reconciliation

## ConfigMap Data Keys

- `client-ca-file`: CA bundle for verifying client certificates
- `requestheader-client-ca-file`: CA for verifying front proxy certificates
- `requestheader-username-headers`, `requestheader-uid-headers`, `requestheader-group-headers`: Header names for user identity
- `requestheader-extra-headers-prefix`: Prefix for extra user info headers
- `requestheader-allowed-names`: Allowed CN names for front proxy

## Design Notes

- Creates kube-system namespace if it doesn't exist
- Handles ConfigMap too large errors by deleting and recreating
- Deduplicates certificates by comparing raw bytes
- Uses dynamic certificate providers that can be updated at runtime
