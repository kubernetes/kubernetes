# Package certificate

Package certificate provides kubelet certificate management for both client and server TLS certificates, including automatic rotation and renewal.

## Key Functions

- `NewKubeletServerCertificateManager`: Creates a certificate manager for kubelet serving certificates with automatic rotation via CSR
- `NewKubeletClientCertificateManager`: Creates a certificate manager for kubelet client certificates used to authenticate to the API server
- `NewKubeletServerCertificateDynamicFileManager`: Creates a certificate manager that watches and reloads certificates from files (no CSR)

## Key Types

- `kubeletServerCertificateDynamicFileManager`: Watches cert/key files and reloads on changes

## Certificate Details

- Server certs: Subject CN=system:node:<nodeName>, O=system:nodes, with node IPs/hostnames as SANs
- Client certs: Subject CN=system:node:<nodeName>, O=system:nodes
- Uses KubeletServingSignerName for server certs
- Uses KubeAPIServerClientKubeletSignerName for client certs

## Metrics

- `server_expiration_renew_errors`: Counter for server cert renewal failures
- `certificate_manager_server_rotation_seconds`: Histogram of certificate lifetimes
- `certificate_manager_server_ttl_seconds`: Gauge of time until server cert expiry
- `client_expiration_renew_errors`: Counter for client cert renewal failures

## Design Notes

- AllowDNSOnlyNodeCSR feature gate allows CSRs with only DNS names (no IPs)
- Certificates stored via FileStore for persistence across restarts
- Dynamic file manager useful when certs are provisioned externally
