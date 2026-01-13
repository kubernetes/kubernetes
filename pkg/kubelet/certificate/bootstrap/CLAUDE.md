# Package bootstrap

Package bootstrap handles kubelet TLS bootstrapping - the process of obtaining initial client certificates using a bootstrap token to authenticate with the API server.

## Key Functions

- `LoadClientConfig`: Loads or creates kubeconfig, preferring existing valid config over bootstrap
- `LoadClientCert`: Requests a new client certificate via CSR if kubeconfig is missing or invalid

## Bootstrap Process

1. Check if existing kubeconfig has valid, non-expired certificates
2. If valid, use existing config; skip bootstrapping
3. If invalid/missing, load bootstrap kubeconfig (with token auth)
4. Generate or reuse private key (cached in kubelet-client.key.tmp)
5. Create CSR with system:node:<nodeName> subject
6. Wait for CSR approval and certificate issuance
7. Write new kubeconfig pointing to issued certificate

## Key Functions Detail

- `isClientConfigStillValid`: Checks if kubeconfig certs are present and not expired
- `requestNodeCertificate`: Creates and submits CSR, waits for approval
- `writeKubeconfigFromBootstrapping`: Writes kubeconfig file with cert references
- `waitForServer`: Polls /healthz endpoint until API server is ready
- `digestedName`: Creates deterministic CSR name from public key and subject

## Design Notes

- Private key is cached separately until CSR succeeds to allow retry
- CSR name includes hash of public key for idempotency
- Supports both RSA and other key types (RSA adds KeyEncipherment usage)
- 1 hour timeout waiting for CSR approval
