# Package: podcertificate

Manages pod certificate projected volumes, handling certificate issuance and renewal via PodCertificateRequest API.

## Key Types

- **Manager**: Interface for tracking pods and retrieving certificate credentials.
- **IssuingManager**: Main implementation using workqueue-based processing for certificate lifecycle.
- **NoOpManager**: Stub implementation for static/detached kubelet mode.
- **MetricReport / SignerAndState**: Metrics reporting types.

## State Machine

Credential states per projection:
- `credStateInitial`: Not yet started
- `credStateWait`: Waiting for initial PCR to be issued
- `credStateFresh`: Certificate issued and valid
- `credStateWaitRefresh`: Waiting for refresh PCR
- `credStateDenied`: Permanently denied
- `credStateFailed`: Permanently failed

## Key Functions

- `NewIssuingManager()`: Creates manager with informers and workqueue.
- `Run()`: Starts projection processing and periodic refresh.
- `TrackPod()`: Queues pod's certificate projections for processing.
- `ForgetPod()`: Cleans up state for removed pods.
- `GetPodCertificateCredentialBundle()`: Returns private key and certificate chain.

## Key Algorithms

- `generateKeyAndProof()`: Generates keypairs (RSA, ECDSA, Ed25519) with proof of possession.
- Automatic refresh before BeginRefreshAt time with jitter.
- Events emitted for overdue refresh (10+ minutes) and expiration.

## Design Notes

- Uses PodCertificateRequest API (v1beta1) for certificate issuance
- State not preserved across restarts; all projections re-queued on startup
- PCR owner references point to pod for garbage collection
- Supports RSA3072, RSA4096, ECDSAP256, ECDSAP384, ECDSAP521, ED25519 key types
