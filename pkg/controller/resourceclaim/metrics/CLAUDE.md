# Package: metrics

Provides Prometheus metrics for the ResourceClaim controller.

## Key Metrics

- **ResourceClaimCreateAttempts**: Counter of attempts to create ResourceClaims from templates (labeled by status: success/failure).
- **ResourceClaimCreateFailures**: Counter of failed ResourceClaim creation attempts (deprecated, use labeled counter).
- **NumResourceClaims**: Custom metric description for total number of ResourceClaims (used for quota/limits).

## Design Patterns

- Registered with the legacy Prometheus registry.
- Uses sync.Once for idempotent registration.
- Labels metrics by operation result for granular observability.
