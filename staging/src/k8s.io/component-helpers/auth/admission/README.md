# Authenticated admission webhook

This toy webhook uses `k8s.io/component-helpers/auth/admission` to verify a
webhook-bound bearer token and its AdmissionReview before running its policy.
It admits only names containing `love-kittens`; the Kitten Council rejects the
rest. Real admission policies should be less arbitrary.

## Example Webhook

The helper adds one package import, one verification option and the
`authadmission.WithTokenVerification` call. See [examples/main.go](examples/main.go)
for the complete toy policy and HTTPS-server setup,
[examples/Dockerfile](examples/Dockerfile) for a minimal container build, and
[examples/in-cluster.yaml](examples/in-cluster.yaml) and
[examples/out-of-cluster.yaml](examples/out-of-cluster.yaml) for minimal
`Service`, `Deployment`, and `ValidatingWebhookConfiguration` manifests showing
how the parameters line up.

Choose the verification option that matches your registration:

- **In-cluster (`WithServiceAudience(serviceName, servicePath, servicePort)`)**:
  Supply the registered `serviceName` (`"kittens"`), `servicePath` (`"/validate"`,
  or `"/"` if omitted in the registration), and `servicePort` (`443`, i.e.
  `Service` `spec.ports[].port` / `clientConfig.service.port`, not the
  container's `targetPort` `8443`).
  The helper reads the Pod's projected ServiceAccount namespace and token to
  construct `https://<serviceName>.<namespace>.svc:<servicePort><servicePath>`,
  fetches signing keys from the local `/openid/v1/jwks`, and queries cluster
  discovery once at startup via a cached discovery client to populate the
  allowed cross-group subresource map. The Pod must mount its projected
  ServiceAccount credentials (`automountServiceAccountToken: true`) and have
  RBAC permission to GET `/openid/v1/jwks` and discovery (`/api` and `/apis`).
- **Explicit issuer (`WithIssuer(issuer, audience, client)`)**: Use for
  off-cluster webhooks, URL-backed `clientConfig.url` registrations, or custom
  issuer discovery. Supply the calling cluster's token issuer URL, the exact
  registered audience (`clientConfig.url` or `https://<name>.<namespace>.svc:<port><path>`),
  and an optional credential-free `*http.Client` (`nil` uses default transport).

The helper authenticates the bearer token before reading the request body,
decodes the v1 `AdmissionReview`, verifies that the token's
`admissionReviewAPIGroups` attestation covers the request's API group and object
kinds, and sets the response UID. Failed authentication returns HTTP 401; policy
denials return an `AdmissionReview` with `Allowed: false`. Use
`failurePolicy: Fail` in your `ValidatingWebhookConfiguration` or
`MutatingWebhookConfiguration` when authentication failures must block admission.

## Cross-Group Subresource Mismatches

For ordinary admission requests, the admitted object's API group (`req.Kind.Group`)
matches the authorized resource's API group (`req.Resource.Group`). Only a small
set of subresources admit an object Kind from a different API group than the
parent resource attested by the token (`/scale` -> `autoscaling/v1, Kind=Scale`,
`pods/eviction` -> `policy/v1, Kind=Eviction`, and `serviceaccounts/token` ->
`authentication.k8s.io/v1, Kind=TokenRequest`). The verifier only consults its
O(1) mismatch map when `req.Kind.Group != req.Resource.Group`.

- `AllowedSubresourceGroupMismatchList()` returns a copy of the built-in
  Kubernetes cross-group subresource map.
- `WithServiceAudience` automatically extends the built-in map once at startup
  using `DiscoverSubresourceGroupMismatches` and a cached discovery client,
  failing startup if discovery is unreachable so the Pod crashloops and logs the
  error.
- Multi-resource or off-cluster webhooks (such as Gatekeeper) that intercept
  custom aggregated API subresources whose Kind group differs from their parent
  resource group can implement `WebhookSubresourceGroupAllowedMismatch` and pass
  `WithSubresourceGroupMismatches(provider)` to seed from
  `AllowedSubresourceGroupMismatchList()` and add custom entries (or call
  `DiscoverSubresourceGroupMismatches` with a `CachedDiscoveryInterface`).
  See [examples/main.go](examples/main.go) for an example implementation.
