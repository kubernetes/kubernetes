# Authenticated admission webhook

This toy webhook uses `k8s.io/component-helpers/auth/admission` to verify a
webhook-bound bearer token and its AdmissionReview before running its policy.
It admits only names containing `love-kittens`; the Kitten Council rejects the
rest. Real admission policies should be less arbitrary.

## Example Webhook

The helper adds one package import, one verification option and the
`authadmission.WithTokenVerification` call. See [examples/main.go](examples/main.go)
for the complete toy policy and HTTPS-server setup.

Choose the verification option that matches your registration:

- **In-cluster (`WithServiceAudience(name, path, port)`)**: Supply the registered
  Service name (`"kittens"`), path (`"/validate"`, or `"/"` if omitted in the
  registration), and **Service** port (`443`, not container `targetPort` `8443`).
  The helper reads the Pod's projected ServiceAccount namespace and token to
  construct `https://<name>.<namespace>.svc:<port><path>` and fetch signing keys
  from the local `/openid/v1/jwks`. The Pod must mount its projected
  ServiceAccount credentials (`automountServiceAccountToken: true`) and have RBAC
  permission to GET `/openid/v1/jwks`.
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
