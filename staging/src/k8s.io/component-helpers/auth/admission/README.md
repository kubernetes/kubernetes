# Authenticated admission webhook

This toy webhook uses `k8s.io/component-helpers/auth/admission` to verify a
webhook-bound bearer token and its AdmissionReview before running its policy.
It admits only names containing `love-kittens`; the Kitten Council rejects the
rest. Real admission policies should be less arbitrary.

## Example Webhook

The helper adds one package import, one verification option and the
`webhookAuth.WithTokenVerification` call. The rest below is the toy policy and
ordinary HTTPS-server setup.

Build this `main.go` as `/webhook` in your container image:

```go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	// Verification helper, distinct from the admission API types above.
	webhookAuth "k8s.io/component-helpers/auth/admission"
)

const resourceNameMustContainForAdmission = "love-kittens"

func main() {
	cert := flag.String("tls-cert", "tls.crt", "PEM certificate for the webhook's HTTPS host")
	key := flag.String("tls-key", "tls.key", "PEM private key for that certificate")
	flag.Parse()

	// An empty issuer selects in-cluster verification; otherwise use OIDC discovery.
	// Both modes verify tokens before invoking policy.
	expectedAudience := os.Getenv("WEBHOOK_LOVE_KITTENS_AUDIENCE")
	expectedIssuer := os.Getenv("WEBHOOK_LOVE_KITTENS_ISSUER")
	verificationOption := webhookAuth.WithAudience(expectedAudience)
	if expectedIssuer != "" {
		verificationOption = webhookAuth.WithRemoteIssuer(expectedIssuer, expectedAudience, nil)
	}
	kittenPolicyHandlerFunc := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		if strings.Contains(req.Name, resourceNameMustContainForAdmission) {
			return &admissionv1.AdmissionResponse{Allowed: true}
		}
		return &admissionv1.AdmissionResponse{
			Allowed: false,
			Result: &metav1.Status{
				Message: fmt.Sprintf("we love kittens but your resource does not. Since %q is not in the name, rejected",
					resourceNameMustContainForAdmission),
			},
		}
	}
	handler, err := webhookAuth.WithTokenVerification(
		context.Background(),
		kittenPolicyHandlerFunc,
		verificationOption)

	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/validate", handler)
	server := &http.Server{Addr: ":8443", Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(server.ListenAndServeTLS(*cert, *key))
}
```

`resourceNameMustContainForAdmission` controls only the toy name policy.
These environment-variable names are specific to this example:

- `WEBHOOK_LOVE_KITTENS_AUDIENCE` is required in both modes.
- `WEBHOOK_LOVE_KITTENS_ISSUER`: unset or empty uses the Pod's in-cluster
  credentials and local `/openid/v1/jwks`; a nonempty value selects remote OIDC
  discovery for that expected token issuer. It is not the webhook's own URL.

This selects the verification mode, not whether authentication is enabled.
For in-cluster `WithAudience`, supply the registered Service audience
`https://<name>.<namespace>.svc:<port><path>`: use the Pod's namespace and
spell out port 443 and path `/` when the registration omits them. The port is
the registered **Service** port, not container port 8443. The helper checks
format and namespace; it cannot tell whether the name, port or path matches the
registration. Never derive the expected audience from the token or request.
URL-backed registrations and selectorless Services in another namespace are
not supported by local `WithAudience`. URL-backed registrations require
`WithRemoteIssuer` with the exact `clientConfig.url` and a client able to reach
issuer discovery and its JWKS; that path may need additional configuration for
a private in-cluster issuer.

The helper decodes the AdmissionReview, encodes the response and sets its UID.
Failed authentication returns HTTP 401 without reading the body; policy denials
return an AdmissionReview with `Allowed: false`.

## 1. On-cluster: infer issuer and signing keys

The webhook runs in the calling cluster. Leave `WEBHOOK_LOVE_KITTENS_ISSUER`
empty (as below) or unset. The helper gets the issuer from the Pod's projected
ServiceAccount token and uses in-cluster configuration to reach the local
`/openid/v1/jwks`. The projected ServiceAccount namespace file must also be
available so the helper can check the Service audience's namespace. The
audience still comes from the webhook registration, not an incoming request.
The ServiceAccount must be allowed to GET the JWKS.

Create `kittens-tls` with a TLS certificate valid for `kittens.kittens-demo.svc`,
signed by the CA you'll put in `caBundle` below:

```sh
kubectl create namespace kittens-demo
kubectl -n kittens-demo create secret tls kittens-tls --cert=tls.crt --key=tls.key
```

Replace `YOUR_IMAGE` with the image containing the example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: {name: kittens, namespace: kittens-demo}
spec:
  replicas: 1
  selector:
    matchLabels: {app: kittens}
  template:
    metadata:
      labels: {app: kittens}
    spec:
      automountServiceAccountToken: true
      containers:
      - name: webhook
        image: YOUR_IMAGE
        command: ["/webhook"]
        args: ["-tls-cert=/tls/tls.crt", "-tls-key=/tls/tls.key"]
        env:
        - name: WEBHOOK_LOVE_KITTENS_AUDIENCE
          value: https://kittens.kittens-demo.svc:443/validate
        - name: WEBHOOK_LOVE_KITTENS_ISSUER
          value: ""
        ports:
        - {containerPort: 8443}
        volumeMounts:
        - {name: tls, mountPath: /tls, readOnly: true}
      volumes:
      - name: tls
        secret: {secretName: kittens-tls}
---
apiVersion: v1
kind: Service
metadata: {name: kittens, namespace: kittens-demo}
spec:
  selector: {app: kittens}
  ports:
  - {port: 443, targetPort: 8443}
```

### Register with the calling API server

The API server must separately be configured to send a webhook-bound bearer token
with the matching audience and an `admissionReviewAPIGroups` attestation. This
library does not mint or attach tokens; an ordinary ServiceAccount token is not
sufficient. The core ConfigMap example requires the `"*"` attestation.

Replace `BASE64_SERVING_CA` with the base64-encoded PEM CA that signed `tls.crt`:

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata: {name: kittens}
webhooks:
- name: kittens.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  failurePolicy: Fail
  namespaceSelector:
    matchLabels: {kubernetes.io/metadata.name: kittens-demo}
  clientConfig:
    service: {name: kittens, namespace: kittens-demo, port: 443, path: /validate}
    caBundle: BASE64_SERVING_CA
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE", "UPDATE"]
    resources: ["configmaps"]
```

The registration is limited to the disposable `kittens-demo` namespace.
`failurePolicy: Fail` prevents authentication/transport failures from being ignored.
Once sender authentication is configured:

```sh
kubectl -n kittens-demo create configmap love-kittens-demo # allowed
kubectl -n kittens-demo create configmap dogs-only         # denied
```

## 2. On-cluster: override issuer and Service port

If projected ServiceAccount credentials or the local JWKS endpoint are
unavailable, explicitly configure the **actual issuer of the webhook tokens**.
This selects remote OIDC discovery even though the webhook runs in-cluster.
Here the Service also uses port 9443 rather than 443. Change only these fields
from scenario 1 (`# ...` marks omitted content; these are not apply-ready YAML):

```yaml
# Deployment
spec:
  template:
    spec:
      automountServiceAccountToken: false
      containers:
      - name: webhook
        # ... keep image, command, TLS args and volume mount ...
        env:
        - name: WEBHOOK_LOVE_KITTENS_AUDIENCE
          value: https://kittens.kittens-demo.svc:9443/validate
        - name: WEBHOOK_LOVE_KITTENS_ISSUER
          value: https://issuer.example.com
---
# Service
spec:
  ports: [{port: 9443, targetPort: 8443}]
---
# ValidatingWebhookConfiguration
webhooks:
- name: kittens.example.com
  clientConfig:
    service: {name: kittens, namespace: kittens-demo, port: 9443, path: /validate}
    caBundle: BASE64_SERVING_CA
  # ... keep the rules, namespaceSelector and failurePolicy ...
```

The issuer's discovery and JWKS endpoints must be reachable without credentials
over HTTPS trusted by the example's default HTTP client. If **only** the Service
port changes, keep the issuer empty and change just the audience, Service port
and registration.

## 3. Off-cluster: explicit issuer and URL audience

For a webhook hosted in another cluster, reuse scenario 1's Deployment there,
changing these fields and exposing `https://kittens.example.com:8443/validate`:

```yaml
# Hosting-cluster Deployment (fragment; other fields are unchanged)
spec:
  template:
    spec:
      automountServiceAccountToken: false
      containers:
      - name: webhook
        # ... keep image, command and TLS volume mount ...
        env:
        - name: WEBHOOK_LOVE_KITTENS_AUDIENCE
          value: https://kittens.example.com:8443/validate
        - name: WEBHOOK_LOVE_KITTENS_ISSUER
          value: https://issuer.example.com
```

Replace the hosting cluster's `kittens-tls` with a certificate valid for
`kittens.example.com`. Put its signing CA in the **calling** cluster's
`caBundle` and register the URL (fragment; other fields are unchanged):

```yaml
# ValidatingWebhookConfiguration in the calling cluster
webhooks:
- name: kittens.example.com
  clientConfig:
    url: https://kittens.example.com:8443/validate
    caBundle: BASE64_SERVING_CA
  # ... keep the rules, namespaceSelector and failurePolicy ...
```

The issuer must match the calling cluster's token issuer. Discovery and JWKS
must permit credential-free access. The example's nil HTTP client trusts system
roots; for a private issuer CA, pass a credential-free client that trusts it to
`webhookAuth.WithRemoteIssuer` instead. No ServiceAccount credential from the
hosting cluster is needed.

## Appendix: Audience inference and trust

An **attestation** is an issuer-signed claim. The current JWT's
`kubernetes.io.attestations.admissionReviewAPIGroups` value says which API
group an authenticated caller may ask **this webhook** to review. The KEP's
draft calls this `webhook-authentication.k8s.io/allowedAPIGroup`; the current
wire name is `admissionReviewAPIGroups`. Its `"*"` value means *any API group
at this webhook*, **not** RBAC `verbs: ["*"]` or permission to read and write
every Kubernetes resource.

The KEP proposes `"*"` for kube-apiserver, which serves resources across API
groups. Its older text permits `""` for the core group, but current validation
rejects that ambiguous value in authorization checks. Thus even this core
ConfigMap example uses the API server's `"*"` attestation. Issuing that token
requires its acquisition ServiceAccount to have the `attest` verb on the
synthetic `authentication.k8s.io/admissionReviewAPIGroups` resource with
literal resource name `"*"`. The KEP calls that synthetic resource
`webhook-authentication.k8s.io/apigroups` and recommends reserving this
broader grant for kube-apiserver;
aggregated API servers should instead attest only to groups they serve. That
is a recommendation, **not** automatic enforcement: granting another
ServiceAccount `attest` on `"*"` broadens what it can ask a webhook about.

For named groups, the current TokenRequest code matches both the group and
the audience against a webhook in the bound configuration. The `"*"` case
skips that **issuer-side** match, but still requires authorization to attest
`"*"` and an existing bound configuration. The receiver must still check
the token's signature, issuer, expiry, configuration binding and **its own
independently established expected audience**. For this Service registration,
that audience is `https://kittens.kittens-demo.svc:443/validate` (Service
port 443, **not** container `targetPort` 8443); a URL registration uses
`clientConfig.url` verbatim. Neither `"*"` nor the token's `aud` tells the
receiver *which* registered endpoint belongs to it.

| Candidate input | What it provides | What it does not prove |
|---|---|---|
| `KITTENS_SERVICE_HOST` / `KITTENS_SERVICE_PORT` | The ClusterIP and **first Service port** of `kittens`, if kubelet discovered it when the container started and Service links were enabled. Named-port and `KITTENS_PORT*` variables may expose other ports. | That `kittens` selects this Pod, is *this* webhook's registered Service, or which port/path the registration uses. |
| `KUBERNETES_SERVICE_HOST` / `KUBERNETES_SERVICE_PORT` | Typically the default API-server Service, even with Service links disabled. | The webhook's own Service. |
| Projected ServiceAccount namespace file; `rest.InClusterConfig` | The file supplies Pod namespace; in-cluster config supplies the local token, issuer hint, API-server connection and CA, **not** the namespace. | Registered webhook Service, port, path, or URL. |
| Downward API fields such as `POD_NAME` / `POD_NAMESPACE` | Pod metadata **if configured** in its manifest. | A Service or webhook registration identity. |
| `r.URL.Path`, HTTP `Host`, TLS SNI | Request-supplied routing hints; an exact mux mount such as `/validate` can restrict paths reaching *that handler*. | The registered Service/port or URL; a reverse proxy may rewrite paths, and clients choose Host/SNI. |
| TLS serving certificate names | Hosts for which this server can present a certificate. | Which registration points here, or its Service port/path; one certificate may cover multiple hosts. |
| `WEBHOOK_LOVE_KITTENS_AUDIENCE` / `WEBHOOK_LOVE_KITTENS_ISSUER` | Values explicitly set by **this example's Deployment**. | They are not kubelet-provided defaults; the audience is trusted only if copied from the registration. |

Kubelet offers Service links for eligible Services in the Pod's namespace, not
just ones selecting this Pod. Headless/no-ClusterIP Services are omitted; links
can be disabled, and a Service created after container startup will not appear
in the existing environment. Another Deployment cannot rewrite this Pod's env,
and an attacker need not run on the same node: the relevant access is the
ability to send an HTTP request to the webhook.

Suppose both `kittens` and `puppies` Services exist and each registers
`/validate`. Those env vars plus a first request for `/validate` do not tell us
which Service registered this handler. A first unauthenticated request could
poison a cached choice and deny subsequent legitimate calls. Accepting a
**genuinely signed** sibling-webhook token after choosing the expected audience
from its `aud` or from a spoofed request Host would also permit cross-webhook
replay; this does **not** mean unsigned tokens pass signature verification.
The server's first request might instead be a readiness probe; even when an
exact mux route restricts the path, it does not identify the Service or port.

**Open question:** What receiver-owned source identifies each handler's exact
registered URL or Service name, namespace, port and path? An installer can
supply it; an authenticated API lookup would require RBAC plus an independently
established, unambiguous mapping. Both must fail closed if identity is missing
or ambiguous. `NewVerifier` for already-decoded requests has no HTTP path.
Until such a source is demonstrated, `WithAudience` supplies the expected
in-cluster audience independently of requests and tokens.

Sources: `pkg/kubelet/kubelet_pods.go` (`getServiceEnvVarMap`),
`pkg/kubelet/envvars/envvars.go` (`FromServices`),
`pkg/registry/core/serviceaccount/storage/token.go` (`getAndValidateWebhookConfig`,
`validateWebhookAudience`), and KEP-6060's
[authorization checks](https://github.com/kubernetes/enhancements/tree/master/keps/sig-auth/6060-api-server-authentication-to-webhooks#authorization-checks)
and [token verification](https://github.com/kubernetes/enhancements/tree/master/keps/sig-auth/6060-api-server-authentication-to-webhooks#token-verification).

## Appendix: Gatekeeper's three webhook audiences

This is the **default Gatekeeper v3.23.1 installation**, not a claim about a
particular cluster. Three controller-manager replicas serve webhooks behind
`gatekeeper-webhook-service` in `gatekeeper-system`; its Service port **443**
forwards to the Pods' HTTPS listener on **8443**. The separate audit Pod is not
selected by that Service. A Service forwards to a Pod and port; it does not
choose a handler based on HTTP path.

| Registered webhook | Path at the same listener | Expected token audience (current issuer) |
|---|---|---|
| Validating admission | `/v1/admit` | `https://gatekeeper-webhook-service.gatekeeper-system.svc:443/v1/admit` |
| Validating namespace labels | `/v1/admitlabel` | `https://gatekeeper-webhook-service.gatekeeper-system.svc:443/v1/admitlabel` |
| Mutating admission | `/v1/mutate` | `https://gatekeeper-webhook-service.gatekeeper-system.svc:443/v1/mutate` |

What might be visible **inside a controller-manager Pod** (values illustrative;
Service links are only a startup snapshot):

```text
# Gatekeeper Service links: only if enabled and the Service existed at Pod start:
GATEKEEPER_WEBHOOK_SERVICE_SERVICE_HOST=<Gatekeeper Service ClusterIP>
GATEKEEPER_WEBHOOK_SERVICE_SERVICE_PORT=443          # first Service port, NOT listener port 8443

# Kubelet's special API server Service (if discovered), even with links disabled:
KUBERNETES_SERVICE_HOST=<API server Service ClusterIP>
KUBERNETES_SERVICE_PORT=<API server Service port>     # NOT Gatekeeper's Service

# Explicitly provided by Gatekeeper's Pod manifest using the Downward API:
POD_NAMESPACE=gatekeeper-system

# Gatekeeper process settings, NOT kubelet Service-link variables:
--cert-service-name=gatekeeper-webhook-service        # default; used for certificate DNS name
--port=8443                                           # container listener, NOT Service port

# NOT provided by these Service links: /v1/admit, /v1/admitlabel, /v1/mutate,
# any webhook-configuration identity, or an exact expected token audience.
```

Gatekeeper's audit Pod can see the same namespace Service links **if** they
exist when it starts, even though the webhook Service does not select it.
Placement on Duck, Chicken or Turtle does not affect that: Service links are
based on namespace, not node or Service selector. Turning off
`spec.template.spec.enableServiceLinks` removes same-namespace Service hints;
it does not reveal the correct audience. Gatekeeper's TLS readiness checker
can also send `GET /` to port 8443 before any admission call. That request
need not reach an admission handler mounted on an exact path, but the first
request to the listener is not a trusted way to discover a webhook path.

The three audience strings can be **rendered from the same installer-owned
Service and webhook-registration values**, without asking an operator to type
them. They are still *three distinct audiences*: a signed token for
`/v1/admit` should not authenticate at `/v1/mutate` merely because both
handlers run in the same process. Each route needs its own expected audience;
sharing issuer or key discovery does not mean sharing authorization scope.

Gatekeeper's `--cert-service-name` and `POD_NAMESPACE` describe its expected
certificate DNS name, **not** proof of the live webhook registrations. Helm can
replace the `/v1/admit` or `/v1/mutate` Service destination with an explicit
URL while keeping `/v1/admitlabel` Service-routed. Any installer-generated
audience must use the *actual* registered `clientConfig` per route, or fail
closed if that mapping is unavailable. The upstream manifests do not currently
inject these three expected-audience values into the Pods.

Upstream evidence (pinned to Gatekeeper
[`v3.23.1`](https://github.com/open-policy-agent/gatekeeper/commit/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0)):
[Service and selector](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/deploy/gatekeeper.yaml#L5725-L5741),
[controller-manager Pod](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/deploy/gatekeeper.yaml#L5852-L5967),
[mutating and validating registrations](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/deploy/gatekeeper.yaml#L5984-L6105),
[certificate Service flag](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/main.go#L114-L123),
[TLS readiness check](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/pkg/webhook/health_check.go#L21-L35),
[Helm validating URL override](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/charts/gatekeeper/templates/gatekeeper-validating-webhook-configuration-validatingwebhookconfiguration.yaml#L19-L27),
[Helm mutating URL override](https://github.com/open-policy-agent/gatekeeper/blob/2d0b6ad97cf3a8b8436a00a1f4b07b3a56b92ca0/charts/gatekeeper/templates/gatekeeper-mutating-webhook-configuration-mutatingwebhookconfiguration.yaml#L19-L27).
