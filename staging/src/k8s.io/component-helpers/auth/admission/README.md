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
Take the audience from the registered webhook endpoint—not the incoming token
or `Host` header. Use the exact `clientConfig.url`, or
`https://<name>.<namespace>.svc:<port><path>` for a Service reference (default
port 443, path `/`). This example uses Service port 443, not container port 8443.

The helper decodes the AdmissionReview, encodes the response and sets its UID.
Failed authentication returns HTTP 401 without reading the body; policy denials
return an AdmissionReview with `Allowed: false`.

## 1. On-cluster: infer issuer and signing keys

The webhook runs in the calling cluster. Leave `WEBHOOK_LOVE_KITTENS_ISSUER`
empty (as below) or unset. The helper gets the issuer from the Pod's projected
ServiceAccount token and uses in-cluster configuration to reach the local
`/openid/v1/jwks`. The audience still comes from the webhook registration, not
an incoming request. The ServiceAccount must be allowed to GET the JWKS.

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
