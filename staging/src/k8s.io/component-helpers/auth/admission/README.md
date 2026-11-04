# Authenticated admission webhook

This toy webhook uses `k8s.io/component-helpers/auth/admission` to verify a
webhook-bound bearer token and its AdmissionReview before running its policy.
It admits only names containing `love-kittens`; the Kitten Council rejects the
rest. Real admission policies should be less arbitrary.

## Example Webhook

The helper adds one package import, one verification option and the
`authadmission.WithTokenVerification` call. The rest below is the toy policy and
ordinary HTTPS-server setup:

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
	authadmission "k8s.io/component-helpers/auth/admission"
)

const resourceNameMustContainForAdmission = "love-kittens"

func main() {
	cert := flag.String("tls-cert", "tls.crt", "PEM certificate for the webhook's HTTPS host")
	key := flag.String("tls-key", "tls.key", "PEM private key for that certificate")
	flag.Parse()

	// Default to in-cluster Service verification; use WithIssuer for explicit OIDC discovery.
	verificationOption := authadmission.WithServiceAudience("kittens", "/validate", 443)
	if issuer := os.Getenv("WEBHOOK_LOVE_KITTENS_ISSUER"); issuer != "" {
		audience := os.Getenv("WEBHOOK_LOVE_KITTENS_AUDIENCE")
		verificationOption = authadmission.WithIssuer(issuer, audience, nil)
	}
	kittenPolicy := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		if strings.Contains(req.Name, resourceNameMustContainForAdmission) || len(req.Name) == 0 {
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
	// Pass the webhook server's root lifecycle context (typically tied to signal handling).
	handler, err := authadmission.WithTokenVerification(context.Background(), kittenPolicy, verificationOption)
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/validate", handler)
	server := &http.Server{Addr: ":8443", Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(server.ListenAndServeTLS(*cert, *key))
}
```

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
