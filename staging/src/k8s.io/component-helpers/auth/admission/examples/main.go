/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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

const (
	resourceNameMustContainForAdmission = "love-kittens"

	// Matches the Service and ValidatingWebhookConfiguration in examples/in-cluster.yaml.
	serviceName = "kittens"
	servicePath = "/validate"
	servicePort = 443     // Service spec.ports[].port (clientConfig.service.port), not container targetPort
	targetPort  = ":8443" // Container listen port (Service spec.ports[].targetPort)
)

// customSubresourceGroupMismatchFunc shows how an off-cluster webhook can check
// the built-in cross-group subresource defaults first and then allow an
// additional custom cross-group subresource. This function is only called when
// req.Kind.Group differs from req.Resource.Group on a subresource; pass nil to
// WithIssuer to use authadmission.DefaultStaticAllowedSubresourceGroupMismatch
// directly.
func customSubresourceGroupMismatchFunc(ctx context.Context, req *admissionv1.AdmissionRequest) (bool, error) {
	allowed, err := authadmission.DefaultStaticAllowedSubresourceGroupMismatch(ctx, req)
	if err != nil || allowed {
		return allowed, err
	}
	return req.Resource == (metav1.GroupVersionResource{Group: "pets.example.com", Version: "v1", Resource: "kittens"}) &&
		req.SubResource == "adopt" &&
		req.Kind == (metav1.GroupVersionKind{Group: "adoption.example.com", Version: "v1", Kind: "AdoptionRequest"}), nil
}

func main() {
	cert := flag.String("tls-cert", "tls.crt", "PEM certificate for the webhook's HTTPS host")
	key := flag.String("tls-key", "tls.key", "PEM private key for that certificate")
	flag.Parse()

	// Default to in-cluster Service verification; use WithIssuer for explicit OIDC discovery.
	verificationOption := authadmission.WithServiceAudience(serviceName, servicePath, servicePort)
	if issuer := os.Getenv("WEBHOOK_LOVE_KITTENS_ISSUER"); issuer != "" {
		audience := os.Getenv("WEBHOOK_LOVE_KITTENS_AUDIENCE")
		verificationOption = authadmission.WithIssuer(issuer, audience, nil, customSubresourceGroupMismatchFunc)
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
	mux.Handle(servicePath, handler)
	server := &http.Server{Addr: targetPort, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(server.ListenAndServeTLS(*cert, *key))
}
