/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/group"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	"k8s.io/client-go/rest"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/bootstrap"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type bootstrapSecrets []*corev1.Secret

func (b bootstrapSecrets) List(selector labels.Selector) (ret []*corev1.Secret, err error) {
	return b, nil
}

func (b bootstrapSecrets) Get(name string) (*corev1.Secret, error) {
	return b[0], nil
}

// TestBootstrapTokenAuth tests the bootstrap token auth provider
func TestBootstrapTokenAuth(t *testing.T) {
	validTokenID := "token1"
	validSecret := "validtokensecret"
	var bootstrapSecretValid = &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceSystem,
			Name:      bootstrapapi.BootstrapTokenSecretPrefix,
		},
		Type: corev1.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:               []byte(validTokenID),
			bootstrapapi.BootstrapTokenSecretKey:           []byte(validSecret),
			bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
		},
	}
	var bootstrapSecretInvalid = &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceSystem,
			Name:      bootstrapapi.BootstrapTokenSecretPrefix,
		},
		Type: corev1.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:               []byte(validTokenID),
			bootstrapapi.BootstrapTokenSecretKey:           []byte("invalid"),
			bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
		},
	}
	tokenExpiredTime := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	var expiredBootstrapToken = &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceSystem,
			Name:      bootstrapapi.BootstrapTokenSecretPrefix,
		},
		Type: corev1.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:               []byte(validTokenID),
			bootstrapapi.BootstrapTokenSecretKey:           []byte("invalid"),
			bootstrapapi.BootstrapTokenUsageAuthentication: []byte("true"),
			bootstrapapi.BootstrapTokenExpirationKey:       []byte(tokenExpiredTime),
		},
	}
	type request struct {
		verb        string
		URL         string
		body        string
		statusCodes map[int]bool // Set of expected resp.StatusCode if all goes well.
	}
	tests := []struct {
		name    string
		request request
		secret  *corev1.Secret
	}{
		{
			name:    "valid token",
			request: request{verb: "GET", URL: path("pods", "", ""), body: "", statusCodes: integration.Code200},
			secret:  bootstrapSecretValid,
		},
		{
			name:    "invalid token format",
			request: request{verb: "GET", URL: path("pods", "", ""), body: "", statusCodes: integration.Code401},
			secret:  bootstrapSecretInvalid,
		},
		{
			name:    "invalid token expired",
			request: request{verb: "GET", URL: path("pods", "", ""), body: "", statusCodes: integration.Code401},
			secret:  expiredBootstrapToken,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			authenticator := group.NewAuthenticatedGroupAdder(bearertoken.New(bootstrap.NewTokenAuthenticator(bootstrapSecrets{test.secret})))

			kubeClient, kubeConfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
				ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
					opts.Authorization.Modes = []string{"AlwaysAllow"}
				},
				ModifyServerConfig: func(config *controlplane.Config) {
					config.GenericConfig.Authentication.Authenticator = authenticator
				},
			})
			defer tearDownFn()

			ns := framework.CreateNamespaceOrDie(kubeClient, "auth-bootstrap-token", t)
			defer framework.DeleteNamespaceOrDie(kubeClient, ns, t)

			previousResourceVersion := make(map[string]float64)
			transport, err := rest.TransportFor(kubeConfig)
			if err != nil {
				t.Fatal(err)
			}

			token := validTokenID + "." + validSecret
			var bodyStr string
			if test.request.body != "" {
				sub := ""
				if test.request.verb == "PUT" {
					// For update operations, insert previous resource version
					if resVersion := previousResourceVersion[getPreviousResourceVersionKey(test.request.URL, "")]; resVersion != 0 {
						sub += fmt.Sprintf(",\r\n\"resourceVersion\": \"%v\"", resVersion)
					}
					sub += fmt.Sprintf(",\r\n\"namespace\": %q", ns.Name)
				}
				bodyStr = fmt.Sprintf(test.request.body, sub)
			}
			test.request.body = bodyStr
			bodyBytes := bytes.NewReader([]byte(bodyStr))
			req, err := http.NewRequest(test.request.verb, kubeConfig.Host+test.request.URL, bodyBytes)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
			if test.request.verb == "PATCH" {
				req.Header.Set("Content-Type", "application/merge-patch+json")
			}

			func() {
				resp, err := transport.RoundTrip(req)
				if err != nil {
					t.Logf("case %v", test.name)
					t.Fatalf("unexpected error: %v", err)
				}
				defer resp.Body.Close()
				b, _ := io.ReadAll(resp.Body)
				if _, ok := test.request.statusCodes[resp.StatusCode]; !ok {
					t.Logf("case %v", test.name)
					t.Errorf("Expected status one of %v, but got %v", test.request.statusCodes, resp.StatusCode)
					t.Errorf("Body: %v", string(b))
				} else {
					if test.request.verb == "POST" {
						// For successful create operations, extract resourceVersion
						id, currentResourceVersion, err := parseResourceVersion(b)
						if err == nil {
							key := getPreviousResourceVersionKey(test.request.URL, id)
							previousResourceVersion[key] = currentResourceVersion
						}
					}
				}

			}()
		})
	}
}
