/*
Copyright 2016 The Kubernetes Authors.

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

package webhook

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/authentication/v1beta1"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api/v1"
)

// Service mocks a remote authentication service.
type Service interface {
	// Review looks at the TokenReviewSpec and provides an authentication
	// response in the TokenReviewStatus.
	Review(*v1beta1.TokenReview)
	HTTPStatusCode() int
}

// NewTestServer wraps a Service as an httptest.Server.
func NewTestServer(s Service, cert, key, caCert []byte) (*httptest.Server, error) {
	var tlsConfig *tls.Config
	if cert != nil {
		cert, err := tls.X509KeyPair(cert, key)
		if err != nil {
			return nil, err
		}
		tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}}
	}

	if caCert != nil {
		rootCAs := x509.NewCertPool()
		rootCAs.AppendCertsFromPEM(caCert)
		if tlsConfig == nil {
			tlsConfig = &tls.Config{}
		}
		tlsConfig.ClientCAs = rootCAs
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	serveHTTP := func(w http.ResponseWriter, r *http.Request) {
		var review v1beta1.TokenReview
		if err := json.NewDecoder(r.Body).Decode(&review); err != nil {
			http.Error(w, fmt.Sprintf("failed to decode body: %v", err), http.StatusBadRequest)
			return
		}
		if s.HTTPStatusCode() < 200 || s.HTTPStatusCode() >= 300 {
			http.Error(w, "HTTP Error", s.HTTPStatusCode())
			return
		}
		s.Review(&review)
		type userInfo struct {
			Username string   `json:"username"`
			UID      string   `json:"uid"`
			Groups   []string `json:"groups"`
		}
		type status struct {
			Authenticated bool     `json:"authenticated"`
			User          userInfo `json:"user"`
		}
		resp := struct {
			APIVersion string `json:"apiVersion"`
			Status     status `json:"status"`
		}{
			APIVersion: v1beta1.SchemeGroupVersion.String(),
			Status: status{
				review.Status.Authenticated,
				userInfo{
					Username: review.Status.User.Username,
					UID:      review.Status.User.UID,
					Groups:   review.Status.User.Groups,
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(serveHTTP))
	server.TLS = tlsConfig
	server.StartTLS()
	return server, nil
}

// A service that can be set to say yes or no to authentication requests.
type mockService struct {
	allow      bool
	statusCode int
}

func (m *mockService) Review(r *v1beta1.TokenReview) {
	r.Status.Authenticated = m.allow
	if m.allow {
		r.Status.User.Username = "realHooman@email.com"
	}
}
func (m *mockService) Allow()              { m.allow = true }
func (m *mockService) Deny()               { m.allow = false }
func (m *mockService) HTTPStatusCode() int { return m.statusCode }

// newTokenAuthenticator creates a temporary kubeconfig file from the provided
// arguments and attempts to load a new WebhookTokenAuthenticator from it.
func newTokenAuthenticator(serverURL string, clientCert, clientKey, ca []byte, cacheTime time.Duration) (*WebhookTokenAuthenticator, error) {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		return nil, err
	}
	p := tempfile.Name()
	defer os.Remove(p)
	config := v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{Server: serverURL, CertificateAuthorityData: ca},
			},
		},
		AuthInfos: []v1.NamedAuthInfo{
			{
				AuthInfo: v1.AuthInfo{ClientCertificateData: clientCert, ClientKeyData: clientKey},
			},
		},
	}
	if err := json.NewEncoder(tempfile).Encode(config); err != nil {
		return nil, err
	}
	return newWithBackoff(p, cacheTime, 0)
}

func TestTLSConfig(t *testing.T) {
	tests := []struct {
		test                            string
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		wantErr                         bool
	}{
		{
			test:       "TLS setup between client and server",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
		},
		{
			test:       "Server does not require client auth",
			clientCA:   caCert,
			serverCert: serverCert, serverKey: serverKey,
		},
		{
			test:       "Server does not require client auth, client provides it",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey,
		},
		{
			test:       "Client does not trust server",
			clientCert: clientCert, clientKey: clientKey,
			serverCert: serverCert, serverKey: serverKey,
			wantErr: true,
		},
		{
			test:       "Server does not trust client",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: badCACert,
			wantErr: true,
		},
		{
			// Plugin does not support insecure configurations.
			test:    "Server is using insecure connection",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		// Use a closure so defer statements trigger between loop iterations.
		func() {
			service := new(mockService)
			service.statusCode = 200

			server, err := NewTestServer(service, tt.serverCert, tt.serverKey, tt.serverCA)
			if err != nil {
				t.Errorf("%s: failed to create server: %v", tt.test, err)
				return
			}
			defer server.Close()

			wh, err := newTokenAuthenticator(server.URL, tt.clientCert, tt.clientKey, tt.clientCA, 0)
			if err != nil {
				t.Errorf("%s: failed to create client: %v", tt.test, err)
				return
			}

			// Allow all and see if we get an error.
			service.Allow()
			_, authenticated, err := wh.AuthenticateToken("t0k3n")
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error making authorization request: %v", err)
				}
				return
			}
			if !authenticated {
				t.Errorf("%s: failed to authenticate token", tt.test)
				return
			}

			service.Deny()
			_, authenticated, err = wh.AuthenticateToken("t0k3n")
			if err != nil {
				t.Errorf("%s: unexpectedly failed AuthenticateToken", tt.test)
			}
			if authenticated {
				t.Errorf("%s: incorrectly authenticated token", tt.test)
			}
		}()
	}
}

// recorderService records all token review requests, and responds with the
// provided TokenReviewStatus.
type recorderService struct {
	lastRequest v1beta1.TokenReview
	response    v1beta1.TokenReviewStatus
}

func (rec *recorderService) Review(r *v1beta1.TokenReview) {
	rec.lastRequest = *r
	r.Status = rec.response
}

func (rec *recorderService) HTTPStatusCode() int { return 200 }

func TestWebhookTokenAuthenticator(t *testing.T) {
	serv := &recorderService{}

	s, err := NewTestServer(serv, serverCert, serverKey, caCert)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	wh, err := newTokenAuthenticator(s.URL, clientCert, clientKey, caCert, 0)
	if err != nil {
		t.Fatal(err)
	}

	expTypeMeta := unversioned.TypeMeta{
		APIVersion: "authentication.k8s.io/v1beta1",
		Kind:       "TokenReview",
	}

	tests := []struct {
		serverResponse        v1beta1.TokenReviewStatus
		expectedAuthenticated bool
		expectedUser          *user.DefaultInfo
	}{
		// Successful response should pass through all user info.
		{
			serverResponse: v1beta1.TokenReviewStatus{
				Authenticated: true,
				User: v1beta1.UserInfo{
					Username: "somebody",
				},
			},
			expectedAuthenticated: true,
			expectedUser: &user.DefaultInfo{
				Name: "somebody",
			},
		},
		{
			serverResponse: v1beta1.TokenReviewStatus{
				Authenticated: true,
				User: v1beta1.UserInfo{
					Username: "person@place.com",
					UID:      "abcd-1234",
					Groups:   []string{"stuff-dev", "main-eng"},
				},
			},
			expectedAuthenticated: true,
			expectedUser: &user.DefaultInfo{
				Name:   "person@place.com",
				UID:    "abcd-1234",
				Groups: []string{"stuff-dev", "main-eng"},
			},
		},
		// Unauthenticated shouldn't even include extra provided info.
		{
			serverResponse: v1beta1.TokenReviewStatus{
				Authenticated: false,
				User: v1beta1.UserInfo{
					Username: "garbage",
					UID:      "abcd-1234",
					Groups:   []string{"not-actually-used"},
				},
			},
			expectedAuthenticated: false,
			expectedUser:          nil,
		},
		{
			serverResponse: v1beta1.TokenReviewStatus{
				Authenticated: false,
			},
			expectedAuthenticated: false,
			expectedUser:          nil,
		},
	}
	token := "my-s3cr3t-t0ken"
	for i, tt := range tests {
		serv.response = tt.serverResponse
		user, authenticated, err := wh.AuthenticateToken(token)
		if err != nil {
			t.Errorf("case %d: authentication failed: %v", i, err)
			continue
		}
		if serv.lastRequest.Spec.Token != token {
			t.Errorf("case %d: Server did not see correct token. Got %q, expected %q.",
				i, serv.lastRequest.Spec.Token, token)
		}
		if !reflect.DeepEqual(serv.lastRequest.TypeMeta, expTypeMeta) {
			t.Errorf("case %d: Server did not see correct TypeMeta. Got %v, expected %v",
				i, serv.lastRequest.TypeMeta, expTypeMeta)
		}
		if authenticated != tt.expectedAuthenticated {
			t.Errorf("case %d: Plugin returned incorrect authentication response. Got %t, expected %t.",
				i, authenticated, tt.expectedAuthenticated)
		}
		if user != nil && tt.expectedUser != nil && !reflect.DeepEqual(user, tt.expectedUser) {
			t.Errorf("case %d: Plugin returned incorrect user. Got %v, expected %v",
				i, user, tt.expectedUser)
		}
	}
}

type authenticationUserInfo v1beta1.UserInfo

func (a *authenticationUserInfo) GetName() string     { return a.Username }
func (a *authenticationUserInfo) GetUID() string      { return a.UID }
func (a *authenticationUserInfo) GetGroups() []string { return a.Groups }

func (a *authenticationUserInfo) GetExtra() map[string][]string {
	if a.Extra == nil {
		return nil
	}
	ret := map[string][]string{}
	for k, v := range a.Extra {
		ret[k] = []string(v)
	}

	return ret
}

// Ensure v1beta1.UserInfo contains the fields necessary to implement the
// user.Info interface.
var _ user.Info = (*authenticationUserInfo)(nil)

// TestWebhookCache verifies that error responses from the server are not
// cached, but successful responses are.
func TestWebhookCache(t *testing.T) {
	serv := new(mockService)
	s, err := NewTestServer(serv, serverCert, serverKey, caCert)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// Create an authenticator that caches successful responses "forever" (100 days).
	wh, err := newTokenAuthenticator(s.URL, clientCert, clientKey, caCert, 2400*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	token := "t0k3n"
	serv.allow = true
	serv.statusCode = 500
	if _, _, err := wh.AuthenticateToken(token); err == nil {
		t.Errorf("Webhook returned HTTP 500, but authorizer reported success.")
	}
	serv.statusCode = 404
	if _, _, err := wh.AuthenticateToken(token); err == nil {
		t.Errorf("Webhook returned HTTP 404, but authorizer reported success.")
	}
	serv.statusCode = 200
	if _, _, err := wh.AuthenticateToken(token); err != nil {
		t.Errorf("Webhook returned HTTP 200, but authorizer reported unauthorized.")
	}
	serv.statusCode = 500
	if _, _, err := wh.AuthenticateToken(token); err != nil {
		t.Errorf("Webhook should have successful response cached, but authorizer reported unauthorized.")
	}
	// For a different request, webhook should be called again.
	token = "an0th3r_t0k3n"
	serv.statusCode = 500
	if _, _, err := wh.AuthenticateToken(token); err == nil {
		t.Errorf("Webhook returned HTTP 500, but authorizer reported success.")
	}
	serv.statusCode = 200
	if _, _, err := wh.AuthenticateToken(token); err != nil {
		t.Errorf("Webhook returned HTTP 200, but authorizer reported unauthorized.")
	}
	serv.statusCode = 500
	if _, _, err := wh.AuthenticateToken(token); err != nil {
		t.Errorf("Webhook should have successful response cached, but authorizer reported unauthorized.")
	}
}
