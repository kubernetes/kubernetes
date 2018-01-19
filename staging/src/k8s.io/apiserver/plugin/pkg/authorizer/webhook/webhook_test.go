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
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"text/template"
	"time"

	"k8s.io/api/authorization/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/tools/clientcmd/api/v1"
)

func TestNewFromConfig(t *testing.T) {
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	data := struct {
		CA   string
		Cert string
		Key  string
	}{
		CA:   filepath.Join(dir, "ca.pem"),
		Cert: filepath.Join(dir, "clientcert.pem"),
		Key:  filepath.Join(dir, "clientkey.pem"),
	}

	files := []struct {
		name string
		data []byte
	}{
		{data.CA, caCert},
		{data.Cert, clientCert},
		{data.Key, clientKey},
	}
	for _, file := range files {
		if err := ioutil.WriteFile(file.name, file.data, 0400); err != nil {
			t.Fatal(err)
		}
	}

	tests := []struct {
		msg        string
		configTmpl string
		wantErr    bool
	}{
		{
			msg: "a single cluster and single user",
			configTmpl: `
clusters:
- cluster:
    certificate-authority: {{ .CA }}
    server: https://authz.example.com
  name: foobar
users:
- name: a cluster
  user:
    client-certificate: {{ .Cert }}
    client-key: {{ .Key }}
`,
			wantErr: true,
		},
		{
			msg: "multiple clusters with no context",
			configTmpl: `
clusters:
- cluster:
    certificate-authority: {{ .CA }}
    server: https://authz.example.com
  name: foobar
- cluster:
    certificate-authority: a bad certificate path
    server: https://authz.example.com
  name: barfoo
users:
- name: a name
  user:
    client-certificate: {{ .Cert }}
    client-key: {{ .Key }}
`,
			wantErr: true,
		},
		{
			msg: "multiple clusters with a context",
			configTmpl: `
clusters:
- cluster:
    certificate-authority: a bad certificate path
    server: https://authz.example.com
  name: foobar
- cluster:
    certificate-authority: {{ .CA }}
    server: https://authz.example.com
  name: barfoo
users:
- name: a name
  user:
    client-certificate: {{ .Cert }}
    client-key: {{ .Key }}
contexts:
- name: default
  context:
    cluster: barfoo
    user: a name
current-context: default
`,
			wantErr: false,
		},
		{
			msg: "cluster with bad certificate path specified",
			configTmpl: `
clusters:
- cluster:
    certificate-authority: a bad certificate path
    server: https://authz.example.com
  name: foobar
- cluster:
    certificate-authority: {{ .CA }}
    server: https://authz.example.com
  name: barfoo
users:
- name: a name
  user:
    client-certificate: {{ .Cert }}
    client-key: {{ .Key }}
contexts:
- name: default
  context:
    cluster: foobar
    user: a name
current-context: default
`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		// Use a closure so defer statements trigger between loop iterations.
		err := func() error {
			tempfile, err := ioutil.TempFile("", "")
			if err != nil {
				return err
			}
			p := tempfile.Name()
			defer os.Remove(p)

			tmpl, err := template.New("test").Parse(tt.configTmpl)
			if err != nil {
				return fmt.Errorf("failed to parse test template: %v", err)
			}
			if err := tmpl.Execute(tempfile, data); err != nil {
				return fmt.Errorf("failed to execute test template: %v", err)
			}
			// Create a new authorizer
			sarClient, err := subjectAccessReviewInterfaceFromKubeconfig(p)
			if err != nil {
				return fmt.Errorf("error building sar client: %v", err)
			}
			_, err = newWithBackoff(sarClient, nil, 0, 0, 0)
			return err
		}()
		if err != nil && !tt.wantErr {
			t.Errorf("failed to load plugin from config %q: %v", tt.msg, err)
		}
		if err == nil && tt.wantErr {
			t.Errorf("wanted an error when loading config, did not get one: %q", tt.msg)
		}
	}
}

// Service mocks a remote service.
type Service interface {
	Review(*v1beta1.SubjectAccessReview)
	HTTPStatusCode() int
}

type RulesReviewService interface {
	Review(v1beta1.SubjectRulesReviewSpec) v1beta1.SubjectRulesReviewStatus
	HTTPStatusCode() int
}

// NewTestServer wraps a Service or a RulesReviewService as an httptest.Server. If a
// Service is passed, the server expectes SubjectAccessReviews, if a RulesReviewService
// is passed, the server expects SubjectRulesReviews.
func NewTestServer(service interface{}, cert, key, caCert []byte) (*httptest.Server, error) {
	var (
		authorizerService  Service
		rulesReviewService RulesReviewService
	)
	switch s := service.(type) {
	case Service:
		authorizerService = s
	case RulesReviewService:
		rulesReviewService = s
	default:
		return nil, fmt.Errorf("service must implement Service or RulesReviewService")
	}

	const webhookPath = "/testserver"
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
		if r.Method != "POST" {
			http.Error(w, fmt.Sprintf("unexpected method: %v", r.Method), http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != webhookPath {
			http.Error(w, fmt.Sprintf("unexpected path: %v", r.URL.Path), http.StatusNotFound)
			return
		}

		bodyData, _ := ioutil.ReadAll(r.Body)

		var resp interface{}

		if authorizerService != nil {
			var review v1beta1.SubjectAccessReview
			if err := json.Unmarshal(bodyData, &review); err != nil {
				http.Error(w, fmt.Sprintf("failed to decode body: %v", err), http.StatusBadRequest)
				return
			}

			// ensure we received the serialized review as expected
			if review.APIVersion != "authorization.k8s.io/v1beta1" {
				http.Error(w, fmt.Sprintf("wrong api version: %s", string(bodyData)), http.StatusBadRequest)
				return
			}
			// once we have a successful request, always call the review to record that we were called
			authorizerService.Review(&review)
			if code := authorizerService.HTTPStatusCode(); code < 200 || code >= 300 {
				http.Error(w, "HTTP Error", code)
				return
			}
			type status struct {
				Allowed         bool   `json:"allowed"`
				Reason          string `json:"reason"`
				EvaluationError string `json:"evaluationError"`
			}
			resp = struct {
				APIVersion string `json:"apiVersion"`
				Status     status `json:"status"`
			}{
				APIVersion: v1beta1.SchemeGroupVersion.String(),
				Status:     status{review.Status.Allowed, review.Status.Reason, review.Status.EvaluationError},
			}
		} else {
			var review v1beta1.SubjectRulesReview
			if err := json.Unmarshal(bodyData, &review); err != nil {
				http.Error(w, fmt.Sprintf("failed to decode body: %v", err), http.StatusBadRequest)
				return
			}

			// ensure we received the serialized review as expected
			if review.APIVersion != "authorization.k8s.io/v1beta1" {
				http.Error(w, fmt.Sprintf("wrong api version: %s", string(bodyData)), http.StatusBadRequest)
				return
			}
			// once we have a successful request, always call the review to record that we were called
			status := rulesReviewService.Review(review.Spec)
			resp = v1beta1.SubjectRulesReview{Status: status}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(serveHTTP))
	server.TLS = tlsConfig
	server.StartTLS()

	// Adjust the path to point to our custom path
	serverURL, _ := url.Parse(server.URL)
	serverURL.Path = webhookPath
	server.URL = serverURL.String()

	return server, nil
}

// A service that can be set to allow all or deny all authorization requests.
type mockService struct {
	allow      bool
	statusCode int
	called     int
}

func (m *mockService) Review(r *v1beta1.SubjectAccessReview) {
	m.called++
	r.Status.Allowed = m.allow
}
func (m *mockService) Allow()              { m.allow = true }
func (m *mockService) Deny()               { m.allow = false }
func (m *mockService) HTTPStatusCode() int { return m.statusCode }

// newAuthorizer creates a temporary kubeconfig file from the provided arguments and attempts to load
// a new WebhookAuthorizer from it.
func newAuthorizer(callbackURL string, clientCert, clientKey, ca []byte, cacheTime time.Duration) (a *WebhookAuthorizer, err error) {
	err = withKubeconfig(callbackURL, clientCert, clientKey, ca, func(path string) error {
		sarClient, err := subjectAccessReviewInterfaceFromKubeconfig(path)
		if err != nil {
			return fmt.Errorf("error building sar client: %v", err)
		}
		a, err = newWithBackoff(sarClient, nil, cacheTime, cacheTime, 0)
		return err
	})
	return a, err
}

func newRulesReviewer(callbackURL string, clientCert, clientKey, ca []byte) (a *WebhookAuthorizer, err error) {
	err = withKubeconfig(callbackURL, clientCert, clientKey, ca, func(path string) error {
		srrClient, err := subjectRulesReviewInterfaceFromKubeconfig(path)
		if err != nil {
			return fmt.Errorf("error building sar client: %v", err)
		}
		a, err = newWithBackoff(nil, srrClient, 0, 0, 0)
		return err
	})
	return a, err
}

func withKubeconfig(url string, clientCert, clientKey, ca []byte, f func(filepath string) error) error {
	tempfile, err := ioutil.TempFile("", "")
	if err != nil {
		return err
	}
	p := tempfile.Name()
	defer os.Remove(p)
	config := v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{Server: url, CertificateAuthorityData: ca},
			},
		},
		AuthInfos: []v1.NamedAuthInfo{
			{
				AuthInfo: v1.AuthInfo{ClientCertificateData: clientCert, ClientKeyData: clientKey},
			},
		},
	}
	if err := json.NewEncoder(tempfile).Encode(config); err != nil {
		return err
	}
	return f(p)
}

func TestTLSConfig(t *testing.T) {
	tests := []struct {
		test                            string
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		wantAuth, wantErr               bool
	}{
		{
			test:       "TLS setup between client and server",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			wantAuth: true,
		},
		{
			test:       "Server does not require client auth",
			clientCA:   caCert,
			serverCert: serverCert, serverKey: serverKey,
			wantAuth: true,
		},
		{
			test:       "Server does not require client auth, client provides it",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey,
			wantAuth: true,
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

			wh, err := newAuthorizer(server.URL, tt.clientCert, tt.clientKey, tt.clientCA, 0)
			if err != nil {
				t.Errorf("%s: failed to create client: %v", tt.test, err)
				return
			}

			attr := authorizer.AttributesRecord{User: &user.DefaultInfo{}}

			// Allow all and see if we get an error.
			service.Allow()
			decision, _, err := wh.Authorize(attr)
			if tt.wantAuth {
				if decision != authorizer.DecisionAllow {
					t.Errorf("expected successful authorization")
				}
			} else {
				if decision == authorizer.DecisionAllow {
					t.Errorf("expected failed authorization")
				}
			}
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error making authorization request: %v", err)
				}
				return
			}
			if err != nil {
				t.Errorf("%s: failed to authorize with AllowAll policy: %v", tt.test, err)
				return
			}

			service.Deny()
			if decision, _, _ := wh.Authorize(attr); decision == authorizer.DecisionAllow {
				t.Errorf("%s: incorrectly authorized with DenyAll policy", tt.test)
			}
		}()
	}
}

// recorderService records all access review requests.
type recorderService struct {
	last v1beta1.SubjectAccessReview
	err  error
}

func (rec *recorderService) Review(r *v1beta1.SubjectAccessReview) {
	rec.last = v1beta1.SubjectAccessReview{}
	rec.last = *r
	r.Status.Allowed = true
}

func (rec *recorderService) Last() (v1beta1.SubjectAccessReview, error) {
	return rec.last, rec.err
}

func (rec *recorderService) HTTPStatusCode() int { return 200 }

func TestWebhook(t *testing.T) {
	serv := new(recorderService)
	s, err := NewTestServer(serv, serverCert, serverKey, caCert)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	wh, err := newAuthorizer(s.URL, clientCert, clientKey, caCert, 0)
	if err != nil {
		t.Fatal(err)
	}

	expTypeMeta := metav1.TypeMeta{
		APIVersion: "authorization.k8s.io/v1beta1",
		Kind:       "SubjectAccessReview",
	}

	tests := []struct {
		attr authorizer.Attributes
		want v1beta1.SubjectAccessReview
	}{
		{
			attr: authorizer.AttributesRecord{User: &user.DefaultInfo{}},
			want: v1beta1.SubjectAccessReview{
				TypeMeta: expTypeMeta,
				Spec: v1beta1.SubjectAccessReviewSpec{
					NonResourceAttributes: &v1beta1.NonResourceAttributes{},
				},
			},
		},
		{
			attr: authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "jane"}},
			want: v1beta1.SubjectAccessReview{
				TypeMeta: expTypeMeta,
				Spec: v1beta1.SubjectAccessReviewSpec{
					User: "jane",
					NonResourceAttributes: &v1beta1.NonResourceAttributes{},
				},
			},
		},
		{
			attr: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name:   "jane",
					UID:    "1",
					Groups: []string{"group1", "group2"},
				},
				Verb:            "GET",
				Namespace:       "kittensandponies",
				APIGroup:        "group3",
				APIVersion:      "v7beta3",
				Resource:        "pods",
				Subresource:     "proxy",
				Name:            "my-pod",
				ResourceRequest: true,
				Path:            "/foo",
			},
			want: v1beta1.SubjectAccessReview{
				TypeMeta: expTypeMeta,
				Spec: v1beta1.SubjectAccessReviewSpec{
					User:   "jane",
					UID:    "1",
					Groups: []string{"group1", "group2"},
					ResourceAttributes: &v1beta1.ResourceAttributes{
						Verb:        "GET",
						Namespace:   "kittensandponies",
						Group:       "group3",
						Version:     "v7beta3",
						Resource:    "pods",
						Subresource: "proxy",
						Name:        "my-pod",
					},
				},
			},
		},
	}

	for i, tt := range tests {
		decision, _, err := wh.Authorize(tt.attr)
		if err != nil {
			t.Fatal(err)
		}
		if decision != authorizer.DecisionAllow {
			t.Errorf("case %d: authorization failed", i)
			continue
		}

		gotAttr, err := serv.Last()
		if err != nil {
			t.Errorf("case %d: failed to deserialize webhook request: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(gotAttr, tt.want) {
			t.Errorf("case %d: got != want:\n%s", i, diff.ObjectGoPrintDiff(gotAttr, tt.want))
		}
	}
}

type mockRulesReviewer struct {
	namespace  string
	userPerms  map[string]v1beta1.SubjectRulesReviewStatus
	groupPerms map[string]v1beta1.SubjectRulesReviewStatus
	extraPerms map[string]map[string]v1beta1.SubjectRulesReviewStatus
}

func (m *mockRulesReviewer) HTTPStatusCode() int { return 200 }

func (m *mockRulesReviewer) Review(spec v1beta1.SubjectRulesReviewSpec) v1beta1.SubjectRulesReviewStatus {
	if spec.Namespace != m.namespace {
		return v1beta1.SubjectRulesReviewStatus{}
	}

	var status v1beta1.SubjectRulesReviewStatus

	appendStatus := func(s v1beta1.SubjectRulesReviewStatus) {
		status.ResourceRules = append(status.ResourceRules, s.ResourceRules...)
		status.NonResourceRules = append(status.NonResourceRules, s.NonResourceRules...)
	}

	appendStatus(m.userPerms[spec.User])
	for _, group := range spec.Groups {
		appendStatus(m.groupPerms[group])
	}
	for key, val := range spec.Extra {
		perms, ok := m.extraPerms[key]
		if ok {
			for _, v := range val {
				appendStatus(perms[string(v)])
			}
		}
	}

	return status
}

func TestRulesReview(t *testing.T) {
	s := &mockRulesReviewer{
		namespace: "my-namespace",
		userPerms: map[string]v1beta1.SubjectRulesReviewStatus{
			"jane": {
				ResourceRules: []v1beta1.ResourceRule{
					{
						Verbs:     []string{"get"},
						APIGroups: []string{""},
						Resources: []string{"pods", "services"},
					},
				},
			},
			"admin": {
				ResourceRules: []v1beta1.ResourceRule{
					{
						Verbs:     []string{"*"},
						APIGroups: []string{"*"},
						Resources: []string{"*"},
					},
				},
			},
		},
		groupPerms: map[string]v1beta1.SubjectRulesReviewStatus{
			"developers": {
				ResourceRules: []v1beta1.ResourceRule{
					{
						Verbs:     []string{"update", "delete"},
						APIGroups: []string{""},
						Resources: []string{"nodes"},
						Denies:    true,
					},
					{
						Verbs:     []string{"update", "delete"},
						APIGroups: []string{""},
						Resources: []string{"pods", "services"},
					},
				},
			},
			"discovery": {
				NonResourceRules: []v1beta1.NonResourceRule{
					{
						Verbs:           []string{"get"},
						NonResourceURLs: []string{"/foo", "/foo/bar"},
					},
				},
			},
		},
		extraPerms: map[string]map[string]v1beta1.SubjectRulesReviewStatus{
			"project": {
				"foo": {
					ResourceRules: []v1beta1.ResourceRule{
						{
							Verbs:         []string{"get", "update"},
							APIGroups:     []string{""},
							Resources:     []string{"namespace"},
							ResourceNames: []string{"foo"},
						},
					},
				},
			},
		},
	}

	server, err := NewTestServer(s, serverCert, serverKey, caCert)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()

	wh, err := newRulesReviewer(server.URL, clientCert, clientKey, caCert)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string

		namespace string
		user      string
		groups    []string
		extra     map[string][]string

		resourceRules    []authorizer.DefaultResourceRuleInfo
		nonResourceRules []authorizer.DefaultNonResourceRuleInfo
	}{
		{
			name:      "user",
			namespace: s.namespace,
			user:      "jane",

			resourceRules: []authorizer.DefaultResourceRuleInfo{
				{
					Verbs:     []string{"get"},
					APIGroups: []string{""},
					Resources: []string{"pods", "services"},
				},
			},
		},
		{
			name:      "user in different namespace",
			namespace: "different-namespace",
			user:      "jane",
		},
		{
			name:      "user with groups",
			namespace: s.namespace,
			user:      "jane",
			groups:    []string{"developers", "discovery"},
			resourceRules: []authorizer.DefaultResourceRuleInfo{
				{
					Verbs:     []string{"get"},
					APIGroups: []string{""},
					Resources: []string{"pods", "services"},
				},
				// Order matters here. The deny rule should always be first.
				{
					Verbs:     []string{"update", "delete"},
					APIGroups: []string{""},
					Resources: []string{"nodes"},
					Denies:    true,
				},
				{
					Verbs:     []string{"update", "delete"},
					APIGroups: []string{""},
					Resources: []string{"pods", "services"},
				},
			},
			nonResourceRules: []authorizer.DefaultNonResourceRuleInfo{
				{
					Verbs:           []string{"get"},
					NonResourceURLs: []string{"/foo", "/foo/bar"},
				},
			},
		},
		{
			name:      "admin",
			namespace: s.namespace,
			user:      "admin",
			resourceRules: []authorizer.DefaultResourceRuleInfo{
				{
					Verbs:     []string{"*"},
					APIGroups: []string{"*"},
					Resources: []string{"*"},
				},
			},
		},
		{
			name:      "user with extras",
			namespace: s.namespace,
			user:      "jane",
			extra: map[string][]string{
				"project": {"foo"},
			},

			resourceRules: []authorizer.DefaultResourceRuleInfo{
				{
					Verbs:     []string{"get"},
					APIGroups: []string{""},
					Resources: []string{"pods", "services"},
				},
				{
					Verbs:         []string{"get", "update"},
					APIGroups:     []string{""},
					Resources:     []string{"namespace"},
					ResourceNames: []string{"foo"},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			u := &user.DefaultInfo{
				Name:   test.user,
				Groups: test.groups,
				Extra:  test.extra,
			}

			resourceRules, nonResourceRules, incomplete, err := wh.RulesFor(u, test.namespace)
			if err != nil {
				t.Fatal(err)
			}
			if incomplete {
				t.Fatal("results incomplete")
			}

			if len(resourceRules) == 0 {
				resourceRules = nil
			}
			if len(nonResourceRules) == 0 {
				nonResourceRules = nil
			}

			assertEq(t, resourceRules, test.resourceRules)
			assertEq(t, nonResourceRules, test.nonResourceRules)
		})
	}
}

func assertEq(t *testing.T, want, got interface{}) {
	wantBytes, err := json.MarshalIndent(want, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(wantBytes, gotBytes) {
		return
	}
	t.Errorf("expected:\n%s\ngot:\n%s", wantBytes, gotBytes)
}

type webhookCacheTestCase struct {
	attr authorizer.AttributesRecord

	allow      bool
	statusCode int

	expectedErr        bool
	expectedAuthorized bool
	expectedCalls      int
}

func testWebhookCacheCases(t *testing.T, serv *mockService, wh *WebhookAuthorizer, tests []webhookCacheTestCase) {
	for i, test := range tests {
		serv.called = 0
		serv.allow = test.allow
		serv.statusCode = test.statusCode
		authorized, _, err := wh.Authorize(test.attr)
		if test.expectedErr && err == nil {
			t.Errorf("%d: Expected error", i)
			continue
		} else if !test.expectedErr && err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}

		if test.expectedAuthorized != (authorized == authorizer.DecisionAllow) {
			t.Errorf("%d: expected authorized=%v, got %v", i, test.expectedAuthorized, authorized)
		}

		if test.expectedCalls != serv.called {
			t.Errorf("%d: expected %d calls, got %d", i, test.expectedCalls, serv.called)
		}
	}
}

// TestWebhookCache verifies that error responses from the server are not
// cached, but successful responses are.
func TestWebhookCache(t *testing.T) {
	serv := new(mockService)
	s, err := NewTestServer(serv, serverCert, serverKey, caCert)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// Create an authorizer that caches successful responses "forever" (100 days).
	wh, err := newAuthorizer(s.URL, clientCert, clientKey, caCert, 2400*time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	aliceAttr := authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "alice"}}
	bobAttr := authorizer.AttributesRecord{User: &user.DefaultInfo{Name: "bob"}}

	tests := []webhookCacheTestCase{
		// server error and 429's retry
		{attr: aliceAttr, allow: false, statusCode: 500, expectedErr: true, expectedAuthorized: false, expectedCalls: 5},
		{attr: aliceAttr, allow: false, statusCode: 429, expectedErr: true, expectedAuthorized: false, expectedCalls: 5},
		// regular errors return errors but do not retry
		{attr: aliceAttr, allow: false, statusCode: 404, expectedErr: true, expectedAuthorized: false, expectedCalls: 1},
		{attr: aliceAttr, allow: false, statusCode: 403, expectedErr: true, expectedAuthorized: false, expectedCalls: 1},
		{attr: aliceAttr, allow: false, statusCode: 401, expectedErr: true, expectedAuthorized: false, expectedCalls: 1},
		// successful responses are cached
		{attr: aliceAttr, allow: true, statusCode: 200, expectedErr: false, expectedAuthorized: true, expectedCalls: 1},
		// later requests within the cache window don't hit the backend
		{attr: aliceAttr, allow: false, statusCode: 500, expectedErr: false, expectedAuthorized: true, expectedCalls: 0},

		// a request with different attributes doesn't hit the cache
		{attr: bobAttr, allow: false, statusCode: 500, expectedErr: true, expectedAuthorized: false, expectedCalls: 5},
		// successful response for other attributes is cached
		{attr: bobAttr, allow: true, statusCode: 200, expectedErr: false, expectedAuthorized: true, expectedCalls: 1},
		// later requests within the cache window don't hit the backend
		{attr: bobAttr, allow: false, statusCode: 500, expectedErr: false, expectedAuthorized: true, expectedCalls: 0},
	}

	testWebhookCacheCases(t, serv, wh, tests)
}
