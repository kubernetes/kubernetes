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

package mutating

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"k8s.io/api/admission/v1beta1"
	registrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/rest"
)

type fakeHookSource struct {
	hooks []registrationv1beta1.Webhook
	err   error
}

func (f *fakeHookSource) Webhooks() (*registrationv1beta1.MutatingWebhookConfiguration, error) {
	if f.err != nil {
		return nil, f.err
	}
	for i, h := range f.hooks {
		if h.NamespaceSelector == nil {
			f.hooks[i].NamespaceSelector = &metav1.LabelSelector{}
		}
	}
	return &registrationv1beta1.MutatingWebhookConfiguration{Webhooks: f.hooks}, nil
}

func (f *fakeHookSource) Run(stopCh <-chan struct{}) {}

type fakeServiceResolver struct {
	base url.URL
}

func (f fakeServiceResolver) ResolveEndpoint(namespace, name string) (*url.URL, error) {
	if namespace == "failResolve" {
		return nil, fmt.Errorf("couldn't resolve service location")
	}
	u := f.base
	return &u, nil
}

type fakeNamespaceLister struct {
	namespaces map[string]*corev1.Namespace
}

func (f fakeNamespaceLister) List(selector labels.Selector) (ret []*corev1.Namespace, err error) {
	return nil, nil
}
func (f fakeNamespaceLister) Get(name string) (*corev1.Namespace, error) {
	ns, ok := f.namespaces[name]
	if ok {
		return ns, nil
	}
	return nil, errors.NewNotFound(corev1.Resource("namespaces"), name)
}

// ccfgSVC returns a client config using the service reference mechanism.
func ccfgSVC(urlPath string) registrationv1beta1.WebhookClientConfig {
	return registrationv1beta1.WebhookClientConfig{
		Service: &registrationv1beta1.ServiceReference{
			Name:      "webhook-test",
			Namespace: "default",
			Path:      &urlPath,
		},
		CABundle: testcerts.CACert,
	}
}

type urlConfigGenerator struct {
	baseURL *url.URL
}

// ccfgURL returns a client config using the URL mechanism.
func (c urlConfigGenerator) ccfgURL(urlPath string) registrationv1beta1.WebhookClientConfig {
	u2 := *c.baseURL
	u2.Path = urlPath
	urlString := u2.String()
	return registrationv1beta1.WebhookClientConfig{
		URL:      &urlString,
		CABundle: testcerts.CACert,
	}
}

// TestAdmit tests that MutatingWebhook#Admit works as expected
func TestAdmit(t *testing.T) {
	scheme := runtime.NewScheme()
	v1beta1.AddToScheme(scheme)
	corev1.AddToScheme(scheme)

	testServer := newTestServer(t)
	testServer.StartTLS()
	defer testServer.Close()
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatalf("this should never happen? %v", err)
	}
	wh, err := NewMutatingWebhook(nil)
	if err != nil {
		t.Fatal(err)
	}
	cm, err := config.NewClientManager()
	if err != nil {
		t.Fatalf("cannot create client manager: %v", err)
	}
	cm.SetAuthenticationInfoResolver(newFakeAuthenticationInfoResolver(new(int32)))
	cm.SetServiceResolver(fakeServiceResolver{base: *serverURL})
	wh.clientManager = cm
	wh.SetScheme(scheme)
	if err = wh.clientManager.Validate(); err != nil {
		t.Fatal(err)
	}
	namespace := "webhook-test"
	wh.namespaceMatcher.NamespaceLister = fakeNamespaceLister{map[string]*corev1.Namespace{
		namespace: {
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"runlevel": "0",
				},
			},
		},
	},
	}

	// Set up a test object for the call
	kind := corev1.SchemeGroupVersion.WithKind("Pod")
	name := "my-pod"
	object := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"pod.name": name,
			},
			Name:      name,
			Namespace: namespace,
		},
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
	}
	oldObject := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	operation := admission.Update
	resource := corev1.Resource("pods").WithVersion("v1")
	subResource := ""
	userInfo := user.DefaultInfo{
		Name: "webhook-test",
		UID:  "webhook-test",
	}

	ccfgURL := urlConfigGenerator{serverURL}.ccfgURL

	type test struct {
		hookSource    fakeHookSource
		path          string
		expectAllow   bool
		errorContains string
	}

	matchEverythingRules := []registrationv1beta1.RuleWithOperations{{
		Operations: []registrationv1beta1.OperationType{registrationv1beta1.OperationAll},
		Rule: registrationv1beta1.Rule{
			APIGroups:   []string{"*"},
			APIVersions: []string{"*"},
			Resources:   []string{"*/*"},
		},
	}}

	policyFail := registrationv1beta1.Fail
	policyIgnore := registrationv1beta1.Ignore

	table := map[string]test{
		"no match": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "nomatch",
					ClientConfig: ccfgSVC("disallow"),
					Rules: []registrationv1beta1.RuleWithOperations{{
						Operations: []registrationv1beta1.OperationType{registrationv1beta1.Create},
					}},
				}},
			},
			expectAllow: true,
		},
		"match & allow": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "allow",
					ClientConfig: ccfgSVC("allow"),
					Rules:        matchEverythingRules,
				}},
			},
			expectAllow: true,
		},
		"match & disallow": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "disallow",
					ClientConfig: ccfgSVC("disallow"),
					Rules:        matchEverythingRules,
				}},
			},
			errorContains: "without explanation",
		},
		"match & disallow ii": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "disallowReason",
					ClientConfig: ccfgSVC("disallowReason"),
					Rules:        matchEverythingRules,
				}},
			},
			errorContains: "you shall not pass",
		},
		"match & disallow & but allowed because namespaceSelector exempt the namespace": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "disallow",
					ClientConfig: ccfgSVC("disallow"),
					Rules:        newMatchEverythingRules(),
					NamespaceSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{{
							Key:      "runlevel",
							Values:   []string{"1"},
							Operator: metav1.LabelSelectorOpIn,
						}},
					},
				}},
			},
			expectAllow: true,
		},
		"match & disallow & but allowed because namespaceSelector exempt the namespace ii": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "disallow",
					ClientConfig: ccfgSVC("disallow"),
					Rules:        newMatchEverythingRules(),
					NamespaceSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{{
							Key:      "runlevel",
							Values:   []string{"0"},
							Operator: metav1.LabelSelectorOpNotIn,
						}},
					},
				}},
			},
			expectAllow: true,
		},
		"match & fail (but allow because fail open)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "internalErr A",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyIgnore,
				}, {
					Name:          "internalErr B",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyIgnore,
				}, {
					Name:          "internalErr C",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
		},
		"match & fail (but disallow because fail closed on nil)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "internalErr A",
					ClientConfig: ccfgSVC("internalErr"),
					Rules:        matchEverythingRules,
				}, {
					Name:         "internalErr B",
					ClientConfig: ccfgSVC("internalErr"),
					Rules:        matchEverythingRules,
				}, {
					Name:         "internalErr C",
					ClientConfig: ccfgSVC("internalErr"),
					Rules:        matchEverythingRules,
				}},
			},
			expectAllow: false,
		},
		"match & fail (but fail because fail closed)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "internalErr A",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyFail,
				}, {
					Name:          "internalErr B",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyFail,
				}, {
					Name:          "internalErr C",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         matchEverythingRules,
					FailurePolicy: &policyFail,
				}},
			},
			expectAllow: false,
		},
		"match & allow (url)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "allow",
					ClientConfig: ccfgURL("allow"),
					Rules:        matchEverythingRules,
				}},
			},
			expectAllow: true,
		},
		"match & disallow (url)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:         "disallow",
					ClientConfig: ccfgURL("disallow"),
					Rules:        matchEverythingRules,
				}},
			},
			errorContains: "without explanation",
		},
		// No need to test everything with the url case, since only the
		// connection is different.
	}

	for name, tt := range table {
		if !strings.Contains(name, "no match") {
			continue
		}
		t.Run(name, func(t *testing.T) {
			wh.hookSource = &tt.hookSource
			err = wh.Admit(admission.NewAttributesRecord(&object, &oldObject, kind, namespace, name, resource, subResource, operation, &userInfo))
			if tt.expectAllow != (err == nil) {
				t.Errorf("expected allowed=%v, but got err=%v", tt.expectAllow, err)
			}
			// ErrWebhookRejected is not an error for our purposes
			if tt.errorContains != "" {
				if err == nil || !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf(" expected an error saying %q, but got %v", tt.errorContains, err)
				}
			}
			if _, isStatusErr := err.(*apierrors.StatusError); err != nil && !isStatusErr {
				t.Errorf("%s: expected a StatusError, got %T", name, err)
			}
		})
	}
}

// TestAdmitCachedClient tests that MutatingWebhook#Admit should cache restClient
func TestAdmitCachedClient(t *testing.T) {
	scheme := runtime.NewScheme()
	v1beta1.AddToScheme(scheme)
	corev1.AddToScheme(scheme)

	testServer := newTestServer(t)
	testServer.StartTLS()
	defer testServer.Close()
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatalf("this should never happen? %v", err)
	}
	wh, err := NewMutatingWebhook(nil)
	if err != nil {
		t.Fatal(err)
	}
	cm, err := config.NewClientManager()
	if err != nil {
		t.Fatalf("cannot create client manager: %v", err)
	}
	cm.SetServiceResolver(fakeServiceResolver{base: *serverURL})
	wh.clientManager = cm
	wh.SetScheme(scheme)
	namespace := "webhook-test"
	wh.namespaceMatcher.NamespaceLister = fakeNamespaceLister{map[string]*corev1.Namespace{
		namespace: {
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"runlevel": "0",
				},
			},
		},
	},
	}

	// Set up a test object for the call
	kind := corev1.SchemeGroupVersion.WithKind("Pod")
	name := "my-pod"
	object := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"pod.name": name,
			},
			Name:      name,
			Namespace: namespace,
		},
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
	}
	oldObject := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	operation := admission.Update
	resource := corev1.Resource("pods").WithVersion("v1")
	subResource := ""
	userInfo := user.DefaultInfo{
		Name: "webhook-test",
		UID:  "webhook-test",
	}
	ccfgURL := urlConfigGenerator{serverURL}.ccfgURL

	type test struct {
		name        string
		hookSource  fakeHookSource
		expectAllow bool
		expectCache bool
	}

	policyIgnore := registrationv1beta1.Ignore
	cases := []test{
		{
			name: "cache 1",
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "cache1",
					ClientConfig:  ccfgSVC("allow"),
					Rules:         newMatchEverythingRules(),
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
			expectCache: true,
		},
		{
			name: "cache 2",
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "cache2",
					ClientConfig:  ccfgSVC("internalErr"),
					Rules:         newMatchEverythingRules(),
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
			expectCache: true,
		},
		{
			name: "cache 3",
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "cache3",
					ClientConfig:  ccfgSVC("allow"),
					Rules:         newMatchEverythingRules(),
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
			expectCache: false,
		},
		{
			name: "cache 4",
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "cache4",
					ClientConfig:  ccfgURL("allow"),
					Rules:         newMatchEverythingRules(),
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
			expectCache: true,
		},
		{
			name: "cache 5",
			hookSource: fakeHookSource{
				hooks: []registrationv1beta1.Webhook{{
					Name:          "cache5",
					ClientConfig:  ccfgURL("allow"),
					Rules:         newMatchEverythingRules(),
					FailurePolicy: &policyIgnore,
				}},
			},
			expectAllow: true,
			expectCache: false,
		},
	}

	for _, testcase := range cases {
		t.Run(testcase.name, func(t *testing.T) {
			wh.hookSource = &testcase.hookSource
			authInfoResolverCount := new(int32)
			r := newFakeAuthenticationInfoResolver(authInfoResolverCount)
			wh.clientManager.SetAuthenticationInfoResolver(r)
			if err = wh.clientManager.Validate(); err != nil {
				t.Fatal(err)
			}

			err = wh.Admit(admission.NewAttributesRecord(&object, &oldObject, kind, namespace, testcase.name, resource, subResource, operation, &userInfo))
			if testcase.expectAllow != (err == nil) {
				t.Errorf("expected allowed=%v, but got err=%v", testcase.expectAllow, err)
			}

			if testcase.expectCache && *authInfoResolverCount != 1 {
				t.Errorf("expected cacheclient, but got none")
			}

			if !testcase.expectCache && *authInfoResolverCount != 0 {
				t.Errorf("expected not cacheclient, but got cache")
			}
		})
	}

}

func newTestServer(t *testing.T) *httptest.Server {
	// Create the test webhook server
	sCert, err := tls.X509KeyPair(testcerts.ServerCert, testcerts.ServerKey)
	if err != nil {
		t.Fatal(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(testcerts.CACert)
	testServer := httptest.NewUnstartedServer(http.HandlerFunc(webhookHandler))
	testServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{sCert},
		ClientCAs:    rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	return testServer
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Printf("got req: %v\n", r.URL.Path)
	switch r.URL.Path {
	case "/internalErr":
		http.Error(w, "webhook internal server error", http.StatusInternalServerError)
		return
	case "/invalidReq":
		w.WriteHeader(http.StatusSwitchingProtocols)
		w.Write([]byte("webhook invalid request"))
		return
	case "/invalidResp":
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("webhook invalid response"))
	case "/disallow":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
			},
		})
	case "/disallowReason":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
				Result: &metav1.Status{
					Message: "you shall not pass",
				},
			},
		})
	case "/allow":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
			},
		})
	default:
		http.NotFound(w, r)
	}
}

func newFakeAuthenticationInfoResolver(count *int32) *fakeAuthenticationInfoResolver {
	return &fakeAuthenticationInfoResolver{
		restConfig: &rest.Config{
			TLSClientConfig: rest.TLSClientConfig{
				CAData:   testcerts.CACert,
				CertData: testcerts.ClientCert,
				KeyData:  testcerts.ClientKey,
			},
		},
		cachedCount: count,
	}
}

type fakeAuthenticationInfoResolver struct {
	restConfig  *rest.Config
	cachedCount *int32
}

func (c *fakeAuthenticationInfoResolver) ClientConfigFor(server string) (*rest.Config, error) {
	atomic.AddInt32(c.cachedCount, 1)
	return c.restConfig, nil
}

func newMatchEverythingRules() []registrationv1beta1.RuleWithOperations {
	return []registrationv1beta1.RuleWithOperations{{
		Operations: []registrationv1beta1.OperationType{registrationv1beta1.OperationAll},
		Rule: registrationv1beta1.Rule{
			APIGroups:   []string{"*"},
			APIVersions: []string{"*"},
			Resources:   []string{"*/*"},
		},
	}}
}
