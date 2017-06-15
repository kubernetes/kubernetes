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

package webhook

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/admission/v1alpha1"
	registrationv1alpha1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1"

	_ "k8s.io/kubernetes/pkg/apis/admission/install"
)

type fakeHookSource struct {
	hooks []registrationv1alpha1.ExternalAdmissionHook
	err   error
}

func (f *fakeHookSource) ExternalAdmissionHooks() (*registrationv1alpha1.ExternalAdmissionHookConfiguration, error) {
	if f.err != nil {
		return nil, f.err
	}
	return &registrationv1alpha1.ExternalAdmissionHookConfiguration{ExternalAdmissionHooks: f.hooks}, nil
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
	u.Path = name
	return &u, nil
}

// TestAdmit tests that GenericAdmissionWebhook#Admit works as expected
func TestAdmit(t *testing.T) {
	// Create the test webhook server
	sCert, err := tls.X509KeyPair(serverCert, serverKey)
	if err != nil {
		t.Fatal(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(caCert)
	testServer := httptest.NewUnstartedServer(http.HandlerFunc(webhookHandler))
	testServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{sCert},
		ClientCAs:    rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	testServer.StartTLS()
	defer testServer.Close()
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatalf("this should never happen? %v", err)
	}
	wh, err := NewGenericAdmissionWebhook()
	if err != nil {
		t.Fatal(err)
	}
	wh.serviceResolver = fakeServiceResolver{*serverURL}
	wh.clientCert = clientCert
	wh.clientKey = clientKey

	// Set up a test object for the call
	kind := api.Kind("Pod").WithVersion("v1")
	name := "my-pod"
	namespace := "webhook-test"
	object := api.Pod{
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
	oldObject := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	operation := admission.Update
	resource := api.Resource("pods").WithVersion("v1")
	subResource := ""
	userInfo := user.DefaultInfo{
		Name: "webhook-test",
		UID:  "webhook-test",
	}

	type test struct {
		hookSource    fakeHookSource
		expectAllow   bool
		errorContains string
	}
	ccfg := func(result string) registrationv1alpha1.AdmissionHookClientConfig {
		return registrationv1alpha1.AdmissionHookClientConfig{
			Service: registrationv1alpha1.ServiceReference{
				Name: result,
			},
			CABundle: caCert,
		}
	}
	matchEverythingRules := []registrationv1alpha1.RuleWithOperations{{
		Operations: []registrationv1alpha1.OperationType{registrationv1alpha1.OperationAll},
		Rule: registrationv1alpha1.Rule{
			APIGroups:   []string{"*"},
			APIVersions: []string{"*"},
			Resources:   []string{"*/*"},
		},
	}}

	table := map[string]test{
		"no match": {
			hookSource: fakeHookSource{
				hooks: []registrationv1alpha1.ExternalAdmissionHook{{
					Name:         "nomatch",
					ClientConfig: ccfg("disallow"),
					Rules: []registrationv1alpha1.RuleWithOperations{{
						Operations: []registrationv1alpha1.OperationType{registrationv1alpha1.Create},
					}},
				}},
			},
			expectAllow: true,
		},
		"match & allow": {
			hookSource: fakeHookSource{
				hooks: []registrationv1alpha1.ExternalAdmissionHook{{
					Name:         "allow",
					ClientConfig: ccfg("allow"),
					Rules:        matchEverythingRules,
				}},
			},
			expectAllow: true,
		},
		"match & disallow": {
			hookSource: fakeHookSource{
				hooks: []registrationv1alpha1.ExternalAdmissionHook{{
					Name:         "disallow",
					ClientConfig: ccfg("disallow"),
					Rules:        matchEverythingRules,
				}},
			},
			errorContains: "without explanation",
		},
		"match & disallow ii": {
			hookSource: fakeHookSource{
				hooks: []registrationv1alpha1.ExternalAdmissionHook{{
					Name:         "disallowReason",
					ClientConfig: ccfg("disallowReason"),
					Rules:        matchEverythingRules,
				}},
			},
			errorContains: "you shall not pass",
		},
		"match & fail (but allow because fail open)": {
			hookSource: fakeHookSource{
				hooks: []registrationv1alpha1.ExternalAdmissionHook{{
					Name:         "internalErr A",
					ClientConfig: ccfg("internalErr"),
					Rules:        matchEverythingRules,
				}, {
					Name:         "invalidReq B",
					ClientConfig: ccfg("invalidReq"),
					Rules:        matchEverythingRules,
				}, {
					Name:         "invalidResp C",
					ClientConfig: ccfg("invalidResp"),
					Rules:        matchEverythingRules,
				}},
			},
			expectAllow: true,
		},
	}

	for name, tt := range table {
		wh.hookSource = &tt.hookSource

		err = wh.Admit(admission.NewAttributesRecord(&object, &oldObject, kind, namespace, name, resource, subResource, operation, &userInfo))
		if tt.expectAllow != (err == nil) {
			t.Errorf("%q: expected allowed=%v, but got err=%v", name, tt.expectAllow, err)
		}
		// ErrWebhookRejected is not an error for our purposes
		if tt.errorContains != "" {
			if err == nil || !strings.Contains(err.Error(), tt.errorContains) {
				t.Errorf("%q: expected an error saying %q, but got %v", name, tt.errorContains, err)
			}
		}
	}
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
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: false,
			},
		})
	case "/disallowReason":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: false,
				Result: &metav1.Status{
					Message: "you shall not pass",
				},
			},
		})
	case "/allow":
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: true,
			},
		})
	default:
		http.NotFound(w, r)
	}
}
