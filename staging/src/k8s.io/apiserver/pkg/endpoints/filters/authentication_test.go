/*
Copyright 2014 The Kubernetes Authors.

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

package filters

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestAuthenticateRequest(t *testing.T) {
	success := make(chan struct{})
	contextMapper := genericapirequest.NewRequestContextMapper()
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			ctx, ok := contextMapper.Get(req)
			if ctx == nil || !ok {
				t.Errorf("no context stored on contextMapper: %#v", contextMapper)
			}
			user, ok := genericapirequest.UserFrom(ctx)
			if user == nil || !ok {
				t.Errorf("no user stored in context: %#v", ctx)
			}
			if req.Header.Get("Authorization") != "" {
				t.Errorf("Authorization header should be removed from request on success: %#v", req)
			}
			close(success)
		}),
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			if req.Header.Get("Authorization") == "Something" {
				return &user.DefaultInfo{Name: "user"}, true, nil
			}
			return nil, false, errors.New("Authorization header is missing.")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Errorf("unexpected call to failed")
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: map[string][]string{"Authorization": {"Something"}}})

	<-success
	empty, err := genericapirequest.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

func TestAuthenticateRequestFailed(t *testing.T) {
	failed := make(chan struct{})
	contextMapper := genericapirequest.NewRequestContextMapper()
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return nil, false, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
	empty, err := genericapirequest.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

func TestAuthenticateRequestError(t *testing.T) {
	failed := make(chan struct{})
	contextMapper := genericapirequest.NewRequestContextMapper()
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
		contextMapper,
		authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
			return nil, false, errors.New("failure")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
	empty, err := genericapirequest.IsEmpty(contextMapper)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !empty {
		t.Fatalf("contextMapper should have no stored requests: %v", contextMapper)
	}
}

func TestCompressUsername(t *testing.T) {
	tests := []struct{ username, expected string }{
		// Known system accounts should be reported as-is.
		{"kubelet", "kubelet"},
		{"admin", "admin"},
		{"system:kube-controller-manager", "system:kube-controller-manager"},
		{"system:kube-scheduler", "system:kube-scheduler"},
		{"system:apiserver", "system:apiserver"},
		{"system:anonymous", "system:anonymous"},
		{"system:kube-proxy", "system:kube-proxy"},
		{"system:serviceaccount:kube-system:default", "system:serviceaccount:kube-system:default"},
		{"system:serviceaccount:kube-system:node-controller", "system:serviceaccount:kube-system:node-controller"},
		{"system:serviceaccount:kube-system:endpoint-controller", "system:serviceaccount:kube-system:endpoint-controller"},
		{"system:serviceaccount:kube-system:cronjob-controller", "system:serviceaccount:kube-system:cronjob-controller"},
		{"system:serviceaccount:kube-system:generic-garbage-collector", "system:serviceaccount:kube-system:generic-garbage-collector"},
		{"system:serviceaccount:kube-system:pod-garbage-collector", "system:serviceaccount:kube-system:pod-garbage-collector"},
		{"system:serviceaccount:kube-system:node-problem-detector", "system:serviceaccount:kube-system:node-problem-detector"},
		{"system:serviceaccount:kube-system:kube-dns", "system:serviceaccount:kube-system:kube-dns"},
		{"system:serviceaccount:kube-system:namespace-controller", "system:serviceaccount:kube-system:namespace-controller"},
		{"system:serviceaccount:kube-system:replicaset-controller", "system:serviceaccount:kube-system:replicaset-controller"},
		{"system:serviceaccount:kube-system:deployment-controller", "system:serviceaccount:kube-system:deployment-controller"},
		{"system:serviceaccount:kube-system:daemon-set-controller", "system:serviceaccount:kube-system:daemon-set-controller"},
		{"system:serviceaccount:kube-system:controller-discovery", "system:serviceaccount:kube-system:controller-discovery"},
		{"system:serviceaccount:kube-system:replication-controller", "system:serviceaccount:kube-system:replication-controller"},
		{"system:serviceaccount:kube-system:ttl-controller", "system:serviceaccount:kube-system:ttl-controller"},
		{"system:serviceaccount:kube-system:route-controller", "system:serviceaccount:kube-system:route-controller"},
		{"system:serviceaccount:kube-system:service-account-controller", "system:serviceaccount:kube-system:service-account-controller"},
		// Custom service & user accounts should be compressed.
		{"foo@bar.com", "email_id"},
		{"foo-bar-baz@k8s.io.edu.gov", "email_id"},
		{"system:serviceaccount:default:foo-service", "system:serviceaccount"},
		{"system:serviceaccount:my-namespace:bar", "system:serviceaccount"},
		{"kubecfg", "other"},
		{"whatsit", "other"},
		{"foo-user", "other"},
	}
	for i, test := range tests {
		if actual := compressUsername(test.username); actual != test.expected {
			t.Errorf("[%d  %s] expected: %q, got: %q", i, test.username, test.expected, actual)
		}
	}
}
