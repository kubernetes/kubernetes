/*
Copyright 2022 The Kubernetes Authors.

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
	"context"
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestGetsSelfAttributes(t *testing.T) {
	tests := []struct {
		name           string
		userInfo       *user.DefaultInfo
		expectedName   string
		expectedUID    string
		expectedGroups []string
		expectedExtra  map[string]authenticationv1.ExtraValue
	}{
		{
			name: "Username",
			userInfo: &user.DefaultInfo{
				Name: "alice",
			},
			expectedName: "alice",
		},
		{
			name: "Username with groups and UID",
			userInfo: &user.DefaultInfo{
				Name:   "alice",
				UID:    "unique-id",
				Groups: []string{"devs", "admins"},
			},
			expectedName:   "alice",
			expectedUID:    "unique-id",
			expectedGroups: []string{"devs", "admins"},
		},
		{
			name: "Username with extra attributes",
			userInfo: &user.DefaultInfo{
				Name: "alice",
				Extra: map[string][]string{
					"nicknames": {"cutie", "bestie"},
				},
			},
			expectedName: "alice",
			expectedExtra: map[string]authenticationv1.ExtraValue{
				"nicknames": authenticationv1.ExtraValue([]string{"cutie", "bestie"}),
			},
		},
		{
			name: "Without username",
			userInfo: &user.DefaultInfo{
				UID: "unique-id",
			},
			expectedUID: "unique-id",
		},
	}

	tCtx := ktesting.Init(t)
	var respMu sync.RWMutex
	response := &user.DefaultInfo{
		Name: "stub",
	}

	kubeClient, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1alpha1=true")
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1beta1=true")
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1=true")
			opts.Authorization.Modes = []string{"AlwaysAllow"}
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			// Unset BearerToken to disable BearerToken authenticator.
			config.ControlPlane.Generic.LoopbackClientConfig.BearerToken = ""
			config.ControlPlane.Generic.Authentication.Authenticator = authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
				respMu.RLock()
				defer respMu.RUnlock()
				return &authenticator.Response{User: response}, true, nil
			})
		},
	})
	defer tearDownFn()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			respMu.Lock()
			response = tc.userInfo
			respMu.Unlock()

			res, err := kubeClient.AuthenticationV1alpha1().
				SelfSubjectReviews().
				Create(tCtx, &authenticationv1alpha1.SelfSubjectReview{}, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if res == nil {
				t.Fatalf("empty response")
			}

			if res.Status.UserInfo.Username != tc.expectedName {
				t.Fatalf("unexpected username: wanted %s, got %s", tc.expectedName, res.Status.UserInfo.Username)
			}

			if res.Status.UserInfo.UID != tc.expectedUID {
				t.Fatalf("unexpected uid: wanted %s, got %s", tc.expectedUID, res.Status.UserInfo.UID)
			}

			if !reflect.DeepEqual(res.Status.UserInfo.Groups, tc.expectedGroups) {
				t.Fatalf("unexpected groups: wanted %v, got %v", tc.expectedGroups, res.Status.UserInfo.Groups)
			}

			if !reflect.DeepEqual(res.Status.UserInfo.Extra, tc.expectedExtra) {
				t.Fatalf("unexpected extra: wanted %v, got %v", tc.expectedExtra, res.Status.UserInfo.Extra)
			}

			res2, err := kubeClient.AuthenticationV1beta1().
				SelfSubjectReviews().
				Create(tCtx, &authenticationv1beta1.SelfSubjectReview{}, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if res2 == nil {
				t.Fatalf("empty response")
			}

			if res2.Status.UserInfo.Username != tc.expectedName {
				t.Fatalf("unexpected username: wanted %s, got %s", tc.expectedName, res.Status.UserInfo.Username)
			}

			if res2.Status.UserInfo.UID != tc.expectedUID {
				t.Fatalf("unexpected uid: wanted %s, got %s", tc.expectedUID, res.Status.UserInfo.UID)
			}

			if !reflect.DeepEqual(res2.Status.UserInfo.Groups, tc.expectedGroups) {
				t.Fatalf("unexpected groups: wanted %v, got %v", tc.expectedGroups, res.Status.UserInfo.Groups)
			}

			if !reflect.DeepEqual(res2.Status.UserInfo.Extra, tc.expectedExtra) {
				t.Fatalf("unexpected extra: wanted %v, got %v", tc.expectedExtra, res.Status.UserInfo.Extra)
			}

			res3, err := kubeClient.AuthenticationV1().
				SelfSubjectReviews().
				Create(context.TODO(), &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if res3 == nil {
				t.Fatalf("empty response")
			}

			if res3.Status.UserInfo.Username != tc.expectedName {
				t.Fatalf("unexpected username: wanted %s, got %s", tc.expectedName, res.Status.UserInfo.Username)
			}

			if res3.Status.UserInfo.UID != tc.expectedUID {
				t.Fatalf("unexpected uid: wanted %s, got %s", tc.expectedUID, res.Status.UserInfo.UID)
			}

			if !reflect.DeepEqual(res3.Status.UserInfo.Groups, tc.expectedGroups) {
				t.Fatalf("unexpected groups: wanted %v, got %v", tc.expectedGroups, res.Status.UserInfo.Groups)
			}

			if !reflect.DeepEqual(res3.Status.UserInfo.Extra, tc.expectedExtra) {
				t.Fatalf("unexpected extra: wanted %v, got %v", tc.expectedExtra, res.Status.UserInfo.Extra)
			}
		})
	}
}

func TestGetsSelfAttributesError(t *testing.T) {
	toggle := &atomic.Value{}
	toggle.Store(true)

	tCtx := ktesting.Init(t)
	kubeClient, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1alpha1=true")
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1beta1=true")
			opts.APIEnablement.RuntimeConfig.Set("authentication.k8s.io/v1=true")
			opts.Authorization.Modes = []string{"AlwaysAllow"}
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			// Unset BearerToken to disable BearerToken authenticator.
			config.ControlPlane.Generic.LoopbackClientConfig.BearerToken = ""
			config.ControlPlane.Generic.Authentication.Authenticator = authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
				if toggle.Load().(bool) {
					return &authenticator.Response{
						User: &user.DefaultInfo{
							Name: "alice",
						},
					}, true, nil
				}

				return nil, false, fmt.Errorf("test error")
			})
		},
	})
	defer tearDownFn()

	expected := fmt.Errorf("Unauthorized")

	{ // v1alpha1
		toggle.Store(!toggle.Load().(bool))

		_, err := kubeClient.AuthenticationV1alpha1().
			SelfSubjectReviews().
			Create(tCtx, &authenticationv1alpha1.SelfSubjectReview{}, metav1.CreateOptions{})
		if err == nil {
			t.Fatalf("expected error: %v, got nil", err)
		}

		toggle.Store(!toggle.Load().(bool))
		if expected.Error() != err.Error() {
			t.Fatalf("expected error: %v, got %v", expected, err)
		}
	}

	{ // v1beta1
		toggle.Store(!toggle.Load().(bool))

		_, err := kubeClient.AuthenticationV1beta1().
			SelfSubjectReviews().
			Create(tCtx, &authenticationv1beta1.SelfSubjectReview{}, metav1.CreateOptions{})
		if err == nil {
			t.Fatalf("expected error: %v, got nil", err)
		}

		toggle.Store(!toggle.Load().(bool))
		if expected.Error() != err.Error() {
			t.Fatalf("expected error: %v, got %v", expected, err)
		}
	}

	{ // v1
		toggle.Store(!toggle.Load().(bool))

		_, err := kubeClient.AuthenticationV1().
			SelfSubjectReviews().
			Create(context.TODO(), &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
		if err == nil {
			t.Fatalf("expected error: %v, got nil", err)
		}

		toggle.Store(!toggle.Load().(bool))
		if expected.Error() != err.Error() {
			t.Fatalf("expected error: %v, got %v", expected, err)
		}
	}
}
