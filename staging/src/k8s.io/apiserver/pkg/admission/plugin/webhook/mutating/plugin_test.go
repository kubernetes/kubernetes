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
	"fmt"
	"net/url"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
)

// TestAdmit tests that MutatingWebhook#Admit works as expected
func TestAdmit(t *testing.T) {
	testServer := webhooktesting.NewTestServer(t)
	testServer.StartTLS()
	defer testServer.Close()
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatalf("this should never happen? %v", err)
	}

	objectInterfaces := webhooktesting.NewObjectInterfacesForTest()

	stopCh := make(chan struct{})
	defer close(stopCh)

	testCases := append(webhooktesting.NewMutatingTestCases(serverURL),
		webhooktesting.ConvertToMutatingTestCases(webhooktesting.NewNonMutatingTestCases(serverURL))...)

	for _, tt := range testCases {
		t.Run(tt.Name, func(t *testing.T) {
			wh, err := NewMutatingWebhook(nil)
			if err != nil {
				t.Errorf("failed to create mutating webhook: %v", err)
				return
			}

			ns := "webhook-test"
			client, informer := webhooktesting.NewFakeMutatingDataSource(ns, tt.Webhooks, stopCh)

			wh.SetAuthenticationInfoResolverWrapper(webhooktesting.Wrapper(webhooktesting.NewAuthenticationInfoResolver(new(int32))))
			wh.SetServiceResolver(webhooktesting.NewServiceResolver(*serverURL))
			wh.SetExternalKubeClientSet(client)
			wh.SetExternalKubeInformerFactory(informer)

			informer.Start(stopCh)
			informer.WaitForCacheSync(stopCh)

			if err = wh.ValidateInitialization(); err != nil {
				t.Errorf("failed to validate initialization: %v", err)
				return
			}

			var attr admission.Attributes
			if tt.IsCRD {
				attr = webhooktesting.NewAttributeUnstructured(ns, tt.AdditionalLabels, tt.IsDryRun)
			} else {
				attr = webhooktesting.NewAttribute(ns, tt.AdditionalLabels, tt.IsDryRun)
			}

			err = wh.Admit(attr, objectInterfaces)
			if tt.ExpectAllow != (err == nil) {
				t.Errorf("expected allowed=%v, but got err=%v", tt.ExpectAllow, err)
			}
			if tt.ExpectLabels != nil {
				if !reflect.DeepEqual(tt.ExpectLabels, attr.GetObject().(metav1.Object).GetLabels()) {
					t.Errorf("expected labels '%v', but got '%v'", tt.ExpectLabels, attr.GetObject().(metav1.Object).GetLabels())
				}
			}
			// ErrWebhookRejected is not an error for our purposes
			if tt.ErrorContains != "" {
				if err == nil || !strings.Contains(err.Error(), tt.ErrorContains) {
					t.Errorf("expected an error saying %q, but got: %v", tt.ErrorContains, err)
				}
			}
			if statusErr, isStatusErr := err.(*errors.StatusError); err != nil && !isStatusErr {
				t.Errorf("expected a StatusError, got %T", err)
			} else if isStatusErr {
				if statusErr.ErrStatus.Code != tt.ExpectStatusCode {
					t.Errorf("expected status code %d, got %d", tt.ExpectStatusCode, statusErr.ErrStatus.Code)
				}
			}
			fakeAttr, ok := attr.(*webhooktesting.FakeAttributes)
			if !ok {
				t.Errorf("Unexpected error, failed to convert attr to webhooktesting.FakeAttributes")
				return
			}
			if len(tt.ExpectAnnotations) == 0 {
				assert.Empty(t, fakeAttr.GetAnnotations(), tt.Name+": annotations not set as expected.")
			} else {
				assert.Equal(t, tt.ExpectAnnotations, fakeAttr.GetAnnotations(), tt.Name+": annotations not set as expected.")
			}
			reinvocationCtx := fakeAttr.Attributes.GetReinvocationContext()
			reinvocationCtx.SetIsReinvoke()
			for webhook, expectReinvoke := range tt.ExpectReinvokeWebhooks {
				shouldReinvoke := reinvocationCtx.Value(PluginName).(*webhookReinvokeContext).ShouldReinvokeWebhook(fmt.Sprintf("test-webhooks/%s/0", webhook))
				if expectReinvoke != shouldReinvoke {
					t.Errorf("expected reinvocationContext.ShouldReinvokeWebhook(%s)=%t, but got %t", webhook, expectReinvoke, shouldReinvoke)
				}
			}
		})
	}
}

// TestAdmitCachedClient tests that MutatingWebhook#Admit should cache restClient
func TestAdmitCachedClient(t *testing.T) {
	testServer := webhooktesting.NewTestServer(t)
	testServer.StartTLS()
	defer testServer.Close()
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatalf("this should never happen? %v", err)
	}

	objectInterfaces := webhooktesting.NewObjectInterfacesForTest()

	stopCh := make(chan struct{})
	defer close(stopCh)

	wh, err := NewMutatingWebhook(nil)
	if err != nil {
		t.Fatalf("Failed to create mutating webhook: %v", err)
	}
	wh.SetServiceResolver(webhooktesting.NewServiceResolver(*serverURL))

	for _, tt := range webhooktesting.NewCachedClientTestcases(serverURL) {
		ns := "webhook-test"
		client, informer := webhooktesting.NewFakeMutatingDataSource(ns, webhooktesting.ConvertToMutatingWebhooks(tt.Webhooks), stopCh)

		// override the webhook source. The client cache will stay the same.
		cacheMisses := new(int32)
		wh.SetAuthenticationInfoResolverWrapper(webhooktesting.Wrapper(webhooktesting.NewAuthenticationInfoResolver(cacheMisses)))
		wh.SetExternalKubeClientSet(client)
		wh.SetExternalKubeInformerFactory(informer)

		informer.Start(stopCh)
		informer.WaitForCacheSync(stopCh)

		if err = wh.ValidateInitialization(); err != nil {
			t.Errorf("%s: failed to validate initialization: %v", tt.Name, err)
			continue
		}

		err = wh.Admit(webhooktesting.NewAttribute(ns, nil, false), objectInterfaces)
		if tt.ExpectAllow != (err == nil) {
			t.Errorf("%s: expected allowed=%v, but got err=%v", tt.Name, tt.ExpectAllow, err)
		}

		if tt.ExpectCacheMiss && *cacheMisses == 0 {
			t.Errorf("%s: expected cache miss, but got no AuthenticationInfoResolver call", tt.Name)
		}

		if !tt.ExpectCacheMiss && *cacheMisses > 0 {
			t.Errorf("%s: expected client to be cached, but got %d AuthenticationInfoResolver calls", tt.Name, *cacheMisses)
		}
	}
}
