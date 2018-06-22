/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"testing"

	"k8s.io/api/admissionregistration/v1beta1"
)

func TestHookClient(t *testing.T) {
	scenarios := []struct {
		clientConfig v1beta1.WebhookClientConfig
		expectedUrl  string
		expectError  bool
	}{
		// scenario_0: A valid https URL with a CABundle provided.
		{
			clientConfig: v1beta1.WebhookClientConfig{
				URL:      strPtr("https://k8s.io:8080/"),
				CABundle: []byte("123"),
			},
			expectedUrl: "https://k8s.io:8080/",
		},
		// scenario_1: A valid https URL without a CABundle provided.
		{
			clientConfig: v1beta1.WebhookClientConfig{
				URL: strPtr("https://k8s.io:8080/"),
			},
			expectedUrl: "https://k8s.io:8080/",
		},
		// scenario_2: A valid http URL without a CABundle provided.
		{
			clientConfig: v1beta1.WebhookClientConfig{
				URL: strPtr("http://k8s.io:8080/"),
			},
			expectedUrl: "http://k8s.io:8080/",
		},
	}

	// act
	for index, scenario := range scenarios {
		t.Run(fmt.Sprintf("scenario %d", index), func(t *testing.T) {
			cm := newClientManagerOrDie(t)
			webhook := toWebhook(scenario.clientConfig)
			client, err := cm.HookClient(webhook)
			if err != nil && !scenario.expectError {
				t.Errorf("unexpected error has occurred = %v", err)
			}
			if err == nil && scenario.expectError {
				t.Error("expected an error but got nothing")
			}
			if !scenario.expectError {
				actualUrl := client.Verb("").URL()
				actualUrl.RawQuery = ""
				if scenario.expectedUrl != actualUrl.String() {
					t.Errorf("expected = %s, got = %s", scenario.expectedUrl, actualUrl.String())
				}
			}
		})
	}
}

func strPtr(s string) *string {
	return &s
}

func newClientManagerOrDie(t *testing.T) ClientManager {
	cm, err := NewClientManager()
	if err != nil {
		t.Errorf("unable to create client manager: %v", err)
	}
	auth, err := NewDefaultAuthenticationInfoResolver("")
	if err != nil {
		t.Errorf("unable to create auth info resolver: %v", err)
	}
	cm.SetAuthenticationInfoResolver(auth)
	return cm
}

func toWebhook(clientConfig v1beta1.WebhookClientConfig) *v1beta1.Webhook {
	return &v1beta1.Webhook{
		Name:         "test-webhook",
		ClientConfig: clientConfig,
	}
}
