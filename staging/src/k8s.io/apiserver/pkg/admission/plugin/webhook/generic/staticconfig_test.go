/*
Copyright 2020 The Kubernetes Authors.

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

package generic

import (
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/admissionregistration/v1"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/client-go/informers"
)

var (
	validatingWebhookConfig = `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: w1
webhooks:
- failurePolicy: Ignore
  name: w11.kube.io
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - '*'
  sideEffects: None
  timeoutSeconds: 5
- failurePolicy: Fail
  name: w12.kube.io
  rules:
  - apiGroups:
    - ""
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - namespaces
  sideEffects: None
  timeoutSeconds: 5
`
	validatingWebhookConfigList = `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfigurationList
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: w2
  webhooks:
  - failurePolicy: Ignore
    name: w21.kube.io
    rules:
    - apiGroups:
      - '*'
      apiVersions:
      - '*'
      operations:
      - CREATE
      - UPDATE
      resources:
      - '*'
    sideEffects: None
    timeoutSeconds: 5
  - failurePolicy: Fail
    name: w22.kube.io
    rules:
    - apiGroups:
      - '*'
      apiVersions:
      - '*'
      operations:
      - CREATE
      - UPDATE
      resources:
      - '*'
    sideEffects: None
    timeoutSeconds: 5
`
	mutatingWebhookConfig = `
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: w1
webhooks:
- name: w31.kube.io
  rules:
  - operations: ["CREATE"]
    apiGroups: ["*"]
    apiVersions: ["*"]
    resources: ["*"]
    scope: "*"
  sideEffects: None
  timeoutSeconds: 5
`
	multiValidatingWebhookConfig = `
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: w1
webhooks:
- failurePolicy: Ignore
  name: w11.kube.io
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - '*'
  sideEffects: None
  timeoutSeconds: 5
- failurePolicy: Fail
  name: w12.kube.io
  rules:
  - apiGroups:
    - ""
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - namespaces
  sideEffects: None
  timeoutSeconds: 5
---
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: w2
webhooks:
- failurePolicy: Ignore
  name: w21.kube.io
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - '*'
  sideEffects: None
  timeoutSeconds: 5
- failurePolicy: Fail
  name: w22.kube.io
  rules:
  - apiGroups:
    - ""
    apiVersions:
    - '*'
    operations:
    - CREATE
    - UPDATE
    resources:
    - namespaces
  sideEffects: None
  timeoutSeconds: 5
`
	listOfValidatingWebhookConfig = `
apiVersion: meta.k8s.io/v1
kind: List
items:
- apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingWebhookConfiguration
  metadata:
    name: w1
  webhooks:
  - failurePolicy: Ignore
    name: w11.kube.io
    rules:
    - apiGroups:
      - '*'
      apiVersions:
      - '*'
      operations:
      - CREATE
      - UPDATE
      resources:
      - '*'
    sideEffects: None
    timeoutSeconds: 5
  - failurePolicy: Fail
    name: w12.kube.io
    rules:
    - apiGroups:
      - '*'
      apiVersions:
      - '*'
      operations:
      - CREATE
      - UPDATE
      resources:
      - '*'
    sideEffects: None
    timeoutSeconds: 5
`
	webhook1 = webhook.NewMutatingWebhookAccessor("uid", "w4", &v1.MutatingWebhook{
		Name: "w41.kube.io",
		Rules: []v1.RuleWithOperations{{
			Operations: []v1.OperationType{v1.Create},
		}},
	})
)

func Test_StaticWebhookConfig(t *testing.T) {
	testCases := []struct {
		name             string
		dynamicWebhooks  []webhook.WebhookAccessor
		staticConfig     string
		expectedWebhooks []string
	}{
		{
			"no dynamic webhooks; validating webhook configuration list",
			nil,
			validatingWebhookConfigList,
			[]string{"w21.kube.io", "w22.kube.io"},
		},
		{
			"no dynamic webhooks; validating webhook configuration",
			nil,
			validatingWebhookConfig,
			[]string{"w11.kube.io", "w12.kube.io"},
		},
		{
			"no dynamic webhooks; validating webhook configuration",
			nil,
			validatingWebhookConfig,
			[]string{"w11.kube.io", "w12.kube.io"},
		},
		{
			"no dynamic webhooks; mutating webhook configuration",
			nil,
			mutatingWebhookConfig,
			[]string{"w31.kube.io"},
		},
		{
			"no dynamic webhooks; multi validating webhook configuration",
			nil,
			multiValidatingWebhookConfig,
			[]string{"w11.kube.io", "w12.kube.io", "w21.kube.io", "w22.kube.io"},
		},
		{
			"one dynamic webhooks; mutating webhook configuration",
			[]webhook.WebhookAccessor{webhook1},
			mutatingWebhookConfig,
			[]string{"w31.kube.io", "w41.kube.io"},
		},
		{
			"no dynamic webhooks; list of validating webhook configuration",
			nil,
			listOfValidatingWebhookConfig,
			[]string{"w11.kube.io", "w12.kube.io"},
		},
	}

	tempFile, err := ioutil.TempFile("", "")
	assert.NoError(t, err)
	defer tempFile.Close()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err = tempFile.Write([]byte(tc.staticConfig))
			assert.NoError(t, err)

			watcher := NewStaticConfigWatcher(tempFile.Name(), nil, nil)
			watcher.Init()
			sf := wrapSourceFactory(testSourceFactory(tc.dynamicWebhooks), watcher.getWebhookAccessors)
			hooks := sf(nil).Webhooks()
			for _, expected := range tc.expectedWebhooks {
				found := false
				for _, hook := range hooks {
					if strings.Contains(hook.GetName(), expected) {
						found = true
					}
				}
				if !found {
					assert.Failf(t, "Missing expected webhook", "could not expected webhook: %v", expected)
				}
			}
		})
	}
}

func Test_StaticConfigWatch(t *testing.T) {
	tempFile, err := ioutil.TempFile("", "")
	assert.NoError(t, err)
	defer tempFile.Close()

	_, err = tempFile.Write([]byte(validatingWebhookConfig))
	assert.NoError(t, err)

	watcher := NewStaticConfigWatcher(tempFile.Name(), nil, nil)
	watcher.Init()
	sf := wrapSourceFactory(testSourceFactory(nil), watcher.getWebhookAccessors)

	expectedHooks := []string{"w11.kube.io", "w12.kube.io"}
	hooks := sf(nil).Webhooks()
	for _, expected := range expectedHooks {
		found := false
		for _, hook := range hooks {
			if strings.Contains(hook.GetName(), expected) {
				found = true
			}
		}
		if !found {
			assert.Failf(t, "Missing expected webhook", "could not expected webhook: %v", expected)
		}
	}

	// Update file
	_, err = tempFile.Write([]byte(validatingWebhookConfigList))
	assert.NoError(t, err)

	expectedHooks = []string{"w21.kube.io", "w22.kube.io"}
	//wait for reload
	time.Sleep(time.Second)
	hooks = sf(nil).Webhooks()
	for _, expected := range expectedHooks {
		found := false
		for _, hook := range hooks {
			if strings.Contains(hook.GetName(), expected) {
				found = true
			}
		}
		if !found {
			assert.Failf(t, "Missing expected webhook", "could not expected webhook: %v", expected)
		}
	}
}

func testSourceFactory(accessors []webhook.WebhookAccessor) sourceFactory {
	return func(f informers.SharedInformerFactory) Source {
		return &testSource{webhooks: accessors}
	}
}

type testSource struct {
	webhooks []webhook.WebhookAccessor
}

func (t *testSource) Webhooks() []webhook.WebhookAccessor {
	return t.webhooks
}

func (t *testSource) HasSynced() bool {
	return true
}
