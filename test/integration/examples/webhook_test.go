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

package apiserver

import (
	"sync/atomic"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/controlplane/reconcilers"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestWebhookLoopback(t *testing.T) {
	webhookPath := "/webhook-test"

	called := int32(0)

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			// Avoid resolvable kubernetes service
			config.Extra.EndpointReconcilerType = reconcilers.NoneEndpointReconcilerType

			// Hook into audit to watch requests
			config.ControlPlane.Generic.AuditBackend = auditSinkFunc(func(events ...*auditinternal.Event) {})
			config.ControlPlane.Generic.AuditPolicyRuleEvaluator = auditPolicyRuleEvaluator(func(attrs authorizer.Attributes) audit.RequestAuditConfig {
				if attrs.GetPath() == webhookPath {
					if attrs.GetUser().GetName() != "system:apiserver" {
						t.Errorf("expected user %q, got %q", "system:apiserver", attrs.GetUser().GetName())
					}
					atomic.AddInt32(&called, 1)
				}
				return audit.RequestAuditConfig{
					Level: auditinternal.LevelNone,
				}
			})
		},
	})
	defer tearDownFn()

	fail := admissionregistrationv1.Fail
	noSideEffects := admissionregistrationv1.SideEffectClassNone
	_, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(tCtx, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "webhooktest.example.com"},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "webhooktest.example.com",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				Service: &admissionregistrationv1.ServiceReference{Namespace: "default", Name: "kubernetes", Path: &webhookPath},
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
			}},
			FailurePolicy:           &fail,
			SideEffects:             &noSideEffects,
			AdmissionReviewVersions: []string{"v1"},
		}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (done bool, err error) {
		_, err = client.CoreV1().ConfigMaps("default").Create(tCtx, &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: "webhook-test"},
			Data:       map[string]string{"invalid key": "value"},
		}, metav1.CreateOptions{})
		if err == nil {
			t.Fatal("Unexpected success")
		}
		if called > 0 {
			return true, nil
		}
		t.Logf("%v", err)
		t.Logf("webhook not called yet, continuing...")
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

type auditPolicyRuleEvaluator func(authorizer.Attributes) audit.RequestAuditConfig

func (f auditPolicyRuleEvaluator) EvaluatePolicyRule(attrs authorizer.Attributes) audit.RequestAuditConfig {
	return f(attrs)
}

type auditSinkFunc func(events ...*auditinternal.Event)

func (f auditSinkFunc) ProcessEvents(events ...*auditinternal.Event) bool {
	f(events...)
	return true
}

func (auditSinkFunc) Run(stopCh <-chan struct{}) error {
	return nil
}

func (auditSinkFunc) Shutdown() {
}

func (auditSinkFunc) String() string {
	return ""
}
