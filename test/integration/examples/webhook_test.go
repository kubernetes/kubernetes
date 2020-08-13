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
	"context"
	"sync/atomic"
	"testing"
	"time"

	admissionv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/master/reconcilers"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestWebhookLoopback(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	webhookPath := "/webhook-test"

	called := int32(0)

	client, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
		},
		ModifyServerConfig: func(config *master.Config) {
			// Avoid resolveable kubernetes service
			config.ExtraConfig.EndpointReconcilerType = reconcilers.NoneEndpointReconcilerType

			// Hook into audit to watch requests
			config.GenericConfig.AuditBackend = auditSinkFunc(func(events ...*auditinternal.Event) {})
			config.GenericConfig.AuditPolicyChecker = auditChecker(func(attrs authorizer.Attributes) (auditinternal.Level, []auditinternal.Stage) {
				if attrs.GetPath() == webhookPath {
					if attrs.GetUser().GetName() != "system:apiserver" {
						t.Errorf("expected user %q, got %q", "system:apiserver", attrs.GetUser().GetName())
					}
					atomic.AddInt32(&called, 1)
				}
				return auditinternal.LevelNone, nil
			})
		},
	})

	fail := admissionv1beta1.Fail
	_, err := client.AdmissionregistrationV1beta1().MutatingWebhookConfigurations().Create(context.TODO(), &admissionv1beta1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "webhooktest.example.com"},
		Webhooks: []admissionv1beta1.MutatingWebhook{{
			Name: "webhooktest.example.com",
			ClientConfig: admissionv1beta1.WebhookClientConfig{
				Service: &admissionv1beta1.ServiceReference{Namespace: "default", Name: "kubernetes", Path: &webhookPath},
			},
			Rules: []admissionv1beta1.RuleWithOperations{{
				Operations: []admissionv1beta1.OperationType{admissionv1beta1.OperationAll},
				Rule:       admissionv1beta1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
			}},
			FailurePolicy: &fail,
		}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (done bool, err error) {
		_, err = client.CoreV1().ConfigMaps("default").Create(context.TODO(), &v1.ConfigMap{
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

type auditChecker func(authorizer.Attributes) (auditinternal.Level, []auditinternal.Stage)

func (f auditChecker) LevelAndStages(attrs authorizer.Attributes) (auditinternal.Level, []auditinternal.Stage) {
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
