/*
Copyright The Kubernetes Authors.

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
	"context"
	"net/url"
	"strings"
	"testing"

	registrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/apiserver/pkg/authentication/user"
)

// podTurboUpdate returns an UPDATE of pods/turbo, whose endpoint declares pods as an admission
// equivalent.
func podTurboUpdate() (*webhooktesting.FakeAttributes, admission.ObjectInterfaces) {
	const ns = "webhook-test"
	obj := &corev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: ns, Labels: map[string]string{"app": "a"}}}
	attr := &webhooktesting.FakeAttributes{Attributes: admission.NewAttributesRecord(obj, obj.DeepCopy(),
		corev1.SchemeGroupVersion.WithKind("Pod"), ns, "p", corev1.SchemeGroupVersion.WithResource("pods"), "turbo",
		admission.Update, &metav1.UpdateOptions{}, false, &user.DefaultInfo{Name: "webhook-test"})}
	o := admissiontesting.ObjectInterfacesWithEquivalents{
		ObjectInterfaces: webhooktesting.NewObjectInterfacesForTest(),
		Equivalents:      []admission.Equivalent{{Subresource: ""}},
	}
	return attr, o
}

func podWebhook(name, path string, objectSelector map[string]string, resources ...string) registrationv1.MutatingWebhook {
	return registrationv1.MutatingWebhook{
		Name: name,
		ClientConfig: registrationv1.WebhookClientConfig{
			Service:  &registrationv1.ServiceReference{Name: "webhook-test", Namespace: "default", Path: &path},
			CABundle: testcerts.CACert,
		},
		Rules: []registrationv1.RuleWithOperations{{
			Operations: []registrationv1.OperationType{registrationv1.OperationAll},
			Rule:       registrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: resources},
		}},
		NamespaceSelector:       &metav1.LabelSelector{},
		ObjectSelector:          &metav1.LabelSelector{MatchLabels: objectSelector},
		AdmissionReviewVersions: []string{"v1beta1"},
	}
}

// newEquivalentsTestWebhook returns a mutating webhook plugin serving webhooks from a test server.
func newEquivalentsTestWebhook(t *testing.T, webhooks []registrationv1.MutatingWebhook) *Plugin {
	testServer := webhooktesting.NewTestServer(t)
	testServer.StartTLS()
	t.Cleanup(testServer.Close)
	serverURL, err := url.ParseRequestURI(testServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	stopCh := make(chan struct{})
	t.Cleanup(func() { close(stopCh) })

	wh, err := NewMutatingWebhook(nil)
	if err != nil {
		t.Fatal(err)
	}
	client, informer := webhooktesting.NewFakeMutatingDataSource("webhook-test", webhooks, stopCh)
	wh.SetAuthenticationInfoResolverWrapper(webhooktesting.Wrapper(webhooktesting.NewAuthenticationInfoResolver(new(int32))))
	wh.SetServiceResolver(webhooktesting.NewServiceResolver(*serverURL))
	wh.SetExternalKubeClientSet(client)
	wh.SetExternalKubeInformerFactory(informer)
	if err := wh.ValidateInitialization(); err != nil {
		t.Fatal(err)
	}
	informer.Start(stopCh)
	informer.WaitForCacheSync(stopCh)
	return wh
}

// The second hook's objectSelector only matches after the first hook's mutation, so the check
// before dispatch passes and the violation is caught when the dispatcher re-evaluates it.
func TestAdmitAdmissionEquivalentsAfterMutation(t *testing.T) {
	wh := newEquivalentsTestWebhook(t, []registrationv1.MutatingWebhook{
		podWebhook("add.example.com", "addLabel", nil, "pods", "pods/turbo"),
		podWebhook("late.example.com", "shouldNotBeCalled", map[string]string{"added": "test"}, "pods"),
	})
	attr, o := podTurboUpdate()

	err := wh.Admit(context.Background(), attr, o)
	const want = `MutatingAdmissionWebhook "late.example.com" (configuration "test-webhooks") applies to pods (UPDATE) but not pods/turbo (UPDATE)`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("expected error containing\n\t%q\ngot\n\t%v", want, err)
	}
}

// A later covering hook's mutation can make an earlier non-covering hook apply to an admission
// equivalent. Round 1 cannot see it (the earlier hook was already skipped), so the check before
// dispatching the reinvocation round must reject. The test simulates reinvocation directly by
// marking the reinvocation context and calling Admit again.
func TestAdmitAdmissionEquivalentsReinvocation(t *testing.T) {
	wh := newEquivalentsTestWebhook(t, []registrationv1.MutatingWebhook{
		podWebhook("early.example.com", "shouldNotBeCalled", map[string]string{"added": "test"}, "pods"),
		podWebhook("add.example.com", "addLabel", nil, "pods", "pods/turbo"),
	})
	attr, o := podTurboUpdate()

	if err := wh.Admit(context.Background(), attr, o); err != nil {
		t.Fatalf("round 1: unexpected error: %v", err)
	}
	if got := attr.GetObject().(*corev1.Pod).Labels["added"]; got != "test" {
		t.Fatalf("round 1: expected the covering hook to add the label, got labels %v", attr.GetObject().(*corev1.Pod).Labels)
	}

	attr.GetReinvocationContext().SetIsReinvoke()
	err := wh.Admit(context.Background(), attr, o)
	const want = `MutatingAdmissionWebhook "early.example.com" (configuration "test-webhooks") applies to pods (UPDATE) but not pods/turbo (UPDATE)`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("round 2: expected coverage error containing\n\t%q\ngot\n\t%v", want, err)
	}
}
