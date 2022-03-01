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

package testing

import (
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"time"

	registrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
)

var matchEverythingRules = []registrationv1.RuleWithOperations{{
	Operations: []registrationv1.OperationType{registrationv1.OperationAll},
	Rule: registrationv1.Rule{
		APIGroups:   []string{"*"},
		APIVersions: []string{"*"},
		Resources:   []string{"*/*"},
	},
}}

var sideEffectsUnknown = registrationv1.SideEffectClassUnknown
var sideEffectsNone = registrationv1.SideEffectClassNone
var sideEffectsSome = registrationv1.SideEffectClassSome
var sideEffectsNoneOnDryRun = registrationv1.SideEffectClassNoneOnDryRun

var reinvokeNever = registrationv1.NeverReinvocationPolicy
var reinvokeIfNeeded = registrationv1.IfNeededReinvocationPolicy

// NewFakeValidatingDataSource returns a mock client and informer returning the given webhooks.
func NewFakeValidatingDataSource(name string, webhooks []registrationv1.ValidatingWebhook, stopCh <-chan struct{}) (clientset kubernetes.Interface, factory informers.SharedInformerFactory) {
	var objs = []runtime.Object{
		&corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"runlevel": "0",
				},
			},
		},
	}
	objs = append(objs, &registrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-webhooks",
		},
		Webhooks: webhooks,
	})

	client := fakeclientset.NewSimpleClientset(objs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	return client, informerFactory
}

// NewFakeMutatingDataSource returns a mock client and informer returning the given webhooks.
func NewFakeMutatingDataSource(name string, webhooks []registrationv1.MutatingWebhook, stopCh <-chan struct{}) (clientset kubernetes.Interface, factory informers.SharedInformerFactory) {
	var objs = []runtime.Object{
		&corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"runlevel": "0",
				},
			},
		},
	}
	objs = append(objs, &registrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-webhooks",
		},
		Webhooks: webhooks,
	})

	client := fakeclientset.NewSimpleClientset(objs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	return client, informerFactory
}

func newAttributesRecord(object metav1.Object, oldObject metav1.Object, kind schema.GroupVersionKind, namespace string, name string, resource string, labels map[string]string, dryRun bool) admission.Attributes {
	object.SetName(name)
	object.SetNamespace(namespace)
	objectLabels := map[string]string{resource + ".name": name}
	for k, v := range labels {
		objectLabels[k] = v
	}
	object.SetLabels(objectLabels)

	oldObject.SetName(name)
	oldObject.SetNamespace(namespace)

	gvr := kind.GroupVersion().WithResource(resource)
	subResource := ""
	userInfo := user.DefaultInfo{
		Name: "webhook-test",
		UID:  "webhook-test",
	}
	options := &metav1.UpdateOptions{}

	return &FakeAttributes{
		Attributes: admission.NewAttributesRecord(object.(runtime.Object), oldObject.(runtime.Object), kind, namespace, name, gvr, subResource, admission.Update, options, dryRun, &userInfo),
	}
}

// FakeAttributes decorate admission.Attributes. It's used to trace the added annotations.
type FakeAttributes struct {
	admission.Attributes
	annotations map[string]string
	mutex       sync.Mutex
}

// AddAnnotation adds an annotation key value pair to FakeAttributes
func (f *FakeAttributes) AddAnnotation(k, v string) error {
	return f.AddAnnotationWithLevel(k, v, auditinternal.LevelMetadata)
}

// AddAnnotationWithLevel adds an annotation key value pair to FakeAttributes
func (f *FakeAttributes) AddAnnotationWithLevel(k, v string, _ auditinternal.Level) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	if err := f.Attributes.AddAnnotation(k, v); err != nil {
		return err
	}
	if f.annotations == nil {
		f.annotations = make(map[string]string)
	}
	f.annotations[k] = v
	return nil
}

// GetAnnotations reads annotations from FakeAttributes
func (f *FakeAttributes) GetAnnotations(level auditinternal.Level) map[string]string {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	return f.annotations
}

// NewAttribute returns static admission Attributes for testing.
func NewAttribute(namespace string, labels map[string]string, dryRun bool) admission.Attributes {
	// Set up a test object for the call
	object := corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
	}
	oldObject := corev1.Pod{}
	kind := corev1.SchemeGroupVersion.WithKind("Pod")
	name := "my-pod"

	return newAttributesRecord(&object, &oldObject, kind, namespace, name, "pod", labels, dryRun)
}

// NewAttributeUnstructured returns static admission Attributes for testing with custom resources.
func NewAttributeUnstructured(namespace string, labels map[string]string, dryRun bool) admission.Attributes {
	// Set up a test object for the call
	object := unstructured.Unstructured{}
	object.SetKind("TestCRD")
	object.SetAPIVersion("custom.resource/v1")
	oldObject := unstructured.Unstructured{}
	oldObject.SetKind("TestCRD")
	oldObject.SetAPIVersion("custom.resource/v1")
	kind := object.GroupVersionKind()
	name := "my-test-crd"

	return newAttributesRecord(&object, &oldObject, kind, namespace, name, "crd", labels, dryRun)
}

type urlConfigGenerator struct {
	baseURL *url.URL
}

func (c urlConfigGenerator) ccfgURL(urlPath string) registrationv1.WebhookClientConfig {
	u2 := *c.baseURL
	u2.Path = urlPath
	urlString := u2.String()
	return registrationv1.WebhookClientConfig{
		URL:      &urlString,
		CABundle: testcerts.CACert,
	}
}

// ValidatingTest is a validating webhook test case.
type ValidatingTest struct {
	Name                   string
	Webhooks               []registrationv1.ValidatingWebhook
	Path                   string
	IsCRD                  bool
	IsDryRun               bool
	AdditionalLabels       map[string]string
	SkipBenchmark          bool
	ExpectLabels           map[string]string
	ExpectAllow            bool
	ErrorContains          string
	ExpectAnnotations      map[string]string
	ExpectStatusCode       int32
	ExpectReinvokeWebhooks map[string]bool
}

// MutatingTest is a mutating webhook test case.
type MutatingTest struct {
	Name                   string
	Webhooks               []registrationv1.MutatingWebhook
	Path                   string
	IsCRD                  bool
	IsDryRun               bool
	AdditionalLabels       map[string]string
	SkipBenchmark          bool
	ExpectLabels           map[string]string
	ExpectAllow            bool
	ErrorContains          string
	ExpectAnnotations      map[string]string
	ExpectStatusCode       int32
	ExpectReinvokeWebhooks map[string]bool
}

// DurationTest is webhook duration test case, used both in mutating and
// validating plugin test cases.
type DurationTest struct {
	Name                string
	Webhooks            []registrationv1.ValidatingWebhook
	InitContext         bool
	IsDryRun            bool
	ExpectedDurationSum time.Duration
	ExpectedDurationMax time.Duration
}

// ConvertToMutatingTestCases converts a validating test case to a mutating one for test purposes.
func ConvertToMutatingTestCases(tests []ValidatingTest, configurationName string) []MutatingTest {
	r := make([]MutatingTest, len(tests))
	for i, t := range tests {
		for idx, hook := range t.Webhooks {
			if t.ExpectAnnotations == nil {
				t.ExpectAnnotations = map[string]string{}
			}
			// Add expected annotation if the converted webhook is intended to match
			if reflect.DeepEqual(hook.NamespaceSelector, &metav1.LabelSelector{}) &&
				reflect.DeepEqual(hook.ObjectSelector, &metav1.LabelSelector{}) &&
				reflect.DeepEqual(hook.Rules, matchEverythingRules) {
				key := fmt.Sprintf("mutation.webhook.admission.k8s.io/round_0_index_%d", idx)
				value := mutationAnnotationValue(configurationName, hook.Name, false)
				t.ExpectAnnotations[key] = value
			}
			// Break if the converted webhook is intended to fail close
			if strings.Contains(hook.Name, "internalErr") && (hook.FailurePolicy == nil || *hook.FailurePolicy == registrationv1.Fail) {
				break
			}
		}
		// Change annotation keys for Validating's fail open to Mutating's fail open.
		failOpenAnnotations := map[string]string{}
		for key, value := range t.ExpectAnnotations {
			if strings.HasPrefix(key, "failed-open.validating.webhook.admission.k8s.io/") {
				failOpenAnnotations[key] = value
			}
		}
		for key, value := range failOpenAnnotations {
			newKey := strings.Replace(key, "failed-open.validating.webhook.admission.k8s.io/", "failed-open.mutation.webhook.admission.k8s.io/", 1)
			t.ExpectAnnotations[newKey] = value
			delete(t.ExpectAnnotations, key)
		}
		r[i] = MutatingTest{t.Name, ConvertToMutatingWebhooks(t.Webhooks), t.Path, t.IsCRD, t.IsDryRun, t.AdditionalLabels, t.SkipBenchmark, t.ExpectLabels, t.ExpectAllow, t.ErrorContains, t.ExpectAnnotations, t.ExpectStatusCode, t.ExpectReinvokeWebhooks}
	}
	return r
}

// ConvertToMutatingWebhooks converts a validating webhook to a mutating one for test purposes.
func ConvertToMutatingWebhooks(webhooks []registrationv1.ValidatingWebhook) []registrationv1.MutatingWebhook {
	mutating := make([]registrationv1.MutatingWebhook, len(webhooks))
	for i, h := range webhooks {
		mutating[i] = registrationv1.MutatingWebhook{h.Name, h.ClientConfig, h.Rules, h.FailurePolicy, h.MatchPolicy, h.NamespaceSelector, h.ObjectSelector, h.SideEffects, h.TimeoutSeconds, h.AdmissionReviewVersions, nil}
	}
	return mutating
}

// NewNonMutatingTestCases returns test cases with a given base url.
// All test cases in NewNonMutatingTestCases have no Patch set in
// AdmissionResponse. The test cases are used by both MutatingAdmissionWebhook
// and ValidatingAdmissionWebhook.
func NewNonMutatingTestCases(url *url.URL) []ValidatingTest {
	policyFail := registrationv1.Fail
	policyIgnore := registrationv1.Ignore
	ccfgURL := urlConfigGenerator{url}.ccfgURL

	return []ValidatingTest{
		{
			Name: "no match",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:         "nomatch",
				ClientConfig: ccfgSVC("disallow"),
				Rules: []registrationv1.RuleWithOperations{{
					Operations: []registrationv1.OperationType{registrationv1.Create},
				}},
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
		{
			Name: "match & allow",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow.example.com",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow.example.com/key1": "value1"},
		},
		{
			Name: "match & disallow",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "disallow",
				ClientConfig:            ccfgSVC("disallow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusForbidden,
			ErrorContains:    "without explanation",
		},
		{
			Name: "match & disallow ii",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "disallowReason",
				ClientConfig:            ccfgSVC("disallowReason"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusForbidden,
			ErrorContains:    "you shall not pass",
		},
		{
			Name: "match & disallow & but allowed because namespaceSelector exempt the ns",
			Webhooks: []registrationv1.ValidatingWebhook{{
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
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},

			ExpectAllow: true,
		},
		{
			Name: "match & disallow & but allowed because namespaceSelector exempt the ns ii",
			Webhooks: []registrationv1.ValidatingWebhook{{
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
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
		{
			Name: "match & fail (but allow because fail open)",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "internalErr A",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr B",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr C",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},

			SkipBenchmark: true,
			ExpectAllow:   true,
			ExpectAnnotations: map[string]string{
				"failed-open.validating.webhook.admission.k8s.io/round_0_index_0": "internalErr A",
				"failed-open.validating.webhook.admission.k8s.io/round_0_index_1": "internalErr B",
				"failed-open.validating.webhook.admission.k8s.io/round_0_index_2": "internalErr C",
			},
		},
		{
			Name: "match & fail (but disallow because fail close on nil FailurePolicy)",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "internalErr A",
				ClientConfig:            ccfgSVC("internalErr"),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr B",
				ClientConfig:            ccfgSVC("internalErr"),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr C",
				ClientConfig:            ccfgSVC("internalErr"),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusInternalServerError,
			ExpectAllow:      false,
		},
		{
			Name: "match & fail (but fail because fail closed)",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "internalErr A",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyFail,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr B",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyFail,
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:                    "internalErr C",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyFail,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusInternalServerError,
			ExpectAllow:      false,
		},
		{
			Name: "match & allow (url)",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow.example.com",
				ClientConfig:            ccfgURL("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow.example.com/key1": "value1"},
		},
		{
			Name: "match & disallow (url)",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "disallow",
				ClientConfig:            ccfgURL("disallow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusForbidden,
			ErrorContains:    "without explanation",
		}, {
			Name: "absent response and fail open",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "nilResponse",
				ClientConfig:            ccfgURL("nilResponse"),
				FailurePolicy:           &policyIgnore,
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			SkipBenchmark:     true,
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"failed-open.validating.webhook.admission.k8s.io/round_0_index_0": "nilResponse"},
		},
		{
			Name: "absent response and fail closed",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "nilResponse",
				ClientConfig:            ccfgURL("nilResponse"),
				FailurePolicy:           &policyFail,
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusInternalServerError,
			ErrorContains:    "webhook response was absent",
		},
		{
			Name: "no match dry run",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:         "nomatch",
				ClientConfig: ccfgSVC("allow"),
				Rules: []registrationv1.RuleWithOperations{{
					Operations: []registrationv1.OperationType{registrationv1.Create},
				}},
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsSome,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:    true,
			ExpectAllow: true,
		},
		{
			Name: "match dry run side effects Unknown",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsUnknown,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:         true,
			ExpectStatusCode: http.StatusBadRequest,
			ErrorContains:    "does not support dry run",
		},
		{
			Name: "match dry run side effects None",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:          true,
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow/key1": "value1"},
		},
		{
			Name: "match dry run side effects Some",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsSome,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:         true,
			ExpectStatusCode: http.StatusBadRequest,
			ErrorContains:    "does not support dry run",
		},
		{
			Name: "match dry run side effects NoneOnDryRun",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsNoneOnDryRun,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:          true,
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow/key1": "value1"},
		},
		{
			Name: "illegal annotation format",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "invalidAnnotation",
				ClientConfig:            ccfgURL("invalidAnnotation"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
		{
			Name: "skip webhook whose objectSelector does not match",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow.example.com",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:              "shouldNotBeCalled",
				ClientConfig:      ccfgSVC("shouldNotBeCalled"),
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"label": "nonexistent",
					},
				},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow.example.com/key1": "value1"},
		},
		{
			Name: "skip webhook whose objectSelector does not match CRD's labels",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "allow.example.com",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:              "shouldNotBeCalled",
				ClientConfig:      ccfgSVC("shouldNotBeCalled"),
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"label": "nonexistent",
					},
				},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsCRD:             true,
			ExpectAllow:       true,
			ExpectAnnotations: map[string]string{"allow.example.com/key1": "value1"},
		},
		// No need to test everything with the url case, since only the
		// connection is different.
	}
}

func mutationAnnotationValue(configuration, webhook string, mutated bool) string {
	return fmt.Sprintf(`{"configuration":"%s","webhook":"%s","mutated":%t}`, configuration, webhook, mutated)
}

func patchAnnotationValue(configuration, webhook string, patch string) string {
	return strings.Replace(fmt.Sprintf(`{"configuration": "%s", "webhook": "%s", "patch": %s, "patchType": "JSONPatch"}`, configuration, webhook, patch), " ", "", -1)
}

// NewMutatingTestCases returns test cases with a given base url.
// All test cases in NewMutatingTestCases have Patch set in
// AdmissionResponse. The test cases are only used by both MutatingAdmissionWebhook.
func NewMutatingTestCases(url *url.URL, configurationName string) []MutatingTest {
	return []MutatingTest{
		{
			Name: "match & remove label",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "removelabel.example.com",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:      true,
			AdditionalLabels: map[string]string{"remove": "me"},
			ExpectLabels:     map[string]string{"pod.name": "my-pod"},
			ExpectAnnotations: map[string]string{
				"removelabel.example.com/key1":                      "value1",
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "removelabel.example.com", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "removelabel.example.com", `[{"op": "remove", "path": "/metadata/labels/remove"}]`),
			},
		},
		{
			Name: "match & add label",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:  true,
			ExpectLabels: map[string]string{"pod.name": "my-pod", "added": "test"},
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "addLabel", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "addLabel", `[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
			},
		},
		{
			Name: "match CRD & add label",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsCRD:        true,
			ExpectAllow:  true,
			ExpectLabels: map[string]string{"crd.name": "my-test-crd", "added": "test"},
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "addLabel", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "addLabel", `[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
			},
		},
		{
			Name: "match CRD & remove label",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "removelabel.example.com",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsCRD:            true,
			ExpectAllow:      true,
			AdditionalLabels: map[string]string{"remove": "me"},
			ExpectLabels:     map[string]string{"crd.name": "my-test-crd"},
			ExpectAnnotations: map[string]string{
				"removelabel.example.com/key1":                      "value1",
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "removelabel.example.com", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "removelabel.example.com", `[{"op": "remove", "path": "/metadata/labels/remove"}]`),
			},
		},
		{
			Name: "match & invalid mutation",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "invalidMutation",
				ClientConfig:            ccfgSVC("invalidMutation"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusInternalServerError,
			ErrorContains:    "invalid character",
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "invalidMutation", false),
			},
		},
		{
			Name: "match & remove label dry run unsupported",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "removeLabel",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				SideEffects:             &sideEffectsUnknown,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsDryRun:         true,
			ExpectStatusCode: http.StatusBadRequest,
			ErrorContains:    "does not support dry run",
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "removeLabel", false),
			},
		},
		{
			Name: "first webhook remove labels, second webhook shouldn't be called",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:              "removelabel.example.com",
				ClientConfig:      ccfgSVC("removeLabel"),
				Rules:             matchEverythingRules,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"remove": "me",
					},
				},
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:              "shouldNotBeCalled",
				ClientConfig:      ccfgSVC("shouldNotBeCalled"),
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"remove": "me",
					},
				},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:      true,
			AdditionalLabels: map[string]string{"remove": "me"},
			ExpectLabels:     map[string]string{"pod.name": "my-pod"},
			ExpectAnnotations: map[string]string{
				"removelabel.example.com/key1":                      "value1",
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "removelabel.example.com", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "removelabel.example.com", `[{"op": "remove", "path": "/metadata/labels/remove"}]`),
			},
		},
		{
			Name: "first webhook remove labels from CRD, second webhook shouldn't be called",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:              "removelabel.example.com",
				ClientConfig:      ccfgSVC("removeLabel"),
				Rules:             matchEverythingRules,
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"remove": "me",
					},
				},
				AdmissionReviewVersions: []string{"v1beta1"},
			}, {
				Name:              "shouldNotBeCalled",
				ClientConfig:      ccfgSVC("shouldNotBeCalled"),
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"remove": "me",
					},
				},
				Rules:                   matchEverythingRules,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsCRD:            true,
			ExpectAllow:      true,
			AdditionalLabels: map[string]string{"remove": "me"},
			ExpectLabels:     map[string]string{"crd.name": "my-test-crd"},
			ExpectAnnotations: map[string]string{
				"removelabel.example.com/key1":                      "value1",
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "removelabel.example.com", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "removelabel.example.com", `[{"op": "remove", "path": "/metadata/labels/remove"}]`),
			},
		},
		// No need to test everything with the url case, since only the
		// connection is different.
		{
			Name: "match & reinvoke if needed policy",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
				ReinvocationPolicy:      &reinvokeIfNeeded,
			}, {
				Name:                    "removeLabel",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
				ReinvocationPolicy:      &reinvokeIfNeeded,
			}},
			AdditionalLabels:       map[string]string{"remove": "me"},
			ExpectAllow:            true,
			ExpectReinvokeWebhooks: map[string]bool{"addLabel": true},
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "addLabel", true),
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue(configurationName, "removeLabel", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "addLabel", `[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
				"patch.webhook.admission.k8s.io/round_0_index_1":    patchAnnotationValue(configurationName, "removeLabel", `[{"op": "remove", "path": "/metadata/labels/remove"}]`),
			},
		},
		{
			Name: "match & never reinvoke policy",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
				ReinvocationPolicy:      &reinvokeNever,
			}},
			ExpectAllow:            true,
			ExpectReinvokeWebhooks: map[string]bool{"addLabel": false},
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "addLabel", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "addLabel", `[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
			},
		},
		{
			Name: "match & never reinvoke policy (by default)",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:            true,
			ExpectReinvokeWebhooks: map[string]bool{"addLabel": false},
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "addLabel", true),
				"patch.webhook.admission.k8s.io/round_0_index_0":    patchAnnotationValue(configurationName, "addLabel", `[{"op": "add", "path": "/metadata/labels/added", "value": "test"}]`),
			},
		},
		{
			Name: "match & no reinvoke",
			Webhooks: []registrationv1.MutatingWebhook{{
				Name:                    "noop",
				ClientConfig:            ccfgSVC("noop"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
			ExpectAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue(configurationName, "noop", false),
			},
		},
	}
}

// CachedTest is a test case for the client manager.
type CachedTest struct {
	Name            string
	Webhooks        []registrationv1.ValidatingWebhook
	ExpectAllow     bool
	ExpectCacheMiss bool
}

// NewCachedClientTestcases returns a set of client manager test cases.
func NewCachedClientTestcases(url *url.URL) []CachedTest {
	policyIgnore := registrationv1.Ignore
	ccfgURL := urlConfigGenerator{url}.ccfgURL

	return []CachedTest{
		{
			Name: "uncached: service webhook, path 'allow'",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "cache1",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   newMatchEverythingRules(),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:     true,
			ExpectCacheMiss: true,
		},
		{
			Name: "uncached: service webhook, path 'internalErr'",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "cache2",
				ClientConfig:            ccfgSVC("internalErr"),
				Rules:                   newMatchEverythingRules(),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:     true,
			ExpectCacheMiss: true,
		},
		{
			Name: "cached: service webhook, path 'allow'",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "cache3",
				ClientConfig:            ccfgSVC("allow"),
				Rules:                   newMatchEverythingRules(),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:     true,
			ExpectCacheMiss: false,
		},
		{
			Name: "uncached: url webhook, path 'allow'",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "cache4",
				ClientConfig:            ccfgURL("allow"),
				Rules:                   newMatchEverythingRules(),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:     true,
			ExpectCacheMiss: true,
		},
		{
			Name: "cached: url webhook, path 'allow'",
			Webhooks: []registrationv1.ValidatingWebhook{{
				Name:                    "cache5",
				ClientConfig:            ccfgURL("allow"),
				Rules:                   newMatchEverythingRules(),
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				FailurePolicy:           &policyIgnore,
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:     true,
			ExpectCacheMiss: false,
		},
	}
}

// ccfgSVC returns a client config using the service reference mechanism.
func ccfgSVC(urlPath string) registrationv1.WebhookClientConfig {
	return registrationv1.WebhookClientConfig{
		Service: &registrationv1.ServiceReference{
			Name:      "webhook-test",
			Namespace: "default",
			Path:      &urlPath,
		},
		CABundle: testcerts.CACert,
	}
}

func newMatchEverythingRules() []registrationv1.RuleWithOperations {
	return []registrationv1.RuleWithOperations{{
		Operations: []registrationv1.OperationType{registrationv1.OperationAll},
		Rule: registrationv1.Rule{
			APIGroups:   []string{"*"},
			APIVersions: []string{"*"},
			Resources:   []string{"*/*"},
		},
	}}
}

// NewObjectInterfacesForTest returns an ObjectInterfaces appropriate for test cases in this file.
func NewObjectInterfacesForTest() admission.ObjectInterfaces {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	return admission.NewObjectInterfacesFromScheme(scheme)
}

// NewValidationDurationTestCases returns test cases for webhook duration test
func NewValidationDurationTestCases(url *url.URL) []DurationTest {
	ccfgURL := urlConfigGenerator{url}.ccfgURL
	webhooks := []registrationv1.ValidatingWebhook{
		{
			Name:                    "allow match",
			ClientConfig:            ccfgURL("allow/100"),
			Rules:                   matchEverythingRules,
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		{
			Name:                    "allow no match",
			ClientConfig:            ccfgURL("allow/200"),
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		{
			Name:                    "disallow match",
			ClientConfig:            ccfgURL("disallow/400"),
			Rules:                   matchEverythingRules,
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			AdmissionReviewVersions: []string{"v1beta1"},
		},
		{
			Name:                    "disallow no match",
			ClientConfig:            ccfgURL("disallow/800"),
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			AdmissionReviewVersions: []string{"v1beta1"},
		},
	}

	return []DurationTest{
		{
			Name:                "duration test",
			IsDryRun:            false,
			InitContext:         true,
			Webhooks:            webhooks,
			ExpectedDurationSum: 500,
			ExpectedDurationMax: 400,
		},
		{
			Name:                "duration dry run",
			IsDryRun:            true,
			InitContext:         true,
			Webhooks:            webhooks,
			ExpectedDurationSum: 0,
			ExpectedDurationMax: 0,
		},
		{
			Name:                "duration no init",
			IsDryRun:            false,
			InitContext:         false,
			Webhooks:            webhooks,
			ExpectedDurationSum: 0,
			ExpectedDurationMax: 0,
		},
	}
}
