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
	"net/http"
	"net/url"
	"sync"

	registrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
)

var matchEverythingRules = []registrationv1beta1.RuleWithOperations{{
	Operations: []registrationv1beta1.OperationType{registrationv1beta1.OperationAll},
	Rule: registrationv1beta1.Rule{
		APIGroups:   []string{"*"},
		APIVersions: []string{"*"},
		Resources:   []string{"*/*"},
	},
}}

var sideEffectsUnknown = registrationv1beta1.SideEffectClassUnknown
var sideEffectsNone = registrationv1beta1.SideEffectClassNone
var sideEffectsSome = registrationv1beta1.SideEffectClassSome
var sideEffectsNoneOnDryRun = registrationv1beta1.SideEffectClassNoneOnDryRun

var reinvokeNever = registrationv1beta1.NeverReinvocationPolicy
var reinvokeIfNeeded = registrationv1beta1.IfNeededReinvocationPolicy

// NewFakeValidatingDataSource returns a mock client and informer returning the given webhooks.
func NewFakeValidatingDataSource(name string, webhooks []registrationv1beta1.ValidatingWebhook, stopCh <-chan struct{}) (clientset kubernetes.Interface, factory informers.SharedInformerFactory) {
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
	objs = append(objs, &registrationv1beta1.ValidatingWebhookConfiguration{
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
func NewFakeMutatingDataSource(name string, webhooks []registrationv1beta1.MutatingWebhook, stopCh <-chan struct{}) (clientset kubernetes.Interface, factory informers.SharedInformerFactory) {
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
	objs = append(objs, &registrationv1beta1.MutatingWebhookConfiguration{
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
func (f *FakeAttributes) GetAnnotations() map[string]string {
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

func (c urlConfigGenerator) ccfgURL(urlPath string) registrationv1beta1.WebhookClientConfig {
	u2 := *c.baseURL
	u2.Path = urlPath
	urlString := u2.String()
	return registrationv1beta1.WebhookClientConfig{
		URL:      &urlString,
		CABundle: testcerts.CACert,
	}
}

// ValidatingTest is a validating webhook test case.
type ValidatingTest struct {
	Name                   string
	Webhooks               []registrationv1beta1.ValidatingWebhook
	Path                   string
	IsCRD                  bool
	IsDryRun               bool
	AdditionalLabels       map[string]string
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
	Webhooks               []registrationv1beta1.MutatingWebhook
	Path                   string
	IsCRD                  bool
	IsDryRun               bool
	AdditionalLabels       map[string]string
	ExpectLabels           map[string]string
	ExpectAllow            bool
	ErrorContains          string
	ExpectAnnotations      map[string]string
	ExpectStatusCode       int32
	ExpectReinvokeWebhooks map[string]bool
}

// ConvertToMutatingTestCases converts a validating test case to a mutating one for test purposes.
func ConvertToMutatingTestCases(tests []ValidatingTest) []MutatingTest {
	r := make([]MutatingTest, len(tests))
	for i, t := range tests {
		r[i] = MutatingTest{t.Name, ConvertToMutatingWebhooks(t.Webhooks), t.Path, t.IsCRD, t.IsDryRun, t.AdditionalLabels, t.ExpectLabels, t.ExpectAllow, t.ErrorContains, t.ExpectAnnotations, t.ExpectStatusCode, t.ExpectReinvokeWebhooks}
	}
	return r
}

// ConvertToMutatingWebhooks converts a validating webhook to a mutating one for test purposes.
func ConvertToMutatingWebhooks(webhooks []registrationv1beta1.ValidatingWebhook) []registrationv1beta1.MutatingWebhook {
	mutating := make([]registrationv1beta1.MutatingWebhook, len(webhooks))
	for i, h := range webhooks {
		mutating[i] = registrationv1beta1.MutatingWebhook{h.Name, h.ClientConfig, h.Rules, h.FailurePolicy, h.MatchPolicy, h.NamespaceSelector, h.ObjectSelector, h.SideEffects, h.TimeoutSeconds, h.AdmissionReviewVersions, nil}
	}
	return mutating
}

// NewNonMutatingTestCases returns test cases with a given base url.
// All test cases in NewNonMutatingTestCases have no Patch set in
// AdmissionResponse. The test cases are used by both MutatingAdmissionWebhook
// and ValidatingAdmissionWebhook.
func NewNonMutatingTestCases(url *url.URL) []ValidatingTest {
	policyFail := registrationv1beta1.Fail
	policyIgnore := registrationv1beta1.Ignore
	ccfgURL := urlConfigGenerator{url}.ccfgURL

	return []ValidatingTest{
		{
			Name: "no match",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
				Name:         "nomatch",
				ClientConfig: ccfgSVC("disallow"),
				Rules: []registrationv1beta1.RuleWithOperations{{
					Operations: []registrationv1beta1.OperationType{registrationv1beta1.Create},
				}},
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
		{
			Name: "match & allow",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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

			ExpectAllow: true,
		},
		{
			Name: "match & fail (but disallow because fail close on nil FailurePolicy)",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
				Name:                    "nilResponse",
				ClientConfig:            ccfgURL("nilResponse"),
				FailurePolicy:           &policyIgnore,
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
		{
			Name: "absent response and fail closed",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
				Name:         "nomatch",
				ClientConfig: ccfgSVC("allow"),
				Rules: []registrationv1beta1.RuleWithOperations{{
					Operations: []registrationv1beta1.OperationType{registrationv1beta1.Create},
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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

// NewMutatingTestCases returns test cases with a given base url.
// All test cases in NewMutatingTestCases have Patch set in
// AdmissionResponse. The test cases are only used by both MutatingAdmissionWebhook.
func NewMutatingTestCases(url *url.URL) []MutatingTest {
	return []MutatingTest{
		{
			Name: "match & remove label",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "removelabel.example.com",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:       true,
			AdditionalLabels:  map[string]string{"remove": "me"},
			ExpectLabels:      map[string]string{"pod.name": "my-pod"},
			ExpectAnnotations: map[string]string{"removelabel.example.com/key1": "value1"},
		},
		{
			Name: "match & add label",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:  true,
			ExpectLabels: map[string]string{"pod.name": "my-pod", "added": "test"},
		},
		{
			Name: "match CRD & add label",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
		},
		{
			Name: "match CRD & remove label",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "removelabel.example.com",
				ClientConfig:            ccfgSVC("removeLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			IsCRD:             true,
			ExpectAllow:       true,
			AdditionalLabels:  map[string]string{"remove": "me"},
			ExpectLabels:      map[string]string{"crd.name": "my-test-crd"},
			ExpectAnnotations: map[string]string{"removelabel.example.com/key1": "value1"},
		},
		{
			Name: "match & invalid mutation",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "invalidMutation",
				ClientConfig:            ccfgSVC("invalidMutation"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectStatusCode: http.StatusInternalServerError,
			ErrorContains:    "invalid character",
		},
		{
			Name: "match & remove label dry run unsupported",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
		},
		{
			Name: "first webhook remove labels, second webhook shouldn't be called",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
			ExpectAllow:       true,
			AdditionalLabels:  map[string]string{"remove": "me"},
			ExpectLabels:      map[string]string{"pod.name": "my-pod"},
			ExpectAnnotations: map[string]string{"removelabel.example.com/key1": "value1"},
		},
		{
			Name: "first webhook remove labels from CRD, second webhook shouldn't be called",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
			IsCRD:             true,
			ExpectAllow:       true,
			AdditionalLabels:  map[string]string{"remove": "me"},
			ExpectLabels:      map[string]string{"crd.name": "my-test-crd"},
			ExpectAnnotations: map[string]string{"removelabel.example.com/key1": "value1"},
		},
		// No need to test everything with the url case, since only the
		// connection is different.
		{
			Name: "match & reinvoke if needed policy",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
		},
		{
			Name: "match & never reinvoke policy",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
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
		},
		{
			Name: "match & never reinvoke policy (by default)",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "addLabel",
				ClientConfig:            ccfgSVC("addLabel"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow:            true,
			ExpectReinvokeWebhooks: map[string]bool{"addLabel": false},
		},
		{
			Name: "match & no reinvoke",
			Webhooks: []registrationv1beta1.MutatingWebhook{{
				Name:                    "noop",
				ClientConfig:            ccfgSVC("noop"),
				Rules:                   matchEverythingRules,
				NamespaceSelector:       &metav1.LabelSelector{},
				ObjectSelector:          &metav1.LabelSelector{},
				AdmissionReviewVersions: []string{"v1beta1"},
			}},
			ExpectAllow: true,
		},
	}
}

// CachedTest is a test case for the client manager.
type CachedTest struct {
	Name            string
	Webhooks        []registrationv1beta1.ValidatingWebhook
	ExpectAllow     bool
	ExpectCacheMiss bool
}

// NewCachedClientTestcases returns a set of client manager test cases.
func NewCachedClientTestcases(url *url.URL) []CachedTest {
	policyIgnore := registrationv1beta1.Ignore
	ccfgURL := urlConfigGenerator{url}.ccfgURL

	return []CachedTest{
		{
			Name: "uncached: service webhook, path 'allow'",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
			Name: "cached: service webhook, path 'allow'",
			Webhooks: []registrationv1beta1.ValidatingWebhook{{
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
func ccfgSVC(urlPath string) registrationv1beta1.WebhookClientConfig {
	return registrationv1beta1.WebhookClientConfig{
		Service: &registrationv1beta1.ServiceReference{
			Name:      "webhook-test",
			Namespace: "default",
			Path:      &urlPath,
		},
		CABundle: testcerts.CACert,
	}
}

func newMatchEverythingRules() []registrationv1beta1.RuleWithOperations {
	return []registrationv1beta1.RuleWithOperations{{
		Operations: []registrationv1beta1.OperationType{registrationv1beta1.OperationAll},
		Rule: registrationv1beta1.Rule{
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
