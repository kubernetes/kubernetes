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

package podpresetrestriction

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/podpresetrestriction/apis/podpresetrestriction"
)

// TestPresetAdmission verifies various scenarios involving preset selector modifications
func TestPresetAdmission(t *testing.T) {
	mockClient := &fake.Clientset{}
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	tests := []struct {
		defaultSelector metav1.LabelSelector
		namespace       api.Namespace
		result          metav1.LabelSelector
		admit           bool
		testName        string
	}{
		{
			defaultSelector: metav1.LabelSelector{MatchLabels: map[string]string{"component": "redis"}},
			namespace: api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "testNamespace1",
				},
			},
			result:   metav1.LabelSelector{MatchLabels: map[string]string{"component": "redis"}},
			admit:    true,
			testName: "label selector from config, no namespace annotation",
		},
		{
			defaultSelector: metav1.LabelSelector{MatchLabels: map[string]string{"component": "redis"}},
			namespace: api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "testNamespace2",
					Annotations: map[string]string{
						annotationDefaultSelector: `{"matchExpressions": [{"key": "openshift.io/build.name", "operator": "DoesNotExist" }]}`,
					},
				},
			},
			result: metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "openshift.io/build.name",
					Operator: metav1.LabelSelectorOpDoesNotExist,
				},
			},
			},
			admit:    true,
			testName: "label selector from config, with namespace annotation",
		},
		{
			defaultSelector: metav1.LabelSelector{},
			namespace: api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "testNamespace3",
				},
			},
			result:   metav1.LabelSelector{},
			admit:    true,
			testName: "no label selector from config, no namespace annotations",
		},
		{
			defaultSelector: metav1.LabelSelector{},
			namespace: api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "testNamespace4",
					Annotations: map[string]string{
						annotationDefaultSelector: `{"matchExpressions": [{"key": "openshift.io/build.name", "operator": "DoesNotExist" }]}`,
					},
				},
			},
			result: metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      "openshift.io/build.name",
					Operator: metav1.LabelSelectorOpDoesNotExist,
				},
			},
			},
			admit:    true,
			testName: "no label selector from config, with namespace annotation",
		},
	}
	for i, test := range tests {
		pp := settings.PodPreset{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pp",
				Namespace: test.namespace.ObjectMeta.Name,
			},
		}

		informerFactory.Core().InternalVersion().Namespaces().Informer().GetStore().Update(&test.namespace)
		informerFactory.Settings().InternalVersion().PodPresets().Informer().GetStore().Add(&pp)
		handler.pluginConfig = &pluginapi.Configuration{DefaultSelector: test.defaultSelector}
		mockClient.AddReactor("get", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
			return true, &test.namespace, nil
		})

		err = handler.Admit(admission.NewAttributesRecord(&pp, nil, settings.Kind("PodPreset").WithVersion("version"), pp.Namespace, test.namespace.ObjectMeta.Name, settings.Resource("podpresets").WithVersion("version"), "", admission.Create, nil))
		if test.admit && err != nil {
			t.Errorf("Test Admit: %s, expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("Test Admit: %s, expected an error", test.testName)
		}

		if !reflect.DeepEqual(test.result, pp.Spec.Selector) {
			t.Errorf("(%d:%s):\nExpected %#v, got %#v", i, test.testName, test.result, pp.Spec.Selector)
		}
		close(stopCh)
	}
}

func TestHandles(t *testing.T) {
	for op, shouldHandle := range map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  true,
		admission.Connect: false,
		admission.Delete:  false,
	} {

		pluginConfig, err := loadConfiguration(nil)
		// must not fail
		if err != nil {
			t.Errorf("%v: error reading default configuration", op)
		}
		prPlugin := NewPodPresetRestrictionPlugin(pluginConfig)
		if e, a := shouldHandle, prPlugin.Handles(op); e != a {
			t.Errorf("%v: shouldHandle=%t, handles=%t", op, e, a)
		}
	}
}

// newHandlerForTest returns the admission controller configured for testing.
func newHandlerForTest(c clientset.Interface) (*podPresetRestrictionPlugin, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(c, 5*time.Minute)
	pluginConfig, err := loadConfiguration(nil)
	// must not fail
	if err != nil {
		return nil, nil, err
	}
	handler := NewPodPresetRestrictionPlugin(pluginConfig)
	pluginInitializer := kubeadmission.NewPluginInitializer(c, f, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	err = admission.Validate(handler)
	return handler, f, err
}
