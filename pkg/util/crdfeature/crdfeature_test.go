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
package crdfeature

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	crdv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	ktesting "k8s.io/client-go/testing"
)

func toUnstructured(t *testing.T, obj interface{}) *unstructured.Unstructured {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		t.Fatalf("Failed to create unstructured: %v", err)
	}
	return &unstructured.Unstructured{
		Object: u,
	}
}

var testCRDAlphaGVR = schema.GroupVersionResource{
	Group:    "test.k8s.io",
	Version:  "v1alpha1",
	Resource: "tests",
}

var testCRDAlpha = &crdv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{
		Name: "tests.test.k8s.io",
	},
	Spec: crdv1.CustomResourceDefinitionSpec{
		Versions: []crdv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1alpha1",
				Served:  true,
				Storage: true,
			},
		},
	},
}

var otherTestCRDAlpha = &crdv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{
		Name: "othertests.test.k8s.io",
	},
	Spec: crdv1.CustomResourceDefinitionSpec{
		Versions: []crdv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1alpha1",
				Served:  true,
				Storage: true,
			},
		},
	},
}

var testCRDBeta = &crdv1.CustomResourceDefinition{
	ObjectMeta: metav1.ObjectMeta{
		Name: "tests.test.k8s.io",
	},
	Spec: crdv1.CustomResourceDefinitionSpec{
		Versions: []crdv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
			},
			{
				Name:    "v1alpha1",
				Served:  false,
				Storage: false,
			},
		},
	},
}

func TestToCRD(t *testing.T) {
	tests := []struct {
		name        string
		input       interface{}
		expectedCRD *crdv1.CustomResourceDefinition
		expectError bool
	}{
		{
			name:        "not unstructured",
			input:       &crdv1.CustomResourceDefinition{},
			expectError: true,
		},
		{
			name:        "valid crd",
			input:       toUnstructured(t, testCRDAlpha),
			expectedCRD: testCRDAlpha,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			crd, err := toCRD(tc.input)
			if tc.expectError {
				if err == nil {
					t.Errorf("toCRD() returned unexpected result. Expected error, got %#v", crd)
				}
				return
			}
			if err != nil {
				t.Fatalf("toCRD() returned unexpected error, got %v", err)
			}
			if diff := cmp.Diff(tc.expectedCRD, crd); diff != "" {
				t.Errorf("toCRD() returned unexpected result, got (+/1):\n%s", diff)
			}
		})
	}
}

func waitForClose(t *testing.T, c <-chan struct{}, eventName string) {
	select {
	case <-c:
		return
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Timed out before %s", eventName)
		return
	}
}

func TestCRDFeatureEnableDisable(t *testing.T) {
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	crdv1.AddToScheme(scheme)
	client := dynamicfake.NewSimpleDynamicClient(scheme)
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("customresourcedefinitions", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)
	enabled := make(chan struct{})
	disabled := make(chan struct{})

	featureWatcher := NewWatcher(client, testCRDAlphaGVR, time.Minute, WithFuncs(func() { close(enabled) }, func() { close(disabled) }))
	go featureWatcher.Start(stopCh)

	fakeWatch.Add(toUnstructured(t, testCRDAlpha))
	waitForClose(t, enabled, "feature enabled")
	fakeWatch.Delete(toUnstructured(t, testCRDAlpha))
	waitForClose(t, enabled, "feature disabled")
}

func TestCRDFeatureChanEnableDisable(t *testing.T) {
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	crdv1.AddToScheme(scheme)
	client := dynamicfake.NewSimpleDynamicClient(scheme)
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("customresourcedefinitions", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)
	enabled := make(chan struct{})
	holder := struct {
		ch <-chan struct{}
	}{}
	feature := WithChan(func(ch <-chan struct{}) {
		holder.ch = ch
		close(enabled)
	})
	featureWatcher := NewWatcher(client, testCRDAlphaGVR, time.Minute, feature)
	go featureWatcher.Start(stopCh)

	fakeWatch.Add(toUnstructured(t, testCRDAlpha))
	waitForClose(t, enabled, "feature enabled")
	if holder.ch == nil {
		t.Fatal("Feature chan not set")
	}
	fakeWatch.Delete(toUnstructured(t, testCRDAlpha))
	waitForClose(t, holder.ch, "feature disabled")
}

func TestCRDFeatureUpdateToInvalid(t *testing.T) {
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	crdv1.AddToScheme(scheme)
	client := dynamicfake.NewSimpleDynamicClient(scheme)
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("customresourcedefinitions", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)
	enabled := make(chan struct{})
	disabled := make(chan struct{})

	featureWatcher := NewWatcher(client, testCRDAlphaGVR, time.Minute, WithFuncs(func() { close(enabled) }, func() { close(disabled) }))
	go featureWatcher.Start(stopCh)

	fakeWatch.Add(toUnstructured(t, testCRDAlpha))
	waitForClose(t, enabled, "feature enabled")
	fakeWatch.Modify(toUnstructured(t, testCRDBeta))
	waitForClose(t, enabled, "feature disabled")
}

func TestCRDFeatureUpdateToValid(t *testing.T) {
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	crdv1.AddToScheme(scheme)
	client := dynamicfake.NewSimpleDynamicClient(scheme)
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("customresourcedefinitions", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)
	enabled := make(chan struct{})
	disabled := make(chan struct{})

	featureWatcher := NewWatcher(client, testCRDAlphaGVR, time.Minute, WithFuncs(func() { close(enabled) }, func() { close(disabled) }))
	go featureWatcher.Start(stopCh)

	fakeWatch.Add(toUnstructured(t, testCRDBeta))
	fakeWatch.Modify(toUnstructured(t, testCRDAlpha))
	waitForClose(t, enabled, "feature enabled")
}

func TestCRDMatchesGVR(t *testing.T) {
	featureWatcher := Watcher{gvr: testCRDAlphaGVR}
	if !featureWatcher.crdMatchesGVR(testCRDAlpha) {
		t.Errorf("Expected match for GVR %v: %#v", featureWatcher.gvr, testCRDAlpha)
	}
	if featureWatcher.crdMatchesGVR(testCRDBeta) {
		t.Errorf("Expected no match for GVR %v: %#v", featureWatcher.gvr, testCRDBeta)
	}
	if featureWatcher.crdMatchesGVR(otherTestCRDAlpha) {
		t.Errorf("Expected no match for GVR %v: %#v", featureWatcher.gvr, otherTestCRDAlpha)
	}
}
