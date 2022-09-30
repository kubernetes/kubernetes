/*
Copyright 2022 The Kubernetes Authors.

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

package globalenv

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestAdmission(t *testing.T) {
	tests := []struct {
		globalenv     string
		namespaceName string
		current       []api.EnvVar
		expected      []api.EnvVar
		admit         bool
		testName      string
	}{
		{
			testName:      "single global env case",
			namespaceName: "test1",
			globalenv:     "TEST_A=a",
			current:       []api.EnvVar{},
			admit:         true,
			expected: []api.EnvVar{
				{Name: "TEST_A", Value: "a"},
			},
		},
		{
			testName:      "multi global env case",
			namespaceName: "test1",
			globalenv:     "TEST-A=a,FOO=foo",
			current:       []api.EnvVar{},
			admit:         true,
			expected: []api.EnvVar{
				{Name: "TEST_A", Value: "a"},
				{Name: "FOO", Value: "foo"},
			},
		},
		{
			testName:      "empty global env value case",
			namespaceName: "test1",
			globalenv:     "TEST=a,FOO=,TEST2=b",
			current:       []api.EnvVar{},
			admit:         true,
			expected: []api.EnvVar{
				{Name: "TEST", Value: "a"},
				{Name: "FOO", Value: ""},
				{Name: "TEST2", Value: "b"},
			},
		},
		{
			testName:      "same env in pod",
			namespaceName: "test1",
			globalenv:     "TEST=a,FOO=foo",
			current:       []api.EnvVar{{Name: "TEST", Value: "a"}},
			admit:         true,
			expected: []api.EnvVar{
				{Name: "TEST", Value: "a"},
				{Name: "FOO", Value: "foo"},
			},
		},
		{
			testName:      "override case",
			namespaceName: "test1",
			globalenv:     "TEST=a,FOO=foo",
			current:       []api.EnvVar{{Name: "TEST", Value: "b"}},
			admit:         true,
			expected: []api.EnvVar{
				{Name: "TEST", Value: "a"},
				{Name: "FOO", Value: "foo"},
			},
		},
		{
			testName:      "no global env, no env",
			namespaceName: "test2",
			globalenv:     "",
			current:       []api.EnvVar{},
			admit:         true,
			expected:      []api.EnvVar{},
		},
		{
			testName:      "no global env, has envs",
			namespaceName: "test2",
			globalenv:     "",
			current: []api.EnvVar{
				{Name: "TEST", Value: "a"},
				{Name: "FOO", Value: "foo"},
			},
			admit: true,
			expected: []api.EnvVar{
				{Name: "TEST", Value: "a"},
				{Name: "FOO", Value: "foo"},
			},
		},
	}
	for _, test := range tests {
		namespace := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:        test.namespaceName,
				Namespace:   "",
				Annotations: map[string]string{},
			},
		}
		if len(test.globalenv) > 0 {
			namespace.Annotations = map[string]string{globalEnvKey: string(test.globalenv)}
		}
		mockClient := fake.NewSimpleClientset(namespace)
		handler, informerFactory, err := newHandlerForTest(mockClient)
		if err != nil {
			t.Fatalf("unexpected error initializing handler: %v", err)
		}
		stopCh := make(chan struct{})
		defer close(stopCh)
		informerFactory.Start(stopCh)

		pod := renderPod(test.namespaceName, test.current)
		err = admissiontesting.WithReinvocationTesting(t, handler).Admit(
			context.TODO(),
			admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), test.namespaceName, namespace.ObjectMeta.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil),
			nil,
		)
		if test.admit && err != nil {
			t.Errorf("Test: %s, expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("Test: %s, expected an error", test.testName)
		}

		for _, c := range pod.Spec.InitContainers {
			if !envSliceEquals(test.expected, c.Env) {
				t.Errorf("Test: %s, Container %v: expected env var  %v, got %v", test.testName, c, test.expected, c.Env)
			}
		}
		for _, c := range pod.Spec.Containers {
			if !envSliceEquals(test.expected, c.Env) {
				t.Errorf("Test: %s, Container %v: expected env var  %v, got %v", test.testName, c, test.expected, c.Env)
			}
		}
	}
}

func TestValidate(t *testing.T) {
	namespaceName := "test3"
	globalenv := "TEST=b,FOO=foo"
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        namespaceName,
			Annotations: map[string]string{globalEnvKey: string(globalenv)},
		},
	}
	mockClient := fake.NewSimpleClientset(namespace)
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Fatalf("unexpected error initializing handler: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	pod := renderPod(namespaceName, []api.EnvVar{{Name: "TEST", Value: "a"}})
	expectedError := `[` +
		`spec.initContainers[0].env: Forbidden: global env should not override current env, ` +
		`spec.containers[0].env: Forbidden: global env should not override current env, ` +
		`spec.containers[1].env: Forbidden: global env should not override current env` +
		`]`

	err = handler.Validate(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
	if err == nil {
		t.Fatal("missing expected error")
	}
	if err.Error() != expectedError {
		t.Fatal(err)
	}
}

// TestOtherResources ensures that this admission controller is a no-op for other resources,
// subresources, and non-pods.
func TestOtherResources(t *testing.T) {
	namespaceName := "test4"
	globalenv := "TEST=b,FOO=foo"
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        namespaceName,
			Annotations: map[string]string{globalEnvKey: string(globalenv)},
		},
	}
	mockClient := fake.NewSimpleClientset(namespace)
	globalEnvHandler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Fatalf("unexpected error initializing handler: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	pod := renderPod(namespaceName, []api.EnvVar{{Name: "TEST", Value: "a"}})

	tests := []struct {
		name        string
		kind        string
		resource    string
		subresource string
		object      runtime.Object
		expectError bool
	}{
		{
			name:     "non-pod resource",
			kind:     "Foo",
			resource: "foos",
			object:   &pod,
		},
		{
			name:        "pod subresource",
			kind:        "Pod",
			resource:    "pods",
			subresource: "exec",
			object:      &pod,
		},
		{
			name:        "non-pod object",
			kind:        "Pod",
			resource:    "pods",
			object:      &api.Service{},
			expectError: true,
		},
	}

	for _, tc := range tests {
		handler := admissiontesting.WithReinvocationTesting(t, globalEnvHandler)

		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespaceName, tc.name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, &metav1.CreateOptions{}, false, nil), nil)

		if tc.expectError {
			if err == nil {
				t.Errorf("%s: unexpected nil error", tc.name)
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
	}

}

// TestUpdatePod ensures that this admission controller is a no-op for update pod if no change.
func TestUpdatePod(t *testing.T) {
	namespaceName := "test4"
	globalenv := "TEST=b,FOO=foo"
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        namespaceName,
			Annotations: map[string]string{globalEnvKey: string(globalenv)},
		},
	}
	mockClient := fake.NewSimpleClientset(namespace)
	globalEnvHandler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Fatalf("unexpected error initializing handler: %v", err)
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)

	oldPod := renderPod(namespaceName, []api.EnvVar{{Name: "TEST", Value: "c"}})
	pod := renderPod(namespaceName, []api.EnvVar{{Name: "TEST", Value: "c"}})
	podWithNoEnv := renderPod(namespaceName, []api.EnvVar{})

	tests := []struct {
		name         string
		kind         string
		resource     string
		subresource  string
		object       runtime.Object
		oldObject    runtime.Object
		expectError  bool
		expectIgnore bool
	}{
		{
			name:         "update pod env",
			kind:         "Pod",
			resource:     "pods",
			object:       &pod,
			oldObject:    &oldPod,
			expectIgnore: true,
		},
		{
			name:      "remove pod env",
			kind:      "Pod",
			resource:  "pods",
			object:    &podWithNoEnv,
			oldObject: &oldPod,
		},
	}

	for _, tc := range tests {
		handler := admissiontesting.WithReinvocationTesting(t, globalEnvHandler)

		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(tc.object, tc.oldObject, api.Kind(tc.kind).WithVersion("version"), namespaceName, tc.name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, &metav1.UpdateOptions{}, false, nil), nil)

		if tc.expectError {
			if err == nil {
				t.Errorf("%s: unexpected nil error", tc.name)
			}
			continue
		}
		if tc.expectIgnore {
			expected := []api.EnvVar{{Name: "TEST", Value: "b"}, {Name: "FOO", Value: "foo"}}
			for _, c := range pod.Spec.InitContainers {
				if !envSliceEquals(expected, c.Env) {
					t.Errorf("Test: %s, Container %v: expected env var  %v, got %v", tc.name, c, expected, c.Env)
				}
			}
			for _, c := range pod.Spec.Containers {
				if !envSliceEquals(expected, c.Env) {
					t.Errorf("Test: %s, Container %v: expected env var  %v, got %v", tc.name, c, expected, c.Env)
				}
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}
	}
}

func renderPod(namespace string, env []api.EnvVar) api.Pod {
	pod := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{Name: "init1", Image: "image", Env: env},
			},
			Containers: []api.Container{
				{Name: "ctr1", Image: "image", Env: env},
				{Name: "ctr2", Image: "image", Env: env},
			},
		},
	}

	return pod
}

// newHandlerForTest returns the admission controller configured for testing.
func newHandlerForTest(c kubernetes.Interface) (*GlobalEnv, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(c, 5*time.Minute)
	handler := NewGlobalEnv()
	pluginInitializer := genericadmissioninitializer.New(c, f, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	err := admission.ValidateInitialization(handler)
	handler.SetExternalKubeInformerFactory(f)
	handler.SetExternalKubeClientSet(c)
	return handler, f, err
}
