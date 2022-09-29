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
			testName:      "override case",
			namespaceName: "test1",
			globalenv:     "TEST=a,FOO=foo",
			current:       []api.EnvVar{},
			admit:         true,
			expected: []api.EnvVar{
				{
					Name:  "TEST",
					Value: "a",
				},
				{
					Name:  "FOO",
					Value: "foo",
				},
			},
		},
		{
			testName:      "admit false",
			namespaceName: "test2",
			globalenv:     "",
			current:       []api.EnvVar{},
			admit:         true,
			expected:      []api.EnvVar{},
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

// func TestValidate(t *testing.T) {
// 	namespace := "test"
// 	handler := &AlwaysPullImages{}
// 	pod := api.Pod{
// 		ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: namespace},
// 		Spec: api.PodSpec{
// 			InitContainers: []api.Container{
// 				{Name: "init1", Image: "image"},
// 				{Name: "init2", Image: "image", ImagePullPolicy: api.PullNever},
// 				{Name: "init3", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
// 				{Name: "init4", Image: "image", ImagePullPolicy: api.PullAlways},
// 			},
// 			Containers: []api.Container{
// 				{Name: "ctr1", Image: "image"},
// 				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullNever},
// 				{Name: "ctr3", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
// 				{Name: "ctr4", Image: "image", ImagePullPolicy: api.PullAlways},
// 			},
// 		},
// 	}
// 	expectedError := `[` +
// 		`pods "123" is forbidden: spec.initContainers[0].imagePullPolicy: Unsupported value: "": supported values: "Always", ` +
// 		`pods "123" is forbidden: spec.initContainers[1].imagePullPolicy: Unsupported value: "Never": supported values: "Always", ` +
// 		`pods "123" is forbidden: spec.initContainers[2].imagePullPolicy: Unsupported value: "IfNotPresent": supported values: "Always", ` +
// 		`pods "123" is forbidden: spec.containers[0].imagePullPolicy: Unsupported value: "": supported values: "Always", ` +
// 		`pods "123" is forbidden: spec.containers[1].imagePullPolicy: Unsupported value: "Never": supported values: "Always", ` +
// 		`pods "123" is forbidden: spec.containers[2].imagePullPolicy: Unsupported value: "IfNotPresent": supported values: "Always"]`
// 	err := handler.Validate(context.TODO(), admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, &metav1.CreateOptions{}, false, nil), nil)
// 	if err == nil {
// 		t.Fatal("missing expected error")
// 	}
// 	if err.Error() != expectedError {
// 		t.Fatal(err)
// 	}
// }

// // TestOtherResources ensures that this admission controller is a no-op for other resources,
// // subresources, and non-pods.
// func TestOtherResources(t *testing.T) {
// 	namespace := "testnamespace"
// 	name := "testname"
// 	pod := &api.Pod{
// 		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
// 		Spec: api.PodSpec{
// 			Containers: []api.Container{
// 				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullNever},
// 			},
// 		},
// 	}
// 	tests := []struct {
// 		name        string
// 		kind        string
// 		resource    string
// 		subresource string
// 		object      runtime.Object
// 		expectError bool
// 	}{
// 		{
// 			name:     "non-pod resource",
// 			kind:     "Foo",
// 			resource: "foos",
// 			object:   pod,
// 		},
// 		{
// 			name:        "pod subresource",
// 			kind:        "Pod",
// 			resource:    "pods",
// 			subresource: "exec",
// 			object:      pod,
// 		},
// 		{
// 			name:        "non-pod object",
// 			kind:        "Pod",
// 			resource:    "pods",
// 			object:      &api.Service{},
// 			expectError: true,
// 		},
// 	}

// 	for _, tc := range tests {
// 		handler := admissiontesting.WithReinvocationTesting(t, &AlwaysPullImages{})

// 		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, &metav1.CreateOptions{}, false, nil), nil)

// 		if tc.expectError {
// 			if err == nil {
// 				t.Errorf("%s: unexpected nil error", tc.name)
// 			}
// 			continue
// 		}

// 		if err != nil {
// 			t.Errorf("%s: unexpected error: %v", tc.name, err)
// 			continue
// 		}

// 		if e, a := api.PullNever, pod.Spec.Containers[0].ImagePullPolicy; e != a {
// 			t.Errorf("%s: image pull policy was changed to %s", tc.name, a)
// 		}
// 	}

// }

// // TestUpdatePod ensures that this admission controller is a no-op for update pod if no
// // images were changed in the new pod spec.
// func TestUpdatePod(t *testing.T) {
// 	namespace := "testnamespace"
// 	name := "testname"
// 	oldPod := &api.Pod{
// 		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
// 		Spec: api.PodSpec{
// 			Containers: []api.Container{
// 				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
// 			},
// 		},
// 	}
// 	// only add new annotation
// 	pod := &api.Pod{
// 		ObjectMeta: metav1.ObjectMeta{
// 			Name:      name,
// 			Namespace: namespace,
// 			Annotations: map[string]string{
// 				"test": "test",
// 			},
// 		},
// 		Spec: api.PodSpec{
// 			Containers: []api.Container{
// 				{Name: "ctr2", Image: "image", ImagePullPolicy: api.PullIfNotPresent},
// 			},
// 		},
// 	}
// 	// add new label and change image
// 	podWithNewImage := &api.Pod{
// 		ObjectMeta: metav1.ObjectMeta{
// 			Name:      name,
// 			Namespace: namespace,
// 			Annotations: map[string]string{
// 				"test": "test",
// 			},
// 		},
// 		Spec: api.PodSpec{
// 			Containers: []api.Container{
// 				{Name: "ctr2", Image: "image2", ImagePullPolicy: api.PullIfNotPresent},
// 			},
// 		},
// 	}
// 	tests := []struct {
// 		name         string
// 		kind         string
// 		resource     string
// 		subresource  string
// 		object       runtime.Object
// 		oldObject    runtime.Object
// 		expectError  bool
// 		expectIgnore bool
// 	}{
// 		{
// 			name:         "update IfNotPresent pod annotations",
// 			kind:         "Pod",
// 			resource:     "pods",
// 			subresource:  "finalizers",
// 			object:       pod,
// 			oldObject:    oldPod,
// 			expectIgnore: true,
// 		},
// 		{
// 			name:        "update IfNotPresent pod image",
// 			kind:        "Pod",
// 			resource:    "pods",
// 			subresource: "finalizers",
// 			object:      podWithNewImage,
// 			oldObject:   oldPod,
// 		},
// 	}

// 	for _, tc := range tests {
// 		handler := admissiontesting.WithReinvocationTesting(t, &AlwaysPullImages{})

// 		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(tc.object, tc.oldObject, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, &metav1.UpdateOptions{}, false, nil), nil)

// 		if tc.expectError {
// 			if err == nil {
// 				t.Errorf("%s: unexpected nil error", tc.name)
// 			}
// 			continue
// 		}
// 		if tc.expectIgnore {
// 			if e, a := api.PullIfNotPresent, pod.Spec.Containers[0].ImagePullPolicy; e != a {
// 				t.Errorf("%s: image pull policy was changed to %s", tc.name, a)
// 			}
// 			continue
// 		}

// 		if err != nil {
// 			t.Errorf("%s: unexpected error: %v", tc.name, err)
// 			continue
// 		}
// 	}
// }

// TODO ignore order
func envSliceEquals(expected, current []api.EnvVar) bool {
	if len(expected) != len(current) {
		return false
	}
	for i := 0; i < len(expected); i++ {
		if expected[i] != current[i] {
			return false
		}
	}
	return true
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
				// {Name: "ctr3", Image: "image"},
				// {Name: "ctr4", Image: "image"},
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
