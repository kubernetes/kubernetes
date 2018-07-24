/*
Copyright 2015 The Kubernetes Authors.

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

package pods

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPodUpdateActiveDeadlineSeconds(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	ns := framework.CreateTestingNamespace("pod-activedeadline-update", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

	var (
		iZero = int64(0)
		i30   = int64(30)
		i60   = int64(60)
		iNeg  = int64(-1)
	)

	prototypePod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "xxx",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	cases := []struct {
		name     string
		original *int64
		update   *int64
		valid    bool
	}{
		{
			name:     "no change, nil",
			original: nil,
			update:   nil,
			valid:    true,
		},
		{
			name:     "no change, set",
			original: &i30,
			update:   &i30,
			valid:    true,
		},
		{
			name:     "change to positive from nil",
			original: nil,
			update:   &i60,
			valid:    true,
		},
		{
			name:     "change to smaller positive",
			original: &i60,
			update:   &i30,
			valid:    true,
		},
		{
			name:     "change to larger positive",
			original: &i30,
			update:   &i60,
			valid:    false,
		},
		{
			name:     "change to negative from positive",
			original: &i30,
			update:   &iNeg,
			valid:    false,
		},
		{
			name:     "change to negative from nil",
			original: nil,
			update:   &iNeg,
			valid:    false,
		},
		// zero is not allowed, must be a positive integer
		{
			name:     "change to zero from positive",
			original: &i30,
			update:   &iZero,
			valid:    false,
		},
		{
			name:     "change to nil from positive",
			original: &i30,
			update:   nil,
			valid:    false,
		},
	}

	for i, tc := range cases {
		pod := prototypePod()
		pod.Spec.ActiveDeadlineSeconds = tc.original
		pod.ObjectMeta.Name = fmt.Sprintf("activedeadlineseconds-test-%v", i)

		if _, err := client.Core().Pods(ns.Name).Create(pod); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		pod.Spec.ActiveDeadlineSeconds = tc.update

		_, err := client.Core().Pods(ns.Name).Update(pod)
		if tc.valid && err != nil {
			t.Errorf("%v: failed to update pod: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to pod", tc.name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestPodReadOnlyFilesystem(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	isReadOnly := true
	ns := framework.CreateTestingNamespace("pod-readonly-root", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "xxx",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					SecurityContext: &v1.SecurityContext{
						ReadOnlyRootFilesystem: &isReadOnly,
					},
				},
			},
		},
	}

	if _, err := client.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}

	integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
}
