// +build integration,!no-etcd

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPodUpdateActiveDeadlineSeconds(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("pod-activedeadline-update", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	var (
		iZero = int64(0)
		i30   = int64(30)
		i60   = int64(60)
		iNeg  = int64(-1)
	)

	prototypePod := func() *api.Pod {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "xxx",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
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

		if _, err := client.Pods(ns.Name).Create(pod); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		pod.Spec.ActiveDeadlineSeconds = tc.update

		_, err := client.Pods(ns.Name).Update(pod)
		if tc.valid && err != nil {
			t.Errorf("%v: failed to update pod: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to pod", tc.name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestPodReadOnlyFilesystem(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	isReadOnly := true
	ns := framework.CreateTestingNamespace("pod-readonly-root", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "xxx",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					SecurityContext: &api.SecurityContext{
						ReadOnlyRootFilesystem: &isReadOnly,
					},
				},
			},
		},
	}

	if _, err := client.Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}

	integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
}
