/*
Copyright 2014 Google Inc. All rights reserved.

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

package config

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

func TestURLErrorNotExistNoUpdate(t *testing.T) {
	ch := make(chan interface{})
	NewSourceURL("http://localhost:49575/_not_found_", "localhost", time.Millisecond, ch)
	select {
	case got := <-ch:
		t.Errorf("Expected no update, Got %#v", got)
	case <-time.After(2 * time.Millisecond):
	}
}

func TestExtractFromHttpBadness(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := sourceURL{"http://localhost:49575/_not_found_", "other", ch, nil}
	if err := c.extractFromURL(); err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractInvalidManifest(t *testing.T) {
	var testCases = []struct {
		desc      string
		manifests interface{}
	}{
		{
			desc:      "No version",
			manifests: []api.ContainerManifest{{Version: ""}},
		},
		{
			desc:      "Invalid version",
			manifests: []api.ContainerManifest{{Version: "v1betta2"}},
		},
		{
			desc: "Invalid volume name",
			manifests: []api.ContainerManifest{
				{Version: "v1beta1", Volumes: []api.Volume{{Name: "_INVALID_"}}},
			},
		},
		{
			desc: "Duplicate volume names",
			manifests: []api.ContainerManifest{
				{
					Version: "v1beta1",
					Volumes: []api.Volume{{Name: "repeated"}, {Name: "repeated"}},
				},
			},
		},
		{
			desc: "Unspecified container name",
			manifests: []api.ContainerManifest{
				{
					Version:    "v1beta1",
					Containers: []api.Container{{Name: ""}},
				},
			},
		},
		{
			desc: "Invalid container name",
			manifests: []api.ContainerManifest{
				{
					Version:    "v1beta1",
					Containers: []api.Container{{Name: "_INVALID_"}},
				},
			},
		},
	}
	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.manifests)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		defer testServer.Close()
		ch := make(chan interface{}, 1)
		c := sourceURL{testServer.URL, "localhost", ch, nil}
		if err := c.extractFromURL(); err == nil {
			t.Errorf("%s: Expected error", testCase.desc)
		}
	}
}

func TestExtractManifestFromHTTP(t *testing.T) {
	hostname := "random-hostname"

	var testCases = []struct {
		desc      string
		manifests interface{}
		expected  kubelet.PodUpdate
	}{
		{
			desc: "Single manifest",
			manifests: v1beta1.ContainerManifest{Version: "v1beta1", ID: "foo", UUID: "111",
				Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta1.PullAlways}}},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				}),
		},
		{
			desc: "Single manifest without ID",
			manifests: v1beta1.ContainerManifest{Version: "v1beta1", UUID: "111",
				Containers: []v1beta1.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}}},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "111" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta2/pods/111-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "ctr",
							Image: "image",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "IfNotPresent"}},
					},
				}),
		},
		{
			desc: "Single manifest with v1beta2",
			manifests: v1beta1.ContainerManifest{Version: "v1beta2", ID: "foo", UUID: "111",
				Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta1.PullAlways}}},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				}),
		},
		{
			desc: "Multiple manifests",
			manifests: []v1beta1.ContainerManifest{
				{Version: "v1beta1", ID: "foo", UUID: "111",
					Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta1.PullAlways}}},
				{Version: "v1beta1", ID: "bar", UUID: "222",
					Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: ""}}},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				},
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "222",
						Name:      "bar" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta2/pods/bar-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "IfNotPresent"}},
					},
				}),
		},
		{
			desc:      "Empty Array",
			manifests: []v1beta1.ContainerManifest{},
			expected:  CreatePodUpdate(kubelet.SET, kubelet.HTTPSource),
		},
	}

	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.manifests)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		defer testServer.Close()
		ch := make(chan interface{}, 1)
		c := sourceURL{testServer.URL, hostname, ch, nil}
		if err := c.extractFromURL(); err != nil {
			t.Errorf("%s: Unexpected error: %v", testCase.desc, err)
			continue
		}
		update := (<-ch).(kubelet.PodUpdate)

		for i := range update.Pods {
			// There's no way to provide namespace in ContainerManifest, so
			// it will be defaulted.
			if update.Pods[i].Namespace != kubelet.NamespaceDefault {
				t.Errorf("Unexpected namespace: %s", update.Pods[0].Namespace)
			}
			update.Pods[i].ObjectMeta.Namespace = "foobar"
		}
		if !api.Semantic.DeepEqual(testCase.expected, update) {
			t.Errorf("%s: Expected: %#v, Got: %#v", testCase.desc, testCase.expected, update)
		}
		for _, pod := range update.Pods {
			if errs := validation.ValidatePod(pod); len(errs) != 0 {
				t.Errorf("%s: Expected no validation errors on %#v, Got %v", testCase.desc, pod, errors.NewAggregate(errs))
			}
		}
	}
}

func TestExtractPodsFromHTTP(t *testing.T) {
	hostname := "different-value"

	var testCases = []struct {
		desc     string
		pods     interface{}
		expected kubelet.PodUpdate
	}{
		{
			desc: "Single pod v1beta1",
			pods: v1beta1.Pod{
				TypeMeta: v1beta1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1beta1",
					ID:         "foo",
					UID:        "111",
					Namespace:  "mynamespace",
				},
				DesiredState: v1beta1.PodState{
					Manifest: v1beta1.ContainerManifest{
						Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta1.PullAlways}},
					},
				},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "mynamespace",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=mynamespace",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				}),
		},
		{
			desc: "Single pod v1beta3",
			pods: v1beta3.Pod{
				TypeMeta: v1beta3.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1beta3",
				},
				ObjectMeta: v1beta3.ObjectMeta{
					Name:      "foo",
					UID:       "111",
					Namespace: "mynamespace",
				},
				Spec: v1beta3.PodSpec{
					Host:       hostname,
					Containers: []v1beta3.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta3.PullAlways}},
				},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "mynamespace",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=mynamespace",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				}),
		},
		{
			desc: "Multiple pods",
			pods: v1beta3.PodList{
				TypeMeta: v1beta3.TypeMeta{
					Kind:       "PodList",
					APIVersion: "v1beta3",
				},
				Items: []v1beta3.Pod{
					{
						ObjectMeta: v1beta3.ObjectMeta{
							Name: "foo",
							UID:  "111",
						},
						Spec: v1beta3.PodSpec{
							Host:       hostname,
							Containers: []v1beta3.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta3.PullAlways}},
						},
					},
					{
						ObjectMeta: v1beta3.ObjectMeta{
							Name: "bar",
							UID:  "222",
						},
						Spec: v1beta3.PodSpec{
							Host:       hostname,
							Containers: []v1beta3.Container{{Name: "2", Image: "bar", ImagePullPolicy: ""}},
						},
					},
				},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "default",
						SelfLink:  "/api/v1beta2/pods/foo-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				},
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "222",
						Name:      "bar" + "-" + hostname,
						Namespace: "default",
						SelfLink:  "/api/v1beta2/pods/bar-" + hostname + "?namespace=default",
					},
					Spec: api.PodSpec{
						Host:          hostname,
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "2",
							Image: "bar",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "IfNotPresent"}},
					},
				}),
		},
		{
			desc:     "Empty Array",
			pods:     []v1beta3.Pod{},
			expected: CreatePodUpdate(kubelet.SET, kubelet.HTTPSource),
		},
	}

	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.pods)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := util.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		defer testServer.Close()
		ch := make(chan interface{}, 1)
		c := sourceURL{testServer.URL, hostname, ch, nil}
		if err := c.extractFromURL(); err != nil {
			t.Errorf("%s: Unexpected error: %v", testCase.desc, err)
			continue
		}
		update := (<-ch).(kubelet.PodUpdate)

		if !api.Semantic.DeepEqual(testCase.expected, update) {
			t.Errorf("%s: Expected: %#v, Got: %#v", testCase.desc, testCase.expected, update)
		}
		for _, pod := range update.Pods {
			if errs := validation.ValidatePod(pod); len(errs) != 0 {
				t.Errorf("%s: Expected no validation errors on %#v, Got %v", testCase.desc, pod, errors.NewAggregate(errs))
			}
		}
	}
}
