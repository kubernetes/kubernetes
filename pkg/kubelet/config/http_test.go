/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
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
				{Version: testapi.Version(), Volumes: []api.Volume{{Name: "_INVALID_"}}},
			},
		},
		{
			desc: "Duplicate volume names",
			manifests: []api.ContainerManifest{
				{
					Version: testapi.Version(),
					Volumes: []api.Volume{{Name: "repeated"}, {Name: "repeated"}},
				},
			},
		},
		{
			desc: "Unspecified container name",
			manifests: []api.ContainerManifest{
				{
					Version:    testapi.Version(),
					Containers: []api.Container{{Name: ""}},
				},
			},
		},
		{
			desc: "Invalid container name",
			manifests: []api.ContainerManifest{
				{
					Version:    testapi.Version(),
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
	// ContainerManifests are not supported v1beta3 onwards.
	if api.PreV1Beta3(testapi.Version()) {
		return
	}

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

						SelfLink: getSelfLink("foo-"+hostname, kubelet.NamespaceDefault),
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

						SelfLink: getSelfLink("111-"+hostname, kubelet.NamespaceDefault),
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

						SelfLink: getSelfLink("foo-"+hostname, kubelet.NamespaceDefault),
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

						SelfLink: getSelfLink("foo-"+hostname, kubelet.NamespaceDefault),
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

						SelfLink: getSelfLink("bar-"+hostname, kubelet.NamespaceDefault),
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
		pods     runtime.Object
		expected kubelet.PodUpdate
	}{
		{
			desc: "Single pod",
			pods: &api.Pod{
				TypeMeta: api.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					UID:       "111",
					Namespace: "mynamespace",
				},
				Spec: api.PodSpec{
					Host:       hostname,
					Containers: []api.Container{{Name: "1", Image: "foo", ImagePullPolicy: api.PullAlways}},
				},
			},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				&api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "mynamespace",

						SelfLink: getSelfLink("foo-"+hostname, "mynamespace"),
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
			pods: &api.PodList{
				TypeMeta: api.TypeMeta{
					Kind:       "PodList",
					APIVersion: "",
				},
				Items: []api.Pod{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							UID:  "111",
						},
						Spec: api.PodSpec{
							Host:       hostname,
							Containers: []api.Container{{Name: "1", Image: "foo", ImagePullPolicy: api.PullAlways}},
						},
					},
					{
						ObjectMeta: api.ObjectMeta{
							Name: "bar",
							UID:  "222",
						},
						Spec: api.PodSpec{
							Host:       hostname,
							Containers: []api.Container{{Name: "2", Image: "bar", ImagePullPolicy: ""}},
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

						SelfLink: getSelfLink("foo-"+hostname, kubelet.NamespaceDefault),
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

						SelfLink: getSelfLink("bar-"+hostname, kubelet.NamespaceDefault),
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
	}

	for _, testCase := range testCases {
		var versionedPods runtime.Object
		err := testapi.Converter().Convert(&testCase.pods, &versionedPods)
		if err != nil {
			t.Fatalf("error in versioning the pods: %s", testCase.desc, err)
		}
		data, err := testapi.Codec().Encode(versionedPods)
		if err != nil {
			t.Fatalf("%s: error in encoding the pod: %v", testCase.desc, err)
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
