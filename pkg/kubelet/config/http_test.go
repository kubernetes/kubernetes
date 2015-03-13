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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

func TestURLErrorNotExistNoUpdate(t *testing.T) {
	ch := make(chan interface{})
	NewSourceURL("http://localhost:49575/_not_found_", time.Millisecond, ch)
	select {
	case got := <-ch:
		t.Errorf("Expected no update, Got %#v", got)
	case <-time.After(2 * time.Millisecond):
	}
}

func TestExtractFromHttpBadness(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := sourceURL{"http://localhost:49575/_not_found_", ch, nil}
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
		c := sourceURL{testServer.URL, ch, nil}
		if err := c.extractFromURL(); err == nil {
			t.Errorf("%s: Expected error", testCase.desc)
		}
	}
}

func TestExtractFromHTTP(t *testing.T) {
	hostname, _ := os.Hostname()
	hostname = strings.ToLower(hostname)

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
				api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta1/pods/foo",
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
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
			desc:      "Single manifest without ID",
			manifests: api.ContainerManifest{Version: "v1beta1", UUID: "111"},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "111" + "-" + hostname,
						Namespace: "foobar",
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
						DNSPolicy:     api.DNSClusterFirst,
					},
				}),
		},
		{
			desc: "Single manifest with v1beta2",
			manifests: v1beta1.ContainerManifest{Version: "v1beta2", ID: "foo", UUID: "111",
				Containers: []v1beta1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1beta1.PullAlways}}},
			expected: CreatePodUpdate(kubelet.SET,
				kubelet.HTTPSource,
				api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta1/pods/foo",
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
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
				api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "111",
						Name:      "foo" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta1/pods/foo",
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
						DNSPolicy:     api.DNSClusterFirst,
						Containers: []api.Container{{
							Name:  "1",
							Image: "foo",
							TerminationMessagePath: "/dev/termination-log",
							ImagePullPolicy:        "Always"}},
					},
				},
				api.Pod{
					ObjectMeta: api.ObjectMeta{
						UID:       "222",
						Name:      "bar" + "-" + hostname,
						Namespace: "foobar",
						SelfLink:  "/api/v1beta1/pods/bar",
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
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
		c := sourceURL{testServer.URL, ch, nil}
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
		for i := range update.Pods {
			if errs := validation.ValidatePod(&update.Pods[i]); len(errs) != 0 {
				t.Errorf("%s: Expected no validation errors on %#v, Got %v", testCase.desc, update.Pods[i], errors.NewAggregate(errs))
			}
		}
	}
}
