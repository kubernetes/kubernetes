/*
Copyright 2014 The Kubernetes Authors.

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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	utiltesting "k8s.io/client-go/util/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func TestURLErrorNotExistNoUpdate(t *testing.T) {
	ch := make(chan interface{})
	NewSourceURL("http://localhost:49575/_not_found_", http.Header{}, "localhost", time.Millisecond, ch)
	select {
	case got := <-ch:
		t.Errorf("Expected no update, Got %#v", got)
	case <-time.After(2 * time.Millisecond):
	}
}

func TestExtractFromHttpBadness(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := sourceURL{"http://localhost:49575/_not_found_", http.Header{}, "other", ch, nil, 0, http.DefaultClient}
	if err := c.extractFromURL(); err == nil {
		t.Errorf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractInvalidPods(t *testing.T) {
	var testCases = []struct {
		desc string
		pod  *v1.Pod
	}{
		{
			desc: "No version",
			pod:  &v1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: ""}},
		},
		{
			desc: "Invalid version",
			pod:  &v1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "v1betta2"}},
		},
		{
			desc: "Invalid volume name",
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{{Name: "_INVALID_"}},
				},
			},
		},
		{
			desc: "Duplicate volume names",
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{{Name: "repeated"}, {Name: "repeated"}},
				},
			},
		},
		{
			desc: "Unspecified container name",
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: ""}},
				},
			},
		},
		{
			desc: "Invalid container name",
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "_INVALID_"}},
				},
			},
		},
	}
	for _, testCase := range testCases {
		data, err := json.Marshal(testCase.pod)
		if err != nil {
			t.Fatalf("%s: Some weird json problem: %v", testCase.desc, err)
		}
		fakeHandler := utiltesting.FakeHandler{
			StatusCode:   http.StatusOK,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		defer testServer.Close()
		ch := make(chan interface{}, 1)
		c := sourceURL{testServer.URL, http.Header{}, "localhost", ch, nil, 0, http.DefaultClient}
		if err := c.extractFromURL(); err == nil {
			t.Errorf("%s: Expected error", testCase.desc)
		}
	}
}

func TestExtractPodsFromHTTP(t *testing.T) {
	nodeName := "different-value"

	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	var testCases = []struct {
		desc     string
		pods     runtime.Object
		expected kubetypes.PodUpdate
	}{
		{
			desc: "Single pod",
			pods: &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					UID:       "111",
					Namespace: "mynamespace",
				},
				Spec: v1.PodSpec{
					NodeName:        string(nodeName),
					Containers:      []v1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1.PullAlways, TerminationMessagePolicy: v1.TerminationMessageReadFile}},
					SecurityContext: &v1.PodSecurityContext{},
					SchedulerName:   api.DefaultSchedulerName,
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expected: CreatePodUpdate(kubetypes.SET,
				kubetypes.HTTPSource,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						UID:         "111",
						Name:        "foo" + "-" + nodeName,
						Namespace:   "mynamespace",
						Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "111"},
						SelfLink:    getSelfLink("foo-"+nodeName, "mynamespace"),
					},
					Spec: v1.PodSpec{
						NodeName:                      nodeName,
						RestartPolicy:                 v1.RestartPolicyAlways,
						DNSPolicy:                     v1.DNSClusterFirst,
						SecurityContext:               &v1.PodSecurityContext{},
						TerminationGracePeriodSeconds: &grace,
						SchedulerName:                 api.DefaultSchedulerName,
						EnableServiceLinks:            &enableServiceLinks,

						Containers: []v1.Container{{
							Name:                     "1",
							Image:                    "foo",
							TerminationMessagePath:   "/dev/termination-log",
							ImagePullPolicy:          "Always",
							TerminationMessagePolicy: v1.TerminationMessageReadFile,
						}},
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				}),
		},
		{
			desc: "Multiple pods",
			pods: &v1.PodList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "PodList",
					APIVersion: "",
				},
				Items: []v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "foo",
							UID:  "111",
						},
						Spec: v1.PodSpec{
							NodeName:        nodeName,
							Containers:      []v1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1.PullAlways, TerminationMessagePolicy: v1.TerminationMessageReadFile}},
							SecurityContext: &v1.PodSecurityContext{},
							SchedulerName:   api.DefaultSchedulerName,
						},
						Status: v1.PodStatus{
							Phase: v1.PodPending,
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "bar",
							UID:  "222",
						},
						Spec: v1.PodSpec{
							NodeName:        nodeName,
							Containers:      []v1.Container{{Name: "2", Image: "bar:bartag", ImagePullPolicy: "", TerminationMessagePolicy: v1.TerminationMessageReadFile}},
							SecurityContext: &v1.PodSecurityContext{},
							SchedulerName:   api.DefaultSchedulerName,
						},
						Status: v1.PodStatus{
							Phase: v1.PodPending,
						},
					},
				},
			},
			expected: CreatePodUpdate(kubetypes.SET,
				kubetypes.HTTPSource,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						UID:         "111",
						Name:        "foo" + "-" + nodeName,
						Namespace:   "default",
						Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "111"},
						SelfLink:    getSelfLink("foo-"+nodeName, metav1.NamespaceDefault),
					},
					Spec: v1.PodSpec{
						NodeName:                      nodeName,
						RestartPolicy:                 v1.RestartPolicyAlways,
						DNSPolicy:                     v1.DNSClusterFirst,
						TerminationGracePeriodSeconds: &grace,
						SecurityContext:               &v1.PodSecurityContext{},
						SchedulerName:                 api.DefaultSchedulerName,
						EnableServiceLinks:            &enableServiceLinks,

						Containers: []v1.Container{{
							Name:                     "1",
							Image:                    "foo",
							TerminationMessagePath:   "/dev/termination-log",
							ImagePullPolicy:          "Always",
							TerminationMessagePolicy: v1.TerminationMessageReadFile,
						}},
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				},
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						UID:         "222",
						Name:        "bar" + "-" + nodeName,
						Namespace:   "default",
						Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "222"},
						SelfLink:    getSelfLink("bar-"+nodeName, metav1.NamespaceDefault),
					},
					Spec: v1.PodSpec{
						NodeName:                      nodeName,
						RestartPolicy:                 v1.RestartPolicyAlways,
						DNSPolicy:                     v1.DNSClusterFirst,
						TerminationGracePeriodSeconds: &grace,
						SecurityContext:               &v1.PodSecurityContext{},
						SchedulerName:                 api.DefaultSchedulerName,
						EnableServiceLinks:            &enableServiceLinks,

						Containers: []v1.Container{{
							Name:                     "2",
							Image:                    "bar:bartag",
							TerminationMessagePath:   "/dev/termination-log",
							ImagePullPolicy:          "IfNotPresent",
							TerminationMessagePolicy: v1.TerminationMessageReadFile,
						}},
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				}),
		},
	}

	for _, testCase := range testCases {
		data, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), testCase.pods)
		if err != nil {
			t.Fatalf("%s: error in encoding the pod: %v", testCase.desc, err)
		}
		fakeHandler := utiltesting.FakeHandler{
			StatusCode:   http.StatusOK,
			ResponseBody: string(data),
		}
		testServer := httptest.NewServer(&fakeHandler)
		defer testServer.Close()
		ch := make(chan interface{}, 1)
		c := sourceURL{testServer.URL, http.Header{}, types.NodeName(nodeName), ch, nil, 0, http.DefaultClient}
		if err := c.extractFromURL(); err != nil {
			t.Errorf("%s: Unexpected error: %v", testCase.desc, err)
			continue
		}
		update := (<-ch).(kubetypes.PodUpdate)

		if !apiequality.Semantic.DeepEqual(testCase.expected, update) {
			t.Errorf("%s: Expected: %#v, Got: %#v", testCase.desc, testCase.expected, update)
		}
		for _, pod := range update.Pods {
			// TODO: remove the conversion when validation is performed on versioned objects.
			internalPod := &api.Pod{}
			if err := k8s_api_v1.Convert_v1_Pod_To_core_Pod(pod, internalPod, nil); err != nil {
				t.Fatalf("%s: Cannot convert pod %#v, %#v", testCase.desc, pod, err)
			}
			if errs := validation.ValidatePodCreate(internalPod, validation.PodValidationOptions{}); len(errs) != 0 {
				t.Errorf("%s: Expected no validation errors on %#v, Got %v", testCase.desc, pod, errs.ToAggregate())
			}
		}
	}
}

func TestURLWithHeader(t *testing.T) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			UID:       "111",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			NodeName:   "localhost",
			Containers: []v1.Container{{Name: "1", Image: "foo", ImagePullPolicy: v1.PullAlways}},
		},
	}
	data, err := json.Marshal(pod)
	if err != nil {
		t.Fatalf("Unexpected json marshalling error: %v", err)
	}
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   http.StatusOK,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	ch := make(chan interface{}, 1)
	header := make(http.Header)
	header.Set("Metadata-Flavor", "Google")
	c := sourceURL{testServer.URL, header, "localhost", ch, nil, 0, http.DefaultClient}
	if err := c.extractFromURL(); err != nil {
		t.Fatalf("Unexpected error extracting from URL: %v", err)
	}
	update := (<-ch).(kubetypes.PodUpdate)

	headerVal := fakeHandler.RequestReceived.Header["Metadata-Flavor"]
	if len(headerVal) != 1 || headerVal[0] != "Google" {
		t.Errorf("Header missing expected entry %v. Got %v", header, fakeHandler.RequestReceived.Header)
	}
	if len(update.Pods) != 1 {
		t.Errorf("Received wrong number of pods, expected one: %v", update.Pods)
	}
}
