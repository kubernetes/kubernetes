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

package config

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/securitycontext"
)

func noDefault(*api.Pod) error { return nil }

func TestDecodeSinglePod(t *testing.T) {
	grace := int64(30)
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				SecurityContext:          securitycontext.ValidSecurityContextWithContainerDefaults(),
			}},
			SecurityContext: &v1.PodSecurityContext{},
			SchedulerName:   api.DefaultSchedulerName,
		},
	}
	json, err := runtime.Encode(testapi.Default.Codec(), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	parsed, podOut, err := tryDecodeSinglePod(json, noDefault)
	if !parsed {
		t.Errorf("expected to have parsed file: (%s)", string(json))
	}
	if err != nil {
		t.Errorf("unexpected error: %v (%s)", err, string(json))
	}
	if !reflect.DeepEqual(pod, podOut) {
		t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, podOut, string(json))
	}

	for _, gv := range api.Registry.EnabledVersionsForGroup(v1.GroupName) {
		info, _ := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), "application/yaml")
		encoder := api.Codecs.EncoderForVersion(info.Serializer, gv)
		yaml, err := runtime.Encode(encoder, pod)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		parsed, podOut, err = tryDecodeSinglePod(yaml, noDefault)
		if !parsed {
			t.Errorf("expected to have parsed file: (%s)", string(yaml))
		}
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", err, string(yaml))
		}
		if !reflect.DeepEqual(pod, podOut) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, podOut, string(yaml))
		}
	}
}

func TestDecodePodList(t *testing.T) {
	grace := int64(30)
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,

				SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
			}},
			SecurityContext: &v1.PodSecurityContext{},
			SchedulerName:   api.DefaultSchedulerName,
		},
	}
	podList := &v1.PodList{
		Items: []v1.Pod{*pod},
	}
	json, err := runtime.Encode(testapi.Default.Codec(), podList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	parsed, podListOut, err := tryDecodePodList(json, noDefault)
	if !parsed {
		t.Errorf("expected to have parsed file: (%s)", string(json))
	}
	if err != nil {
		t.Errorf("unexpected error: %v (%s)", err, string(json))
	}
	if !reflect.DeepEqual(podList, &podListOut) {
		t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", podList, &podListOut, string(json))
	}

	for _, gv := range api.Registry.EnabledVersionsForGroup(v1.GroupName) {
		info, _ := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), "application/yaml")
		encoder := api.Codecs.EncoderForVersion(info.Serializer, gv)
		yaml, err := runtime.Encode(encoder, podList)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		parsed, podListOut, err = tryDecodePodList(yaml, noDefault)
		if !parsed {
			t.Errorf("expected to have parsed file: (%s): %v", string(yaml), err)
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", err, string(yaml))
			continue
		}
		if !reflect.DeepEqual(podList, &podListOut) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, &podListOut, string(yaml))
		}
	}
}

func TestGetSelfLink(t *testing.T) {
	var testCases = []struct {
		desc             string
		name             string
		namespace        string
		expectedSelfLink string
	}{
		{
			desc:             "No namespace specified",
			name:             "foo",
			namespace:        "",
			expectedSelfLink: "/api/v1/namespaces/default/pods/foo",
		},
		{
			desc:             "Namespace specified",
			name:             "foo",
			namespace:        "bar",
			expectedSelfLink: "/api/v1/namespaces/bar/pods/foo",
		},
	}
	for _, testCase := range testCases {
		selfLink := getSelfLink(testCase.name, testCase.namespace)
		if testCase.expectedSelfLink != selfLink {
			t.Errorf("%s: getSelfLink error, expected: %s, got: %s", testCase.desc, testCase.expectedSelfLink, selfLink)
		}
	}
}
