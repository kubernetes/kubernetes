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
	"io/ioutil"
	"os"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/util/wait"
)

func TestExtractFromNonExistentFile(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := sourceFile{"/some/fake/file", "localhost", ch}
	err := c.extractFromPath()
	if err == nil {
		t.Errorf("Expected error")
	}
}

func TestUpdateOnNonExistentFile(t *testing.T) {
	ch := make(chan interface{})
	NewSourceFile("random_non_existent_path", "localhost", time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubetypes.PodUpdate)
		expected := CreatePodUpdate(kubetypes.SET, kubetypes.FileSource)
		if !api.Semantic.DeepDerivative(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Expected update, timeout instead")
	}
}

func writeTestFile(t *testing.T, dir, name string, contents string) *os.File {
	file, err := ioutil.TempFile(os.TempDir(), "test_pod_config")
	if err != nil {
		t.Fatalf("Unable to create test file %#v", err)
	}
	file.Close()
	if err := ioutil.WriteFile(file.Name(), []byte(contents), 0555); err != nil {
		t.Fatalf("Unable to write test file %#v", err)
	}
	return file
}

func TestReadPodsFromFile(t *testing.T) {
	hostname := "random-test-hostname"
	grace := int64(30)
	var testCases = []struct {
		desc     string
		pod      runtime.Object
		expected kubetypes.PodUpdate
	}{
		{
			desc: "Simple pod",
			pod: &api.Pod{
				TypeMeta: unversioned.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: api.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
				},
				Spec: api.PodSpec{
					Containers:      []api.Container{{Name: "image", Image: "test/image", SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults()}},
					SecurityContext: &api.PodSecurityContext{},
				},
				Status: api.PodStatus{
					Phase: api.PodPending,
				},
			},
			expected: CreatePodUpdate(kubetypes.SET, kubetypes.FileSource, &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:        "test-" + hostname,
					UID:         "12345",
					Namespace:   "mynamespace",
					Annotations: map[string]string{kubetypes.ConfigHashAnnotationKey: "12345"},
					SelfLink:    getSelfLink("test-"+hostname, "mynamespace"),
				},
				Spec: api.PodSpec{
					NodeName:                      hostname,
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					TerminationGracePeriodSeconds: &grace,
					Containers: []api.Container{{
						Name:  "image",
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        "Always",
						SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults()}},
					SecurityContext: &api.PodSecurityContext{},
				},
				Status: api.PodStatus{
					Phase: api.PodPending,
				},
			}),
		},
	}

	for _, testCase := range testCases {
		func() {
			var versionedPod runtime.Object
			err := testapi.Default.Converter().Convert(&testCase.pod, &versionedPod, nil)
			if err != nil {
				t.Fatalf("%s: error in versioning the pod: %v", testCase.desc, err)
			}
			fileContents, err := runtime.Encode(testapi.Default.Codec(), versionedPod)
			if err != nil {
				t.Fatalf("%s: error in encoding the pod: %v", testCase.desc, err)
			}

			file := writeTestFile(t, os.TempDir(), "test_pod_config", string(fileContents))
			defer os.Remove(file.Name())

			ch := make(chan interface{})
			NewSourceFile(file.Name(), hostname, time.Millisecond, ch)
			select {
			case got := <-ch:
				update := got.(kubetypes.PodUpdate)
				for _, pod := range update.Pods {
					if errs := validation.ValidatePod(pod); len(errs) > 0 {
						t.Errorf("%s: Invalid pod %#v, %#v", testCase.desc, pod, errs)
					}
				}
				if !api.Semantic.DeepEqual(testCase.expected, update) {
					t.Errorf("%s: Expected %#v, Got %#v", testCase.desc, testCase.expected, update)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("%s: Expected update, timeout instead", testCase.desc)
			}
		}()
	}
}

func TestExtractFromBadDataFile(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config", string([]byte{1, 2, 3}))
	defer os.Remove(file.Name())

	ch := make(chan interface{}, 1)
	c := sourceFile{file.Name(), "localhost", ch}
	err := c.extractFromPath()
	if err == nil {
		t.Fatalf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractFromEmptyDir(t *testing.T) {
	dirName, err := utiltesting.MkTmpdir("file-test")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(dirName)

	ch := make(chan interface{}, 1)
	c := sourceFile{dirName, "localhost", ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubetypes.PodUpdate)
	expected := CreatePodUpdate(kubetypes.SET, kubetypes.FileSource)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}
