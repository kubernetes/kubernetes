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
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
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
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource)
		if !api.Semantic.DeepDerivative(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(time.Second):
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

func TestReadContainerManifestFromFile(t *testing.T) {
	// ContainerManifest is supported only for pre v1beta3 versions.
	if !api.PreV1Beta3(testapi.Version()) {
		return
	}
	hostname := "random-test-hostname"
	var testCases = []struct {
		desc         string
		fileContents string
		expected     kubelet.PodUpdate
	}{
		{
			desc: "Manifest",
			fileContents: fmt.Sprintf(`{
					"version": "%s",
					"uuid": "12345",
					"id": "test",
					"containers": [{ "name": "image", "image": "test/image", "imagePullPolicy": "PullAlways"}]
				}`, testapi.Version()),
			expected: CreatePodUpdate(kubelet.SET, kubelet.FileSource, &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "test-" + hostname,
					UID:       "12345",
					Namespace: kubelet.NamespaceDefault,
					SelfLink:  getSelfLink("test-"+hostname, kubelet.NamespaceDefault),
				},
				Spec: api.PodSpec{
					Host:          hostname,
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers: []api.Container{{
						Name:  "image",
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        "Always"}},
				},
			}),
		},
		{
			desc: "Manifest without ID",
			fileContents: fmt.Sprintf(`{
						"version": "%s",
						"uuid": "12345",
						"containers": [{ "name": "image", "image": "test/image", "imagePullPolicy": "PullAlways"}]
					}`, testapi.Version()),
			expected: CreatePodUpdate(kubelet.SET, kubelet.FileSource, &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "12345-" + hostname,
					UID:       "12345",
					Namespace: kubelet.NamespaceDefault,
					SelfLink:  getSelfLink("12345-"+hostname, kubelet.NamespaceDefault),
				},
				Spec: api.PodSpec{
					Host:          hostname,
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers: []api.Container{{
						Name:  "image",
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        "Always"}},
				},
			}),
		},
	}

	for _, testCase := range testCases {
		func() {
			file := writeTestFile(t, os.TempDir(), "test_pod_config", testCase.fileContents)
			defer os.Remove(file.Name())

			ch := make(chan interface{})
			NewSourceFile(file.Name(), hostname, time.Millisecond, ch)
			select {
			case got := <-ch:
				update := got.(kubelet.PodUpdate)
				for _, pod := range update.Pods {
					if errs := validation.ValidatePod(pod); len(errs) > 0 {
						t.Errorf("%s: Invalid pod %#v, %#v", testCase.desc, pod, errs)
					}
				}
				if !api.Semantic.DeepEqual(testCase.expected, update) {
					t.Errorf("%s: Expected %#v, Got %#v", testCase.desc, testCase.expected, update)
				}
			case <-time.After(time.Second):
				t.Errorf("%s: Expected update, timeout instead", testCase.desc)
			}
		}()
	}
}

func TestReadPodsFromFile(t *testing.T) {
	hostname := "random-test-hostname"
	var testCases = []struct {
		desc     string
		pod      runtime.Object
		expected kubelet.PodUpdate
	}{
		{
			desc: "Simple pod",
			pod: &api.Pod{
				TypeMeta: api.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: api.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Name: "image", Image: "test/image"}},
				},
			},
			expected: CreatePodUpdate(kubelet.SET, kubelet.FileSource, &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "test-" + hostname,
					UID:       "12345",
					Namespace: "mynamespace",
					SelfLink:  getSelfLink("test-"+hostname, "mynamespace"),
				},
				Spec: api.PodSpec{
					Host:          hostname,
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers: []api.Container{{
						Name:  "image",
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        "IfNotPresent"}},
				},
			}),
		},
		{
			desc: "Pod without ID",
			pod: &api.Pod{
				TypeMeta: api.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: api.ObjectMeta{
					// No name
					UID: "12345",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Name: "image", Image: "test/image"}},
				},
			},
			expected: CreatePodUpdate(kubelet.SET, kubelet.FileSource, &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "12345-" + hostname,
					UID:       "12345",
					Namespace: kubelet.NamespaceDefault,
					SelfLink:  getSelfLink("12345-"+hostname, kubelet.NamespaceDefault),
				},
				Spec: api.PodSpec{
					Host:          hostname,
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers: []api.Container{{
						Name:  "image",
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        "IfNotPresent"}},
				},
			}),
		},
	}

	for _, testCase := range testCases {
		func() {
			var versionedPod runtime.Object
			err := testapi.Converter().Convert(&testCase.pod, &versionedPod)
			if err != nil {
				t.Fatalf("error in versioning the pod: %s", testCase.desc, err)
			}
			fileContents, err := testapi.Codec().Encode(versionedPod)
			if err != nil {
				t.Fatalf("%s: error in encoding the pod: %v", testCase.desc, err)
			}

			file := writeTestFile(t, os.TempDir(), "test_pod_config", string(fileContents))
			defer os.Remove(file.Name())

			ch := make(chan interface{})
			NewSourceFile(file.Name(), hostname, time.Millisecond, ch)
			select {
			case got := <-ch:
				update := got.(kubelet.PodUpdate)
				for _, pod := range update.Pods {
					if errs := validation.ValidatePod(pod); len(errs) > 0 {
						t.Errorf("%s: Invalid pod %#v, %#v", testCase.desc, pod, errs)
					}
				}
				if !api.Semantic.DeepEqual(testCase.expected, update) {
					t.Errorf("%s: Expected %#v, Got %#v", testCase.desc, testCase.expected, update)
				}
			case <-time.After(time.Second):
				t.Errorf("%s: Expected update, timeout instead", testCase.desc)
			}
		}()
	}
}

func TestReadManifestFromFileWithDefaults(t *testing.T) {
	if !api.PreV1Beta3(testapi.Version()) {
		return
	}
	file := writeTestFile(t, os.TempDir(), "test_pod_config",
		fmt.Sprintf(`{
			"version": "%s",
			"id": "test",
			"containers": [{ "name": "image", "image": "test/image" }]
		}`, testapi.Version()))
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), "localhost", time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		if update.Pods[0].UID == "" {
			t.Errorf("Unexpected UID: %s", update.Pods[0].UID)
		}

	case <-time.After(time.Second):
		t.Errorf("Expected update, timeout instead")
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
	dirName, err := ioutil.TempDir("", "foo")
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

	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}

func exampleManifestAndPod(id string) (v1beta1.ContainerManifest, *api.Pod) {
	hostname := "an-example-host"

	manifest := v1beta1.ContainerManifest{
		Version: "v1beta1",
		ID:      id,
		UUID:    types.UID(id),
		Containers: []v1beta1.Container{
			{
				Name:  "c" + id,
				Image: "foo",
				TerminationMessagePath: "/somepath",
			},
		},
		Volumes: []v1beta1.Volume{
			{
				Name: "host-dir",
				Source: v1beta1.VolumeSource{
					HostDir: &v1beta1.HostPathVolumeSource{"/dir/path"},
				},
			},
		},
	}
	expectedPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      id + "-" + hostname,
			UID:       types.UID(id),
			Namespace: kubelet.NamespaceDefault,
			SelfLink:  getSelfLink(id+"-"+hostname, kubelet.NamespaceDefault),
		},
		Spec: api.PodSpec{
			Host: hostname,
			Containers: []api.Container{
				{
					Name:  "c" + id,
					Image: "foo",
				},
			},
			Volumes: []api.Volume{
				{
					Name: "host-dir",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{"/dir/path"},
					},
				},
			},
		},
	}
	return manifest, expectedPod
}

func TestExtractFromDir(t *testing.T) {
	if !api.PreV1Beta3(testapi.Version()) {
		return
	}
	manifest, expectedPod := exampleManifestAndPod("1")
	manifest2, expectedPod2 := exampleManifestAndPod("2")

	manifests := []v1beta1.ContainerManifest{manifest, manifest2}
	pods := []*api.Pod{expectedPod, expectedPod2}
	files := make([]*os.File, len(manifests))

	dirName, err := ioutil.TempDir("", "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	for i, manifest := range manifests {
		data, err := json.Marshal(manifest)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		file, err := ioutil.TempFile(dirName, manifest.ID)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		name := file.Name()
		if err := file.Close(); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		ioutil.WriteFile(name, data, 0755)
		files[i] = file
	}

	ch := make(chan interface{}, 1)
	c := sourceFile{dirName, "an-example-host", ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource, pods...)
	sort.Sort(sortedPods(update.Pods))
	sort.Sort(sortedPods(expected.Pods))
	if !api.Semantic.DeepDerivative(expected, update) {
		t.Fatalf("Expected %#v, Got %#v", expected, update)
	}
	for _, pod := range update.Pods {
		if errs := validation.ValidatePod(pod); len(errs) != 0 {
			t.Errorf("Expected no validation errors on %#v, Got %q", pod, errs)
		}
	}
}
