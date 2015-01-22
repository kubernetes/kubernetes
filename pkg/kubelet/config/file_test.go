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
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func ExampleManifestAndPod(id string) (v1beta1.ContainerManifest, api.BoundPod) {
	manifest := v1beta1.ContainerManifest{
		ID:   id,
		UUID: types.UID(id),
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
					HostDir: &v1beta1.HostPath{"/dir/path"},
				},
			},
		},
	}
	expectedPod := api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			Name: id,
			UID:  types.UID(id),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "c" + id,
					Image: "foo",
					TerminationMessagePath: "/somepath",
				},
			},
			Volumes: []api.Volume{
				{
					Name: "host-dir",
					Source: api.VolumeSource{
						HostPath: &api.HostPath{"/dir/path"},
					},
				},
			},
		},
	}
	return manifest, expectedPod
}

func TestExtractFromNonExistentFile(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := sourceFile{"/some/fake/file", ch}
	err := c.extractFromPath()
	if err == nil {
		t.Errorf("Expected error")
	}
}

func TestUpdateOnNonExistentFile(t *testing.T) {
	ch := make(chan interface{})
	NewSourceFile("random_non_existent_path", time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource)
		if !api.Semantic.DeepEqual(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(2 * time.Millisecond):
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

func TestReadFromFile(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config",
		`{
			"version": "v1beta1",
			"uuid": "12345",
			"id": "test",
			"containers": [{ "image": "test/image", imagePullPolicy: "PullAlways"}]
		}`)
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource, api.BoundPod{
			ObjectMeta: api.ObjectMeta{
				Name:      "test",
				UID:       "12345",
				Namespace: "",
				SelfLink:  "",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        api.PullAlways,
					},
				},
			},
		})

		// There's no way to provide namespace in ContainerManifest, so
		// it will be defaulted.
		if !strings.HasPrefix(update.Pods[0].ObjectMeta.Namespace, "file-") {
			t.Errorf("Unexpected namespace: %s", update.Pods[0].ObjectMeta.Namespace)
		}
		update.Pods[0].ObjectMeta.Namespace = ""

		// SelfLink depends on namespace.
		if !strings.HasPrefix(update.Pods[0].ObjectMeta.SelfLink, "/api/") {
			t.Errorf("Unexpected selflink: %s", update.Pods[0].ObjectMeta.SelfLink)
		}
		update.Pods[0].ObjectMeta.SelfLink = ""

		if !api.Semantic.DeepEqual(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(2 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestReadFromFileWithoutID(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config",
		`{
			"version": "v1beta1",
			"uuid": "12345",
			"containers": [{ "image": "test/image", imagePullPolicy: "PullAlways"}]
		}`)
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource, api.BoundPod{
			ObjectMeta: api.ObjectMeta{
				Name:      "",
				UID:       "12345",
				Namespace: "",
				SelfLink:  "",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        api.PullAlways,
					},
				},
			},
		})

		if len(update.Pods[0].ObjectMeta.Name) == 0 {
			t.Errorf("Name did not get defaulted")
		}
		update.Pods[0].ObjectMeta.Name = ""
		update.Pods[0].ObjectMeta.Namespace = ""
		update.Pods[0].ObjectMeta.SelfLink = ""

		if !api.Semantic.DeepEqual(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(2 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestReadV1Beta2FromFile(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config",
		`{
			"version": "v1beta2",
			"uuid": "12345",
			"id": "test",
			"containers": [{ "image": "test/image", imagePullPolicy: "PullAlways"}]
		}`)
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource, api.BoundPod{
			ObjectMeta: api.ObjectMeta{
				Name:      "test",
				UID:       "12345",
				Namespace: "",
				SelfLink:  "",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Image: "test/image",
						TerminationMessagePath: "/dev/termination-log",
						ImagePullPolicy:        api.PullAlways,
					},
				},
			},
		})

		update.Pods[0].ObjectMeta.Namespace = ""
		update.Pods[0].ObjectMeta.SelfLink = ""

		if !api.Semantic.DeepEqual(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(2 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestReadFromFileWithDefaults(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config",
		`{
			"version": "v1beta1",
			"id": "test",
			"containers": [{ "image": "test/image" }]
		}`)
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		if update.Pods[0].ObjectMeta.UID == "" {
			t.Errorf("Unexpected UID: %s", update.Pods[0].ObjectMeta.UID)
		}

	case <-time.After(2 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestExtractFromBadDataFile(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config", string([]byte{1, 2, 3}))
	defer os.Remove(file.Name())

	ch := make(chan interface{}, 1)
	c := sourceFile{file.Name(), ch}
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
	c := sourceFile{dirName, ch}
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

func TestExtractFromDir(t *testing.T) {
	manifest, expectedPod := ExampleManifestAndPod("1")
	manifest2, expectedPod2 := ExampleManifestAndPod("2")

	manifests := []v1beta1.ContainerManifest{manifest, manifest2}
	pods := []api.BoundPod{expectedPod, expectedPod2}
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
	c := sourceFile{dirName, ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubelet.PodUpdate)
	for i := range update.Pods {
		update.Pods[i].Namespace = "foobar"
		update.Pods[i].SelfLink = ""
	}
	expected := CreatePodUpdate(kubelet.SET, kubelet.FileSource, pods...)
	for i := range expected.Pods {
		expected.Pods[i].Namespace = "foobar"
	}
	sort.Sort(sortedPods(update.Pods))
	sort.Sort(sortedPods(expected.Pods))
	if !api.Semantic.DeepEqual(expected, update) {
		t.Fatalf("Expected %#v, Got %#v", expected, update)
	}
	for i := range update.Pods {
		if errs := validation.ValidateBoundPod(&update.Pods[i]); len(errs) != 0 {
			t.Errorf("Expected no validation errors on %#v, Got %#v", update.Pods[i], errs)
		}
	}
}
