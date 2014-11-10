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
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"

	"gopkg.in/v1/yaml"
)

func ExampleManifestAndPod(id string) (api.ContainerManifest, api.BoundPod) {
	manifest := api.ContainerManifest{
		ID:   id,
		UUID: "uid",
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
				Source: &api.VolumeSource{
					HostDir: &api.HostDir{"/dir/path"},
				},
			},
		},
	}
	expectedPod := api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			Name:      id,
			UID:       "uid",
			Namespace: "default",
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
					Source: &api.VolumeSource{
						HostDir: &api.HostDir{"/dir/path"},
					},
				},
			},
		},
	}
	return manifest, expectedPod
}

func TestExtractFromNonExistentFile(t *testing.T) {
	ch := make(chan interface{}, 1)
	c := SourceFile{"/some/fake/file", ch}
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
		t.Errorf("Expected no update, Got %#v", got)
	case <-time.After(2 * time.Millisecond):
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
	file := writeTestFile(t, os.TempDir(), "test_pod_config", "version: v1beta1\nid: test\ncontainers:\n- image: test/image")
	defer os.Remove(file.Name())

	ch := make(chan interface{})
	NewSourceFile(file.Name(), time.Millisecond, ch)
	select {
	case got := <-ch:
		update := got.(kubelet.PodUpdate)
		expected := CreatePodUpdate(kubelet.SET, api.BoundPod{
			ObjectMeta: api.ObjectMeta{
				Name:      simpleSubdomainSafeHash(file.Name()),
				UID:       simpleSubdomainSafeHash(file.Name()),
				Namespace: "default",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{{Image: "test/image", TerminationMessagePath: "/dev/termination-log"}},
			},
		})
		if !reflect.DeepEqual(expected, update) {
			t.Fatalf("Expected %#v, Got %#v", expected, update)
		}

	case <-time.After(2 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestExtractFromBadDataFile(t *testing.T) {
	file := writeTestFile(t, os.TempDir(), "test_pod_config", string([]byte{1, 2, 3}))
	defer os.Remove(file.Name())

	ch := make(chan interface{}, 1)
	c := SourceFile{file.Name(), ch}
	err := c.extractFromPath()
	if err == nil {
		t.Fatalf("Expected error")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractFromValidDataFile(t *testing.T) {
	manifest, expectedPod := ExampleManifestAndPod("id")

	text, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	file := writeTestFile(t, os.TempDir(), "test_pod_config", string(text))
	defer os.Remove(file.Name())

	expectedPod.Name = simpleSubdomainSafeHash(file.Name())

	ch := make(chan interface{}, 1)
	c := SourceFile{file.Name(), ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, expectedPod)
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}

func TestExtractFromEmptyDir(t *testing.T) {
	dirName, err := ioutil.TempDir("", "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(dirName)

	ch := make(chan interface{}, 1)
	c := SourceFile{dirName, ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET)
	if !reflect.DeepEqual(expected, update) {
		t.Errorf("Expected %#v, Got %#v", expected, update)
	}
}

func TestExtractFromDir(t *testing.T) {
	manifest, expectedPod := ExampleManifestAndPod("1")
	manifest2, expectedPod2 := ExampleManifestAndPod("2")

	manifests := []api.ContainerManifest{manifest, manifest2}
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
		pods[i].Name = simpleSubdomainSafeHash(name)
	}

	ch := make(chan interface{}, 1)
	c := SourceFile{dirName, ch}
	err = c.extractFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubelet.PodUpdate)
	expected := CreatePodUpdate(kubelet.SET, pods...)
	sort.Sort(sortedPods(update.Pods))
	sort.Sort(sortedPods(expected.Pods))
	if !reflect.DeepEqual(expected, update) {
		t.Fatalf("Expected %#v, Got %#v", expected, update)
	}
	for i := range update.Pods {
		if errs := validation.ValidateBoundPod(&update.Pods[i]); len(errs) != 0 {
			t.Errorf("Expected no validation errors on %#v, Got %#v", update.Pods[i], errs)
		}
	}
}

func TestSubdomainSafeName(t *testing.T) {
	type Case struct {
		Input    string
		Expected string
	}
	testCases := []Case{
		{"/some/path/invalidUPPERCASE", "invaliduppercasa6hlenc0vpqbbdtt26ghneqsq3pvud"},
		{"/some/path/_-!%$#&@^&*(){}", "nvhc03p016m60huaiv3avts372rl2p"},
	}
	for _, testCase := range testCases {
		value := simpleSubdomainSafeHash(testCase.Input)
		if value != testCase.Expected {
			t.Errorf("Expected %s, Got %s", testCase.Expected, value)
		}
		value2 := simpleSubdomainSafeHash(testCase.Input)
		if value != value2 {
			t.Errorf("Value for %s was not stable across runs: %s %s", testCase.Input, value, value2)
		}
	}
}

// These are used for testing extract json (below)
type TestData struct {
	Value  string
	Number int
}

type TestObject struct {
	Name string
	Data TestData
}

func verifyStringEquals(t *testing.T, actual, expected string) {
	if actual != expected {
		t.Errorf("Verification failed.  Expected: %s, Found %s", expected, actual)
	}
}

func verifyIntEquals(t *testing.T, actual, expected int) {
	if actual != expected {
		t.Errorf("Verification failed.  Expected: %d, Found %d", expected, actual)
	}
}

func TestExtractJSON(t *testing.T) {
	obj := TestObject{}
	data := `{ "name": "foo", "data": { "value": "bar", "number": 10 } }`

	if err := yaml.Unmarshal([]byte(data), &obj); err != nil {
		t.Fatalf("Could not unmarshal JSON: %v", err)
	}

	verifyStringEquals(t, obj.Name, "foo")
	verifyStringEquals(t, obj.Data.Value, "bar")
	verifyIntEquals(t, obj.Data.Number, 10)
}
