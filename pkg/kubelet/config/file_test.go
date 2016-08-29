/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
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
	c := new("/some/fake/file", "localhost", time.Millisecond, ch)
	err := c.watch()
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
		t.Fatalf("Expected update, timeout instead")
	}
}

func TestReadPodsFromFileExistAlready(t *testing.T) {
	hostname := "random-test-hostname"
	var testCases = getTestCases(hostname)

	for _, testCase := range testCases {
		func() {
			dirName, err := utiltesting.MkTmpdir("file-test")
			if err != nil {
				t.Fatalf("Unable to create temp dir: %v", err)
			}
			defer os.RemoveAll(dirName)
			file := testCase.writeToFile(dirName, "test_pod_config", t)

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
				t.Fatalf("%s: Expected update, timeout instead", testCase.desc)
			}
		}()
	}
}

func TestReadPodsFromFileExistLater(t *testing.T) {
	watchFileAdded(false, t)
}

func TestReadPodsFromFileChanged(t *testing.T) {
	watchFileChanged(false, t)
}

func TestReadPodsFromFileInDirAdded(t *testing.T) {
	watchFileAdded(true, t)
}

func TestReadPodsFromFileInDirChanged(t *testing.T) {
	watchFileChanged(true, t)
}

func TestExtractFromBadDataFile(t *testing.T) {
	dirName, err := utiltesting.MkTmpdir("file-test")
	if err != nil {
		t.Fatalf("Unable to create temp dir: %v", err)
	}
	defer os.RemoveAll(dirName)

	fileName := filepath.Join(dirName, "test_pod_config")
	err = ioutil.WriteFile(fileName, []byte{1, 2, 3}, 0555)
	if err != nil {
		t.Fatalf("Unable to write test file %#v", err)
	}

	ch := make(chan interface{}, 1)
	c := new(fileName, "localhost", time.Millisecond, ch)
	err = c.resetStoreFromPath()
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
	c := new(dirName, "localhost", time.Millisecond, ch)
	err = c.resetStoreFromPath()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	update := (<-ch).(kubetypes.PodUpdate)
	expected := CreatePodUpdate(kubetypes.SET, kubetypes.FileSource)
	if !api.Semantic.DeepEqual(expected, update) {
		t.Fatalf("Expected %#v, Got %#v", expected, update)
	}
}

type testCase struct {
	desc     string
	pod      runtime.Object
	expected kubetypes.PodUpdate
}

func getTestCases(hostname string) []*testCase {
	grace := int64(30)
	return []*testCase{
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
					Containers:      []api.Container{{Name: "image", Image: "test/image", TerminationMessagePath: "/dev/termination-log", ImagePullPolicy: "Always", SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults()}},
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
}

func (tc *testCase) writeToFile(dir, name string, t *testing.T) *os.File {
	var versionedPod runtime.Object
	err := testapi.Default.Converter().Convert(&tc.pod, &versionedPod, nil)
	if err != nil {
		t.Fatalf("%s: error in versioning the pod: %v", tc.desc, err)
	}
	fileContents, err := runtime.Encode(testapi.Default.Codec(), versionedPod)
	if err != nil {
		t.Fatalf("%s: error in encoding the pod: %v", tc.desc, err)
	}

	fileName := filepath.Join(dir, name)
	file, err := os.Create(fileName)
	if err != nil {
		t.Fatalf("Unable to create test file %#v", err)
	}
	file.Close()
	if err := ioutil.WriteFile(file.Name(), []byte(fileContents), 0555); err != nil {
		t.Fatalf("Unable to write test file %#v", err)
	}
	return file
}

func watchFileAdded(watchDir bool, t *testing.T) {
	hostname := "random-test-hostname"
	var testCases = getTestCases(hostname)

	fileNamePre := "test_pod_config"
	for index, testCase := range testCases {
		func() {
			dirName, err := utiltesting.MkTmpdir("dir-test")
			if err != nil {
				t.Fatalf("Unable to create temp dir: %v", err)
			}
			defer os.RemoveAll(dirName)
			fileName := fmt.Sprintf("%s_%d", fileNamePre, index)

			ch := make(chan interface{})
			if watchDir {
				NewSourceFile(dirName, hostname, time.Second, ch)
			} else {
				NewSourceFile(filepath.Join(dirName, fileName), hostname, time.Second, ch)
			}

			finishEdit := make(chan struct{})
			changeFile := func() {
				// Add a file
				testCase.writeToFile(dirName, fileName, t)

				if watchDir {
					// Change the file name
					from := fileName
					fileName = fileName + "_ch"
					if err = changeFileName(dirName, from, fileName); err != nil {
						t.Errorf("Fail to change file name: %s", err)
					}
				}
				finishEdit <- struct{}{}
			}
			go changeFile()

			if watchDir {
				defer func() {
					<-finishEdit
					// Remove the file
					deleteFile(dirName, fileName, ch, t)
				}()
			}

			timer := time.After(5 * time.Second)
			for {
				select {
				case got := <-ch:
					update := got.(kubetypes.PodUpdate)
					for _, pod := range update.Pods {
						if errs := validation.ValidatePod(pod); len(errs) > 0 {
							t.Errorf("%s: Invalid pod %#v, %#v", testCase.desc, pod, errs)
						}
					}
					if api.Semantic.DeepEqual(testCase.expected, update) {
						return
					}
				case <-timer:
					t.Fatalf("%s: Expected update, timeout instead", testCase.desc)
				}
			}
		}()
	}
}

func watchFileChanged(watchDir bool, t *testing.T) {
	hostname := "random-test-hostname"
	var testCases = getTestCases(hostname)

	fileNamePre := "test_pod_config"
	for index, testCase := range testCases {
		func() {
			dirName, err := utiltesting.MkTmpdir("dir-test")
			fileName := fmt.Sprintf("%s_%d", fileNamePre, index)
			if err != nil {
				t.Fatalf("Unable to create temp dir: %v", err)
			}
			defer os.RemoveAll(dirName)

			var file *os.File
			lock := &sync.Mutex{}
			ch := make(chan interface{})
			finishEdit := make(chan struct{})
			func() {
				lock.Lock()
				defer lock.Unlock()
				file = testCase.writeToFile(dirName, fileName, t)
			}()

			if watchDir {
				NewSourceFile(dirName, hostname, time.Second, ch)
				defer func() {
					<-finishEdit
					// Remove the file
					deleteFile(dirName, fileName, ch, t)
				}()
			} else {
				NewSourceFile(file.Name(), hostname, time.Second, ch)
			}
			testCase.expected.Pods[0].Spec.Containers[0].Image = "test/newImage"

			changeFile := func() {
				// Edit the file content
				lock.Lock()
				defer lock.Unlock()
				pod := testCase.pod.(*api.Pod)
				pod.Spec.Containers[0].Image = "test/newImage"
				testCase.pod = pod
				testCase.writeToFile(dirName, fileName, t)

				// Change the file name
				if watchDir {
					from := fileName
					fileName = fileName + "_ch"
					if err = changeFileName(dirName, from, fileName); err != nil {
						t.Errorf("Fail to change file name: %s", err)
					}
				}
				finishEdit <- struct{}{}
			}
			go changeFile()

			timer := time.After(5 * time.Second)
			for {
				select {
				case got := <-ch:
					update := got.(kubetypes.PodUpdate)
					for _, pod := range update.Pods {
						if errs := validation.ValidatePod(pod); len(errs) > 0 {
							t.Errorf("%s: Invalid pod %#v, %#v", testCase.desc, pod, errs)
						}
					}
					if api.Semantic.DeepEqual(testCase.expected, update) {
						return
					}
				case <-timer:
					t.Fatalf("%s: Expected update, timeout instead", testCase.desc)
				}
			}
		}()
	}
}

func deleteFile(dir, file string, ch chan interface{}, t *testing.T) {
	finishDel := make(chan struct{})
	defer func() {
		<-finishDel
	}()

	go func() {
		path := filepath.Join(dir, file)
		err := os.Remove(path)
		finishDel <- struct{}{}
		if err != nil {
			t.Errorf("Unable to remove test file %s: %s", path, err)
		}
	}()

	timer := time.After(5 * time.Second)
	for {
		select {
		case got := <-ch:
			update := got.(kubetypes.PodUpdate)
			if len(update.Pods) == 0 {
				return
			}
		case <-timer:
			t.Fatalf("Deleting static pod file: Expected update with empty pods, timeout instead")
		}
	}
}

func changeFileName(dir, from, to string) error {
	fromPath := filepath.Join(dir, from)
	toPath := filepath.Join(dir, to)
	return exec.Command("mv", fromPath, toPath).Run()
}
