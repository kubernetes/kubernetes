/*
Copyright 2015 Google Inc. All rights reserved.

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

package rkt

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

var enableTests bool

func init() {
	// Disabled by default since these tests require root privilege.
	rand.Seed(time.Now().UnixNano())
	flag.BoolVar(&enableTests, "enable-rkt-tests", true, "Whether the rkt tests should be enabled")
}

const (
	testACI1              = "http://users.developer.core-os.net/k8s_tests/test1.aci"
	testACI2              = "http://users.developer.core-os.net/k8s_tests/test2.aci"
	mountTestACI          = "http://users.developer.core-os.net/k8s_tests/mount_test.aci"
	runInContainerTestACI = "docker://nginx"
)

var (
	defaultTestTimeout = time.Second * 5
)

func newRktOrFail(t *testing.T) *Runtime {
	rkt, err := New(&Config{
		InsecureSkipVerify: true,
	})
	if err != nil {
		t.Fatalf("Cannot create rkt: %v", err)
	}
	return rkt
}

func TestVersion(t *testing.T) {
	rkt := newRktOrFail(t)
	_, err := rkt.Version()
	if err != nil {
		t.Errorf("Cannot get rkt version: %v", err)
	}
}

func TestParseLine(t *testing.T) {
	tests := []struct {
		input  string
		output []string
	}{
		{
			input:  "foo",
			output: []string{"foo"},
		},
		{
			input:  "\tfoo",
			output: []string{"foo"},
		},
		{
			input:  "\tfoo\t",
			output: []string{"foo"},
		},
		{
			input:  "\tfoo\tbar\t",
			output: []string{"foo", "bar"},
		},
		{
			input:  "\tfoo\t\n",
			output: []string{"foo"},
		},
		{
			input:  "uuid\taci\tstate\tnetwork",
			output: []string{"uuid", "aci", "state", "network"},
		},
		{
			input:  "uuid\t\t\t\t\taci\t\tstate\tnetwork",
			output: []string{"uuid", "aci", "state", "network"},
		},
	}

	for i, tt := range tests {
		if !reflect.DeepEqual(splitLine(tt.input), tt.output) {
			t.Errorf("test %d fail, expect: %v, saw: %v", i, tt.output, tt.input)
		}
	}
}

// verifyContainer returns true if container a's info is correct according to
// container b.
func verifyContainer(a *kubecontainer.Container, b *api.Container) bool {
	return a.Name == b.Name &&
		a.Image == b.Image &&
		a.Hash == HashContainer(b)
}

// verifyContainers returns true if the containers are correct according to the
// expectedContainers. It returns false if not.
func verifyContainers(containers []*kubecontainer.Container, expectedContainers []api.Container) bool {
	if len(containers) != len(expectedContainers) {
		return false
	}

	ok := 0
	for i := range containers {
		for j := range expectedContainers {
			if verifyContainer(containers[i], &expectedContainers[j]) {
				ok += 1
			}
		}
	}
	return ok == len(containers)
}

// checkAllContainerStates returns true if all containers have
// the expected state, false otherwise.
func checkAllContainerStates(p *kubecontainer.Pod, expectedState string) bool {
	for _, status := range p.Status.ContainerStatuses {
		switch expectedState {
		case "running":
			if status.State.Running == nil {
				return false
			}
		case "termination":
			if status.State.Termination == nil {
				return false
			}
		case "waiting":
			if status.State.Waiting == nil {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// tryFindPod tries to find a pod which has the expected information and containers.
func tryFindPod(expectedPod *api.Pod, pods []*kubecontainer.Pod) *kubecontainer.Pod {
	for _, p := range pods {
		if p.ID == expectedPod.ObjectMeta.UID &&
			p.Name == expectedPod.ObjectMeta.Name &&
			p.Namespace == expectedPod.ObjectMeta.Namespace &&
			verifyContainers(p.Containers, expectedPod.Spec.Containers) {
			return p
		}
	}
	return nil
}

// verifyPod tests if the expected pod with expected state exists.
func verifyPod(rkt *Runtime, expectedPod *api.Pod, expectedState string, t *testing.T) {
	pods, err := rkt.GetPods(true)
	if err != nil {
		t.Errorf("Cannot list pods: %v", err)
	}

	foundPod := tryFindPod(expectedPod, pods)
	if foundPod == nil {
		t.Errorf("Cannot find the pod: %v", expectedPod.Name)
	}

	timeout := time.Now().Add(defaultTestTimeout)
	for !checkAllContainerStates(foundPod, expectedState) {
		if time.Now().After(timeout) {
			t.Errorf("Timeout Waiting expected state: %q", expectedState)
			return
		}

		time.Sleep(time.Second)
		pods, err = rkt.GetPods(true)
		if err != nil {
			t.Errorf("Cannot list pods: %v", err)
		}
		foundPod = tryFindPod(expectedPod, pods)
		if foundPod == nil {
			t.Errorf("Cannot find pod: %v", expectedPod.Name)
		}
	}
}

// TestRunListKillListPod runs and kills a pod, and verify the pod is
// being correctly run and killed.
func TestRunListKillListPod(t *testing.T) {
	if !enableTests {
		return
	}
	rkt := newRktOrFail(t)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       types.UID(fmt.Sprintf("testRkt_%d", rand.Int())),
			Name:      "foo",
			Namespace: "default",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "testImage1",
					Image: testACI1,
				},
				{
					Name:  "testImage2",
					Image: testACI2,
				},
			},
		},
	}

	if err := rkt.RunPod(pod, nil); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	verifyPod(rkt, pod, "running", t)
	if err := rkt.KillPod(pod); err != nil {
		t.Errorf("Cannot kill pod %v", err)
	}

	verifyPod(rkt, pod, "termination", t)
}

type stubVolume struct {
	path string
}

func (f *stubVolume) GetPath() string {
	return f.path
}

// TestRunPodWithMountVolumes starts a pod that will mount volumes and
// generate files on the host file system.
func TestRunPodWithMountVolumes(t *testing.T) {
	if !enableTests {
		return
	}
	// The output file name is hardcoded.
	tmpDirPath := "/tmp"
	outputFiles := []string{
		"outputFoo.txt",
		"outputBar.txt",
	}

	rkt := newRktOrFail(t)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID(fmt.Sprintf("testRkt_%d", rand.Int())),
			Name:        "foo",
			Namespace:   "default",
			Annotations: make(map[string]string),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "mountTest",
					Image: mountTestACI,
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "foo",
							ReadOnly:  false,
							MountPath: "/foo",
						},
						{
							Name:      "bar",
							ReadOnly:  false,
							MountPath: "/bar",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "foo",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{"/tmp"},
					},
				},
				{
					Name: "bar",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{"/tmp"},
					},
				},
			},
		},
	}

	volumeMap := map[string]volume.Volume{
		"foo": &stubVolume{"/tmp"},
		"bar": &stubVolume{"/tmp"},
	}

	// Clean the dir.
	if err := os.Remove(path.Join(tmpDirPath, outputFiles[0])); err != nil {
		if !os.IsNotExist(err) {
			t.Error("Cannot remove output file: %v", err)
		}
	}

	if err := os.Remove(path.Join(tmpDirPath, outputFiles[1])); err != nil {
		if !os.IsNotExist(err) {
			t.Error("Cannot remove output file: %v", err)
		}
	}

	if err := rkt.RunPod(pod, volumeMap); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	verifyPod(rkt, pod, "running", t)

	// Wait for the execution of the pod.
	due := time.Now().Add(defaultTestTimeout)
	for {
		time.Sleep(time.Second)
		if time.Now().After(due) {
			t.Errorf("Timeout waiting the output files")
			break
		}
		if _, err := os.Stat(path.Join(tmpDirPath, outputFiles[0])); err != nil {
			continue
		}
		if _, err := os.Stat(path.Join(tmpDirPath, outputFiles[1])); err != nil {
			continue
		}
		break
	}
	if err := rkt.KillPod(pod); err != nil {
		t.Errorf("Cannot kill pod: %v", err)
	}
}

func TestRunInContainer(t *testing.T) {
	if !enableTests {
		return
	}
	tests := []struct {
		cmds           []string
		expectedOutput []byte
		expectedErr    error
	}{
		{
			[]string{"/bin/echo", "hello"},
			[]byte("hello"),
			nil,
		},
		{
			[]string{"/bin/echo", "rkt"},
			[]byte("rkt"),
			nil,
		},
	}

	rkt := newRktOrFail(t)

	containers := []api.Container{
		{
			Name:  "testRunInContainer",
			Image: runInContainerTestACI,
		},
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID(fmt.Sprintf("testRkt_%d", rand.Int())),
			Name:        "foo",
			Namespace:   "default",
			Annotations: make(map[string]string),
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	if err := rkt.RunPod(pod, nil); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	verifyPod(rkt, pod, "running", t)

	pods, err := rkt.GetPods(true)
	if err != nil {
		t.Errorf("Cannot list pods: %v", err)
	}
	runningPod := tryFindPod(pod, pods)
	if runningPod == nil {
		t.Errorf("Cannot find pod: %v", pod)
	}
	runningContainer := runningPod.FindContainerByName(containers[0].Name)
	if runningContainer == nil {
		t.Errorf("Cannot find container: %v", containers[0])
	}

	for i, tt := range tests {
		output, err := rkt.RunInContainer(string(runningContainer.ID), tt.cmds)
		if err != tt.expectedErr {
			t.Errorf("%d: Expected: %v, saw: %v", i, tt.expectedErr, err)
		}
		if !reflect.DeepEqual(tt.expectedOutput, output) {
			t.Errorf("%d: Expected: %v, saw: %v", i, tt.expectedOutput, output)
		}
	}

	if err := rkt.KillPod(pod); err != nil {
		t.Errorf("Cannot kill pod %v", err)
	}
}

func TestExecInContainer(t *testing.T) {
	if !enableTests {
		return
	}
	tests := []struct {
		cmds           []byte
		expectedOutput []byte
		expectedErr    error
	}{
		{
			[]byte("echo hello\n"),
			[]byte("hello\n"),
			nil,
		},
		{
			[]byte("echo rkt\n"),
			[]byte("rkt\n"),
			nil,
		},
		{
			[]byte("exit"),
			[]byte{},
			nil,
		},
	}

	rkt := newRktOrFail(t)

	containers := []api.Container{
		{
			Name:  "testRunInContainer",
			Image: runInContainerTestACI,
		},
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID(fmt.Sprintf("testRkt_%d", rand.Int())),
			Name:        "foo",
			Namespace:   "default",
			Annotations: make(map[string]string),
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	if err := rkt.RunPod(pod, nil); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	verifyPod(rkt, pod, "running", t)

	pods, err := rkt.GetPods(true)
	if err != nil {
		t.Errorf("Cannot list pods: %v", err)
	}
	runningPod := tryFindPod(pod, pods)
	if runningPod == nil {
		t.Errorf("Cannot find pod: %v", pod)
	}
	runningContainer := runningPod.FindContainerByName(containers[0].Name)
	if runningContainer == nil {
		t.Errorf("Cannot find container: %v", containers[0])
	}

	// Create pipes for stdin and stdout.
	inR, inW, err := os.Pipe()
	if err != nil {
		t.Errorf("Cannot create pipe: %v", err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Errorf("Cannot create pipe: %v", err)
	}

	for i, tt := range tests {
		go rkt.ExecInContainer(string(runningContainer.ID), []string{"/bin/bash"}, inR, outW, outW, false)
		// Wait for exec, however, this is racy.
		time.Sleep(time.Second)

		// Make some input
		n, err := inW.Write(tt.cmds)
		if err != nil || n != len(tt.cmds) {
			t.Errorf("%d: Cannot write to pipe: %v", i, err)
		}

		output := make([]byte, len(tt.expectedOutput))
		_, err = outR.Read(output)
		if err != nil {
			t.Errorf("%d: Cannot read from pipe: %v", i, err)
		}

		if !reflect.DeepEqual(tt.expectedOutput, output) {
			t.Errorf("%d: Expected: %q, saw: %q", i, tt.expectedOutput, output)
		}
	}

	if err := rkt.KillPod(pod); err != nil {
		t.Errorf("Cannot kill pod %v", err)
	}
}

func TestIsImagePresentAndPullImage(t *testing.T) {
	testImage := "nginx"
	rkt := newRktOrFail(t)

	if err := exec.Command("rkt", "gc", "--expire-prepared=0", "--grace-period=0").Run(); err != nil {
		t.Errorf("Failed to rkt gc: %v", err)
	}
	if err := os.RemoveAll("/var/lib/rkt"); err != nil {
		t.Errorf("Failed to remove rkt dir: %v", err)
	}

	ok, err := rkt.IsImagePresent(testImage)
	if err != nil {
		t.Errorf("IsImagePresent failed: %v", err)
	}
	if ok {
		t.Errorf("Should not find the image present!")
	}

	if err := rkt.PullImage(testImage); err != nil {
		t.Errorf("Failed to pull image: %v", err)
	}

	ok, err = rkt.IsImagePresent(testImage)
	if err != nil {
		t.Errorf("IsImagePresent failed: %v", err)
	}
	if !ok {
		t.Errorf("Should find the image present!")
	}
}
