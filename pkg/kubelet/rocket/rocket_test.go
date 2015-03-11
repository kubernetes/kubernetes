/*
Copyright 2015 CoreOS Inc. All rights reserved.

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

package rocket

import (
	"flag"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

var enableTests bool

func init() {
	// Disabled by default since these tests require root privilege.
	flag.BoolVar(&enableTests, "enable-rocket-tests", false, "Whether the rocket tests should be enabled")
}

const (
	testACI1 = "coreos.com/etcd:v2.0.4"
	testACI2 = "docker://busybox"
)

func newRocketOrFail(t *testing.T) *Runtime {
	rkt, err := New(&Config{
		InsecureSkipVerify: true,
	})
	if err != nil {
		t.Fatalf("Cannot create rocket: %v", err)
	}
	return rkt
}

func TestVersion(t *testing.T) {
	rkt := newRocketOrFail(t)
	_, err := rkt.Version()
	if err != nil {
		t.Errorf("Cannot get rocket version: %v", err)
	}
}

// fetchAndVerifyPod tests if the expected pod is in the expected state.
// If so returns the pod we found.
func fetchAndVerifyPod(rkt *Runtime, expectedPod *api.BoundPod, expectedState string, t *testing.T) *api.Pod {
	pods, err := rkt.ListPods()
	if err != nil {
		t.Errorf("Cannot list pods: %v", err)
	}
	var foundPod *api.Pod
	for _, p := range pods {
		if p.UID == expectedPod.UID &&
			p.Name == expectedPod.Name &&
			p.Namespace == expectedPod.Namespace &&
			reflect.DeepEqual(p.Spec, expectedPod.Spec) {
			foundPod = p
			break
		}
	}
	if foundPod == nil {
		t.Errorf("Cannot find the pod: %v", expectedPod.Name)
	}
	for _, status := range foundPod.Status.Info {
		switch expectedState {
		case "running":
			if status.State.Running == nil {
				t.Errorf("Container status is not %v", expectedState)
				return nil
			}
		case "termination":
			if status.State.Termination == nil {
				t.Errorf("Container status is not %v", expectedState)
				return nil
			}
		case "waiting":
			if status.State.Waiting == nil {
				t.Errorf("Container status is not %v", expectedState)
				return nil
			}
		default:
			t.Errorf("Wrong state: %v", expectedState)
			return nil
		}
	}
	return foundPod
}

// TestRunListKillListPod runs and kills a pod, and verify the pod is
// being correctly run and killed.
func TestRunListKillListPod(t *testing.T) {
	if !enableTests {
		return
	}
	rkt := newRocketOrFail(t)

	pod := &api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID("testRocket_" + time.Now().String()),
			Name:        "foo",
			Namespace:   "default",
			Annotations: make(map[string]string),
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

	// TODO(yifan): Add volumes.
	if err := rkt.RunPod(pod, nil); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	time.Sleep(time.Second * 1) // Wait for the pod becomes running.

	toKill := fetchAndVerifyPod(rkt, pod, "running", t)
	if err := rkt.KillPod(toKill); err != nil {
		t.Errorf("Cannot kill pod %v", err)
	}

	time.Sleep(time.Second * 1) // Wait for the pod becomes dead.
	fetchAndVerifyPod(rkt, pod, "termination", t)
}

// TestKillAndRunContainerInPod runs a pod, and kill/restart a container
// in that pod.
func TestKillAndRunContainerInPod(t *testing.T) {
	if !enableTests {
		return
	}
	rkt := newRocketOrFail(t)

	containers := []api.Container{
		{
			Name:  "testImage1",
			Image: testACI1,
		},
		{
			Name:  "testImage2",
			Image: testACI2,
		},
	}

	pod := &api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID("testRocket_" + time.Now().String()),
			Name:        "foo",
			Namespace:   "default",
			Annotations: make(map[string]string),
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	// TODO(yifan): Add volumes.
	if err := rkt.RunPod(pod, nil); err != nil {
		t.Errorf("Cannot run pod: %v", err)
	}

	time.Sleep(time.Second * 1) // Wait for the pod becomes running.
	runningPod := fetchAndVerifyPod(rkt, pod, "running", t)

	if err := rkt.KillContainerInPod(containers[1], runningPod); err != nil {
		t.Errorf("Cannot kill container: %v", err)
	}

	time.Sleep(time.Second * 1) // Wait for the pod to be updated.
	pod.Spec.Containers = []api.Container{containers[0]}
	runningPod = fetchAndVerifyPod(rkt, pod, "running", t)

	if err := rkt.RunContainerInPod(containers[1], runningPod, nil); err != nil {
		t.Errorf("Cannot run container: %v", err)
	}

	time.Sleep(time.Second * 1) // Wait for the pod to be updated.
	pod.Spec.Containers = containers
	runningPod = fetchAndVerifyPod(rkt, pod, "running", t)

	// Tear down the pod
	if err := rkt.KillPod(runningPod); err != nil {
		t.Errorf("Cannot kill pod: %v", err)
	}
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

	rkt := newRocketOrFail(t)

	pod := &api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID("testRocket_" + time.Now().String()),
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

	volumeMap := map[string]volume.Interface{
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

	time.Sleep(time.Second * 1)
	runningPod := fetchAndVerifyPod(rkt, pod, "running", t)

	if _, err := os.Stat(path.Join(tmpDirPath, outputFiles[0])); err != nil {
		t.Errorf("Cannot stat the output file: %v", err)
	}
	if _, err := os.Stat(path.Join(tmpDirPath, outputFiles[1])); err != nil {
		t.Errorf("Cannot stat the output file: %v", err)
	}
	if err := rkt.KillPod(runningPod); err != nil {
		t.Errorf("Cannot kill pod: %v", err)
	}
}
