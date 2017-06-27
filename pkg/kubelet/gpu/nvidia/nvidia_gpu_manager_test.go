/*
Copyright 2017 The Kubernetes Authors.

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

package nvidia

import (
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/container"
	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

type testActivePodsLister struct {
	activePods     []*v1.Pod
	dockerClient   libdocker.Interface
	nvidiaGPUPaths []string
}

func (tapl *testActivePodsLister) GetActivePods() []*v1.Pod {
	return tapl.activePods
}

func (tapl *testActivePodsLister) InitiateActivePods() {
	dockerClient := tapl.dockerClient

	for i, pod := range tapl.activePods {
		var gpusNeeded int64
		for _, container := range pod.Spec.Containers {
			gpusNeeded = container.Resources.Limits.NvidiaGPU().Value()
		}
		for j, containerStatus := range pod.Status.ContainerStatuses {
			var gpuDevices []string
			if gpusNeeded <= int64(len(tapl.nvidiaGPUPaths)) {
				gpuDevices = tapl.nvidiaGPUPaths[:gpusNeeded]
				tapl.nvidiaGPUPaths = tapl.nvidiaGPUPaths[gpusNeeded:]
			}
			devices := makeDevices(gpuDevices)

			conConfig := types.ContainerCreateConfig{
				Name: containerStatus.Name,
				Config: &container.Config{
					Image:  "foo",
					Labels: make(map[string]string),
				},
				HostConfig: &container.HostConfig{
					Resources: container.Resources{
						Devices: devices,
					},
				},
			}

			cr, _ := dockerClient.CreateContainer(conConfig)
			tapl.activePods[i].Status.ContainerStatuses[j].ContainerID = cr.ID
			dockerClient.StartContainer(cr.ID)
		}
	}
}

func makeDevices(nvidiaGPUPaths []string) []container.DeviceMapping {
	var devices []container.DeviceMapping
	for _, path := range nvidiaGPUPaths {
		// Devices have to be mapped one to one because of nvidia CUDA library requirements.
		devices = append(devices, container.DeviceMapping{PathOnHost: path, PathInContainer: path, CgroupPermissions: "mrw"})
	}

	return devices
}

func makeTestPod(numContainers, gpusPerContainer int) *v1.Pod {
	quantity := resource.NewQuantity(int64(gpusPerContainer), resource.DecimalSI)
	resources := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceNvidiaGPU: *quantity,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{},
		},
	}
	for ; numContainers > 0; numContainers-- {

		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name:      string(uuid.NewUUID()),
			Resources: resources,
		})
	}
	return pod
}

func TestNewNvidiaGPUManager(t *testing.T) {
	podLister := &testActivePodsLister{}

	// Expects nil GPUManager and an error with nil dockerClient.
	testGpuManager1, err := NewNvidiaGPUManager(podLister, nil)
	as := assert.New(t)
	as.Nil(testGpuManager1)
	as.NotNil(err)

	// Expects a GPUManager to be created with non-nil dockerClient.
	fakeDocker := libdocker.NewFakeDockerClient()
	testGpuManager2, err := NewNvidiaGPUManager(podLister, fakeDocker)
	as.NotNil(testGpuManager2)
	as.Nil(err)

	// Expects zero capacity without any GPUs.
	gpuCapacity := testGpuManager2.Capacity()
	as.Equal(len(gpuCapacity), 1)
	rgpu := gpuCapacity[v1.ResourceNvidiaGPU]
	as.Equal(rgpu.Value(), int64(0))

	err2 := testGpuManager2.Start()
	if !os.IsNotExist(err2) {
		gpus := reflect.ValueOf(testGpuManager2).Elem().FieldByName("allGPUs").Len()
		as.NotZero(gpus)
	}
}

func makeTestPodWithStatuses(numContainers, gpusPerContainer int) *v1.Pod {
	quantity := resource.NewQuantity(int64(gpusPerContainer), resource.DecimalSI)
	resources := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceNvidiaGPU: *quantity,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: uuid.NewUUID(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{},
		},
	}
	for ; numContainers > 0; numContainers-- {
		con := string(uuid.NewUUID())

		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name:      con,
			Resources: resources,
		})

		pod.Status.ContainerStatuses = append(pod.Status.ContainerStatuses, v1.ContainerStatus{
			Name:        con,
			Ready:       true,
			ContainerID: con,
			State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
		})
	}
	return pod
}

func TestMultiContainerPodGPUAllocation(t *testing.T) {
	podLister := &testActivePodsLister{}

	testGpuManager := &nvidiaGPUManager{
		activePodsLister: podLister,
		allGPUs:          sets.NewString("/dev/nvidia0", "/dev/nvidia1"),
		allocated:        newPodGPUs(),
	}

	// Expect that no devices are in use.
	gpusInUse := testGpuManager.gpusInUse()
	as := assert.New(t)
	as.Equal(len(gpusInUse.devices()), 0)

	// Allocated GPUs for a pod with two containers.
	pod := makeTestPod(2, 1)
	// Allocate for the first container.
	devices1, err := testGpuManager.AllocateGPU(pod, &pod.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devices1), 1)

	podLister.activePods = append(podLister.activePods, pod)
	// Allocate for the second container.
	devices2, err := testGpuManager.AllocateGPU(pod, &pod.Spec.Containers[1])
	as.Nil(err)
	as.Equal(len(devices2), 1)

	as.NotEqual(devices1, devices2, "expected containers to get different devices")

	// further allocations should fail.
	newPod := makeTestPod(2, 1)
	devices1, err = testGpuManager.AllocateGPU(newPod, &newPod.Spec.Containers[0])
	as.NotNil(err, "expected gpu allocation to fail. got: %v", devices1)

	// Now terminate the original pod and observe that GPU allocation for new pod succeeds.
	podLister.activePods = podLister.activePods[:0]

	devices1, err = testGpuManager.AllocateGPU(newPod, &newPod.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devices1), 1)

	podLister.activePods = append(podLister.activePods, newPod)

	devices2, err = testGpuManager.AllocateGPU(newPod, &newPod.Spec.Containers[1])
	as.Nil(err)
	as.Equal(len(devices2), 1)

	as.NotEqual(devices1, devices2, "expected containers to get different devices")
}

func TestMultiPodGPUAllocation(t *testing.T) {
	podLister := &testActivePodsLister{}

	testGpuManager := &nvidiaGPUManager{
		activePodsLister: podLister,
		allGPUs:          sets.NewString("/dev/nvidia0", "/dev/nvidia1"),
		allocated:        newPodGPUs(),
	}

	// Expect that no devices are in use.
	gpusInUse := testGpuManager.gpusInUse()
	as := assert.New(t)
	as.Equal(len(gpusInUse.devices()), 0)

	// Allocated GPUs for a pod with two containers.
	podA := makeTestPod(1, 1)
	// Allocate for the first container.
	devicesA, err := testGpuManager.AllocateGPU(podA, &podA.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devicesA), 1)

	podLister.activePods = append(podLister.activePods, podA)

	// further allocations should fail.
	podB := makeTestPod(1, 1)
	// Allocate for the first container.
	devicesB, err := testGpuManager.AllocateGPU(podB, &podB.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devicesB), 1)
	as.NotEqual(devicesA, devicesB, "expected pods to get different devices")
}

func TestPodContainerRestart(t *testing.T) {
	podLister := &testActivePodsLister{}

	testGpuManager := &nvidiaGPUManager{
		activePodsLister: podLister,
		allGPUs:          sets.NewString("/dev/nvidia0", "/dev/nvidia1"),
		allocated:        newPodGPUs(),
		defaultDevices:   []string{"/dev/nvidia-smi"},
	}

	// Expect that no devices are in use.
	gpusInUse := testGpuManager.gpusInUse()
	as := assert.New(t)
	as.Equal(len(gpusInUse.devices()), 0)

	// Make a pod with one containers that requests two GPUs.
	podA := makeTestPod(1, 2)
	// Allocate GPUs
	devicesA, err := testGpuManager.AllocateGPU(podA, &podA.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devicesA), 3)

	podLister.activePods = append(podLister.activePods, podA)

	// further allocations should fail.
	podB := makeTestPod(1, 1)
	_, err = testGpuManager.AllocateGPU(podB, &podB.Spec.Containers[0])
	as.NotNil(err)

	// Allcate GPU for existing Pod A.
	// The same gpus must be returned.
	devicesAretry, err := testGpuManager.AllocateGPU(podA, &podA.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devicesA), 3)
	as.True(sets.NewString(devicesA...).Equal(sets.NewString(devicesAretry...)))
}

func TestGpuAllocationWithActivePods(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Time{})
	dc := libdocker.NewFakeDockerClient().WithClock(fakeClock).WithVersion("1.11.2", "1.23")

	// Make sure the GPUs in active pods match the numGpus
	podLister := &testActivePodsLister{
		activePods:     []*v1.Pod{makeTestPodWithStatuses(1, 2)},
		dockerClient:   dc,
		nvidiaGPUPaths: sets.NewString("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3").UnsortedList(),
	}

	testGpuManager := &nvidiaGPUManager{
		activePodsLister: podLister,
		allGPUs:          sets.NewString("/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3"),
		dockerClient:     dc,
		allocated:        newPodGPUs(),
	}

	// At first initiate the active pods.
	podLister.InitiateActivePods()

	// Expect that no devices are in use.
	gpusInUse := testGpuManager.gpusInUse()
	as := assert.New(t)
	as.Equal(len(gpusInUse.devices()), 2)

	// Update the allocated list of GpuManager
	testGpuManager.allocated = gpusInUse

	// Make a pod with one containers that requests one GPU.
	podA := makeTestPodWithStatuses(1, 1)
	// Allocate GPUs
	devices, err := testGpuManager.AllocateGPU(podA, &podA.Spec.Containers[0])
	as.Nil(err)
	as.Equal(len(devices), 1)
	podLister.activePods = append(podLister.activePods, podA)

	// Make a pod with one containers that requests two GPU.
	podA = makeTestPodWithStatuses(1, 2)
	// Gpu allocation will be failed as no more devices can be allocated.
	_, err = testGpuManager.AllocateGPU(podA, &podA.Spec.Containers[0])
	as.NotNil(err)
}
