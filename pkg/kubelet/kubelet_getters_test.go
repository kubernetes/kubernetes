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

package kubelet

import (
	"fmt"
	"path/filepath"
	"testing"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"net"
)

func TestKubeletDirs(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory

	var exp, got string

	got = kubelet.getPodsDir()
	exp = filepath.Join(root, "pods")
	assert.Equal(t, exp, got)

	got = kubelet.getPluginsDir()
	exp = filepath.Join(root, "plugins")
	assert.Equal(t, exp, got)

	got = kubelet.getPluginsRegistrationDir()
	exp = filepath.Join(root, "plugins_registry")
	assert.Equal(t, exp, got)

	got = kubelet.getPluginDir("foobar")
	exp = filepath.Join(root, "plugins/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.GetPodDir("abc123")
	exp = filepath.Join(root, "pods/abc123")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumesDir("abc123")
	exp = filepath.Join(root, "pods/abc123/volumes")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumeDir("abc123", "plugin", "foobar")
	exp = filepath.Join(root, "pods/abc123/volumes/plugin/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumeDevicesDir("abc123")
	exp = filepath.Join(root, "pods/abc123/volumeDevices")
	assert.Equal(t, exp, got)

	got = kubelet.getPodVolumeDeviceDir("abc123", "plugin")
	exp = filepath.Join(root, "pods/abc123/volumeDevices/plugin")
	assert.Equal(t, exp, got)

	got = kubelet.getPodPluginsDir("abc123")
	exp = filepath.Join(root, "pods/abc123/plugins")
	assert.Equal(t, exp, got)

	got = kubelet.getPodPluginDir("abc123", "foobar")
	exp = filepath.Join(root, "pods/abc123/plugins/foobar")
	assert.Equal(t, exp, got)

	got = kubelet.getVolumeDevicePluginsDir()
	exp = filepath.Join(root, "plugins")
	assert.Equal(t, exp, got)

	got = kubelet.getVolumeDevicePluginDir("foobar")
	exp = filepath.Join(root, "plugins", "foobar", "volumeDevices")
	assert.Equal(t, exp, got)

	got = kubelet.getPodContainerDir("abc123", "def456")
	exp = filepath.Join(root, "pods/abc123/containers/def456")
	assert.Equal(t, exp, got)

	got = kubelet.getPodResourcesDir()
	exp = filepath.Join(root, "pod-resources")
	assert.Equal(t, exp, got)
}

func TestGetPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet, expectedPods, expectedRunningPods := newKubeletAndPods(testKubelet)

	testCases := []struct {
		name       string
		assertFunc func(t *testing.T)
	}{
		{
			name: "GetPods",
			assertFunc: func(t *testing.T) {
				got := kubelet.GetPods()
				exp := expectedPods
				assert.Equal(t, len(exp), len(got))
			},
		},
		{
			name: "GetRunningPods",
			assertFunc: func(t *testing.T) {
				got, err := kubelet.GetRunningPods()
				exp := expectedRunningPods
				assert.NoError(t, err, "unexpected GetRunningPods error")
				assert.Equal(t, len(exp), len(got))
			},
		},
		{
			name: "GetPodByFullName",
			assertFunc: func(t *testing.T) {
				got, ok := kubelet.GetPodByFullName("pod0_")
				exp := expectedPods[0]
				assert.Equal(t, true, ok)
				assert.Equal(t, exp, got)
			},
		},
		{
			name: "GetPodByName",
			assertFunc: func(t *testing.T) {
				got, ok := kubelet.GetPodByName("", "pod1")
				exp := expectedPods[1]
				assert.Equal(t, true, ok)
				assert.Equal(t, exp, got)
			},
		},
		{
			name: "GetPodByCgroupfs",
			assertFunc: func(t *testing.T) {
				// TODO: Currently FakePodContainerManager.IsPodCgroup only returns false, types.UID("").
				// Improving FakePodContainerManager.IsPodCgroup makes this more meaningful.
				got, ok := kubelet.GetPodByCgroupfs("10001")
				assert.Equal(t, false, ok)
				assert.Equal(t, (*v1.Pod)(nil), got)
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.assertFunc(t)
		})
	}
}

func TestGetHost(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	got := kubelet.GetHostname()
	exp := "127.0.0.1"
	assert.Equal(t, exp, got, "GetHostname")

	gotIPs, err := kubelet.GetHostIPs()
	assert.NoError(t, err, "GetHostIPs")
	expIPs := []net.IP{
		net.ParseIP("127.0.0.1"),
	}
	assert.Equal(t, expIPs, gotIPs, "GetHostIPs")

	kubelet.nodeName = "127.0.0.2"
	_, err = kubelet.GetHostIPs()
	expErr := fmt.Errorf("cannot get node: Node with name: 127.0.0.2 does not exist")
	assert.ErrorAs(t, expErr, &err, "unexpected GetHostIPs error")
}

func TestGetCadvisor(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet

	got, err := kubelet.GetVersionInfo()
	assert.NoError(t, err, "GetVersionInfo error")
	exp := &cadvisorapiv1.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.13.1",
	}
	assert.Equal(t, exp, got, "GetVersionInfo")

	// TODO: Currently FakecCadvisor.GetRequestedContainersInfo only returns
	// map[string]*cadvisorapi.ContainerInfo{}, nil.
	// Improving FakecCadvisor.GetRequestedContainersInfo makes this more meaningful.
	containerInfo, err := kubelet.GetRequestedContainersInfo("test", cadvisorv2.RequestOptions{})
	expContainerInfo := map[string]*cadvisorapiv1.ContainerInfo{}
	assert.NoError(t, err, "unexpected GetRequestedContainersInfo error")
	assert.Equal(t, expContainerInfo, containerInfo, "GetRequestedContainersInfo")
}

func newKubeletAndPods(testKubelet *TestKubelet) (*Kubelet, []*v1.Pod, []*containertest.FakePod) {
	kubelet := testKubelet.kubelet

	expectedPods := newTestPods(2)
	expectedPods[0].ObjectMeta.Annotations = make(map[string]string)
	expectedPods[0].ObjectMeta.Annotations["kubernetes.io/config.source"] = "file"

	kubelet.podManager.SetPods(expectedPods)
	kubelet.statusManager.SetPodStatus(expectedPods[0], v1.PodStatus{Phase: v1.PodSucceeded})

	return kubelet, expectedPods, testKubelet.fakeRuntime.PodList
}
