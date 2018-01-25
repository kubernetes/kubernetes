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
package deviceplugin

import (
	"reflect"
	"sort"
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha"
	"k8s.io/kubernetes/pkg/kubelet/container"
)

func getMounts(prefix string) []*pluginapi.Mount {
	ret := []*pluginapi.Mount{}
	for i := 0; i < 5; i++ {
		str := strconv.Itoa(i)
		mount := &pluginapi.Mount{
			ContainerPath: prefix + "containerpath" + str,
			HostPath:      prefix + "hostpath" + str,
			ReadOnly:      false,
		}
		ret = append(ret, mount)
	}
	return ret
}

func getEnvs(prefix string) map[string]string {
	ret := map[string]string{}
	for i := 0; i < 5; i++ {
		str := strconv.Itoa(i)
		key := prefix + "envkey" + str
		value := prefix + "envvalue" + str
		ret[key] = value
	}
	return ret
}

func getDeviceSpec(prefix string) []*pluginapi.DeviceSpec {
	ret := []*pluginapi.DeviceSpec{}
	for i := 0; i < 5; i++ {
		str := strconv.Itoa(i)
		device := &pluginapi.DeviceSpec{
			ContainerPath: prefix + "containerpath" + str,
			HostPath:      prefix + "hostpath" + str,
			Permissions:   prefix + "permissions" + str,
		}
		ret = append(ret, device)
	}
	return ret
}

func getDevices(prefix string) sets.String {
	ret := sets.String{}
	for i := 0; i < 5; i++ {
		ret.Insert(prefix + "devices" + strconv.Itoa(i))
	}
	return ret
}

func getPodDevices() podDevices {
	datas := []struct {
		podUID   string
		contName string
		resource string
		devices  sets.String
		resp     *pluginapi.AllocateResponse
	}{
		{
			podUID:   "UID-1",
			contName: "cont1",
			resource: "resource1",
			devices:  getDevices("first"),
			resp: &pluginapi.AllocateResponse{
				Envs:    getEnvs("first"),
				Mounts:  getMounts("first"),
				Devices: getDeviceSpec("first"),
			},
		},
		{
			podUID:   "UID-2",
			contName: "cont2",
			resource: "resource2",
			devices:  getDevices("second"),
			resp: &pluginapi.AllocateResponse{
				Envs:    getEnvs("second"),
				Mounts:  getMounts("second"),
				Devices: getDeviceSpec("second"),
			},
		},
		{
			podUID:   "UID-3",
			contName: "cont3",
			resource: "resource3",
			devices:  getDevices("third"),
			resp: &pluginapi.AllocateResponse{
				Envs:    getEnvs("third"),
				Mounts:  getMounts("third"),
				Devices: getDeviceSpec("third"),
			},
		},
	}
	podDevices := podDevices{}

	for _, data := range datas {
		podDevices.insert(data.podUID, data.contName, data.resource, data.devices, data.resp)
	}
	return podDevices

}

func TestDeviceRunContainerOptions(t *testing.T) {
	testCases := map[string]struct {
		podUID   string
		contName string
		expected *DeviceRunContainerOptions
	}{
		"get-uid-1": {
			podUID:   "UID-1",
			contName: "cont1",
			expected: &DeviceRunContainerOptions{
				Envs: []container.EnvVar{
					{Name: "firstenvkey3", Value: "firstenvvalue3"},
					{Name: "firstenvkey4", Value: "firstenvvalue4"},
					{Name: "firstenvkey0", Value: "firstenvvalue0"},
					{Name: "firstenvkey1", Value: "firstenvvalue1"},
					{Name: "firstenvkey2", Value: "firstenvvalue2"},
				},
				Mounts: []container.Mount{
					{Name: "firstcontainerpath0", ContainerPath: "firstcontainerpath0", HostPath: "firsthostpath0", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "firstcontainerpath1", ContainerPath: "firstcontainerpath1", HostPath: "firsthostpath1", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "firstcontainerpath2", ContainerPath: "firstcontainerpath2", HostPath: "firsthostpath2", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "firstcontainerpath3", ContainerPath: "firstcontainerpath3", HostPath: "firsthostpath3", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "firstcontainerpath4", ContainerPath: "firstcontainerpath4", HostPath: "firsthostpath4", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
				},
				Devices: []container.DeviceInfo{
					{PathOnHost: "firsthostpath0", PathInContainer: "firstcontainerpath0", Permissions: "firstpermissions0"},
					{PathOnHost: "firsthostpath1", PathInContainer: "firstcontainerpath1", Permissions: "firstpermissions1"},
					{PathOnHost: "firsthostpath2", PathInContainer: "firstcontainerpath2", Permissions: "firstpermissions2"},
					{PathOnHost: "firsthostpath3", PathInContainer: "firstcontainerpath3", Permissions: "firstpermissions3"},
					{PathOnHost: "firsthostpath4", PathInContainer: "firstcontainerpath4", Permissions: "firstpermissions4"},
				},
			},
		},
		"get-uid-3": {
			podUID:   "UID-3",
			contName: "cont3",
			expected: &DeviceRunContainerOptions{
				Envs: []container.EnvVar{
					{Name: "thirdenvkey3", Value: "thirdenvvalue3"},
					{Name: "thirdenvkey4", Value: "thirdenvvalue4"},
					{Name: "thirdenvkey0", Value: "thirdenvvalue0"},
					{Name: "thirdenvkey1", Value: "thirdenvvalue1"},
					{Name: "thirdenvkey2", Value: "thirdenvvalue2"},
				},
				Mounts: []container.Mount{
					{Name: "thirdcontainerpath0", ContainerPath: "thirdcontainerpath0", HostPath: "thirdhostpath0", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "thirdcontainerpath1", ContainerPath: "thirdcontainerpath1", HostPath: "thirdhostpath1", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "thirdcontainerpath2", ContainerPath: "thirdcontainerpath2", HostPath: "thirdhostpath2", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "thirdcontainerpath3", ContainerPath: "thirdcontainerpath3", HostPath: "thirdhostpath3", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
					{Name: "thirdcontainerpath4", ContainerPath: "thirdcontainerpath4", HostPath: "thirdhostpath4", ReadOnly: false, SELinuxRelabel: false, Propagation: 0},
				},
				Devices: []container.DeviceInfo{
					{PathOnHost: "thirdhostpath0", PathInContainer: "thirdcontainerpath0", Permissions: "thirdpermissions0"},
					{PathOnHost: "thirdhostpath1", PathInContainer: "thirdcontainerpath1", Permissions: "thirdpermissions1"},
					{PathOnHost: "thirdhostpath2", PathInContainer: "thirdcontainerpath2", Permissions: "thirdpermissions2"},
					{PathOnHost: "thirdhostpath3", PathInContainer: "thirdcontainerpath3", Permissions: "thirdpermissions3"},
					{PathOnHost: "thirdhostpath4", PathInContainer: "thirdcontainerpath4", Permissions: "thirdpermissions4"},
				},
			},
		},
		"get-none-wrong": {
			podUID:   "gakki",
			contName: "masami",
			expected: nil,
		},
		"get-mix-wrong": {
			podUID:   "UID-3",
			contName: "cont1",
			expected: nil,
		},
		"get-nil-wrong": {
			podUID:   "",
			contName: "",
			expected: nil,
		},
	}

	testPodDevices := getPodDevices()
	for testName, testCase := range testCases {
		actual := testPodDevices.deviceRunContainerOptions(testCase.podUID, testCase.contName)
		if actual == nil || testCase.expected == nil {
			if actual != testCase.expected {
				t.Errorf("unexpected result, test: %v", testName)
			}
			continue
		}

		if !reflect.DeepEqual(actual.Mounts, testCase.expected.Mounts) {
			t.Errorf("unexpected result, test: %v, Mounts not as expected", testName)
		}
		if !reflect.DeepEqual(actual.Devices, testCase.expected.Devices) {
			t.Errorf("unexpected result, test: %v, Devices not as expected", testName)
		}
		if !envEqual(actual.Envs, testCase.expected.Envs) {
			t.Errorf("unexpected result, test: %v, Envs not as expected", testName)
		}

	}
}

type sortInterface struct {
	//use only for sort
	envs []container.EnvVar
}

func (c sortInterface) Len() int {
	return len(c.envs)
}

func (c sortInterface) Swap(i, j int) {
	c.envs[i], c.envs[j] = c.envs[j], c.envs[i]
}

func (c sortInterface) Less(i, j int) bool {
	return c.envs[i].Name < c.envs[j].Name
}

func envEqual(envs1, envs2 []container.EnvVar) bool {
	//since Envs is non order guguaranteed, so compare them handly
	if envs1 == nil || envs2 == nil {
		if envs1 == nil && envs2 == nil {
			return true
		}
		return false
	}

	if len(envs1) != len(envs2) {
		return false
	}
	sort1 := sortInterface{
		envs: envs1,
	}
	sort2 := sortInterface{
		envs: envs2,
	}
	sort.Sort(sort1)
	sort.Sort(sort2)
	return reflect.DeepEqual(sort1.envs, sort2.envs)
}
