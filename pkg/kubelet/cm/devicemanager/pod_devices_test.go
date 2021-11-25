/*
Copyright 2020 The Kubernetes Authors.

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

package devicemanager

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/sets"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/checkpoint"
)

func TestGetContainerDevices(t *testing.T) {
	podDevices := newPodDevices()
	resourceName1 := "domain1.com/resource1"
	podID := "pod1"
	contID := "con1"
	devices := checkpoint.DevicesPerNUMA{0: []string{"dev1"}, 1: []string{"dev1"}}

	podDevices.insert(podID, contID, resourceName1,
		devices,
		constructAllocResp(map[string]string{"/dev/r1dev1": "/dev/r1dev1", "/dev/r1dev2": "/dev/r1dev2"}, map[string]string{"/home/r1lib1": "/usr/r1lib1"}, map[string]string{}))

	resContDevices := podDevices.getContainerDevices(podID, contID)
	contDevices, ok := resContDevices[resourceName1]
	require.True(t, ok, "resource %q not present", resourceName1)

	for devId, plugInfo := range contDevices {
		nodes := plugInfo.GetTopology().GetNodes()
		require.Equal(t, len(nodes), len(devices), "Incorrect container devices: %v - %v (nodes %v)", devices, contDevices, nodes)

		for _, node := range plugInfo.GetTopology().GetNodes() {
			dev, ok := devices[node.ID]
			require.True(t, ok, "NUMA id %v doesn't exist in result", node.ID)
			require.Equal(t, devId, dev[0], "Can't find device %s in result", dev[0])
		}
	}
}

func TestResourceDeviceInstanceFilter(t *testing.T) {
	var expected string
	var cond map[string]sets.String
	var resp ResourceDeviceInstances
	devs := ResourceDeviceInstances{
		"foo": DeviceInstances{
			"dev-foo1": pluginapi.Device{
				ID: "foo1",
			},
			"dev-foo2": pluginapi.Device{
				ID: "foo2",
			},
			"dev-foo3": pluginapi.Device{
				ID: "foo3",
			},
		},
		"bar": DeviceInstances{
			"dev-bar1": pluginapi.Device{
				ID: "bar1",
			},
			"dev-bar2": pluginapi.Device{
				ID: "bar2",
			},
			"dev-bar3": pluginapi.Device{
				ID: "bar3",
			},
		},
		"baz": DeviceInstances{
			"dev-baz1": pluginapi.Device{
				ID: "baz1",
			},
			"dev-baz2": pluginapi.Device{
				ID: "baz2",
			},
			"dev-baz3": pluginapi.Device{
				ID: "baz3",
			},
		},
	}

	resp = devs.Filter(map[string]sets.String{})
	expected = `{}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.String{
		"foo": sets.NewString("dev-foo1", "dev-foo2"),
		"bar": sets.NewString("dev-bar1"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.String{
		"foo": sets.NewString("dev-foo1", "dev-foo2", "dev-foo3"),
		"bar": sets.NewString("dev-bar1", "dev-bar2", "dev-bar3"),
		"baz": sets.NewString("dev-baz1", "dev-baz2", "dev-baz3"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"},"dev-bar2":{"ID":"bar2"},"dev-bar3":{"ID":"bar3"}},"baz":{"dev-baz1":{"ID":"baz1"},"dev-baz2":{"ID":"baz2"},"dev-baz3":{"ID":"baz3"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"},"dev-foo3":{"ID":"foo3"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.String{
		"foo": sets.NewString("dev-foo1", "dev-foo2", "dev-foo3", "dev-foo4"),
		"bar": sets.NewString("dev-bar1", "dev-bar2", "dev-bar3", "dev-bar4"),
		"baz": sets.NewString("dev-baz1", "dev-baz2", "dev-baz3", "dev-bar4"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"},"dev-bar2":{"ID":"bar2"},"dev-bar3":{"ID":"bar3"}},"baz":{"dev-baz1":{"ID":"baz1"},"dev-baz2":{"ID":"baz2"},"dev-baz3":{"ID":"baz3"}},"foo":{"dev-foo1":{"ID":"foo1"},"dev-foo2":{"ID":"foo2"},"dev-foo3":{"ID":"foo3"}}}`
	expectResourceDeviceInstances(t, resp, expected)

	cond = map[string]sets.String{
		"foo": sets.NewString("dev-foo1", "dev-foo4", "dev-foo7"),
		"bar": sets.NewString("dev-bar1", "dev-bar4", "dev-bar7"),
		"baz": sets.NewString("dev-baz1", "dev-baz4", "dev-baz7"),
	}
	resp = devs.Filter(cond)
	expected = `{"bar":{"dev-bar1":{"ID":"bar1"}},"baz":{"dev-baz1":{"ID":"baz1"}},"foo":{"dev-foo1":{"ID":"foo1"}}}`
	expectResourceDeviceInstances(t, resp, expected)

}

func expectResourceDeviceInstances(t *testing.T, resp ResourceDeviceInstances, expected string) {
	// per docs in https://pkg.go.dev/encoding/json#Marshal
	// "Map values encode as JSON objects. The map's key type must either be a string, an integer type, or
	// implement encoding.TextMarshaler. The map keys are sorted [...]"
	// so this check is expected to be stable and not flaky
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("unexpected JSON marshalling error: %v", err)
	}
	got := string(data)
	if got != expected {
		t.Errorf("expected %q got %q", expected, got)
	}
}
