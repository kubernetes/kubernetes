/*
Copyright 2015 The Kubernetes Authors.

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

package cadvisor

import (
	"errors"
	"fmt"
	"net"
	"syscall"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapi2 "github.com/google/cadvisor/info/v2"
	"github.com/vishvananda/netlink"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

const (
	// CrioSocket is the path to the CRI-O socket.
	// Please keep this in sync with the one in:
	// github.com/google/cadvisor/tree/master/container/crio/client.go
	CrioSocket = "/var/run/crio/crio.sock"
)

// CapacityFromMachineInfo returns the capacity of the resources from the machine info.
func CapacityFromMachineInfo(info *cadvisorapi.MachineInfo) v1.ResourceList {
	c := v1.ResourceList{
		v1.ResourceCPU: *resource.NewMilliQuantity(
			int64(info.NumCores*1000),
			resource.DecimalSI),
		v1.ResourceMemory: *resource.NewQuantity(
			int64(info.MemoryCapacity),
			resource.BinarySI),
	}

	// if huge pages are enabled, we report them as a schedulable resource on the node
	for _, hugepagesInfo := range info.HugePages {
		pageSizeBytes := int64(hugepagesInfo.PageSize * 1024)
		hugePagesBytes := pageSizeBytes * int64(hugepagesInfo.NumPages)
		pageSizeQuantity := resource.NewQuantity(pageSizeBytes, resource.BinarySI)
		c[v1helper.HugePageResourceName(*pageSizeQuantity)] = *resource.NewQuantity(hugePagesBytes, resource.BinarySI)
	}

	// if network io are enabled, we report them as two schedulable resources on the node
	var iface *net.Interface
	var err error
	if iface, err = GetDefaultGatewayInterface(); err != nil {
		return c
	}

	for _, netInfo := range info.NetworkDevices {
		// fmt.Println("iface name", iface.Name, "cadvisor interface name", netInfo.Name)
		if netInfo.Name != iface.Name {
			continue
		}
		networkSpeed := int64(netInfo.Speed)
		c[v1.ResourceName(fmt.Sprintf("%s-ingress", v1.ResourceNetworkIOPrefix))] = *resource.NewQuantity(networkSpeed, resource.BinarySI)
		c[v1.ResourceName(fmt.Sprintf("%s-egress", v1.ResourceNetworkIOPrefix))] = *resource.NewQuantity(networkSpeed, resource.BinarySI)
	}

	return c
}

// find the default gateway's interface from the route table
func GetDefaultGatewayInterface() (*net.Interface, error) {
	routes, err := netlink.RouteList(nil, syscall.AF_INET)
	if err != nil {
		return nil, err
	}

	for _, route := range routes {
		if route.Dst == nil || route.Dst.String() == "0.0.0.0/0" {
			if route.LinkIndex <= 0 {
				return nil, errors.New("found default route but could not determine interface")
			}
			return net.InterfaceByIndex(route.LinkIndex)
		}
	}

	return nil, errors.New("unable to find default route")
}

// EphemeralStorageCapacityFromFsInfo returns the capacity of the ephemeral storage from the FsInfo.
func EphemeralStorageCapacityFromFsInfo(info cadvisorapi2.FsInfo) v1.ResourceList {
	c := v1.ResourceList{
		v1.ResourceEphemeralStorage: *resource.NewQuantity(
			int64(info.Capacity),
			resource.BinarySI),
	}
	return c
}

// UsingLegacyCadvisorStats returns true if container stats are provided by cadvisor instead of through the CRI.
// CRI integrations should get container metrics via CRI.
// TODO: cri-o relies on cadvisor as a temporary workaround. The code should
// be removed. Related issue:
// https://github.com/kubernetes/kubernetes/issues/51798
func UsingLegacyCadvisorStats(runtimeEndpoint string) bool {
	return runtimeEndpoint == CrioSocket || runtimeEndpoint == "unix://"+CrioSocket
}
