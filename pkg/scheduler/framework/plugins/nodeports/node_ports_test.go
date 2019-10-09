/*
Copyright 2019 The Kubernetes Authors.

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

package nodeports

import (
	"reflect"
	"strconv"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func newPod(host string, hostPortInfos ...string) *v1.Pod {
	networkPorts := []v1.ContainerPort{}
	for _, portInfo := range hostPortInfos {
		splited := strings.Split(portInfo, "/")
		hostPort, _ := strconv.Atoi(splited[2])

		networkPorts = append(networkPorts, v1.ContainerPort{
			HostIP:   splited[1],
			HostPort: int32(hostPort),
			Protocol: v1.Protocol(splited[0]),
		})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			NodeName: host,
			Containers: []v1.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestNodePorts(t *testing.T) {
	tests := []struct {
		pod        *v1.Pod
		nodeInfo   *schedulernodeinfo.NodeInfo
		name       string
		wantStatus *framework.Status
	}{
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulernodeinfo.NewNodeInfo(),
			name:     "nothing running",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/9090")),
			name: "other port",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			name:       "same udp port",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			name:       "same tcp port",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.2/8080")),
			name: "different host ip",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			name: "different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8000", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			name:       "second udp port conflict",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8081")),
			name:       "first tcp port conflict",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			name:       "first tcp port conflict due to 0.0.0.0 hostIP",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/10.0.10.10/8001", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			name:       "TCP hostPort conflict due to 0.0.0.0 hostIP",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name:       "second tcp port conflict to 0.0.0.0 hostIP",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name: "second different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001", "UDP/0.0.0.0/8001")),
			name:       "UDP hostPort conflict due to 0.0.0.0 hostIP",
			wantStatus: framework.NewStatus(framework.Unschedulable, predicates.ErrPodNotFitsHostPorts.GetReason()),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, _ := New(nil, nil)
			gotStatus := p.(framework.FilterPlugin).Filter(nil, test.pod, test.nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}
