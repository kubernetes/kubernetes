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
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
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
	return st.MakePod().Node(host).ContainerPort(networkPorts).Obj()
}

func TestNodePorts(t *testing.T) {
	tests := []struct {
		pod                 *v1.Pod
		nodeInfo            *framework.NodeInfo
		name                string
		wantPreFilterStatus *fwk.Status
		wantFilterStatus    *fwk.Status
	}{
		{
			pod:                 &v1.Pod{},
			nodeInfo:            framework.NewNodeInfo(),
			name:                "skip filter",
			wantPreFilterStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/9090")),
			name: "other port",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			name:             "same udp port",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			name:             "same tcp port",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.2/8080")),
			name: "different host ip",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			name: "different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8000", "UDP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			name:             "second udp port conflict",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8080"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8081")),
			name:             "first tcp port conflict",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/0.0.0.0/8001"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			name:             "first tcp port conflict due to 0.0.0.0 hostIP",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/10.0.10.10/8001", "TCP/0.0.0.0/8001"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			name:             "TCP hostPort conflict due to 0.0.0.0 hostIP",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name:             "second tcp port conflict to 0.0.0.0 hostIP",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name: "second different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001", "UDP/0.0.0.0/8001")),
			name:             "UDP hostPort conflict due to 0.0.0.0 hostIP",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
		{
			pod: st.MakePod().
				InitContainerPort(false /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					},
				}).Obj(),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name:                "non-sidecar initContainer using hostPort",
			wantPreFilterStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			pod: st.MakePod().
				InitContainerPort(true /* sidecar */, []v1.ContainerPort{
					{
						ContainerPort: 8001,
						HostPort:      8001,
						Protocol:      v1.ProtocolTCP,
					},
				}).Obj(),
			nodeInfo: framework.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			name:             "TCP hostPort conflict from sidecar initContainer",
			wantFilterStatus: fwk.NewStatus(fwk.Unschedulable, ErrReason),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			cycleState := framework.NewCycleState()
			_, preFilterStatus := p.(framework.PreFilterPlugin).PreFilter(ctx, cycleState, test.pod, nil)
			if diff := cmp.Diff(test.wantPreFilterStatus, preFilterStatus); diff != "" {
				t.Errorf("preFilter status does not match (-want,+got): %s", diff)
			}
			if preFilterStatus.IsSkip() {
				return
			}
			if !preFilterStatus.IsSuccess() {
				t.Errorf("prefilter failed with status: %v", preFilterStatus)
			}
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, test.pod, test.nodeInfo)
			if diff := cmp.Diff(test.wantFilterStatus, gotStatus); diff != "" {
				t.Errorf("filter status does not match (-want, +got): %s", diff)
			}
		})
	}
}

func TestPreFilterDisabled(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	pod := &v1.Pod{}
	nodeInfo := framework.NewNodeInfo()
	node := v1.Node{}
	nodeInfo.SetNode(&node)
	p, err := New(ctx, nil, nil, feature.Features{})
	if err != nil {
		t.Fatalf("creating plugin: %v", err)
	}
	cycleState := framework.NewCycleState()
	gotStatus := p.(framework.FilterPlugin).Filter(ctx, cycleState, pod, nodeInfo)
	wantStatus := fwk.AsStatus(fwk.ErrNotFound)
	if diff := cmp.Diff(wantStatus, gotStatus); diff != "" {
		t.Errorf("status does not match (-want,+got):\n%s", diff)
	}
}

func Test_isSchedulableAfterPodDeleted(t *testing.T) {
	podWithHostPort := st.MakePod().HostPort(8080)

	testcases := map[string]struct {
		pod          *v1.Pod
		oldObj       interface{}
		expectedHint fwk.QueueingHint
		expectedErr  bool
	}{
		"backoff-wrong-old-object": {
			pod:          podWithHostPort.Obj(),
			oldObj:       "not-a-pod",
			expectedHint: fwk.Queue,
			expectedErr:  true,
		},
		"skip-queue-on-unscheduled": {
			pod:          podWithHostPort.Obj(),
			oldObj:       st.MakePod().Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"skip-queue-on-non-hostport": {
			pod:          podWithHostPort.Obj(),
			oldObj:       st.MakePod().Node("fake-node").Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"skip-queue-on-unrelated-hostport": {
			pod:          podWithHostPort.Obj(),
			oldObj:       st.MakePod().Node("fake-node").HostPort(8081).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-released-hostport": {
			pod:          podWithHostPort.Obj(),
			oldObj:       st.MakePod().Node("fake-node").HostPort(8080).Obj(),
			expectedHint: fwk.Queue,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("Creating plugin: %v", err)
			}
			actualHint, err := p.(*NodePorts).isSchedulableAfterPodDeleted(logger, tc.pod, tc.oldObj, nil)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}
