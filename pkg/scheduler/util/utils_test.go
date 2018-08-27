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

package util

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

// TestGetPodPriority tests GetPodPriority function.
func TestGetPodPriority(t *testing.T) {
	p := int32(20)
	tests := []struct {
		name             string
		pod              *v1.Pod
		expectedPriority int32
	}{
		{
			name: "no priority pod resolves to static default priority",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
				},
			},
			expectedPriority: scheduling.DefaultPriorityWhenNoDefaultClassExists,
		},
		{
			name: "pod with priority resolves correctly",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
					Priority: &p,
				},
			},
			expectedPriority: p,
		},
	}
	for _, test := range tests {
		if GetPodPriority(test.pod) != test.expectedPriority {
			t.Errorf("expected pod priority: %v, got %v", test.expectedPriority, GetPodPriority(test.pod))
		}

	}
}

// TestSortableList tests SortableList by storing pods in the list and sorting
// them by their priority.
func TestSortableList(t *testing.T) {
	higherPriority := func(pod1, pod2 interface{}) bool {
		return GetPodPriority(pod1.(*v1.Pod)) > GetPodPriority(pod2.(*v1.Pod))
	}
	podList := SortableList{CompFunc: higherPriority}
	// Add a few Pods with different priorities from lowest to highest priority.
	for i := 0; i < 10; i++ {
		var p = int32(i)
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container",
						Image: "image",
					},
				},
				Priority: &p,
			},
		}
		podList.Items = append(podList.Items, pod)
	}
	podList.Sort()
	if len(podList.Items) != 10 {
		t.Errorf("expected length of list was 10, got: %v", len(podList.Items))
	}
	var prevPriority = int32(10)
	for _, p := range podList.Items {
		if *p.(*v1.Pod).Spec.Priority >= prevPriority {
			t.Errorf("Pods are not soreted. Current pod pririty is %v, while previous one was %v.", *p.(*v1.Pod).Spec.Priority, prevPriority)
		}
	}
}

type hostPortInfoParam struct {
	protocol, ip string
	port         int32
}

func TestHostPortInfo_AddRemove(t *testing.T) {
	tests := []struct {
		desc    string
		added   []hostPortInfoParam
		removed []hostPortInfoParam
		length  int
	}{
		{
			desc: "normal add case",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				// this might not make sense in real case, but the struct doesn't forbid it.
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
				{"TCP", "0.0.0.0", 0},
				{"TCP", "0.0.0.0", -1},
			},
			length: 8,
		},
		{
			desc: "empty ip and protocol add should work",
			added: []hostPortInfoParam{
				{"", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"", "127.0.0.1", 81},
				{"", "127.0.0.1", 82},
				{"", "", 79},
				{"UDP", "", 80},
				{"", "", 81},
				{"", "", 82},
				{"", "", 0},
				{"", "", -1},
			},
			length: 8,
		},
		{
			desc: "normal remove case",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			removed: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			length: 0,
		},
		{
			desc: "empty ip and protocol remove should work",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			removed: []hostPortInfoParam{
				{"", "127.0.0.1", 79},
				{"", "127.0.0.1", 81},
				{"", "127.0.0.1", 82},
				{"UDP", "127.0.0.1", 80},
				{"", "", 79},
				{"", "", 81},
				{"", "", 82},
				{"UDP", "", 80},
			},
			length: 0,
		},
	}

	for _, test := range tests {
		hp := make(HostPortInfo)
		for _, param := range test.added {
			hp.Add(param.ip, param.protocol, param.port)
		}
		for _, param := range test.removed {
			hp.Remove(param.ip, param.protocol, param.port)
		}
		if hp.Len() != test.length {
			t.Errorf("%v failed: expect length %d; got %d", test.desc, test.length, hp.Len())
			t.Error(hp)
		}
	}
}

func TestHostPortInfo_Check(t *testing.T) {
	tests := []struct {
		desc   string
		added  []hostPortInfoParam
		check  hostPortInfoParam
		expect bool
	}{
		{
			desc: "empty check should check 0.0.0.0 and TCP",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 81},
			expect: false,
		},
		{
			desc: "empty check should check 0.0.0.0 and TCP (conflicted)",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 80},
			expect: true,
		},
		{
			desc: "empty port check should pass",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 0},
			expect: false,
		},
		{
			desc: "0.0.0.0 should check all registered IPs",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: true,
		},
		{
			desc: "0.0.0.0 with different protocol should be allowed",
			added: []hostPortInfoParam{
				{"UDP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: false,
		},
		{
			desc: "0.0.0.0 with different port should be allowed",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: false,
		},
		{
			desc: "normal ip should check all registered 0.0.0.0",
			added: []hostPortInfoParam{
				{"TCP", "0.0.0.0", 80},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: true,
		},
		{
			desc: "normal ip with different port/protocol should be allowed (0.0.0.0)",
			added: []hostPortInfoParam{
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: false,
		},
		{
			desc: "normal ip with different port/protocol should be allowed",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: false,
		},
	}

	for _, test := range tests {
		hp := make(HostPortInfo)
		for _, param := range test.added {
			hp.Add(param.ip, param.protocol, param.port)
		}
		if hp.CheckConflict(test.check.ip, test.check.protocol, test.check.port) != test.expect {
			t.Errorf("%v failed, expected %t; got %t", test.desc, test.expect, !test.expect)
		}
	}
}

func TestGetContainerPorts(t *testing.T) {
	tests := []struct {
		pod1     *v1.Pod
		pod2     *v1.Pod
		expected []*v1.ContainerPort
	}{
		{
			pod1: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 8001,
									Protocol:      v1.ProtocolTCP,
								},
								{
									ContainerPort: 8002,
									Protocol:      v1.ProtocolTCP,
								},
							},
						},
						{
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 8003,
									Protocol:      v1.ProtocolTCP,
								},
								{
									ContainerPort: 8004,
									Protocol:      v1.ProtocolTCP,
								},
							},
						},
					},
				},
			},
			pod2: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 8011,
									Protocol:      v1.ProtocolTCP,
								},
								{
									ContainerPort: 8012,
									Protocol:      v1.ProtocolTCP,
								},
							},
						},
						{
							Ports: []v1.ContainerPort{
								{
									ContainerPort: 8013,
									Protocol:      v1.ProtocolTCP,
								},
								{
									ContainerPort: 8014,
									Protocol:      v1.ProtocolTCP,
								},
							},
						},
					},
				},
			},
			expected: []*v1.ContainerPort{
				{
					ContainerPort: 8001,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8002,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8003,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8004,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8011,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8012,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8013,
					Protocol:      v1.ProtocolTCP,
				},
				{
					ContainerPort: 8014,
					Protocol:      v1.ProtocolTCP,
				},
			},
		},
	}

	for _, test := range tests {
		result := GetContainerPorts(test.pod1, test.pod2)
		if !reflect.DeepEqual(test.expected, result) {
			t.Errorf("Got different result than expected.\nDifference detected on:\n%s", diff.ObjectGoPrintSideBySide(test.expected, result))
		}
	}
}
