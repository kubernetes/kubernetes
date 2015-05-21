/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package predicates

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm"
)

type FakeNodeInfo api.Node

func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*api.Node, error) {
	node := api.Node(n)
	return &node, nil
}

type FakeNodeListInfo []api.Node

func (nodes FakeNodeListInfo) GetNodeInfo(nodeName string) (*api.Node, error) {
	for _, node := range nodes {
		if node.Name == nodeName {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("Unable to find node: %s", nodeName)
}

func makeResources(milliCPU int64, memory int64, pods int64) api.NodeResources {
	return api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
			api.ResourcePods:   *resource.NewQuantity(pods, resource.DecimalSI),
		},
	}
}

func newResourcePod(usage ...resourceRequest) *api.Pod {
	containers := []api.Container{}
	for _, req := range usage {
		containers = append(containers, api.Container{
			Resources: api.ResourceRequirements{
				Limits: api.ResourceList{
					api.ResourceCPU:    *resource.NewMilliQuantity(req.milliCPU, resource.DecimalSI),
					api.ResourceMemory: *resource.NewQuantity(req.memory, resource.BinarySI),
				},
			},
		})
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
}

func TestPodFitsResources(t *testing.T) {

	enoughPodsTests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		fits         bool
		test         string
	}{
		{
			pod: &api.Pod{},
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 10, memory: 20}),
			},
			fits: true,
			test: "no resources requested always fits",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 1, memory: 1}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 10, memory: 20}),
			},
			fits: false,
			test: "too many resources fails",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 1, memory: 1}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 5, memory: 5}),
			},
			fits: true,
			test: "both resources fit",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 1, memory: 2}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 5, memory: 19}),
			},
			fits: false,
			test: "one resources fits",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 5, memory: 1}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 5, memory: 19}),
			},
			fits: true,
			test: "equal edge case",
		},
	}

	for _, test := range enoughPodsTests {
		node := api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 32).Capacity}}

		fit := ResourceFit{FakeNodeInfo(node)}
		fits, err := fit.PodFitsResources(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}

	notEnoughPodsTests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		fits         bool
		test         string
	}{
		{
			pod: &api.Pod{},
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 10, memory: 20}),
			},
			fits: false,
			test: "even without specified resources predicate fails when there's no available ips",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 1, memory: 1}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 5, memory: 5}),
			},
			fits: false,
			test: "even if both resources fit predicate fails when there's no available ips",
		},
		{
			pod: newResourcePod(resourceRequest{milliCPU: 5, memory: 1}),
			existingPods: []*api.Pod{
				newResourcePod(resourceRequest{milliCPU: 5, memory: 19}),
			},
			fits: false,
			test: "even for equal edge case predicate fails when there's no available ips",
		},
	}
	for _, test := range notEnoughPodsTests {
		node := api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 1).Capacity}}

		fit := ResourceFit{FakeNodeInfo(node)}
		fits, err := fit.PodFitsResources(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestPodFitsHost(t *testing.T) {
	tests := []struct {
		pod  *api.Pod
		node string
		fits bool
		test string
	}{
		{
			pod:  &api.Pod{},
			node: "foo",
			fits: true,
			test: "no host specified",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					Host: "foo",
				},
			},
			node: "foo",
			fits: true,
			test: "host matches",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					Host: "bar",
				},
			},
			node: "foo",
			fits: false,
			test: "host doesn't match",
		},
	}

	for _, test := range tests {
		result, err := PodFitsHost(test.pod, []*api.Pod{}, test.node)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != test.fits {
			t.Errorf("unexpected difference for %s: got: %v expected %v", test.test, test.fits, result)
		}
	}
}

func newPod(host string, hostPorts ...int) *api.Pod {
	networkPorts := []api.ContainerPort{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, api.ContainerPort{HostPort: port})
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Host: host,
			Containers: []api.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestPodFitsPorts(t *testing.T) {
	tests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		fits         bool
		test         string
	}{
		{
			pod:          &api.Pod{},
			existingPods: []*api.Pod{},
			fits:         true,
			test:         "nothing running",
		},
		{
			pod: newPod("m1", 8080),
			existingPods: []*api.Pod{
				newPod("m1", 9090),
			},
			fits: true,
			test: "other port",
		},
		{
			pod: newPod("m1", 8080),
			existingPods: []*api.Pod{
				newPod("m1", 8080),
			},
			fits: false,
			test: "same port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			existingPods: []*api.Pod{
				newPod("m1", 8080),
			},
			fits: false,
			test: "second port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			existingPods: []*api.Pod{
				newPod("m1", 8001, 8080),
			},
			fits: false,
			test: "second port",
		},
	}
	for _, test := range tests {
		fits, err := PodFitsPorts(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.fits != fits {
			t.Errorf("%s: expected %v, saw %v", test.test, test.fits, fits)
		}
	}
}

func TestGetUsedPorts(t *testing.T) {
	tests := []struct {
		pods []*api.Pod

		ports map[int]bool
	}{
		{
			[]*api.Pod{
				newPod("m1", 9090),
			},
			map[int]bool{9090: true},
		},
		{
			[]*api.Pod{
				newPod("m1", 9090),
				newPod("m1", 9091),
			},
			map[int]bool{9090: true, 9091: true},
		},
		{
			[]*api.Pod{
				newPod("m1", 9090),
				newPod("m2", 9091),
			},
			map[int]bool{9090: true, 9091: true},
		},
	}

	for _, test := range tests {
		ports := getUsedPorts(test.pods...)
		if !reflect.DeepEqual(test.ports, ports) {
			t.Errorf("expect %v, got %v", test.ports, ports)
		}
	}
}

func TestDiskConflicts(t *testing.T) {
	volState := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "foo",
					},
				},
			},
		},
	}
	volState2 := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		isOk         bool
		test         string
	}{
		{&api.Pod{}, []*api.Pod{}, true, "nothing"},
		{&api.Pod{}, []*api.Pod{{Spec: volState}}, true, "one state"},
		{&api.Pod{Spec: volState}, []*api.Pod{{Spec: volState}}, false, "same state"},
		{&api.Pod{Spec: volState2}, []*api.Pod{{Spec: volState}}, true, "different state"},
	}

	for _, test := range tests {
		ok, err := NoDiskConflict(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if test.isOk && !ok {
			t.Errorf("expected ok, got none.  %v %v %s", test.pod, test.existingPods, test.test)
		}
		if !test.isOk && ok {
			t.Errorf("expected no ok, got one.  %v %v %s", test.pod, test.existingPods, test.test)
		}
	}
}

func TestAWSDiskConflicts(t *testing.T) {
	volState := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: "foo",
					},
				},
			},
		},
	}
	volState2 := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		isOk         bool
		test         string
	}{
		{&api.Pod{}, []*api.Pod{}, true, "nothing"},
		{&api.Pod{}, []*api.Pod{{Spec: volState}}, true, "one state"},
		{&api.Pod{Spec: volState}, []*api.Pod{{Spec: volState}}, false, "same state"},
		{&api.Pod{Spec: volState2}, []*api.Pod{{Spec: volState}}, true, "different state"},
	}

	for _, test := range tests {
		ok, err := NoDiskConflict(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if test.isOk && !ok {
			t.Errorf("expected ok, got none.  %v %v %s", test.pod, test.existingPods, test.test)
		}
		if !test.isOk && ok {
			t.Errorf("expected no ok, got one.  %v %v %s", test.pod, test.existingPods, test.test)
		}
	}
}

func TestPodFitsSelector(t *testing.T) {
	tests := []struct {
		pod    *api.Pod
		labels map[string]string
		fits   bool
		test   string
	}{
		{
			pod:  &api.Pod{},
			fits: true,
			test: "no selector",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			fits: false,
			test: "missing labels",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "same labels",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
				"baz": "blah",
			},
			fits: true,
			test: "node labels are superset",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "node labels are subset",
		},
	}
	for _, test := range tests {
		node := api.Node{ObjectMeta: api.ObjectMeta{Labels: test.labels}}

		fit := NodeSelector{FakeNodeInfo(node)}
		fits, err := fit.PodSelectorMatches(test.pod, []*api.Pod{}, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestNodeLabelPresence(t *testing.T) {
	label := map[string]string{"foo": "bar", "bar": "foo"}
	tests := []struct {
		pod          *api.Pod
		existingPods []*api.Pod
		labels       []string
		presence     bool
		fits         bool
		test         string
	}{
		{
			labels:   []string{"baz"},
			presence: true,
			fits:     false,
			test:     "label does not match, presence true",
		},
		{
			labels:   []string{"baz"},
			presence: false,
			fits:     true,
			test:     "label does not match, presence false",
		},
		{
			labels:   []string{"foo", "baz"},
			presence: true,
			fits:     false,
			test:     "one label matches, presence true",
		},
		{
			labels:   []string{"foo", "baz"},
			presence: false,
			fits:     false,
			test:     "one label matches, presence false",
		},
		{
			labels:   []string{"foo", "bar"},
			presence: true,
			fits:     true,
			test:     "all labels match, presence true",
		},
		{
			labels:   []string{"foo", "bar"},
			presence: false,
			fits:     false,
			test:     "all labels match, presence false",
		},
	}
	for _, test := range tests {
		node := api.Node{ObjectMeta: api.ObjectMeta{Labels: label}}
		labelChecker := NodeLabelChecker{FakeNodeInfo(node), test.labels, test.presence}
		fits, err := labelChecker.CheckNodeLabelPresence(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestServiceAffinity(t *testing.T) {
	selector := map[string]string{"foo": "bar"}
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	labels2 := map[string]string{
		"region": "r1",
		"zone":   "z12",
	}
	labels3 := map[string]string{
		"region": "r2",
		"zone":   "z21",
	}
	labels4 := map[string]string{
		"region": "r2",
		"zone":   "z22",
	}
	node1 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: labels1}}
	node2 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: labels2}}
	node3 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: labels3}}
	node4 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine4", Labels: labels4}}
	node5 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine5", Labels: labels4}}
	tests := []struct {
		pod      *api.Pod
		pods     []*api.Pod
		services []api.Service
		node     string
		labels   []string
		fits     bool
		test     string
	}{
		{
			pod:    new(api.Pod),
			node:   "machine1",
			fits:   true,
			labels: []string{"region"},
			test:   "nothing scheduled",
		},
		{
			pod:    &api.Pod{Spec: api.PodSpec{NodeSelector: map[string]string{"region": "r1"}}},
			node:   "machine1",
			fits:   true,
			labels: []string{"region"},
			test:   "pod with region label match",
		},
		{
			pod:    &api.Pod{Spec: api.PodSpec{NodeSelector: map[string]string{"region": "r2"}}},
			node:   "machine1",
			fits:   false,
			labels: []string{"region"},
			test:   "pod with region label mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on same minion",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine2"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on different minion, region match",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service pod on different minion, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns2"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service in different namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns2"}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "pod in different namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service and pod in same namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine2"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     "machine1",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region", "zone"},
			test:     "service pod on different minion, multiple labels, not all match",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{Host: "machine5"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     "machine4",
			services: []api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region", "zone"},
			test:     "service pod on different minion, multiple labels, all match",
		},
	}

	for _, test := range tests {
		nodes := []api.Node{node1, node2, node3, node4, node5}
		serviceAffinity := ServiceAffinity{algorithm.FakePodLister(test.pods), algorithm.FakeServiceLister(test.services), FakeNodeListInfo(nodes), test.labels}
		fits, err := serviceAffinity.CheckServiceAffinity(test.pod, []*api.Pod{}, test.node)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}
