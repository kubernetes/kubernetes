/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package podtask

import (
	"testing"

	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/pkg/api"
)

func TestDefaultHostPortMatching(t *testing.T) {
	t.Parallel()
	task, _ := fakePodTask("foo")
	pod := &task.Pod

	offer := &mesos.Offer{
		Resources: []*mesos.Resource{
			rangeResource("ports", []uint64{1, 1}),
		},
	}
	mapping, err := defaultHostPortMapping(task, offer)
	if err != nil {
		t.Fatal(err)
	}
	if len(mapping) > 0 {
		t.Fatalf("Found mappings for a pod without ports: %v", pod)
	}

	//--
	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 123,
			}, {
				HostPort: 123,
			}},
		}},
	}
	task, err = New(api.NewDefaultContext(), "", pod)
	if err != nil {
		t.Fatal(err)
	}
	_, err = defaultHostPortMapping(task, offer)
	if err, _ := err.(*DuplicateHostPortError); err == nil {
		t.Fatal("Expected duplicate port error")
	} else if err.m1.OfferPort != 123 {
		t.Fatal("Expected duplicate host port 123")
	}
}

func TestWildcardHostPortMatching(t *testing.T) {
	t.Parallel()
	task, _ := fakePodTask("foo")
	pod := &task.Pod

	offer := &mesos.Offer{}
	mapping, err := wildcardHostPortMapping(task, offer)
	if err != nil {
		t.Fatal(err)
	}
	if len(mapping) > 0 {
		t.Fatalf("Found mappings for an empty offer and a pod without ports: %v", pod)
	}

	//--
	offer = &mesos.Offer{
		Resources: []*mesos.Resource{
			rangeResource("ports", []uint64{1, 1}),
		},
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err != nil {
		t.Fatal(err)
	}
	if len(mapping) > 0 {
		t.Fatalf("Found mappings for a pod without ports: %v", pod)
	}

	//--
	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 123,
			}},
		}},
	}
	task, err = New(api.NewDefaultContext(), "", pod)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err == nil {
		t.Fatalf("expected error instead of mappings: %#v", mapping)
	} else if err, _ := err.(*PortAllocationError); err == nil {
		t.Fatal("Expected port allocation error")
	} else if !(len(err.Ports) == 1 && err.Ports[0] == 123) {
		t.Fatal("Expected port allocation error for host port 123")
	}

	//--
	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 0,
			}, {
				HostPort: 123,
			}},
		}},
	}
	task, err = New(api.NewDefaultContext(), "", pod)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err, _ := err.(*PortAllocationError); err == nil {
		t.Fatal("Expected port allocation error")
	} else if !(len(err.Ports) == 1 && err.Ports[0] == 123) {
		t.Fatal("Expected port allocation error for host port 123")
	}

	//--
	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 0,
			}, {
				HostPort: 1,
			}},
		}},
	}
	task, err = New(api.NewDefaultContext(), "", pod)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err, _ := err.(*PortAllocationError); err == nil {
		t.Fatal("Expected port allocation error")
	} else if len(err.Ports) != 0 {
		t.Fatal("Expected port allocation error for wildcard port")
	}

	//--
	offer = &mesos.Offer{
		Resources: []*mesos.Resource{
			rangeResource("ports", []uint64{1, 2}),
		},
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err != nil {
		t.Fatal(err)
	} else if len(mapping) != 2 {
		t.Fatal("Expected both ports allocated")
	}
	valid := 0
	for _, entry := range mapping {
		if entry.ContainerIdx == 0 && entry.PortIdx == 0 && entry.OfferPort == 2 {
			valid++
		}
		if entry.ContainerIdx == 0 && entry.PortIdx == 1 && entry.OfferPort == 1 {
			valid++
		}
	}
	if valid < 2 {
		t.Fatalf("Expected 2 valid port mappings, not %d", valid)
	}

	//-- port mapping in case of multiple discontinuous port ranges in mesos offer
	pod.Spec = api.PodSpec{
		Containers: []api.Container{{
			Ports: []api.ContainerPort{{
				HostPort: 0,
			}, {
				HostPort: 0,
			}},
		}},
	}
	task, err = New(api.NewDefaultContext(), "", pod)
	if err != nil {
		t.Fatal(err)
	}
	offer = &mesos.Offer{
		Resources: []*mesos.Resource{
			mesosutil.NewRangesResource("ports", []*mesos.Value_Range{mesosutil.NewValueRange(1, 1), mesosutil.NewValueRange(3, 5)}),
		},
	}
	mapping, err = wildcardHostPortMapping(task, offer)
	if err != nil {
		t.Fatal(err)
	} else if len(mapping) != 2 {
		t.Fatal("Expected both ports allocated")
	}
	valid = 0
	for _, entry := range mapping {
		if entry.ContainerIdx == 0 && entry.PortIdx == 0 && entry.OfferPort == 1 {
			valid++
		}
		if entry.ContainerIdx == 0 && entry.PortIdx == 1 && entry.OfferPort == 3 {
			valid++
		}
	}
	if valid < 2 {
		t.Fatalf("Expected 2 valid port mappings, not %d", valid)
	}
}

func TestMappingTypeForPod(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{},
		},
	}
	mt := MappingTypeForPod(pod)
	if mt != HostPortMappingWildcard {
		t.Fatalf("expected wildcard mapping")
	}

	pod.Labels[PortMappingLabelKey] = string(HostPortMappingFixed)
	mt = MappingTypeForPod(pod)
	if mt != HostPortMappingFixed {
		t.Fatalf("expected fixed mapping")
	}

	pod.Labels[PortMappingLabelKey] = string(HostPortMappingWildcard)
	mt = MappingTypeForPod(pod)
	if mt != HostPortMappingWildcard {
		t.Fatalf("expected wildcard mapping")
	}
}
