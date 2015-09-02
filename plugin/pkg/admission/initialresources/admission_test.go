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

package initialresources

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

type fakeSource struct {
	f func(kind api.ResourceName, perc int64, image string, exactMatch bool, start, end time.Time) (int64, int64, error)
}

func (s *fakeSource) GetUsagePercentile(kind api.ResourceName, perc int64, image string, exactMatch bool, start, end time.Time) (usage int64, samples int64, err error) {
	return s.f(kind, perc, image, exactMatch, start, end)
}

func parseReq(cpu, mem string) api.ResourceList {
	if cpu == "" && mem == "" {
		return nil
	}
	req := api.ResourceList{}
	if cpu != "" {
		req[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if mem != "" {
		req[api.ResourceMemory] = resource.MustParse(mem)
	}
	return req
}

func addContainer(pod *api.Pod, name, image string, request api.ResourceList) {
	pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
		Name:      name,
		Image:     image,
		Resources: api.ResourceRequirements{Requests: request},
	})
}

func createPod(name string, image string, request api.ResourceList) api.Pod {
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = []api.Container{}
	addContainer(&pod, "c0", image, request)
	return pod
}

func getPods() []api.Pod {
	return []api.Pod{
		createPod("p0", "image:v0", parseReq("", "")),
		createPod("p1", "image:v1", parseReq("", "300")),
		createPod("p2", "image:v2", parseReq("300m", "")),
		createPod("p3", "image:v3", parseReq("300m", "300")),
	}
}

func verifyContainer(t *testing.T, c *api.Container, cpu, mem int64) {
	req := c.Resources.Requests
	if req.Cpu().MilliValue() != cpu {
		t.Errorf("Wrong CPU request for container %v. Expected %v, got %v.", c.Name, cpu, req.Cpu().MilliValue())
	}
	if req.Memory().Value() != mem {
		t.Errorf("Wrong memory request for container %v. Expected %v, got %v.", c.Name, mem, req.Memory().Value())
	}
}

func verifyPod(t *testing.T, pod *api.Pod, cpu, mem int64) {
	verifyContainer(t, &pod.Spec.Containers[0], cpu, mem)
}

func admit(t *testing.T, ir admission.Interface, pods []api.Pod) {
	for i := range pods {
		p := &pods[i]
		if err := ir.Admit(admission.NewAttributesRecord(p, "Pod", "test", p.ObjectMeta.Name, "pods", "", admission.Create, nil)); err != nil {
			t.Error(err)
		}
	}
}

func TestEstimationBasedOnTheSameImage7d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _ string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && end.Sub(start) == week {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	ir := newInitialResources(&fakeSource{f: f})

	pods := getPods()
	admit(t, ir, pods)

	verifyPod(t, &pods[0], 100, 100)
	verifyPod(t, &pods[1], 100, 300)
	verifyPod(t, &pods[2], 300, 100)
	verifyPod(t, &pods[3], 300, 300)
}

func TestEstimationBasedOnTheSameImage30d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _ string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && end.Sub(start) == week {
			return 200, 20, nil
		}
		if exactMatch && end.Sub(start) == month {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	ir := newInitialResources(&fakeSource{f: f})

	pods := getPods()
	admit(t, ir, pods)

	verifyPod(t, &pods[0], 100, 100)
	verifyPod(t, &pods[1], 100, 300)
	verifyPod(t, &pods[2], 300, 100)
	verifyPod(t, &pods[3], 300, 300)
}

func TestEstimationBasedOnOtherImages(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, image string, exactMatch bool, _, _ time.Time) (int64, int64, error) {
		if image == "image" && !exactMatch {
			return 100, 5, nil
		}
		return 200, 20, nil
	}
	ir := newInitialResources(&fakeSource{f: f})

	pods := getPods()
	admit(t, ir, pods)

	verifyPod(t, &pods[0], 100, 100)
	verifyPod(t, &pods[1], 100, 300)
	verifyPod(t, &pods[2], 300, 100)
	verifyPod(t, &pods[3], 300, 300)
}

func TestNoData(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _ string, _ bool, _, _ time.Time) (int64, int64, error) {
		return 200, 0, nil
	}
	ir := newInitialResources(&fakeSource{f: f})

	pods := []api.Pod{
		createPod("p0", "image:v0", parseReq("", "")),
	}
	admit(t, ir, pods)

	if pods[0].Spec.Containers[0].Resources.Requests != nil {
		t.Errorf("Unexpected resource estimation")
	}
}

func TestManyContainers(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _ string, exactMatch bool, _, _ time.Time) (int64, int64, error) {
		if exactMatch {
			return 100, 120, nil
		}
		return 200, 30, nil
	}
	ir := newInitialResources(&fakeSource{f: f})

	pod := createPod("p", "image:v0", parseReq("", ""))
	addContainer(&pod, "c1", "image:v1", parseReq("", "300"))
	addContainer(&pod, "c2", "image:v2", parseReq("300m", ""))
	addContainer(&pod, "c3", "image:v3", parseReq("300m", "300"))
	admit(t, ir, []api.Pod{pod})

	verifyContainer(t, &pod.Spec.Containers[0], 100, 100)
	verifyContainer(t, &pod.Spec.Containers[1], 100, 300)
	verifyContainer(t, &pod.Spec.Containers[2], 300, 100)
	verifyContainer(t, &pod.Spec.Containers[3], 300, 300)
}
