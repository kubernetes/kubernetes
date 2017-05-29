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

package initialresources

import (
	"errors"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

type fakeSource struct {
	f func(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (int64, int64, error)
}

func (s *fakeSource) GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (usage int64, samples int64, err error) {
	return s.f(kind, perc, image, namespace, exactMatch, start, end)
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

func createPod(name string, image string, request api.ResourceList) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test-ns"},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = []api.Container{}
	addContainer(pod, "i0", image, request)
	pod.Spec.InitContainers = pod.Spec.Containers
	pod.Spec.Containers = []api.Container{}
	addContainer(pod, "c0", image, request)
	return pod
}

func getPods() []*api.Pod {
	return []*api.Pod{
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
	verifyContainer(t, &pod.Spec.InitContainers[0], cpu, mem)
}

func verifyAnnotation(t *testing.T, pod *api.Pod, expected string) {
	a, ok := pod.ObjectMeta.Annotations[initialResourcesAnnotation]
	if !ok {
		t.Errorf("No annotation but expected %v", expected)
	}
	if a != expected {
		t.Errorf("Wrong annotation set by Initial Resources: got %v, expected %v", a, expected)
	}
}

func expectNoAnnotation(t *testing.T, pod *api.Pod) {
	if a, ok := pod.ObjectMeta.Annotations[initialResourcesAnnotation]; ok {
		t.Errorf("Expected no annotation but got %v", a)
	}
}

func admit(t *testing.T, ir admission.Interface, pods []*api.Pod) {
	for i := range pods {
		p := pods[i]

		podKind := api.Kind("Pod").WithVersion("version")
		podRes := api.Resource("pods").WithVersion("version")
		attrs := admission.NewAttributesRecord(p, nil, podKind, "test", p.ObjectMeta.Name, podRes, "", admission.Create, nil)
		if err := ir.Admit(attrs); err != nil {
			t.Error(err)
		}
	}
}

func testAdminScenarios(t *testing.T, ir admission.Interface, p *api.Pod) {
	podKind := api.Kind("Pod").WithVersion("version")
	podRes := api.Resource("pods").WithVersion("version")

	var tests = []struct {
		attrs       admission.Attributes
		expectError bool
	}{
		{
			admission.NewAttributesRecord(p, nil, podKind, "test", p.ObjectMeta.Name, podRes, "foo", admission.Create, nil),
			false,
		},
		{
			admission.NewAttributesRecord(&api.ReplicationController{}, nil, podKind, "test", "", podRes, "", admission.Create, nil),
			true,
		},
	}

	for _, test := range tests {
		err := ir.Admit(test.attrs)
		if err != nil && test.expectError == false {
			t.Error(err)
		} else if err == nil && test.expectError == true {
			t.Error("Error expected for Admit but received none")
		}
	}
}

func performTest(t *testing.T, ir admission.Interface) {
	pods := getPods()
	admit(t, ir, pods)
	testAdminScenarios(t, ir, pods[0])

	verifyPod(t, pods[0], 100, 100)
	verifyPod(t, pods[1], 100, 300)
	verifyPod(t, pods[2], 300, 100)
	verifyPod(t, pods[3], 300, 300)

	verifyAnnotation(t, pods[0], "Initial Resources plugin set: cpu, memory request for init container i0; cpu, memory request for container c0")
	verifyAnnotation(t, pods[1], "Initial Resources plugin set: cpu request for init container i0")
	verifyAnnotation(t, pods[2], "Initial Resources plugin set: memory request for init container i0")
	expectNoAnnotation(t, pods[3])
}

func TestEstimateReturnsErrorFromSource(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		return 0, 0, errors.New("Example error")
	}
	ir := newInitialResources(&fakeSource{f: f}, 90, false)
	admit(t, ir, getPods())
}

func TestEstimationBasedOnTheSameImageSameNamespace7d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && end.Sub(start) == week && ns == "test-ns" {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	performTest(t, newInitialResources(&fakeSource{f: f}, 90, false))
}

func TestEstimationBasedOnTheSameImageSameNamespace30d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && end.Sub(start) == week && ns == "test-ns" {
			return 200, 20, nil
		}
		if exactMatch && end.Sub(start) == month && ns == "test-ns" {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	performTest(t, newInitialResources(&fakeSource{f: f}, 90, false))
}

func TestEstimationBasedOnTheSameImageAllNamespaces7d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && ns == "test-ns" {
			return 200, 20, nil
		}
		if exactMatch && end.Sub(start) == week && ns == "" {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	performTest(t, newInitialResources(&fakeSource{f: f}, 90, false))
}

func TestEstimationBasedOnTheSameImageAllNamespaces30d(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if exactMatch && ns == "test-ns" {
			return 200, 20, nil
		}
		if exactMatch && end.Sub(start) == week && ns == "" {
			return 200, 20, nil
		}
		if exactMatch && end.Sub(start) == month && ns == "" {
			return 100, 120, nil
		}
		return 200, 120, nil
	}
	performTest(t, newInitialResources(&fakeSource{f: f}, 90, false))
}

func TestEstimationBasedOnOtherImages(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, image, ns string, exactMatch bool, _, _ time.Time) (int64, int64, error) {
		if image == "image" && !exactMatch && ns == "" {
			return 100, 5, nil
		}
		return 200, 20, nil
	}
	performTest(t, newInitialResources(&fakeSource{f: f}, 90, false))
}

func TestNoData(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, _ bool, _, _ time.Time) (int64, int64, error) {
		return 200, 0, nil
	}
	ir := newInitialResources(&fakeSource{f: f}, 90, false)

	pods := []*api.Pod{
		createPod("p0", "image:v0", parseReq("", "")),
	}
	admit(t, ir, pods)

	if pods[0].Spec.Containers[0].Resources.Requests != nil {
		t.Errorf("Unexpected resource estimation")
	}

	expectNoAnnotation(t, pods[0])
}

func TestManyContainers(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, _, _ time.Time) (int64, int64, error) {
		if exactMatch {
			return 100, 120, nil
		}
		return 200, 30, nil
	}
	ir := newInitialResources(&fakeSource{f: f}, 90, false)

	pod := createPod("p", "image:v0", parseReq("", ""))
	addContainer(pod, "c1", "image:v1", parseReq("", "300"))
	addContainer(pod, "c2", "image:v2", parseReq("300m", ""))
	addContainer(pod, "c3", "image:v3", parseReq("300m", "300"))
	admit(t, ir, []*api.Pod{pod})

	verifyContainer(t, &pod.Spec.Containers[0], 100, 100)
	verifyContainer(t, &pod.Spec.Containers[1], 100, 300)
	verifyContainer(t, &pod.Spec.Containers[2], 300, 100)
	verifyContainer(t, &pod.Spec.Containers[3], 300, 300)

	verifyAnnotation(t, pod, "Initial Resources plugin set: cpu, memory request for init container i0; cpu, memory request for container c0; cpu request for container c1; memory request for container c2")
}

func TestNamespaceAware(t *testing.T) {
	f := func(_ api.ResourceName, _ int64, _, ns string, exactMatch bool, start, end time.Time) (int64, int64, error) {
		if ns == "test-ns" {
			return 200, 0, nil
		}
		return 200, 120, nil
	}
	ir := newInitialResources(&fakeSource{f: f}, 90, true)

	pods := []*api.Pod{
		createPod("p0", "image:v0", parseReq("", "")),
	}
	admit(t, ir, pods)

	if pods[0].Spec.Containers[0].Resources.Requests != nil {
		t.Errorf("Unexpected resource estimation")
	}

	expectNoAnnotation(t, pods[0])
}
