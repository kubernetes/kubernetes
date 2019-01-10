/*
Copyright 2016 The Kubernetes Authors.
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
package decorategpupod

import (
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/util/tolerations"
)

func getPod(name string, numContainers int, needGPU bool) *api.Pod {
	res := api.ResourceRequirements{}
	if needGPU {
		res.Requests = api.ResourceList{api.ResourceNvidiaGPU: resource.MustParse("1")}
		res.Limits = api.ResourceList{api.ResourceNvidiaGPU: resource.MustParse("1")}
	}
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test"},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = make([]api.Container, 0, numContainers)
	for i := 0; i < numContainers; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: res,
		})
	}
	return pod
}

func tolerationContains(toleration api.Toleration, tolerationSlice []api.Toleration) bool {
	for _, t := range tolerationSlice {
		if tolerations.AreEqual(t, toleration) {
			return true
		}
	}
	return false
}

func TestAdmitNeedGPUOnCreateShouldAddToleration(t *testing.T) {
	handler := NewDecorateGPUPodPlugin()
	newPod := getPod("test", 2, true)
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	// check toleration
	if !tolerationContains(toleration, newPod.Spec.Tolerations) {
		t.Errorf("Check toleration failed")
	}
}

func TestAdmitNeedGPUOnCreateShouldNotAddToleration(t *testing.T) {
	handler := NewDecorateGPUPodPlugin()
	newPod := getPod("test", 2, false)
	err := handler.Admit(admission.NewAttributesRecord(newPod, nil, api.Kind("Pod").WithVersion("version"), newPod.Namespace, newPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}
	// check toleration
	if tolerationContains(toleration, newPod.Spec.Tolerations) {
		t.Errorf("Check toleration failed")
	}
}

func TestHandles(t *testing.T) {
	for op, shouldHandle := range map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  false,
		admission.Connect: false,
		admission.Delete:  false,
	} {
		handler := NewDecorateGPUPodPlugin()
		if e, a := shouldHandle, handler.Handles(op); e != a {
			t.Errorf("%v: shouldHandle=%t, handles=%t", op, e, a)
		}
	}
}
