/*
Copyright 2021 The Kubernetes Authors.

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

package managed

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

var (
	pinnedManagementEnabled  bool
	pinnedManagementFilename = "/etc/kubernetes/openshift-workload-pinning"
)

const (
	qosWarning                = "skipping pod CPUs requests modifications because it has guaranteed QoS class"
	WorkloadsAnnotationPrefix = "target.workload.openshift.io/"
	WorkloadsCapacitySuffix   = "workload.openshift.io/cores"
	ContainerAnnotationPrefix = "resources.workload.openshift.io/"
	WorkloadAnnotationWarning = "workload.openshift.io/warning"
)

type WorkloadContainerAnnotation struct {
	CpuShares uint64 `json:"cpushares"`
}

func NewWorkloadContainerAnnotation(cpushares uint64) WorkloadContainerAnnotation {
	return WorkloadContainerAnnotation{
		CpuShares: cpushares,
	}
}

func (w WorkloadContainerAnnotation) Serialize() ([]byte, error) {
	return json.Marshal(w)
}

func init() {
	readEnablementFile()
}

func readEnablementFile() {
	if _, err := os.Stat(pinnedManagementFilename); err == nil {
		pinnedManagementEnabled = true
	}
}

// TestOnlySetEnabled allows changing the state of management partition enablement
// This method MUST NOT be used outside of test code
func TestOnlySetEnabled(enabled bool) {
	pinnedManagementEnabled = enabled
}

func IsEnabled() bool {
	return pinnedManagementEnabled
}

// IsPodManaged returns true and the name of the workload if enabled.
// returns true, workload name, and the annotation payload.
func IsPodManaged(pod *v1.Pod) (bool, string, string) {
	if pod.ObjectMeta.Annotations == nil {
		return false, "", ""
	}
	for annotation, value := range pod.ObjectMeta.Annotations {
		if strings.HasPrefix(annotation, WorkloadsAnnotationPrefix) {
			return true, strings.TrimPrefix(annotation, WorkloadsAnnotationPrefix), value
		}
	}
	return false, "", ""
}

// ModifyStaticPodForPinnedManagement will modify a pod for pod management
func ModifyStaticPodForPinnedManagement(pod *v1.Pod) (*v1.Pod, string, error) {
	pod = pod.DeepCopy()
	enabled, workloadName, value := IsPodManaged(pod)
	if !enabled {
		return nil, "", nil
	}
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	if isPodGuaranteed(pod) {
		stripWorkloadAnnotations(pod.Annotations)
		pod.Annotations[WorkloadAnnotationWarning] = qosWarning
		return pod, "", nil
	}
	pod.Annotations[fmt.Sprintf("%v%v", WorkloadsAnnotationPrefix, workloadName)] = value
	if err := updateContainers(workloadName, pod); err != nil {
		return nil, "", err
	}
	return pod, workloadName, nil
}

func GenerateResourceName(workloadName string) v1.ResourceName {
	return v1.ResourceName(fmt.Sprintf("%v.%v", workloadName, WorkloadsCapacitySuffix))
}

func updateContainers(workloadName string, pod *v1.Pod) error {
	updateContainer := func(container *v1.Container) error {
		if container.Resources.Requests == nil {
			return fmt.Errorf("managed container %v does not have Resource.Requests", container.Name)
		}
		if _, ok := container.Resources.Requests[v1.ResourceCPU]; !ok {
			return fmt.Errorf("managed container %v does not have cpu requests", container.Name)
		}
		if _, ok := container.Resources.Requests[v1.ResourceMemory]; !ok {
			return fmt.Errorf("managed container %v does not have memory requests", container.Name)
		}
		if container.Resources.Limits == nil {
			container.Resources.Limits = v1.ResourceList{}
		}
		cpuRequest := container.Resources.Requests[v1.ResourceCPU]
		cpuRequestInMilli := cpuRequest.MilliValue()

		containerAnnotation := NewWorkloadContainerAnnotation(MilliCPUToShares(cpuRequestInMilli))
		jsonAnnotation, _ := containerAnnotation.Serialize()
		containerNameKey := fmt.Sprintf("%v%v", ContainerAnnotationPrefix, container.Name)

		newCPURequest := resource.NewMilliQuantity(cpuRequestInMilli*1000, cpuRequest.Format)

		pod.Annotations[containerNameKey] = string(jsonAnnotation)
		container.Resources.Requests[GenerateResourceName(workloadName)] = *newCPURequest
		container.Resources.Limits[GenerateResourceName(workloadName)] = *newCPURequest

		delete(container.Resources.Requests, v1.ResourceCPU)
		return nil
	}
	for idx := range pod.Spec.Containers {
		if err := updateContainer(&pod.Spec.Containers[idx]); err != nil {
			return err
		}
	}
	for idx := range pod.Spec.InitContainers {
		if err := updateContainer(&pod.Spec.InitContainers[idx]); err != nil {
			return err
		}
	}
	return nil
}

// isPodGuaranteed checks if the pod has a guaranteed QoS.
// This QoS check is different from the library versions that check QoS,
// this is because of the order at which changes get observed.
// (i.e. the library assumes the defaulter has ran on the pod resource before calculating QoS).
//
// The files will get interpreted before they reach the API server and before the defaulter applies changes,
// this function takes into account the case where only `limits.cpu` are provided but no `requests.cpu` are since that
// counts as a Guaranteed QoS after the defaulter runs.
func isPodGuaranteed(pod *v1.Pod) bool {
	isGuaranteed := func(containers []v1.Container) bool {
		for _, c := range containers {
			// only memory and CPU resources are relevant to decide pod QoS class
			for _, r := range []v1.ResourceName{v1.ResourceMemory, v1.ResourceCPU} {
				limit := c.Resources.Limits[r]
				request, requestExist := c.Resources.Requests[r]
				if limit.IsZero() {
					return false
				}
				if !requestExist {
					continue
				}
				// In some corner case, when you set CPU requests to 0 the k8s defaulter will change it to the value
				// specified under the limit.
				if r == v1.ResourceCPU && request.IsZero() {
					continue
				}
				if !limit.Equal(request) {
					return false
				}
			}
		}
		return true
	}
	return isGuaranteed(pod.Spec.InitContainers) && isGuaranteed(pod.Spec.Containers)
}

func stripWorkloadAnnotations(annotations map[string]string) {
	for k := range annotations {
		if strings.HasPrefix(k, WorkloadsAnnotationPrefix) {
			delete(annotations, k)
		}
		if strings.HasPrefix(k, ContainerAnnotationPrefix) {
			delete(annotations, k)
		}
	}
}
