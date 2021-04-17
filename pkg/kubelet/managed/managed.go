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
	"fmt"
	"os"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

var (
	pinnedManagementEnabled      bool
	pinnedManagementFilename     = "/etc/kubernetes/openshift-workload-pinning"
	WorkloadsAnnotationPrefix    = "workload.openshift.io/"
	WorkloadsCapacitySuffix      = "workload.openshift.io/cores"
	ContainerCpuAnnotationFormat = "io.openshift.workload.%v.cpushares/%v"
)

func init() {
	readEnablementFile()
}

func readEnablementFile() {
	if _, err := os.Stat(pinnedManagementFilename); err == nil {
		pinnedManagementEnabled = true
	}
}

func IsEnabled() bool {
	return pinnedManagementEnabled
}

/// IsPodManaged returns true and the name of the workload if enabled
func IsPodManaged(pod *v1.Pod) (bool, string) {
	if pod.ObjectMeta.Annotations == nil {
		return false, ""
	}
	for annotation := range pod.ObjectMeta.Annotations {
		if strings.HasPrefix(annotation, WorkloadsAnnotationPrefix) {
			return true, strings.TrimPrefix(annotation, WorkloadsAnnotationPrefix)
		}
	}
	return false, ""
}

func ModifyStaticPodForPinnedManagement(pod *v1.Pod) (bool, string) {
	enabled, workloadName := IsPodManaged(pod)
	if !enabled {
		return false, ""
	}
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[fmt.Sprintf("%v%v", WorkloadsAnnotationPrefix, workloadName)] = ""
	updateContainers(workloadName, pod, pod.Spec.Containers)
	updateContainers(workloadName, pod, pod.Spec.InitContainers)
	return true, workloadName
}

func GenerateResourceName(workloadName string) v1.ResourceName {
	return v1.ResourceName(fmt.Sprintf("%v.%v", workloadName, WorkloadsCapacitySuffix))
}

func updateContainers(workloadName string, pod *v1.Pod, containers []v1.Container) {
	for _, container := range containers {
		if _, ok := container.Resources.Requests[v1.ResourceCPU]; !ok {
			continue
		}
		if _, ok := container.Resources.Requests[v1.ResourceMemory]; !ok {
			continue
		}
		cpuRequest := container.Resources.Requests[v1.ResourceCPU]
		cpuRequestInMilli := cpuRequest.MilliValue()
		resourceLimit := fmt.Sprintf("%v", MilliCPUToShares(cpuRequestInMilli))

		containerNameKey := fmt.Sprintf(ContainerCpuAnnotationFormat, workloadName, container.Name)
		pod.Annotations[containerNameKey] = resourceLimit

		newCPURequest := resource.NewMilliQuantity(cpuRequestInMilli*1000, cpuRequest.Format)
		container.Resources.Requests[GenerateResourceName(workloadName)] = *newCPURequest

		delete(container.Resources.Requests, v1.ResourceCPU)
	}
}
