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

package service

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
)

// StaticPodValidator discards a pod if we can't calculate resource limits for it.
func StaticPodValidator(
	defaultContainerCPULimit resources.CPUShares,
	defaultContainerMemLimit resources.MegaBytes,
	accumCPU, accumMem *float64,
) podutil.FilterFunc {
	return podutil.FilterFunc(func(pod *api.Pod) (bool, error) {
		_, cpu, _, err := resources.LimitPodCPU(pod, defaultContainerCPULimit)
		if err != nil {
			return false, err
		}

		_, mem, _, err := resources.LimitPodMem(pod, defaultContainerMemLimit)
		if err != nil {
			return false, err
		}

		log.V(2).Infof("reserving %.2f cpu shares and %.2f MB of memory to static pod %s/%s", cpu, mem, pod.Namespace, pod.Name)

		*accumCPU += float64(cpu)
		*accumMem += float64(mem)
		return true, nil
	})
}
