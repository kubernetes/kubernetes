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

package service

import (
	"fmt"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/pkg/api"
)

// staticPodValidator discards a pod if that pod has a container that fails to declare either
// a CPU or Memory limit. if minimalResources is true then podtask.MinimalCpus and
// podtask.MinimalMem are used when accounting for pod resources (but are not applied as pod
// resource limits). minimalResources is part of a bigger hack/work-around that attempts
// to address a systemd/mesos conflict w/ respect to resource accounting.
func staticPodValidator(accumCPU, accumMem *float64, minimalResources bool) podutil.FilterFunc {
	return podutil.FilterFunc(func(pod *api.Pod) (bool, error) {
		// TODO(sttts): allow unlimited static pods as well and patch in the default resource limits
		// for now use of the value 1 is a placeholder: we're discarding pods if they have containers
		// without limits so it doesn't really matter what number we use here.
		unlimitedCPU := resource.LimitPodCPU(pod, 1)
		if unlimitedCPU {
			return false, fmt.Errorf("found static pod without limit on cpu resources")
		}

		unlimitedMem := resource.LimitPodMem(pod, 1)
		if unlimitedMem {
			return false, fmt.Errorf("found static pod without limit on memory resources")
		}

		cpu := resource.PodCPULimit(pod)
		mem := resource.PodMemLimit(pod)
		if minimalResources {
			// see SchedulerServer.AccountForPodResources
			cpu = podtask.MinimalCpus
			mem = podtask.MinimalMem
		}

		log.V(2).Infof("reserving %.2f cpu shares and %.2f MB of memory to static pod %s/%s", cpu, mem, pod.Namespace, pod.Name)

		*accumCPU += float64(cpu)
		*accumMem += float64(mem)
		return true, nil
	})
}
