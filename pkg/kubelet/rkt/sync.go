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

package rkt

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

// SyncPod syncs the running pod to match the specified desired pod.
func (r *Runtime) SyncPod(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus) error {
	podFullName := kubecontainer.GetPodFullName(pod)
	if len(runningPod.Containers) == 0 {
		glog.V(4).Infof("Pod %q is not running, will start it", podFullName)
		// TODO(yifan): Use RunContainerOptionsGeneratior to get volumeMaps, etc.
		return r.RunPod(pod, nil)
	}

	// Add references to all containers.
	unidentifiedContainers := make(map[types.UID]*kubecontainer.Container)
	for _, c := range runningPod.Containers {
		unidentifiedContainers[c.ID] = c
	}

	restartPod := false
	for _, container := range pod.Spec.Containers {
		expectedHash := hashContainer(&container)

		c := runningPod.FindContainerByName(container.Name)
		if c == nil {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, &podStatus, r.readinessManager) {
				glog.V(3).Infof("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				// TODO(yifan): Containers in one pod are fate-sharing at this moment, see:
				// https://github.com/appc/spec/issues/276.
				restartPod = true
				break
			}
			continue
		}

		// TODO(yifan): Take care of host network change.
		containerChanged := c.Hash != 0 && c.Hash != expectedHash
		if containerChanged {
			glog.Infof("Pod %q container %q hash changed (%d vs %d), it will be killed and re-created.", podFullName, container.Name, c.Hash, expectedHash)
			restartPod = true
			break
		}

		result, err := r.prober.Probe(pod, podStatus, container, string(c.ID), c.Created)
		// TODO(vmarmol): examine this logic.
		if err == nil && result != probe.Success {
			glog.Infof("Pod %q container %q is unhealthy (probe result: %v), it will be killed and re-created.", podFullName, container.Name, result)
			restartPod = true
			break
		}

		if err != nil {
			glog.V(2).Infof("Probe container %q failed: %v", container.Name, err)
		}
		delete(unidentifiedContainers, c.ID)
	}

	// If there is any unidentified containers, restart the pod.
	if len(unidentifiedContainers) > 0 {
		restartPod = true
	}

	if restartPod {
		// TODO(yifan): Handle network plugin.
		if err := r.KillPod(runningPod); err != nil {
			return err
		}
		if err := r.RunPod(pod, nil); err != nil {
			return err
		}
	}
	return nil
}
