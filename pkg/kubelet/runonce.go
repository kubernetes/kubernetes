/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/golang/glog"
)

const (
	RunOnceManifestDelay     = 1 * time.Second
	RunOnceMaxRetries        = 10
	RunOnceRetryDelay        = 1 * time.Second
	RunOnceRetryDelayBackoff = 2
)

type RunPodResult struct {
	Pod *api.BoundPod
	Err error
}

// RunOnce polls from one configuration update and run the associated pods.
func (kl *Kubelet) RunOnce(updates <-chan PodUpdate) ([]RunPodResult, error) {
	select {
	case u := <-updates:
		glog.Infof("processing manifest with %d pods", len(u.Pods))
		result, err := kl.runOnce(u.Pods)
		glog.Infof("finished processing %d pods", len(u.Pods))
		return result, err
	case <-time.After(RunOnceManifestDelay):
		return nil, fmt.Errorf("no pod manifest update after %v", RunOnceManifestDelay)
	}
}

// runOnce runs a given set of pods and returns their status.
func (kl *Kubelet) runOnce(pods []api.BoundPod) (results []RunPodResult, err error) {
	if kl.dockerPuller == nil {
		kl.dockerPuller = dockertools.NewDockerPuller(kl.dockerClient, kl.pullQPS, kl.pullBurst)
	}
	pods = filterHostPortConflicts(pods)

	ch := make(chan RunPodResult)
	for i := range pods {
		pod := pods[i] // Make a copy
		go func() {
			err := kl.runPod(pod)
			ch <- RunPodResult{&pod, err}
		}()
	}

	glog.Infof("waiting for %d pods", len(pods))
	failedPods := []string{}
	for i := 0; i < len(pods); i++ {
		res := <-ch
		results = append(results, res)
		if res.Err != nil {
			// TODO(proppy): report which containers failed the pod.
			glog.Infof("failed to start pod %q: %v", res.Pod.Name, res.Err)
			failedPods = append(failedPods, res.Pod.Name)
		} else {
			glog.Infof("started pod %q", res.Pod.Name)
		}
	}
	if len(failedPods) > 0 {
		return results, fmt.Errorf("error running pods: %v", failedPods)
	}
	glog.Infof("%d pods started", len(pods))
	return results, err
}

// runPod runs a single pod and wait until all containers are running.
func (kl *Kubelet) runPod(pod api.BoundPod) error {
	delay := RunOnceRetryDelay
	retry := 0
	for {
		dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
		if err != nil {
			return fmt.Errorf("failed to get kubelet docker containers: %v", err)
		}
		running, err := kl.isPodRunning(pod, dockerContainers)
		if err != nil {
			return fmt.Errorf("failed to check pod status: %v", err)
		}
		if running {
			glog.Infof("pod %q containers running", pod.Name)
			return nil
		}
		glog.Infof("pod %q containers not running: syncing", pod.Name)
		if err = kl.syncPod(&pod, dockerContainers); err != nil {
			return fmt.Errorf("error syncing pod: %v", err)
		}
		if retry >= RunOnceMaxRetries {
			return fmt.Errorf("timeout error: pod %q containers not running after %d retries", pod.Name, RunOnceMaxRetries)
		}
		// TODO(proppy): health checking would be better than waiting + checking the state at the next iteration.
		glog.Infof("pod %q containers synced, waiting for %v", pod.Name, delay)
		<-time.After(delay)
		retry++
		delay *= RunOnceRetryDelayBackoff
	}
}

// isPodRunning returns true if all containers of a manifest are running.
func (kl *Kubelet) isPodRunning(pod api.BoundPod, dockerContainers dockertools.DockerContainers) (bool, error) {
	for _, container := range pod.Spec.Containers {
		dockerContainer, found, _ := dockerContainers.FindPodContainer(GetPodFullName(&pod), pod.UID, container.Name)
		if !found {
			glog.Infof("container %q not found", container.Name)
			return false, nil
		}
		inspectResult, err := kl.dockerClient.InspectContainer(dockerContainer.ID)
		if err != nil {
			glog.Infof("failed to inspect container %q: %v", container.Name, err)
			return false, err
		}
		if !inspectResult.State.Running {
			glog.Infof("container %q not running: %#v", container.Name, inspectResult.State)
			return false, nil
		}
	}
	return true, nil
}
