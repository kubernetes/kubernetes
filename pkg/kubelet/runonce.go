/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"github.com/golang/glog"
)

const (
	RunOnceManifestDelay     = 1 * time.Second
	RunOnceMaxRetries        = 10
	RunOnceRetryDelay        = 1 * time.Second
	RunOnceRetryDelayBackoff = 2
)

type RunPodResult struct {
	Pod *api.Pod
	Err error
}

// RunOnce polls from one configuration update and run the associated pods.
func (kl *Kubelet) RunOnce(updates <-chan PodUpdate) ([]RunPodResult, error) {
	select {
	case u := <-updates:
		glog.Infof("processing manifest with %d pods", len(u.Pods))
		result, err := kl.runOnce(u.Pods, RunOnceRetryDelay)
		glog.Infof("finished processing %d pods", len(u.Pods))
		return result, err
	case <-time.After(RunOnceManifestDelay):
		return nil, fmt.Errorf("no pod manifest update after %v", RunOnceManifestDelay)
	}
}

// runOnce runs a given set of pods and returns their status.
func (kl *Kubelet) runOnce(pods []*api.Pod, retryDelay time.Duration) (results []RunPodResult, err error) {
	kl.handleNotFittingPods(pods)

	ch := make(chan RunPodResult)
	for _, pod := range pods {
		go func(pod *api.Pod) {
			err := kl.runPod(pod, retryDelay)
			ch <- RunPodResult{pod, err}
		}(pod)
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
func (kl *Kubelet) runPod(pod *api.Pod, retryDelay time.Duration) error {
	delay := retryDelay
	retry := 0
	for {
		pods, err := kl.containerRuntime.GetPods(false)
		if err != nil {
			return fmt.Errorf("failed to get kubelet pods: %v", err)
		}
		p := container.Pods(pods).FindPodByID(pod.UID)
		running, err := kl.isPodRunning(pod, p)
		if err != nil {
			return fmt.Errorf("failed to check pod status: %v", err)
		}
		if running {
			glog.Infof("pod %q containers running", pod.Name)
			return nil
		}
		glog.Infof("pod %q containers not running: syncing", pod.Name)
		// We don't create mirror pods in this mode; pass a dummy boolean value
		// to sycnPod.
		if err = kl.syncPod(pod, nil, p, SyncPodUpdate); err != nil {
			return fmt.Errorf("error syncing pod: %v", err)
		}
		if retry >= RunOnceMaxRetries {
			return fmt.Errorf("timeout error: pod %q containers not running after %d retries", pod.Name, RunOnceMaxRetries)
		}
		// TODO(proppy): health checking would be better than waiting + checking the state at the next iteration.
		glog.Infof("pod %q containers synced, waiting for %v", pod.Name, delay)
		time.Sleep(delay)
		retry++
		delay *= RunOnceRetryDelayBackoff
	}
}

// isPodRunning returns true if all containers of a manifest are running.
func (kl *Kubelet) isPodRunning(pod *api.Pod, runningPod container.Pod) (bool, error) {
	status, err := kl.containerRuntime.GetPodStatus(pod)
	if err != nil {
		glog.Infof("Failed to get the status of pod %q: %v", kubecontainer.GetPodFullName(pod), err)
		return false, err
	}
	for _, st := range status.ContainerStatuses {
		if st.State.Running == nil {
			glog.Infof("Container %q not running: %#v", st.Name, st.State)
			return false, nil
		}
	}
	return true, nil
}
