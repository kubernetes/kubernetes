/*
Copyright 2014 The Kubernetes Authors.

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
	"os"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

const (
	runOnceManifestDelay     = 1 * time.Second
	runOnceMaxRetries        = 10
	runOnceRetryDelay        = 1 * time.Second
	runOnceRetryDelayBackoff = 2
)

type RunPodResult struct {
	Pod *api.Pod
	Err error
}

// RunOnce polls from one configuration update and run the associated pods.
func (kl *Kubelet) RunOnce(updates <-chan kubetypes.PodUpdate) ([]RunPodResult, error) {
	// Setup filesystem directories.
	if err := kl.setupDataDirs(); err != nil {
		return nil, err
	}

	// If the container logs directory does not exist, create it.
	if _, err := os.Stat(ContainerLogsDir); err != nil {
		if err := kl.os.MkdirAll(ContainerLogsDir, 0755); err != nil {
			glog.Errorf("Failed to create directory %q: %v", ContainerLogsDir, err)
		}
	}

	select {
	case u := <-updates:
		glog.Infof("processing manifest with %d pods", len(u.Pods))
		result, err := kl.runOnce(u.Pods, runOnceRetryDelay)
		glog.Infof("finished processing %d pods", len(u.Pods))
		return result, err
	case <-time.After(runOnceManifestDelay):
		return nil, fmt.Errorf("no pod manifest update after %v", runOnceManifestDelay)
	}
}

// runOnce runs a given set of pods and returns their status.
func (kl *Kubelet) runOnce(pods []*api.Pod, retryDelay time.Duration) (results []RunPodResult, err error) {
	ch := make(chan RunPodResult)
	admitted := []*api.Pod{}
	for _, pod := range pods {
		// Check if we can admit the pod.
		if ok, reason, message := kl.canAdmitPod(admitted, pod); !ok {
			kl.rejectPod(pod, reason, message)
			results = append(results, RunPodResult{pod, nil})
			continue
		}

		admitted = append(admitted, pod)
		go func(pod *api.Pod) {
			err := kl.runPod(pod, retryDelay)
			ch <- RunPodResult{pod, err}
		}(pod)
	}

	glog.Infof("Waiting for %d pods", len(admitted))
	failedPods := []string{}
	for i := 0; i < len(admitted); i++ {
		res := <-ch
		results = append(results, res)
		if res.Err != nil {
			// TODO(proppy): report which containers failed the pod.
			glog.Infof("failed to start pod %q: %v", format.Pod(res.Pod), res.Err)
			failedPods = append(failedPods, format.Pod(res.Pod))
		} else {
			glog.Infof("started pod %q", format.Pod(res.Pod))
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
		status, err := kl.containerRuntime.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			return fmt.Errorf("Unable to get status for pod %q: %v", format.Pod(pod), err)
		}

		if kl.isPodRunning(pod, status) {
			glog.Infof("pod %q containers running", format.Pod(pod))
			return nil
		}
		glog.Infof("pod %q containers not running: syncing", format.Pod(pod))

		glog.Infof("Creating a mirror pod for static pod %q", format.Pod(pod))
		if err := kl.podManager.CreateMirrorPod(pod); err != nil {
			glog.Errorf("Failed creating a mirror pod %q: %v", format.Pod(pod), err)
		}
		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		if err = kl.syncPod(syncPodOptions{
			pod:        pod,
			mirrorPod:  mirrorPod,
			podStatus:  status,
			updateType: kubetypes.SyncPodUpdate,
		}); err != nil {
			return fmt.Errorf("error syncing pod %q: %v", format.Pod(pod), err)
		}
		if retry >= runOnceMaxRetries {
			return fmt.Errorf("timeout error: pod %q containers not running after %d retries", format.Pod(pod), runOnceMaxRetries)
		}
		// TODO(proppy): health checking would be better than waiting + checking the state at the next iteration.
		glog.Infof("pod %q containers synced, waiting for %v", format.Pod(pod), delay)
		time.Sleep(delay)
		retry++
		delay *= runOnceRetryDelayBackoff
	}
}

// isPodRunning returns true if all containers of a manifest are running.
func (kl *Kubelet) isPodRunning(pod *api.Pod, status *kubecontainer.PodStatus) bool {
	for _, c := range pod.Spec.Containers {
		cs := status.FindContainerStatusByName(c.Name)
		if cs == nil || cs.State != kubecontainer.ContainerStateRunning {
			glog.Infof("Container %q for pod %q not running", c.Name, format.Pod(pod))
			return false
		}
	}
	return true
}
