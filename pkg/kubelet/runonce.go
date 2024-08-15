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
	"context"
	"fmt"
	"os"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
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

// RunPodResult defines the running results of a Pod.
type RunPodResult struct {
	Pod *v1.Pod
	Err error
}

// RunOnce polls from one configuration update and run the associated pods.
func (kl *Kubelet) RunOnce(updates <-chan kubetypes.PodUpdate) ([]RunPodResult, error) {
	ctx := context.Background()
	// Setup filesystem directories.
	if err := kl.setupDataDirs(); err != nil {
		return nil, err
	}

	// If the container logs directory does not exist, create it.
	if _, err := os.Stat(ContainerLogsDir); err != nil {
		if err := kl.os.MkdirAll(ContainerLogsDir, 0755); err != nil {
			klog.ErrorS(err, "Failed to create directory", "path", ContainerLogsDir)
		}
	}

	select {
	case u := <-updates:
		klog.InfoS("Processing manifest with pods", "numPods", len(u.Pods))
		result, err := kl.runOnce(ctx, u.Pods, runOnceRetryDelay)
		klog.InfoS("Finished processing pods", "numPods", len(u.Pods))
		return result, err
	case <-time.After(runOnceManifestDelay):
		return nil, fmt.Errorf("no pod manifest update after %v", runOnceManifestDelay)
	}
}

// runOnce runs a given set of pods and returns their status.
func (kl *Kubelet) runOnce(ctx context.Context, pods []*v1.Pod, retryDelay time.Duration) (results []RunPodResult, err error) {
	ch := make(chan RunPodResult)
	admitted := []*v1.Pod{}
	for _, pod := range pods {
		// Check if we can admit the pod.
		if ok, reason, message := kl.canAdmitPod(admitted, pod); !ok {
			kl.rejectPod(pod, reason, message)
			results = append(results, RunPodResult{pod, nil})
			continue
		}

		admitted = append(admitted, pod)
		go func(pod *v1.Pod) {
			err := kl.runPod(ctx, pod, retryDelay)
			ch <- RunPodResult{pod, err}
		}(pod)
	}

	klog.InfoS("Waiting for pods", "numPods", len(admitted))
	failedPods := []string{}
	for i := 0; i < len(admitted); i++ {
		res := <-ch
		results = append(results, res)
		if res.Err != nil {
			failedContainerName, err := kl.getFailedContainers(ctx, res.Pod)
			if err != nil {
				klog.InfoS("Unable to get failed containers' names for pod", "pod", klog.KObj(res.Pod), "err", err)
			} else {
				klog.InfoS("Unable to start pod because container failed", "pod", klog.KObj(res.Pod), "containerName", failedContainerName)
			}
			failedPods = append(failedPods, format.Pod(res.Pod))
		} else {
			klog.InfoS("Started pod", "pod", klog.KObj(res.Pod))
		}
	}
	if len(failedPods) > 0 {
		return results, fmt.Errorf("error running pods: %v", failedPods)
	}
	klog.InfoS("Pods started", "numPods", len(pods))
	return results, err
}

// runPod runs a single pod and waits until all containers are running.
func (kl *Kubelet) runPod(ctx context.Context, pod *v1.Pod, retryDelay time.Duration) error {
	var isTerminal bool
	delay := retryDelay
	retry := 0
	for !isTerminal {
		status, err := kl.containerRuntime.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			return fmt.Errorf("unable to get status for pod %q: %v", format.Pod(pod), err)
		}

		if kl.isPodRunning(pod, status) {
			klog.InfoS("Pod's containers running", "pod", klog.KObj(pod))
			return nil
		}
		klog.InfoS("Pod's containers not running: syncing", "pod", klog.KObj(pod))

		klog.InfoS("Creating a mirror pod for static pod", "pod", klog.KObj(pod))
		if err := kl.mirrorPodClient.CreateMirrorPod(pod); err != nil {
			klog.ErrorS(err, "Failed creating a mirror pod", "pod", klog.KObj(pod))
		}
		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		if isTerminal, err = kl.SyncPod(ctx, kubetypes.SyncPodUpdate, pod, mirrorPod, status); err != nil {
			return fmt.Errorf("error syncing pod %q: %v", format.Pod(pod), err)
		}
		if retry >= runOnceMaxRetries {
			return fmt.Errorf("timeout error: pod %q containers not running after %d retries", format.Pod(pod), runOnceMaxRetries)
		}
		// TODO(proppy): health checking would be better than waiting + checking the state at the next iteration.
		klog.InfoS("Pod's containers synced, waiting", "pod", klog.KObj(pod), "duration", delay)
		time.Sleep(delay)
		retry++
		delay *= runOnceRetryDelayBackoff
	}
	return nil
}

// isPodRunning returns true if all containers of a manifest are running.
func (kl *Kubelet) isPodRunning(pod *v1.Pod, status *kubecontainer.PodStatus) bool {
	for _, c := range pod.Spec.Containers {
		cs := status.FindContainerStatusByName(c.Name)
		if cs == nil || cs.State != kubecontainer.ContainerStateRunning {
			klog.InfoS("Container not running", "pod", klog.KObj(pod), "containerName", c.Name)
			return false
		}
	}
	return true
}

// getFailedContainers returns failed container name for pod.
func (kl *Kubelet) getFailedContainers(ctx context.Context, pod *v1.Pod) ([]string, error) {
	status, err := kl.containerRuntime.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	if err != nil {
		return nil, fmt.Errorf("unable to get status for pod %q: %v", format.Pod(pod), err)
	}
	var containerNames []string
	for _, cs := range status.ContainerStatuses {
		if cs.State != kubecontainer.ContainerStateRunning && cs.ExitCode != 0 {
			containerNames = append(containerNames, cs.Name)
		}
	}
	return containerNames, nil
}
