/*
Copyright The Kubernetes Authors.

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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// restorePodState completes reconciliation before API configuration updates can
// admit new pods. Static sources remain available to bootstrap the API server.
func (kl *Kubelet) restorePodState(ctx context.Context, updates <-chan kubetypes.PodUpdate) ([]kubetypes.PodUpdate, error) {
	if kl.kubeClient == nil {
		return nil, nil
	}
	var checkpointedPods []*v1.Pod
	for _, pod := range kl.allocationManager.GetCheckpointedPods() {
		// V1 entries lack a complete spec and must go through normal admission to
		// migrate. Static pods must come from their current local configuration.
		if state.IsMigratedFromV1(pod) || kubetypes.IsStaticPod(pod) {
			continue
		}
		checkpointedPods = append(checkpointedPods, pod)
	}
	if len(checkpointedPods) == 0 {
		return nil, nil
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	listedPods := make(chan *v1.PodList, 1)
	go func() {
		_ = wait.PollUntilContextCancel(ctx, time.Second, true, func(ctx context.Context) (bool, error) {
			// An empty resourceVersion requests a consistent read, rather than an
			// arbitrarily old informer snapshot. A failed list is never an empty list.
			requestCtx, requestCancel := context.WithTimeout(ctx, 30*time.Second)
			defer requestCancel()
			pods, err := kl.kubeClient.CoreV1().Pods(metav1.NamespaceAll).List(requestCtx, metav1.ListOptions{
				FieldSelector: fields.OneTermEqualSelector("spec.nodeName", string(kl.nodeName)).String(),
			})
			if err != nil {
				klog.FromContext(ctx).Error(err, "Unable to list pods for startup reconciliation; retrying")
				return false, nil
			}
			listedPods <- pods
			return true, nil
		})
	}()

	var pending []kubetypes.PodUpdate
	monitor := time.NewTicker(time.Second)
	defer monitor.Stop()
	for {
		kl.syncLoopMonitor.Store(kl.clock.Now())
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-monitor.C:
			// Static control-plane pods may need retries before the API can answer.
			kl.HandlePodSyncs(ctx, kl.getPodsToSync())
		case pods := <-listedPods:
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			kl.restoreCheckpointedPods(ctx, checkpointedPods, pods.Items)
			return pending, nil
		case update, open := <-updates:
			if !open {
				return nil, fmt.Errorf("pod update channel closed during startup reconciliation")
			}
			if update.Source == kubetypes.ApiserverSource {
				pending = append(pending, update)
			} else {
				kl.dispatchPodUpdate(ctx, update, kl)
			}
		}
	}
}

func (kl *Kubelet) restoreCheckpointedPods(ctx context.Context, checkpointedPods []*v1.Pod, apiPods []v1.Pod) {
	logger := klog.FromContext(ctx)
	podsByUID := make(map[types.UID]*v1.Pod, len(apiPods))
	for i := range apiPods {
		pod := &apiPods[i]
		if !kubetypes.IsMirrorPod(pod) {
			podsByUID[pod.UID] = pod
		}
	}
	var restoredPods []*v1.Pod
	for _, allocatedPod := range checkpointedPods {
		if pod, found := podsByUID[allocatedPod.UID]; found {
			// Keep current metadata, status and desired resources. The allocation
			// manager supplies the checkpointed spec to pod workers and resource accounting.
			pod = pod.DeepCopy()
			if pod.Annotations == nil {
				pod.Annotations = make(map[string]string)
			}
			pod.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.ApiserverSource
			if allocatedPod.Annotations[kubetypes.ConfigSourceAnnotationKey] == "" {
				// Upgrade old V2 entries without replacing allocated resources with
				// a resize that was requested while the kubelet was stopped.
				allocated, _ := kl.allocationManager.UpdatePodFromAllocation(pod)
				if err := kl.allocationManager.SetAllocatedResources(logger, allocated); err != nil {
					logger.Error(err, "Unable to checkpoint pod source after startup reconciliation", "pod", klog.KObj(pod))
				}
			}
			kl.podManager.AddPod(pod)
			restoredPods = append(restoredPods, pod)
		} else if allocatedPod.Annotations[kubetypes.ConfigSourceAnnotationKey] == kubetypes.ApiserverSource {
			// The consistent API list proves this pod was deleted. Do not wait for
			// sourcesReady, or briefly start it with a CREATE before requesting termination.
			kl.podWorkers.UpdatePod(ctx, UpdatePodOptions{
				Pod:        allocatedPod,
				UpdateType: kubetypes.SyncPodKill,
			})
			kl.allocationManager.RemovePod(logger, allocatedPod.UID)
		}
		// Older V2 checkpoints have no source annotation. An absent pod may be a
		// static pod, so leave its cleanup to housekeeping once all sources are ready.
	}
	// Hydrate all statuses before starting any workers, including resize conditions
	// that the first SyncPod must not accidentally overwrite.
	kl.statusManager.BackfillPodResizeConditions(restoredPods)
	for _, pod := range restoredPods {
		kl.statusManager.SetPodStatus(logger, pod, pod.Status)
	}
	kl.handlePodAdditions(ctx, restoredPods, true)
}
