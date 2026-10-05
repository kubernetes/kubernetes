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
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	schedulinghelper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/sliceutils"
)

// Share one deadline across the runtime lists used only to reorder a batch.
// The deadline bounds runtime requests; it is not a bound on local processing.
const podStartOrderRuntimeTimeout = time.Second

func (kl *Kubelet) sortPodAdditions(ctx context.Context, pods []*v1.Pod) {
	sort.Sort(sliceutils.PodsByCreationTime(pods))
	if len(pods) < 2 || !utilfeature.DefaultFeatureGate.Enabled(features.PodStartingOrderByPriority) {
		return
	}

	logger := klog.FromContext(ctx)
	batchUIDs := make(sets.Set[types.UID], len(pods))
	for _, pod := range pods {
		// Static pods can preempt other pods, and mirror pods update a different
		// worker. Keeping their positions while reordering ordinary pods would
		// still change resource competition across these admission paths.
		if pod.UID == "" || batchUIDs.Has(pod.UID) || kubetypes.IsStaticPod(pod) || kubetypes.IsMirrorPod(pod) ||
			pod.DeletionTimestamp != nil || podutil.IsPodPhaseTerminal(pod.Status.Phase) || kl.podWorkers.IsPodTerminationRequested(pod.UID) {
			logger.V(4).Info("Keeping pod creation order for a batch outside priority ordering", "pod", klog.KObj(pod), "podUID", pod.UID)
			return
		}
		batchUIDs.Insert(pod.UID)
	}

	// A cached absence is not sufficient to conclude that a pod has no running
	// containers. Copy one runtime observation into a fixed set before sorting.
	runtimeCtx, cancel := context.WithTimeout(ctx, podStartOrderRuntimeTimeout)
	defer cancel()
	runtimePods, err := kl.containerRuntime.GetPods(runtimeCtx, true)
	if err == nil {
		err = runtimeCtx.Err()
	}
	if err != nil {
		logger.V(4).Info("Keeping pod creation order because runtime state could not be read", "err", err)
		return
	}
	protectedUIDs, err := podsWithActiveRuntimeState(runtimePods)
	if err != nil {
		logger.V(4).Info("Keeping pod creation order because runtime state is incomplete", "err", err)
		return
	}

	// Sources can deliver independent batches. Do not give new candidates
	// priority while another observed pod is missing from admission accounting.
	var allocatedUIDs sets.Set[types.UID]
	for uid := range protectedUIDs {
		if batchUIDs.Has(uid) {
			continue
		}
		// In-batch runtime state is accounted for during admission. Reading
		// allocations cannot affect ordering unless an active UID is outside.
		if allocatedUIDs == nil {
			allocatedUIDs = sets.New[types.UID]()
			for _, pod := range kl.allocationManager.GetAllocatedPods() {
				if pod == nil || pod.UID == "" || kubetypes.IsMirrorPod(pod) || pod.DeletionTimestamp != nil ||
					podutil.IsPodPhaseTerminal(pod.Status.Phase) || kl.podWorkers.IsPodTerminationRequested(pod.UID) {
					continue
				}
				allocatedUIDs.Insert(pod.UID)
			}
		}
		if !allocatedUIDs.Has(uid) {
			logger.V(4).Info("Keeping pod creation order because an active runtime pod is not accounted for", "podUID", uid)
			return
		}
	}

	// Existing runtime state is accounted for first without changing the order
	// among those pods. Stable sorting retains creation time and UID tie breaks.
	sort.SliceStable(pods, func(i, j int) bool {
		iProtected, jProtected := protectedUIDs.Has(pods[i].UID), protectedUIDs.Has(pods[j].UID)
		if iProtected != jProtected {
			return iProtected
		}
		if iProtected {
			return false
		}
		return schedulinghelper.PodPriority(pods[i]) > schedulinghelper.PodPriority(pods[j])
	})
}

func podsWithActiveRuntimeState(pods []*kubecontainer.Pod) (sets.Set[types.UID], error) {
	protectedUIDs := sets.New[types.UID]()
	seenUIDs := make(sets.Set[types.UID], len(pods))
	for _, pod := range pods {
		if pod == nil || pod.ID == "" {
			return nil, fmt.Errorf("runtime pod is missing its UID")
		}
		if seenUIDs.Has(pod.ID) {
			return nil, fmt.Errorf("runtime returned multiple entries for pod UID %q", pod.ID)
		}
		seenUIDs.Insert(pod.ID)
		if len(pod.Containers) == 0 && len(pod.Sandboxes) == 0 {
			return nil, fmt.Errorf("runtime pod %q has no container or sandbox state", pod.ID)
		}
		for _, container := range pod.Containers {
			if container == nil {
				return nil, fmt.Errorf("runtime pod %q has a missing container", pod.ID)
			}
			switch container.State {
			case kubecontainer.ContainerStateCreated, kubecontainer.ContainerStateRunning:
				protectedUIDs.Insert(pod.ID)
			case kubecontainer.ContainerStateExited:
				// Historical containers alone do not establish an allocation.
			default:
				return nil, fmt.Errorf("runtime pod %q has an unresolved container state %q", pod.ID, container.State)
			}
		}
		for _, sandbox := range pod.Sandboxes {
			if sandbox == nil {
				return nil, fmt.Errorf("runtime pod %q has a missing sandbox", pod.ID)
			}
			switch sandbox.State {
			case kubecontainer.ContainerStateRunning:
				// Ready sandboxes use ContainerStateRunning. Protect work in
				// progress even when an application container has not started.
				protectedUIDs.Insert(pod.ID)
			case kubecontainer.ContainerStateExited:
				// NotReady sandboxes alone do not establish an allocation.
			default:
				return nil, fmt.Errorf("runtime pod %q has an unresolved sandbox state %q", pod.ID, sandbox.State)
			}
		}
	}
	return protectedUIDs, nil
}
