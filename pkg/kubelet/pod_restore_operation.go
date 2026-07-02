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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// A runtime can lose discoverable resources before its abandoned restore has
// stopped using prepared paths. Recover from the durable marker as well as the
// API/runtime lists, without replaying RestorePod or occupying Pod workers.
func (kl *Kubelet) recoverOrphanedPodRestores(ctx context.Context) {
	uids, err := kl.listPodsFromDisk()
	if err != nil {
		klog.FromContext(ctx).Error(err, "Cannot list restore recovery records")
		return
	}
	for _, uid := range uids {
		if ctx.Err() != nil {
			return
		}
		if _, exists := kl.podManager.GetPodByUID(uid); exists {
			continue
		}
		if _, owned := kl.restoreOperationOwners.LoadOrStore(uid, struct{}{}); owned {
			continue
		}
		func() {
			defer kl.restoreOperationOwners.Delete(uid)
			state, err := kl.ReadPodRestoreState(uid)
			if err != nil || state != kubecontainer.PodRestoreStarted && state != kubecontainer.PodRestoreCleanupPending {
				return
			}
			pod := &v1.Pod{}
			pod.UID = uid
			// Keep each orphan attempt short; later passes retry uncertainty.
			timeout := min(5*time.Minute, podCheckpointAPITimeout)
			if _, err := kl.settlePodRestoreWithTimeout(ctx, pod, nil, timeout); err != nil {
				klog.FromContext(ctx).Error(err, "Orphaned restore runtime recovery remains pending", "podUID", uid)
			}
		}()
	}
}
