/*
Copyright 2022 The KCP Authors.

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

package resourcequota

import (
	"context"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// UpdateMonitors determines if there are any newly available or removed API resources, and if so, starts/stops monitors
// for them. This is similar to Sync, but instead of polling discovery every 30 seconds, this method is invoked by kcp
// whenever the set of APIs is known to change (CRDs added or removed).
func (rq *Controller) UpdateMonitors(ctx context.Context, discoveryFunc NamespacedResourcesFunc) {
	// Something has changed, so track the new state and perform a sync.
	oldResources := make(map[schema.GroupVersionResource]struct{})
	func() {
		// Get the current resource list from discovery.
		newResources, err := GetQuotableResources(discoveryFunc)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("%s: %v", rq.clusterName, err))

			if discovery.IsGroupDiscoveryFailedError(err) && len(newResources) > 0 {
				// In partial discovery cases, don't remove any existing informers, just add new ones
				for k, v := range oldResources {
					newResources[k] = v
				}
			} else {
				// short circuit in non-discovery error cases or if discovery returned zero resources
				return
			}
		}

		// Decide whether discovery has reported a change.
		if reflect.DeepEqual(oldResources, newResources) {
			klog.V(4).Infof("%s: no resource updates from discovery, skipping resource quota sync", rq.clusterName)
			return
		}

		// Ensure workers are paused to avoid processing events before informers
		// have resynced.
		rq.workerLock.Lock()
		defer rq.workerLock.Unlock()

		// Something has changed, so track the new state and perform a sync.
		if klogV := klog.V(2); klogV.Enabled() {
			klogV.Infof("%s: syncing resource quota controller with updated resources from discovery: %s", rq.clusterName, printDiff(oldResources, newResources))
		}

		// Perform the monitor resync and wait for controllers to report cache sync.
		if err := rq.resyncMonitors(newResources); err != nil {
			utilruntime.HandleError(fmt.Errorf("%s: failed to sync resource monitors: %v", rq.clusterName, err))
			return
		}
		// wait for caches to fill for a while (our sync period).
		// this protects us from deadlocks where available resources changed and one of our informer caches will never fill.
		// informers keep attempting to sync in the background, so retrying doesn't interrupt them.
		// the call to resyncMonitors on the reattempt will no-op for resources that still exist.
		if rq.quotaMonitor != nil && !cache.WaitForNamedCacheSync(fmt.Sprintf("%q resource quota", rq.clusterName), ctx.Done(), rq.quotaMonitor.IsSynced) {
			utilruntime.HandleError(fmt.Errorf("%s: timed out waiting for quota monitor sync", rq.clusterName))
			return
		}

		// success, remember newly synced resources
		oldResources = newResources
		klog.V(2).Infof("%s: synced quota controller", rq.clusterName)
	}()

	// List all the quotas (this is scoped to the workspace)
	quotas, err := rq.rqLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("%s: error listing all resourcequotas: %v", rq.clusterName, err))
	}

	// Requeue all quotas in the workspace
	for i := range quotas {
		quota := quotas[i]
		klog.V(2).Infof("%s: enqueuing resourcequota %s/%s because the list of available APIs changed", rq.clusterName, quota.Namespace, quota.Name)
		rq.addQuota(quota)
	}
}
