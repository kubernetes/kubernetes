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

package garbagecollector

func (gc *GarbageCollector) Sync(ctx context.Context, discoveryClient discovery.ServerResourcesInterface, period time.Duration) {
	oldResources := make(map[schema.GroupVersionResource]struct{})
	wait.UntilWithContext(ctx, func(ctx context.Context) {
		logger := klog.FromContext(ctx)

		// Get the current resource list from discovery.
		newResources := GetDeletableResources(logger, discoveryClient)

		// This can occur if there is an internal error in GetDeletableResources.
		if len(newResources) == 0 {
			logger.V(2).Info("no resources reported by discovery, skipping garbage collector sync")
			metrics.GarbageCollectorResourcesSyncError.Inc()
			return
		}

		// Decide whether discovery has reported a change.
		if reflect.DeepEqual(oldResources, newResources) {
			logger.V(5).Info("no resource updates from discovery, skipping garbage collector sync")
			return
		}

		// Ensure workers are paused to avoid processing events before informers
		// have resynced.
		gc.workerLock.Lock()
		defer gc.workerLock.Unlock()

		// Once we get here, we should not unpause workers until we've successfully synced
		attempt := 0
		wait.PollImmediateUntilWithContext(ctx, 100*time.Millisecond, func(ctx context.Context) (bool, error) {
			attempt++

			// On a reattempt, check if available resources have changed
			if attempt > 1 {
				newResources = GetDeletableResources(logger, discoveryClient)
				if len(newResources) == 0 {
					logger.V(2).Info("no resources reported by discovery", "attempt", attempt)
					metrics.GarbageCollectorResourcesSyncError.Inc()
					return false, nil
				}
			}

			logger.V(2).Info(
				"syncing garbage collector with updated resources from discovery",
				"attempt", attempt,
				"diff", printDiff(oldResources, newResources),
			)

			// Resetting the REST mapper will also invalidate the underlying discovery
			// client. This is a leaky abstraction and assumes behavior about the REST
			// mapper, but we'll deal with it for now.
			gc.restMapper.Reset()
			logger.V(4).Info("reset restmapper")

			// Perform the monitor resync and wait for controllers to report cache sync.
			//
			// NOTE: It's possible that newResources will diverge from the resources
			// discovered by restMapper during the call to Reset, since they are
			// distinct discovery clients invalidated at different times. For example,
			// newResources may contain resources not returned in the restMapper's
			// discovery call if the resources appeared in-between the calls. In that
			// case, the restMapper will fail to map some of newResources until the next
			// attempt.
			if err := gc.resyncMonitors(logger, newResources); err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to sync resource monitors (attempt %d): %v", attempt, err))
				metrics.GarbageCollectorResourcesSyncError.Inc()
				return false, nil
			}
			logger.V(4).Info("resynced monitors")

			// wait for caches to fill for a while (our sync period) before attempting to rediscover resources and retry syncing.
			// this protects us from deadlocks where available resources changed and one of our informer caches will never fill.
			// informers keep attempting to sync in the background, so retrying doesn't interrupt them.
			// the call to resyncMonitors on the reattempt will no-op for resources that still exist.
			// note that workers stay paused until we successfully resync.
			if !cache.WaitForNamedCacheSync("garbage collector", waitForStopOrTimeout(ctx.Done(), period), func() bool {
				return gc.dependencyGraphBuilder.IsSynced(logger)
			}) {
				utilruntime.HandleError(fmt.Errorf("timed out waiting for dependency graph builder sync during GC sync (attempt %d)", attempt))
				metrics.GarbageCollectorResourcesSyncError.Inc()
				return false, nil
			}

			// success, break out of the loop
			return true, nil
		})

		// Finally, keep track of our new state. Do this after all preceding steps
		// have succeeded to ensure we'll retry on subsequent syncs if an error
		// occurred.
		oldResources = newResources
		logger.V(2).Info("synced garbage collector")
	}, period)
}
