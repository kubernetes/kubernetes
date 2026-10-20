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

package dynamicresources

import (
	"context"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// affinityWaiters belongs to one scheduler profile. A resolved pool removes a
// Pod from every index, including unrelated pools which may never become ready.
type affinityWaiters struct {
	mu        sync.Mutex
	byPool    map[poolName]map[types.UID]*v1.Pod
	byPod     map[types.UID]sets.Set[poolName]
	closed    bool
	podLister corelisters.PodLister
}

var affinityPodLifecycleHandlerHook func(cache.ResourceEventHandler)

func (w *affinityWaiters) register(pod *v1.Pod, pools sets.Set[poolName]) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return false
	}
	if len(pools) == 0 {
		return true
	}
	if w.byPool == nil {
		w.byPool = make(map[poolName]map[types.UID]*v1.Pod)
		w.byPod = make(map[types.UID]sets.Set[poolName])
	}
	if w.byPod[pod.UID] == nil {
		w.byPod[pod.UID] = sets.New[poolName]()
	}
	for pool := range pools {
		if w.byPool[pool] == nil {
			w.byPool[pool] = make(map[types.UID]*v1.Pod)
		}
		w.byPool[pool][pod.UID] = pod
		w.byPod[pod.UID].Insert(pool)
	}
	return true
}

func (w *affinityWaiters) close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closed = true
	w.byPod = nil
	w.byPool = nil
}

func (w *affinityWaiters) remove(uid types.UID) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.removeLocked(uid)
}

func (w *affinityWaiters) removeLocked(uid types.UID) {
	for pool := range w.byPod[uid] {
		delete(w.byPool[pool], uid)
		if len(w.byPool[pool]) == 0 {
			delete(w.byPool, pool)
		}
	}
	delete(w.byPod, uid)
}

func (w *affinityWaiters) takeResolved(pools sets.Set[poolName], resolved func(poolName) bool) []*v1.Pod {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	var pods []*v1.Pod
	for pool := range pools {
		if len(w.byPool[pool]) == 0 || !resolved(pool) {
			continue
		}
		for uid, pod := range w.byPool[pool] {
			pods = append(pods, pod)
			w.removeLocked(uid)
		}
	}
	return pods
}

func (pl *DynamicResources) initAffinityWaiters(ctx context.Context) error {
	manager, ok := pl.draManager.(*DefaultDRAManager)
	if !ok {
		return nil
	}
	pods := pl.fh.SharedInformerFactory().Core().V1().Pods()
	pl.affinityWaiters = &affinityWaiters{podLister: pods.Lister()}
	pl.affinityTracker = manager.resourceClaimTracker
	handler := cache.ResourceEventHandlerFuncs{
		DeleteFunc: func(obj interface{}) {
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = tombstone.Obj
			}
			if pod, ok := obj.(*v1.Pod); ok {
				pl.affinityWaiters.remove(pod.UID)
			}
		},
		UpdateFunc: func(_, obj interface{}) {
			if pod, ok := obj.(*v1.Pod); ok && (pod.Spec.NodeName != "" || pod.DeletionTimestamp != nil) {
				pl.affinityWaiters.remove(pod.UID)
			}
		},
	}
	if affinityPodLifecycleHandlerHook != nil {
		affinityPodLifecycleHandlerHook(handler)
	}
	registration, err := pods.Informer().AddEventHandler(handler)
	if err != nil {
		return fmt.Errorf("register sharing affinity Pod lifecycle handler: %w", err)
	}
	unsubscribe := pl.affinityTracker.subscribeExclusionChanges(func(logger klog.Logger, changed sets.Set[poolName], startupDone bool) {
		if startupDone {
			changed = changed.Clone()
			if changed == nil {
				changed = sets.New[poolName]()
			}
			changed.Insert(poolName{})
		}
		pl.activateAffinityWaiters(logger, changed)
	})
	go func() {
		<-ctx.Done()
		pl.affinityWaiters.close()
		unsubscribe()
		if err := pods.Informer().RemoveEventHandler(registration); err != nil {
			klog.FromContext(ctx).Error(err, "Remove sharing affinity Pod lifecycle handler")
		}
	}()
	return nil
}

func (pl *DynamicResources) registerAffinityWaiters(logger klog.Logger, pod *v1.Pod, pools sets.Set[poolName]) {
	w := pl.affinityWaiters
	if w == nil || len(pools) == 0 {
		return
	}
	if !w.register(pod, pools) {
		return
	}
	// The informer store is updated before handlers run. Validating after
	// insertion closes the delete/bind-before-registration race.
	cached, err := w.podLister.Pods(pod.Namespace).Get(pod.Name)
	if err != nil || cached.UID != pod.UID || cached.Spec.NodeName != "" || cached.DeletionTimestamp != nil {
		w.remove(pod.UID)
		return
	}
	pl.activateAffinityWaiters(logger, pools)
}

func (pl *DynamicResources) activateAffinityWaiters(logger klog.Logger, changed sets.Set[poolName]) {
	pods := pl.affinityWaiters.takeResolved(changed, func(pool poolName) bool {
		resolved, err := pl.affinityTracker.exclusionResolved(pool)
		if err != nil {
			logger.V(5).Info("Cannot check sharing affinity pool readiness", "driver", pool.driver, "pool", pool.pool, "err", err)
		}
		return err == nil && resolved
	})
	if len(pods) == 0 {
		return
	}
	activate := make(map[string]*v1.Pod, len(pods))
	for _, pod := range pods {
		activate[pod.Namespace+"/"+pod.Name] = pod
	}
	pl.fh.Activate(logger, activate)
}
