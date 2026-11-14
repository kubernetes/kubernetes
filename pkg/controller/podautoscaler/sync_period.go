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

package podautoscaler

import (
	"context"
	"sync"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/monitor"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/clock"
)

// pendingRequeue describes the periodic requeue scheduled after a reconciliation.
type pendingRequeue struct {
	completedAt time.Time
	delay       time.Duration
	source      monitor.PeriodSourceLabel
}

func (r pendingRequeue) dueAt() time.Time {
	return r.completedAt.Add(r.delay)
}

// requeueTracker keeps the state needed to measure how late periodic
// reconciliations start compared with the delay they requested. Each pending
// requeue is removed by the reconciliation it schedules. The work queue cannot
// cancel that reconciliation, so the state of a deleted HPA goes away with its
// last periodic pass.
type requeueTracker struct {
	clock clock.PassiveClock
	// globalPeriod is the delay that the rate limiter of the work queue applies
	// to HPAs without spec.syncPeriodSeconds.
	globalPeriod time.Duration

	mu          sync.Mutex
	pending     map[string]pendingRequeue
	eventDriven sets.Set[string]
}

func newRequeueTracker(globalPeriod time.Duration) *requeueTracker {
	return &requeueTracker{
		clock:        clock.RealClock{},
		globalPeriod: globalPeriod,
		pending:      map[string]pendingRequeue{},
		eventDriven:  sets.New[string](),
	}
}

// markEventDriven records that the next reconciliation of the key was requested
// by an event rather than by a periodic requeue.
func (t *requeueTracker) markEventDriven(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.eventDriven.Insert(key)
}

// record stores the periodic requeue scheduled for the HPA after a
// reconciliation. The delaying queue keeps only the earliest requeue of a key,
// so a pending requeue that is due earlier is kept.
func (t *requeueTracker) record(key string, hpa *autoscalingv2.HorizontalPodAutoscaler) {
	requeue := pendingRequeue{completedAt: t.clock.Now(), delay: t.globalPeriod, source: monitor.PeriodSourceGlobal}
	if period, ok := syncPeriodFor(hpa); ok {
		requeue.delay, requeue.source = period, monitor.PeriodSourceSpec
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if pending, ok := t.pending[key]; ok && pending.dueAt().Before(requeue.dueAt()) {
		return
	}
	t.pending[key] = requeue
}

// take removes the pending requeue of the key and returns it with the time that
// has passed since it was scheduled, unless the reconciliation that is about to
// start was requested by an event. Such a reconciliation keeps a requeue that
// is not due yet: the queue cannot cancel it, so it still fires and is measured
// against its own delay.
func (t *requeueTracker) take(key string) (pendingRequeue, time.Duration, bool) {
	now := t.clock.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	requeue, ok := t.pending[key]
	if t.eventDriven.Has(key) {
		t.eventDriven.Delete(key)
		if ok && !now.Before(requeue.dueAt()) {
			delete(t.pending, key)
		}
		return pendingRequeue{}, 0, false
	}
	if !ok {
		return pendingRequeue{}, 0, false
	}
	delete(t.pending, key)
	return requeue, now.Sub(requeue.completedAt), true
}

// syncPeriodFor returns spec.syncPeriodSeconds of the HPA as a duration, or
// false if the HPA uses the global sync period.
func syncPeriodFor(hpa *autoscalingv2.HorizontalPodAutoscaler) (time.Duration, bool) {
	// A non-positive delay would requeue the HPA immediately, in a loop.
	if hpa == nil || !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) ||
		hpa.Spec.SyncPeriodSeconds == nil || *hpa.Spec.SyncPeriodSeconds <= 0 {
		return 0, false
	}
	return time.Duration(*hpa.Spec.SyncPeriodSeconds) * time.Second, true
}

// addAfterSyncPeriod adds the key to the queue after the sync period of the
// HPA. HPAs without spec.syncPeriodSeconds use the global period of the rate
// limiter of the queue.
func (a *HorizontalController) addAfterSyncPeriod(key string, hpa *autoscalingv2.HorizontalPodAutoscaler) {
	if period, ok := syncPeriodFor(hpa); ok {
		a.queue.AddAfter(key, period)
		return
	}
	a.queue.AddRateLimited(key)
}

// requeueAfterSyncPeriod schedules the periodic reconciliation that follows a
// reconciliation of the key. The HPA is read from the lister because
// reconcileKey can return before reading it, and the requeue must still use
// the HPA's own period.
func (a *HorizontalController) requeueAfterSyncPeriod(key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		a.queue.AddRateLimited(key)
		return
	}
	var hpa *autoscalingv2.HorizontalPodAutoscaler
	if namespace, name, err := cache.SplitMetaNamespaceKey(key); err == nil {
		hpa, _ = a.hpaLister.HorizontalPodAutoscalers(namespace).Get(name)
	}
	a.requeues.record(key, hpa)
	a.addAfterSyncPeriod(key, hpa)
}

// markEventDriven excludes the next reconciliation of the key from the delay measurement.
// It must be called before the key is added to the queue.
func (a *HorizontalController) markEventDriven(key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		return
	}
	a.requeues.markEventDriven(key)
}

// observeRequeueDelay reports how the delay before a periodic reconciliation of
// the key compares with the requested delay. It must be called when the key is
// taken from the queue.
func (a *HorizontalController) observeRequeueDelay(ctx context.Context, key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		return
	}
	requeue, observed, ok := a.requeues.take(key)
	if !ok || requeue.delay <= 0 {
		return
	}
	a.monitor.ObserveReconciliationDelayRatio(requeue.source, observed.Seconds()/requeue.delay.Seconds())

	if logger := klog.FromContext(ctx).V(4); logger.Enabled() {
		namespace, name, _ := cache.SplitMetaNamespaceKey(key)
		logger.Info("Starting periodic HPA reconciliation", "HPA", klog.KRef(namespace, name),
			"requestedDelay", requeue.delay, "observedDelay", observed, "periodSource", requeue.source)
	}
}
