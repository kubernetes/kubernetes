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
)

// pendingRequeue describes the periodic requeue scheduled after a reconciliation.
type pendingRequeue struct {
	completedAt time.Time
	delay       time.Duration
	source      monitor.PeriodSourceLabel
}

// requeueTracker keeps the state needed to measure how late periodic
// reconciliations start compared with the delay they requested.
type requeueTracker struct {
	mu          sync.Mutex
	pending     map[string]pendingRequeue
	eventDriven sets.Set[string]
}

func newRequeueTracker() *requeueTracker {
	return &requeueTracker{
		pending:     map[string]pendingRequeue{},
		eventDriven: sets.New[string](),
	}
}

// markEventDriven records that the next reconciliation of the key was requested
// by an event rather than by a periodic requeue.
func (t *requeueTracker) markEventDriven(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.eventDriven.Insert(key)
}

// record stores the periodic requeue scheduled for the key.
func (t *requeueTracker) record(key string, requeue pendingRequeue) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pending[key] = requeue
}

// take removes the pending requeue of the key and returns it, unless the
// reconciliation that is about to start was requested by an event.
func (t *requeueTracker) take(key string) (pendingRequeue, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	requeue, ok := t.pending[key]
	delete(t.pending, key)
	if t.eventDriven.Has(key) {
		t.eventDriven.Delete(key)
		return pendingRequeue{}, false
	}
	return requeue, ok
}

// forget removes all state of the key.
func (t *requeueTracker) forget(key string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.pending, key)
	t.eventDriven.Delete(key)
}

// refreshSyncPeriod sets the requeue interval of the HPA from spec.syncPeriodSeconds,
// or restores the global interval if the field is unset or the feature is disabled.
func (a *HorizontalController) refreshSyncPeriod(key string, hpa *autoscalingv2.HorizontalPodAutoscaler) {
	if utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) &&
		hpa.Spec.SyncPeriodSeconds != nil && *hpa.Spec.SyncPeriodSeconds > 0 {
		a.rateLimiter.SetInterval(key, time.Duration(*hpa.Spec.SyncPeriodSeconds)*time.Second)
		return
	}
	a.rateLimiter.RemoveInterval(key)
}

// markEventDriven excludes the next reconciliation of the key from the delay measurement.
// It must be called before the key is added to the queue.
func (a *HorizontalController) markEventDriven(key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		return
	}
	a.requeues.markEventDriven(key)
}

// recordRequeue stores the completion time and the requested delay of the
// periodic requeue that follows a reconciliation of the key.
func (a *HorizontalController) recordRequeue(key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		return
	}
	delay, fromSpec := a.rateLimiter.Interval(key)
	source := monitor.PeriodSourceGlobal
	if fromSpec {
		source = monitor.PeriodSourceSpec
	}
	a.requeues.record(key, pendingRequeue{
		completedAt: a.clock.Now(),
		delay:       delay,
		source:      source,
	})
}

// observeRequeueDelay reports how the delay before a periodic reconciliation of
// the key compares with the requested delay. It must be called when the key is
// taken from the queue.
func (a *HorizontalController) observeRequeueDelay(ctx context.Context, key string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HPAConfigurableSyncPeriod) {
		return
	}
	requeue, ok := a.requeues.take(key)
	if !ok || requeue.delay <= 0 {
		return
	}
	observed := a.clock.Since(requeue.completedAt)
	a.monitor.ObserveReconciliationDelayRatio(requeue.source, observed.Seconds()/requeue.delay.Seconds())

	if logger := klog.FromContext(ctx).V(4); logger.Enabled() {
		namespace, name, _ := cache.SplitMetaNamespaceKey(key)
		logger.Info("Starting periodic HPA reconciliation", "HPA", klog.KRef(namespace, name),
			"requestedDelay", requeue.delay, "observedDelay", observed, "periodSource", requeue.source)
	}
}

// forgetSyncPeriod removes the requeue interval and the timing state of the key.
func (a *HorizontalController) forgetSyncPeriod(key string) {
	a.rateLimiter.RemoveInterval(key)
	a.requeues.forget(key)
}
