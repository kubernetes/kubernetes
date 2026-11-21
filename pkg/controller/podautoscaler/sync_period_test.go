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
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	autoscalinglisters "k8s.io/client-go/listers/autoscaling/v2"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	consistencyutil "k8s.io/client-go/util/consistency"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/monitor"
	"k8s.io/kubernetes/pkg/features"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

const (
	testGlobalSyncPeriod = 15 * time.Second
	testHPAKey           = "test-ns/test-hpa"
	delayRatioMetricName = "horizontal_pod_autoscaler_controller_reconciliation_delay_ratio"
)

func TestReconciliationDelayRatio(t *testing.T) {
	tests := []struct {
		name              string
		gateDisabled      bool
		syncPeriodSeconds *int32
		delay             time.Duration
		expected          map[monitor.PeriodSourceLabel][]float64
	}{
		{
			name:              "per-HPA period",
			syncPeriodSeconds: ptr.To[int32](5),
			delay:             6 * time.Second,
			expected:          map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceSpec: {1.2}},
		},
		{
			name:     "global period",
			delay:    testGlobalSyncPeriod,
			expected: map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceGlobal: {1}},
		},
		{
			name:     "late reconciliation",
			delay:    3 * testGlobalSyncPeriod,
			expected: map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceGlobal: {3}},
		},
		{
			name:              "gate disabled",
			gateDisabled:      true,
			syncPeriodSeconds: ptr.To[int32](5),
			delay:             testGlobalSyncPeriod,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, !tt.gateDisabled)
			tCtx := ktesting.Init(t)
			hpa := newUnreconcilableHPA(tt.syncPeriodSeconds)
			ctrl, clk := newSyncPeriodTestController(t, hpa)

			ctrl.enqueueHPA(hpa)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertDelayRatio(t, nil)

			clk.Step(tt.delay)
			waitForQueueLen(t, ctrl.queue, 1)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertDelayRatio(t, tt.expected)
			if tt.gateDisabled {
				assert.Empty(t, ctrl.requeues.pending)
			}
		})
	}
}

func TestReconciliationDelayRatioAfterEventDrivenReconciliation(t *testing.T) {
	tests := []struct {
		name                     string
		syncPeriodSeconds        int32
		updatedSyncPeriodSeconds int32
		// updateAt and periodicAt are measured from the first reconciliation.
		updateAt   time.Duration
		periodicAt time.Duration
	}{
		{
			name:                     "update before the requeue is due",
			syncPeriodSeconds:        5,
			updatedSyncPeriodSeconds: 5,
			updateAt:                 2 * time.Second,
			periodicAt:               5 * time.Second,
		},
		{
			name:                     "update to a longer period before the requeue is due",
			syncPeriodSeconds:        5,
			updatedSyncPeriodSeconds: 30,
			updateAt:                 2 * time.Second,
			periodicAt:               5 * time.Second,
		},
		{
			name:                     "update to a shorter period before the requeue is due",
			syncPeriodSeconds:        30,
			updatedSyncPeriodSeconds: 5,
			updateAt:                 2 * time.Second,
			periodicAt:               7 * time.Second,
		},
		{
			name:                     "update when the requeue is due",
			syncPeriodSeconds:        5,
			updatedSyncPeriodSeconds: 5,
			updateAt:                 5 * time.Second,
			periodicAt:               10 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
			tCtx := ktesting.Init(t)
			hpa := newUnreconcilableHPA(new(tt.syncPeriodSeconds))
			ctrl, clk := newSyncPeriodTestController(t, hpa)

			ctrl.enqueueHPA(hpa)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertDelayRatio(t, nil)

			clk.Step(tt.updateAt)
			if tt.updateAt >= time.Duration(tt.syncPeriodSeconds)*time.Second {
				// Let the due requeue reach the queue first, so that the
				// event-driven reconciliation consumes it.
				waitForQueueLen(t, ctrl.queue, 1)
			}
			updated := hpa.DeepCopy()
			updated.Generation++
			updated.Spec.SyncPeriodSeconds = new(tt.updatedSyncPeriodSeconds)
			setTestHPALister(t, ctrl, updated)
			ctrl.updateHPA(hpa, updated)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertDelayRatio(t, nil)

			// The queue keeps the earliest requeue of the key, and the periodic
			// reconciliation starts exactly when that requeue is due.
			clk.Step(tt.periodicAt - tt.updateAt)
			waitForQueueLen(t, ctrl.queue, 1)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertDelayRatio(t, map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceSpec: {1}})
		})
	}
}

func TestRequeueStateIsRemovedWithHPA(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
	tCtx := ktesting.Init(t)
	hpa := newUnreconcilableHPA(ptr.To[int32](5))
	ctrl, clk := newSyncPeriodTestController(t, hpa)

	ctrl.enqueueHPA(hpa)
	require.True(t, ctrl.processNextWorkItem(tCtx))
	assertDelayRatio(t, nil)
	require.Contains(t, ctrl.requeues.pending, testHPAKey)

	// The work queue cannot cancel the requeue of a deleted HPA, and the
	// reconciliation it starts removes the state of the HPA.
	ctrl.deleteHPA(hpa)
	setTestHPALister(t, ctrl)
	clk.Step(5 * time.Second)
	waitForQueueLen(t, ctrl.queue, 1)
	require.True(t, ctrl.processNextWorkItem(tCtx))

	assertDelayRatio(t, map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceSpec: {1}})
	assert.Empty(t, ctrl.requeues.pending)
	assert.Empty(t, ctrl.requeues.eventDriven)
}

func TestUpdateHPARequeuesAfterSyncPeriod(t *testing.T) {
	tests := []struct {
		name              string
		gateEnabled       bool
		syncPeriodSeconds *int32
		// expectedDelay is zero if the update must use the rate limiter of the queue.
		expectedDelay time.Duration
	}{
		{
			name:              "per-HPA period",
			gateEnabled:       true,
			syncPeriodSeconds: ptr.To[int32](30),
			expectedDelay:     30 * time.Second,
		},
		{
			name:        "global period without the field",
			gateEnabled: true,
		},
		{
			name:              "global period for a non-positive value",
			gateEnabled:       true,
			syncPeriodSeconds: ptr.To[int32](0),
		},
		{
			name:              "gate disabled ignores the field",
			syncPeriodSeconds: ptr.To[int32](30),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, tt.gateEnabled)
			spy := newSpyWorkQueue(testGlobalSyncPeriod)
			defer spy.ShutDown()
			ctrl := newTestEnqueueController(spy)
			oldHPA := newUnreconcilableHPA(tt.syncPeriodSeconds)
			curHPA := oldHPA.DeepCopy()
			curHPA.Status.CurrentReplicas = 5

			ctrl.updateHPA(oldHPA, curHPA)

			assert.Empty(t, spy.getAddCalls())
			if tt.expectedDelay == 0 {
				assert.Equal(t, []string{testHPAKey}, spy.getAddRateLimitedCalls())
				assert.Empty(t, spy.getAddAfterDelays())
			} else {
				assert.Empty(t, spy.getAddRateLimitedCalls())
				assert.Equal(t, []time.Duration{tt.expectedDelay}, spy.getAddAfterDelays())
			}
		})
	}
}

// newUnreconcilableHPA returns an HPA whose reconciliation fails right after
// the controller reads it, because its scale target has an invalid API version.
func newUnreconcilableHPA(syncPeriodSeconds *int32) *autoscalingv2.HorizontalPodAutoscaler {
	return &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-hpa",
			Namespace: "test-ns",
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "invalid/api/version",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			MaxReplicas:       5,
			SyncPeriodSeconds: syncPeriodSeconds,
		},
	}
}

// newSyncPeriodTestController returns a controller that serves the given HPAs
// from its lister and uses a fake clock for its queue and delay measurements.
func newSyncPeriodTestController(t *testing.T, hpas ...*autoscalingv2.HorizontalPodAutoscaler) (*HorizontalController, *testingclock.FakeClock) {
	t.Helper()
	monitor.Register()
	monitor.ReconciliationDelayRatio.Reset()

	objects := make([]runtime.Object, 0, len(hpas))
	for _, hpa := range hpas {
		objects = append(objects, hpa)
	}

	clk := testingclock.NewFakeClock(time.Now())
	queue := workqueue.NewTypedRateLimitingQueueWithConfig(
		NewDefaultHPARateLimiter(testGlobalSyncPeriod),
		workqueue.TypedRateLimitingQueueConfig[string]{Clock: clk},
	)
	t.Cleanup(queue.ShutDown)
	ctrl := &HorizontalController{
		hpaNamespacer:    fake.NewClientset(objects...).AutoscalingV2(),
		eventRecorder:    &record.FakeRecorder{},
		monitor:          monitor.New(),
		recommendations:  map[string][]timestampedRecommendation{},
		scaleUpEvents:    map[string][]timestampedScaleEvent{},
		scaleDownEvents:  map[string][]timestampedScaleEvent{},
		consistencyStore: consistencyutil.NewNoopConsistencyStore(),
		queue:            queue,
		requeues:         newRequeueTracker(testGlobalSyncPeriod),
	}
	ctrl.requeues.clock = clk
	setTestHPALister(t, ctrl, hpas...)
	if utilfeature.DefaultFeatureGate.Enabled(features.HPAOptimizedSelectorStore) {
		ctrl.selectorTracker = newHPASelectorStore()
	} else {
		ctrl.selectorTracker = newBiMultimapSelectorTracker()
	}
	return ctrl, clk
}

// setTestHPALister makes the lister of the controller serve exactly the given HPAs.
func setTestHPALister(t *testing.T, ctrl *HorizontalController, hpas ...*autoscalingv2.HorizontalPodAutoscaler) {
	t.Helper()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	for _, hpa := range hpas {
		require.NoError(t, indexer.Add(hpa))
	}
	ctrl.hpaLister = autoscalinglisters.NewHorizontalPodAutoscalerLister(indexer)
}

// assertDelayRatio checks that the reconciliation delay ratio histogram holds
// exactly the given observations, in observation order per period source.
func assertDelayRatio(t *testing.T, observations map[monitor.PeriodSourceLabel][]float64) {
	t.Helper()
	var expected strings.Builder
	expected.WriteString(`# HELP horizontal_pod_autoscaler_controller_reconciliation_delay_ratio [ALPHA] Ratio of the observed delay to the requested delay between the end of an HPA reconciliation and the start of the next periodic reconciliation. The label 'period_source' is 'global' for HPAs that use the controller-wide sync period and 'spec' for HPAs that set spec.syncPeriodSeconds.
# TYPE horizontal_pod_autoscaler_controller_reconciliation_delay_ratio histogram
`)
	for _, source := range []monitor.PeriodSourceLabel{monitor.PeriodSourceGlobal, monitor.PeriodSourceSpec} {
		ratios, ok := observations[source]
		if !ok {
			continue
		}
		sum := 0.0
		for _, ratio := range ratios {
			sum += ratio
		}
		for _, le := range []float64{1, 1.1, 1.25, 1.5, 2, 3, 5, 10} {
			count := 0
			for _, ratio := range ratios {
				if ratio <= le {
					count++
				}
			}
			fmt.Fprintf(&expected, "%s_bucket{period_source=%q,le=\"%v\"} %d\n", delayRatioMetricName, source, le, count)
		}
		fmt.Fprintf(&expected, "%s_bucket{period_source=%q,le=\"+Inf\"} %d\n", delayRatioMetricName, source, len(ratios))
		fmt.Fprintf(&expected, "%s_sum{period_source=%q} %v\n", delayRatioMetricName, source, sum)
		fmt.Fprintf(&expected, "%s_count{period_source=%q} %d\n", delayRatioMetricName, source, len(ratios))
	}
	err := metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected.String()), delayRatioMetricName)
	assert.NoError(t, err)
}

func waitForQueueLen(t *testing.T, queue workqueue.TypedRateLimitingInterface[string], length int) {
	t.Helper()
	err := wait.PollUntilContextTimeout(context.Background(), time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		return queue.Len() == length, nil
	})
	require.NoError(t, err, "queue length did not become %d, it is %d", length, queue.Len())
}

// BenchmarkRequeueTrackerParallel measures the tracker work of a periodic
// reconciliation under contention: every iteration takes the pending requeue
// of an HPA and records the next one, with no other work outside the lock.
func BenchmarkRequeueTrackerParallel(b *testing.B) {
	featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
	for _, hpaCount := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("%dhpas", hpaCount), func(b *testing.B) {
			tracker := newRequeueTracker(testGlobalSyncPeriod)
			keys := make([]string, hpaCount)
			hpas := make([]*autoscalingv2.HorizontalPodAutoscaler, hpaCount)
			for i := range hpaCount {
				keys[i] = fmt.Sprintf("ns-%d/hpa-%d", i%10, i)
				hpas[i] = &autoscalingv2.HorizontalPodAutoscaler{}
				if i%2 == 0 {
					hpas[i].Spec.SyncPeriodSeconds = ptr.To[int32](30)
				}
				tracker.record(keys[i], hpas[i])
			}
			var next atomic.Int64
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					i := int(next.Add(1) % int64(hpaCount))
					tracker.take(keys[i])
					tracker.record(keys[i], hpas[i])
				}
			})
		})
	}
}
