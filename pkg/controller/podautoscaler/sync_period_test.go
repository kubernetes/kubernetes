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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	autoscalinglisters "k8s.io/client-go/listers/autoscaling/v2"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	consistencyutil "k8s.io/client-go/util/consistency"
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
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
		})
	}
}

func TestReconciliationDelayRatioIgnoresEventDrivenReconciliations(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
	tCtx := ktesting.Init(t)
	hpa := newUnreconcilableHPA(ptr.To[int32](5))
	ctrl, clk := newSyncPeriodTestController(t, hpa)

	ctrl.enqueueHPA(hpa)
	require.True(t, ctrl.processNextWorkItem(tCtx))

	clk.Step(2 * time.Second)
	ctrl.enqueueHPA(hpa)
	require.True(t, ctrl.processNextWorkItem(tCtx))
	assertDelayRatio(t, nil)

	// The requeue scheduled by the first reconciliation is still waiting. It
	// fires 3s after the event-driven reconciliation, which it is measured against.
	clk.Step(3 * time.Second)
	waitForQueueLen(t, ctrl.queue, 1)
	require.True(t, ctrl.processNextWorkItem(tCtx))
	assertDelayRatio(t, map[monitor.PeriodSourceLabel][]float64{monitor.PeriodSourceSpec: {0.6}})
}

func TestReconciliationDelayRatioGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, false)
	tCtx := ktesting.Init(t)
	hpa := newUnreconcilableHPA(ptr.To[int32](5))
	ctrl, clk := newSyncPeriodTestController(t, hpa)

	ctrl.enqueueHPA(hpa)
	require.True(t, ctrl.processNextWorkItem(tCtx))
	interval, override := ctrl.rateLimiter.Interval(testHPAKey)
	assert.Equal(t, testGlobalSyncPeriod, interval, "spec.syncPeriodSeconds must be ignored")
	assert.False(t, override)

	clk.Step(testGlobalSyncPeriod)
	waitForQueueLen(t, ctrl.queue, 1)
	require.True(t, ctrl.processNextWorkItem(tCtx))
	assertDelayRatio(t, nil)
	assert.Empty(t, ctrl.requeues.pending)
	assert.Empty(t, ctrl.requeues.eventDriven)
}

func TestSyncPeriodStateIsRemovedWithHPA(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)

	tests := []struct {
		name   string
		remove func(t *testing.T, ctx context.Context, ctrl *HorizontalController)
	}{
		{
			name: "deleted HPA",
			remove: func(t *testing.T, ctx context.Context, ctrl *HorizontalController) {
				ctrl.deleteHPA(newUnreconcilableHPA(ptr.To[int32](5)))
				require.True(t, ctrl.processNextWorkItem(ctx))
			},
		},
		{
			name: "reconciled HPA that is no longer in the lister",
			remove: func(t *testing.T, ctx context.Context, ctrl *HorizontalController) {
				deleted, err := ctrl.reconcileKey(ctx, testHPAKey)
				require.NoError(t, err)
				require.True(t, deleted)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			ctrl, clk := newSyncPeriodTestController(t)
			ctrl.rateLimiter.SetInterval(testHPAKey, 5*time.Second)
			ctrl.requeues.record(testHPAKey, pendingRequeue{completedAt: clk.Now(), delay: 5 * time.Second, source: monitor.PeriodSourceSpec})
			ctrl.requeues.markEventDriven(testHPAKey)

			tt.remove(t, tCtx, ctrl)

			_, override := ctrl.rateLimiter.Interval(testHPAKey)
			assert.False(t, override, "the per-HPA interval must be removed")
			assert.NotContains(t, ctrl.requeues.pending, testHPAKey)
			assert.False(t, ctrl.requeues.eventDriven.Has(testHPAKey))
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

	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	objects := make([]runtime.Object, 0, len(hpas))
	for _, hpa := range hpas {
		require.NoError(t, indexer.Add(hpa))
		objects = append(objects, hpa)
	}

	rateLimiter := NewDefaultHPARateLimiter(testGlobalSyncPeriod)
	clk, queue := newFakeClockQueue(t, rateLimiter)
	ctrl := &HorizontalController{
		hpaNamespacer:    fake.NewClientset(objects...).AutoscalingV2(),
		hpaLister:        autoscalinglisters.NewHorizontalPodAutoscalerLister(indexer),
		eventRecorder:    &record.FakeRecorder{},
		monitor:          monitor.New(),
		recommendations:  map[string][]timestampedRecommendation{},
		scaleUpEvents:    map[string][]timestampedScaleEvent{},
		scaleDownEvents:  map[string][]timestampedScaleEvent{},
		consistencyStore: consistencyutil.NewNoopConsistencyStore(),
		queue:            queue,
		rateLimiter:      rateLimiter,
		requeues:         newRequeueTracker(),
		clock:            clk,
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.HPAOptimizedSelectorStore) {
		ctrl.selectorTracker = newHPASelectorStore()
	} else {
		ctrl.selectorTracker = newBiMultimapSelectorTracker()
	}
	return ctrl, clk
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
