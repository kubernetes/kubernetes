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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	autoscalinglisters "k8s.io/client-go/listers/autoscaling/v2"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestDeleteHPACleansHistoryBeforeSyncPeriod(t *testing.T) {
	for _, tombstone := range []bool{false, true} {
		name := "HPA"
		if tombstone {
			name = "tombstone"
		}
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
			tCtx := ktesting.Init(t)
			hpa := newUnreconcilableHPA(ptr.To[int32](3600))
			ctrl, clk := newSyncPeriodTestController(t, hpa)
			indexer := setLifecycleTestLister(t, ctrl, hpa)

			ctrl.enqueueHPA(hpa)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			flushWaitingItems(t, clk, ctrl.queue)
			seedLifecycleTestHistory(ctrl)

			require.NoError(t, indexer.Delete(hpa))
			var deletedObj interface{} = hpa
			if tombstone {
				deletedObj = cache.DeletedFinalStateUnknown{Key: testHPAKey, Obj: hpa}
			}
			ctrl.deleteHPA(deletedObj)

			require.Equal(t, 1, ctrl.queue.Len(), "deletion must not wait for the one-hour timer")
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertLifecycleTestStateRemoved(t, ctrl)

			// The delaying queue cannot cancel the old timer; processing it again
			// must leave the deleted HPA's state empty.
			clk.Step(time.Hour)
			waitForQueueLen(t, ctrl.queue, 1)
			require.True(t, ctrl.processNextWorkItem(tCtx))
			assertLifecycleTestStateRemoved(t, ctrl)
		})
	}
}

func TestDeleteHPADuringReconciliation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HPAConfigurableSyncPeriod, true)
	tCtx := ktesting.Init(t)
	hpa := newUnreconcilableHPA(ptr.To[int32](3600))
	ctrl, _ := newSyncPeriodTestController(t, hpa)
	indexer := setLifecycleTestLister(t, ctrl, hpa)
	seedLifecycleTestHistory(ctrl)

	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	ctrl.hpaLister = &blockingLifecycleHPALister{
		HorizontalPodAutoscalerLister: ctrl.hpaLister,
		block: func() {
			close(entered)
			<-release
		},
	}

	ctrl.enqueueHPA(hpa)
	finished := make(chan bool, 1)
	go func() { finished <- ctrl.processNextWorkItem(tCtx) }()
	select {
	case <-entered:
	case <-time.After(time.Minute):
		t.Fatal("reconciliation did not read the HPA from the lister")
	}

	require.NoError(t, indexer.Delete(hpa))
	ctrl.deleteHPA(hpa)
	unblock()
	select {
	case processed := <-finished:
		require.True(t, processed)
	case <-time.After(time.Minute):
		t.Fatal("reconciliation did not finish after the lister was released")
	}

	// The in-flight reconciliation used the object read before deletion and
	// restored its interval. A second pass must remove that state promptly.
	_, override := ctrl.rateLimiter.Interval(testHPAKey)
	require.True(t, override)
	require.Equal(t, 1, ctrl.queue.Len())
	require.True(t, ctrl.processNextWorkItem(tCtx))
	assertLifecycleTestStateRemoved(t, ctrl)
	assertDelayRatio(t, nil)
}

func setLifecycleTestLister(t *testing.T, ctrl *HorizontalController, hpa *autoscalingv2.HorizontalPodAutoscaler) cache.Indexer {
	t.Helper()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	require.NoError(t, indexer.Add(hpa))
	ctrl.hpaLister = autoscalinglisters.NewHorizontalPodAutoscalerLister(indexer)
	return indexer
}

func seedLifecycleTestHistory(ctrl *HorizontalController) {
	ctrl.recordInitialRecommendation(9, testHPAKey)
	ctrl.scaleUpEvents[testHPAKey] = []timestampedScaleEvent{{replicaChange: 2, timestamp: time.Now()}}
	ctrl.scaleDownEvents[testHPAKey] = []timestampedScaleEvent{{replicaChange: 1, timestamp: time.Now()}}
}

func assertLifecycleTestStateRemoved(t *testing.T, ctrl *HorizontalController) {
	t.Helper()
	assert.NotContains(t, ctrl.recommendations, testHPAKey)
	assert.NotContains(t, ctrl.scaleUpEvents, testHPAKey)
	assert.NotContains(t, ctrl.scaleDownEvents, testHPAKey)
	_, override := ctrl.rateLimiter.Interval(testHPAKey)
	assert.False(t, override)
	assert.NotContains(t, ctrl.requeues.pending, testHPAKey)
	assert.False(t, ctrl.requeues.eventDriven.Has(testHPAKey))
}

type blockingLifecycleHPALister struct {
	autoscalinglisters.HorizontalPodAutoscalerLister
	block func()
}

func (l *blockingLifecycleHPALister) HorizontalPodAutoscalers(namespace string) autoscalinglisters.HorizontalPodAutoscalerNamespaceLister {
	return &blockingLifecycleHPANamespaceLister{
		HorizontalPodAutoscalerNamespaceLister: l.HorizontalPodAutoscalerLister.HorizontalPodAutoscalers(namespace),
		block:                                  l.block,
	}
}

type blockingLifecycleHPANamespaceLister struct {
	autoscalinglisters.HorizontalPodAutoscalerNamespaceLister
	block func()
}

func (l *blockingLifecycleHPANamespaceLister) Get(name string) (*autoscalingv2.HorizontalPodAutoscaler, error) {
	hpa, err := l.HorizontalPodAutoscalerNamespaceLister.Get(name)
	if err == nil {
		l.block()
	}
	return hpa, err
}
