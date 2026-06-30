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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/utils/ktesting"
	clocktesting "k8s.io/utils/clock/testing"
)

func TestTerminationDeadlineDoesNotReset(t *testing.T) {
	now := time.Unix(1000, 0)
	pod := &v1.Pod{}
	status := &podSyncStatus{terminatingAt: now}
	deadline := calculateTerminationDeadline(status, pod, 120, false, now)
	require.Equal(t, now.Add(120*time.Second), deadline)
	status.terminationDeadline = deadline
	// A shorter request made a minute later gets five seconds, not five
	// seconds from the start of termination and not a fresh original budget.
	deadline = calculateTerminationDeadline(status, pod, 5, true, now.Add(time.Minute))
	require.Equal(t, now.Add(65*time.Second), deadline)
	status.terminationDeadline = deadline
	require.Equal(t, deadline, calculateTerminationDeadline(status, pod, 5, false, now.Add(61*time.Second)))
	require.Equal(t, deadline, calculateTerminationDeadline(status, pod, 120, false, now.Add(62*time.Second)))
	// Reconstructing a worker from an API pod must preserve an expired deadline.
	pod.DeletionTimestamp = &metav1.Time{Time: deadline}
	restarted := &podSyncStatus{terminatingAt: now.Add(70 * time.Second)}
	require.Equal(t, deadline, calculateTerminationDeadline(restarted, pod, 5, false, now.Add(70*time.Second)))
}

func TestTerminatingPodRequeuesWithoutCompleting(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarsRestartableDuringPodTermination, true)
	logger, ctx := ktesting.NewTestContext(t)
	workers, _, _ := createPodWorkers(logger)
	syncer := newPodSyncerFuncs(workers.podSyncer)
	workers.podSyncer = syncer
	grace := int64(60)
	restart := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "0", Name: "terminating", Namespace: "default"}, Spec: v1.PodSpec{
		TerminationGracePeriodSeconds: &grace,
		InitContainers:                []v1.Container{{Name: "sidecar", RestartPolicy: &restart}},
	}}
	expectedDeadline := workers.clock.Now().Add(time.Minute)
	complete := make(chan struct{})
	invocations := 0
	syncer.syncTerminatingPod = func(ctx context.Context, pod *v1.Pod, status *kubecontainer.PodStatus, grace *int64, deadline time.Time, fn func(*v1.PodStatus)) (bool, error) {
		require.Equal(t, expectedDeadline, deadline)
		invocations++
		return invocations > 1, nil
	}
	workers.podSyncer = syncer
	workers.UpdatePod(ctx, UpdatePodOptions{Pod: pod, UpdateType: kubetypes.SyncPodKill, KillPodOptions: &KillPodOptions{CompletedCh: complete}})
	drainWorkers(workers, 1)
	require.False(t, workers.IsPodKnownTerminated(pod.UID))
	require.False(t, workers.ShouldPodRuntimeBeRemoved(pod.UID))
	select {
	case <-complete:
		t.Fatal("kill waiter completed before containers stopped")
	default:
	}
	queued := workers.workQueue.(*fakeQueue).Items()
	require.NotEmpty(t, queued)
	require.Equal(t, time.Second, queued[len(queued)-1].Delay)
	fakeClock := workers.clock.(*clocktesting.FakeClock)
	require.Eventually(t, fakeClock.HasWaiters, time.Second, time.Millisecond)
	fakeClock.Step(time.Second)
	// No UpdatePod call or shared work queue consumer is involved in this retry.
	require.Eventually(t, func() bool { return workers.IsPodKnownTerminated(pod.UID) }, time.Second, time.Millisecond)
	drainWorkers(workers, 1)
	require.True(t, workers.IsPodKnownTerminated(pod.UID))
	select {
	case <-complete:
	default:
		t.Fatal("kill waiter not notified after termination")
	}
}

func TestTerminationRetryCannotPassDeadline(t *testing.T) {
	for _, syncErr := range []error{nil, errors.New("runtime unavailable")} {
		logger, _ := ktesting.NewTestContext(t)
		workers, _, _ := createPodWorkers(logger)
		workers.backOffPeriod = time.Minute
		deadline := workers.clock.Now().Add(500 * time.Millisecond)
		workers.completeWork(logger, "pod", false, syncErr, deadline)
		queued := workers.workQueue.(*fakeQueue).Items()
		require.Len(t, queued, 1)
		require.Equal(t, 500*time.Millisecond, queued[0].Delay)
	}
}
