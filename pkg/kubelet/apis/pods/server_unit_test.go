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

package pods_test

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/component-base/metrics/testutil"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	corefuzzer "k8s.io/kubernetes/pkg/apis/core/fuzzer"
	podsapi "k8s.io/kubernetes/pkg/kubelet/apis/pods"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubepodtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
)

func TestStartEventLoop(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	mockManager := new(kubepodtest.MockManager)
	server := podsapi.NewPodsServerForTest(broadcaster, mockManager)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}

	clientChannel := make(chan podsapi.PodWatchEvent, 100)
	broadcaster.Register(clientChannel)
	defer broadcaster.Unregister(clientChannel)

	server.OnPodUpdated(pod1, v1.PodStatus{Phase: v1.PodPending}, watch.Added)
	server.OnPodUpdated(pod1, v1.PodStatus{Phase: v1.PodSucceeded}, watch.Modified)
	server.OnPodRemoved(pod1)

	event := <-clientChannel
	assert.Equal(t, watch.Added, event.Type)
	assert.Equal(t, pod1.UID, event.UID)
	assert.Equal(t, v1.PodPending, event.Pod.Status.Phase)

	event = <-clientChannel
	assert.Equal(t, watch.Modified, event.Type)
	assert.Equal(t, pod1.UID, event.UID)
	assert.Equal(t, v1.PodSucceeded, event.Pod.Status.Phase)

	event = <-clientChannel
	assert.Equal(t, watch.Deleted, event.Type)
	assert.Equal(t, pod1.UID, event.UID)
	require.NotNil(t, event.Pod)
	assert.Equal(t, pod1.Name, event.Pod.Name)
	assert.Equal(t, pod1.Namespace, event.Pod.Namespace)
	assert.Equal(t, pod1.UID, event.Pod.UID)
	assert.Empty(t, event.Pod.Spec.Containers)

	select {
	case event := <-clientChannel:
		t.Errorf("Unexpected event received: %v", event)
	default:
	}
}

type MockWatchPodsServer struct {
	grpc.ServerStream
	Ctx     context.Context
	EventCh chan *podsv1alpha1.WatchPodsEvent
}

func (m *MockWatchPodsServer) Send(event *podsv1alpha1.WatchPodsEvent) error {
	select {
	case m.EventCh <- event:
		return nil
	case <-m.Ctx.Done():
		return m.Ctx.Err()
	}
}

func (m *MockWatchPodsServer) Context() context.Context {
	return m.Ctx
}

func TestWatchPods(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	mockManager := new(kubepodtest.MockManager)
	server := podsapi.NewPodsServerForTest(broadcaster, mockManager)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}

	mockManager.On("GetPods").Return([]*v1.Pod{pod1})
	mockManager.On("GetPodByUID", types.UID("pod1-uid")).Return(pod1, true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mockStream := &MockWatchPodsServer{
		Ctx:     ctx,
		EventCh: make(chan *podsv1alpha1.WatchPodsEvent, 10),
	}

	go func() {
		server.WatchPods(&podsv1alpha1.WatchPodsRequest{}, mockStream)
	}()

	// Verify initial ADDED event
	event := <-mockStream.EventCh
	assert.Equal(t, podsv1alpha1.EventType_ADDED, event.Type)
	podOut := &v1.Pod{}
	err := podOut.Unmarshal(event.Pod)
	require.NoError(t, err)
	assert.Equal(t, "pod1", podOut.Name)

	// Verify INITIAL_SYNC event
	event = <-mockStream.EventCh
	assert.Equal(t, podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE, event.Type)
	assert.Empty(t, event.Pod)

	// Trigger an update
	server.OnPodUpdated(pod1, pod1.Status, watch.Modified)

	// Verify MODIFIED event
	event = <-mockStream.EventCh
	assert.Equal(t, podsv1alpha1.EventType_MODIFIED, event.Type)
	podOut = &v1.Pod{}
	err = podOut.Unmarshal(event.Pod)
	require.NoError(t, err)
	assert.Equal(t, "pod1", podOut.Name)

	// Trigger a removal
	server.OnPodRemoved(pod1)

	// Verify DELETED event
	event = <-mockStream.EventCh
	assert.Equal(t, podsv1alpha1.EventType_DELETED, event.Type)
	podOut = &v1.Pod{}
	err = podOut.Unmarshal(event.Pod)
	require.NoError(t, err)
	assert.Equal(t, "pod1", podOut.Name)
	assert.Equal(t, "ns1", podOut.Namespace)
	assert.Equal(t, pod1.UID, podOut.UID)
	assert.Empty(t, podOut.Spec.Containers)
}

func TestListPods(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	mockManager := new(kubepodtest.MockManager)
	server := podsapi.NewPodsServerForTest(broadcaster, mockManager)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod2-uid", Name: "pod2", Namespace: "ns2"}}

	mockManager.On("GetPods").Return([]*v1.Pod{pod1, pod2})

	resp, err := server.ListPods(context.Background(), &podsv1alpha1.ListPodsRequest{})
	require.NoError(t, err)
	pod1Out := &v1.Pod{}
	err = pod1Out.Unmarshal(resp.Pods[0])
	require.NoError(t, err)
	pod2Out := &v1.Pod{}
	err = pod2Out.Unmarshal(resp.Pods[1])
	require.NoError(t, err)
	require.Equal(t, "pod1", pod1Out.Name)
	assert.Equal(t, "pod2", pod2Out.Name)
}

func TestGetPod(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	mockManager := new(kubepodtest.MockManager)
	server := podsapi.NewPodsServerForTest(broadcaster, mockManager)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
	pod1.Spec.EphemeralContainers = []v1.EphemeralContainer{
		{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    "debugger",
				Image:   "busybox",
				Command: []string{"sh"},
				Stdin:   true,
				TTY:     true,
			},
		},
	}
	pod1.Status = v1.PodStatus{Phase: v1.PodRunning}

	mockManager.On("GetPodByUID", types.UID("pod1-uid")).Return(pod1, true)

	resp, err := server.GetPod(context.Background(), &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
	require.NoError(t, err)
	podOut := &v1.Pod{}
	err = podOut.Unmarshal(resp.Pod)
	require.NoError(t, err)
	require.Equal(t, "pod1", podOut.Name)
	assert.Equal(t, v1.PodRunning, podOut.Status.Phase)
	require.Len(t, podOut.Spec.EphemeralContainers, 1)
	assert.Equal(t, "debugger", podOut.Spec.EphemeralContainers[0].Name)
	assert.Equal(t, "busybox", podOut.Spec.EphemeralContainers[0].Image)
	assert.Equal(t, []string{"sh"}, podOut.Spec.EphemeralContainers[0].Command)
	assert.True(t, podOut.Spec.EphemeralContainers[0].Stdin)
	assert.True(t, podOut.Spec.EphemeralContainers[0].TTY)
}

func TestStaticPod(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	mockManager := new(kubepodtest.MockManager)
	server := podsapi.NewPodsServerForTest(broadcaster, mockManager)

	staticPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "static-pod-uid",
			Name:      "static-pod",
			Namespace: "default",
			Annotations: map[string]string{
				"kubernetes.io/config.source": "file",
			},
		},
	}

	mockManager.On("GetPodByUID", types.UID("static-pod-uid")).Return(staticPod, true)

	resp, err := server.GetPod(context.Background(), &podsv1alpha1.GetPodRequest{PodUID: "static-pod-uid"})
	require.NoError(t, err)

	podOut := &v1.Pod{}
	err = podOut.Unmarshal(resp.Pod)
	require.NoError(t, err)
	assert.Equal(t, "static-pod", podOut.Name)
	assert.Equal(t, "file", podOut.Annotations["kubernetes.io/config.source"])
}

func TestErrorsAndMetrics(t *testing.T) {
	metrics.Register()

	t.Run("DroppedWatchEventIncrementsMetric", func(t *testing.T) {
		broadcaster := podsapi.NewBroadcaster()
		mockManager := new(kubepodtest.MockManager)
		server := podsapi.NewPodsServerForTest(broadcaster, mockManager)
		pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
		server.OnPodUpdated(pod1, pod1.Status, watch.Added)

		// Reset the metric before the test
		metrics.PodWatchEventsDroppedTotal.Reset()
		clientChannel := make(chan podsapi.PodWatchEvent, 1) // Buffered channel of size 1
		broadcaster.Register(clientChannel)
		defer broadcaster.Unregister(clientChannel)

		// Send two events. The first one should fill the buffer.
		broadcaster.Broadcast(podsapi.PodWatchEvent{UID: "1"})
		// The second one should be dropped and increment the metric.
		broadcaster.Broadcast(podsapi.PodWatchEvent{UID: "2"})

		// Wait for the background goroutine to process the events and update metrics
		err := wait.PollImmediate(10*time.Millisecond, 2*time.Second, func() (bool, error) {
			count, err := testutil.GetCounterMetricValue(metrics.PodWatchEventsDroppedTotal)
			if err != nil {
				return false, err
			}
			return count == 1, nil
		})
		require.NoError(t, err, "Metric PodWatchEventsDroppedTotal should be 1")

		err = testutil.CollectAndCompare(metrics.PodWatchEventsDroppedTotal, strings.NewReader(`
			# HELP kubelet_pod_watch_events_dropped_total [ALPHA] Cumulative number of pod watch events dropped.
			# TYPE kubelet_pod_watch_events_dropped_total counter
			kubelet_pod_watch_events_dropped_total 1
		`))
		require.NoError(t, err)
	})
}

func TestSerialize(t *testing.T) {
	apiObjectFuzzer := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, corefuzzer.Funcs), rand.NewSource(152), legacyscheme.Codecs)
	for i := 0; i < 100; i++ {
		pod := &v1.Pod{}
		apiObjectFuzzer.Fill(pod)
		podBytes, err := pod.Marshal()
		if err != nil {
			t.Fatal(err)
		}

		resp := &podsv1alpha1.ListPodsResponse{Pods: [][]byte{podBytes}}
		data, err := proto.Marshal(resp)
		if err != nil {
			t.Fatal(err)
		}

		resp2 := &podsv1alpha1.ListPodsResponse{}
		if err := proto.Unmarshal(data, resp2); err != nil {
			t.Fatal(err)
		}

		if !proto.Equal(resp, resp2) {
			t.Fatal("round-tripped objects were different")
		}
		pod2 := &v1.Pod{}
		if err := pod2.Unmarshal(resp2.Pods[0]); err != nil {
			t.Fatal(err)
		}
		if !apiequality.Semantic.DeepEqual(pod, pod2) {
			t.Fatal("round-tripped objects were different")
		}
	}
}

func TestBroadcaster_SlowClient(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()

	numFastClients := 3
	numEvents := 10
	fastClients := make([]chan podsapi.PodWatchEvent, numFastClients)
	for i := 0; i < numFastClients; i++ {
		fastClients[i] = make(chan podsapi.PodWatchEvent, numEvents)
		broadcaster.Register(fastClients[i])
	}

	slowClient := make(chan podsapi.PodWatchEvent)
	broadcaster.Register(slowClient)

	for i := 0; i < numEvents; i++ {
		broadcaster.Broadcast(podsapi.PodWatchEvent{UID: types.UID(fmt.Sprintf("event-%d", i))})
	}

	for i := 0; i < numFastClients; i++ {
		for j := 0; j < numEvents; j++ {
			select {
			case event := <-fastClients[i]:
				expectedUID := types.UID(fmt.Sprintf("event-%d", j))
				assert.Equal(t, expectedUID, event.UID, "Fast client %d missed or got wrong event", i)
			case <-time.After(2 * time.Second):
				t.Fatalf("Fast client %d timed out waiting for event %d", i, j)
			}
		}
	}

	err := wait.PollImmediate(10*time.Millisecond, 2*time.Second, func() (bool, error) {
		select {
		case _, ok := <-slowClient:
			if !ok {
				return true, nil
			}
			return false, nil
		default:
			count, _ := testutil.GetCounterMetricValue(metrics.PodWatchEventsDroppedTotal)
			return count >= 1, nil
		}
	})
	assert.NoError(t, err, "Slow client should have been dropped")
}
