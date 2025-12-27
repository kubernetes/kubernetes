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
	"math/rand"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	corefuzzer "k8s.io/kubernetes/pkg/apis/core/fuzzer"
	podsapi "k8s.io/kubernetes/pkg/kubelet/apis/pods"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

func TestStartEventLoop(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
	pod1Running := pod1.DeepCopy()
	pod1Running.Status = v1.PodStatus{Phase: v1.PodRunning}
	pod1Succeeded := pod1.DeepCopy()
	pod1Succeeded.Status = v1.PodStatus{Phase: v1.PodSucceeded}
	clientChannel := make(chan podsapi.PodWatchEvent, 100)
	broadcaster.Register(clientChannel)
	defer broadcaster.Unregister(clientChannel)
	server.OnPodAdded(pod1Running)
	server.OnPodStatusUpdated(pod1, pod1Succeeded.Status)
	server.OnPodRemoved(pod1)
	event := <-clientChannel
	assert.Equal(t, "ADDED", string(event.Type))
	assert.Equal(t, pod1.UID, event.Pod.UID)
	event = <-clientChannel
	assert.Equal(t, "MODIFIED", string(event.Type))
	assert.Equal(t, v1.PodSucceeded, event.Pod.Status.Phase)
	event = <-clientChannel
	assert.Equal(t, "DELETED", string(event.Type))
	assert.Equal(t, pod1.UID, event.Pod.UID)
}

func TestListPods(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod2-uid", Name: "pod2", Namespace: "ns2"}}
	status1 := v1.PodStatus{Phase: v1.PodRunning}
	status2 := v1.PodStatus{Phase: v1.PodSucceeded}
	server.OnPodAdded(pod1)
	server.OnPodAdded(pod2)
	server.OnPodStatusUpdated(pod1, status1)
	server.OnPodStatusUpdated(pod2, status2)
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
	server := podsapi.NewPodsServerForTest(broadcaster)
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
	status1 := v1.PodStatus{Phase: v1.PodRunning}
	server.OnPodAdded(pod1)
	server.OnPodStatusUpdated(pod1, status1)
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

func TestErrorsAndMetrics(t *testing.T) {
	metrics.Register()

	t.Run("DroppedWatchEventIncrementsMetric", func(t *testing.T) {
		broadcaster := podsapi.NewBroadcaster()
		server := podsapi.NewPodsServerForTest(broadcaster)
		pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
		server.OnPodAdded(pod1)

		// Reset the metric before the test
		metrics.PodWatchEventsDroppedTotal.Reset()
		clientChannel := make(chan podsapi.PodWatchEvent, 1) // Buffered channel of size 1
		broadcaster.Register(clientChannel)
		defer broadcaster.Unregister(clientChannel)

		// Send two events. The first one should fill the buffer.
		broadcaster.Broadcast(podsapi.PodWatchEvent{})
		// The second one should be dropped and increment the metric.
		broadcaster.Broadcast(podsapi.PodWatchEvent{})

		err := testutil.CollectAndCompare(metrics.PodWatchEventsDroppedTotal, strings.NewReader(`
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
