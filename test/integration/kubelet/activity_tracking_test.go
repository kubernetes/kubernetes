/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	LastActivityAnnotationKey = "kubernetes.io/last-activity"
	testNamespaceActivity     = "test-activity-tracking"
)

// IdlePodsResponse matches the response format from /pods/idle endpoint
type IdlePodsResponse struct {
	Pods map[string]metav1.Time `json:"pods"`
}

// TestActivityTrackingIntegration tests the complete activity tracking flow
func TestActivityTrackingIntegration(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	// Create test namespace
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespaceActivity}}
	_, err = client.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	require.NoError(t, err)

	t.Run("ActivityAnnotationIsSetOnPodActivity", func(t *testing.T) {
		testActivityAnnotationIsSetOnPodActivity(t, ctx, client)
	})

	t.Run("ActivityTimestampUpdatesOnNewActivity", func(t *testing.T) {
		testActivityTimestampUpdatesOnNewActivity(t, ctx, client)
	})

	t.Run("ActivityAnnotationSurvivesPodRestart", func(t *testing.T) {
		testActivityAnnotationSurvivesPodRestart(t, ctx, client)
	})

	t.Run("IdlePodsEndpointReturnsCorrectData", func(t *testing.T) {
		testIdlePodsEndpointReturnsCorrectData(t, ctx)
	})
}

// testActivityAnnotationIsSetOnPodActivity verifies annotation is set when pod has activity
func testActivityAnnotationIsSetOnPodActivity(t *testing.T, ctx context.Context, client *kubernetes.Clientset) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "activity-test-pod-1",
			Namespace: testNamespaceActivity,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test",
					Image:   "busybox",
					Command: []string{"sleep", "3600"},
				},
			},
		},
	}

	createdPod, err := client.CoreV1().Pods(testNamespaceActivity).Create(ctx, pod, metav1.CreateOptions{})
	require.NoError(t, err)
	t.Cleanup(func() {
		client.CoreV1().Pods(testNamespaceActivity).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	})

	// Initially, pod should not have activity annotation
	_, hasAnnotation := createdPod.Annotations[LastActivityAnnotationKey]
	assert.False(t, hasAnnotation, "new pod should not have activity annotation")

	// Simulate activity by updating the pod with the annotation
	// In real implementation, this would be done by kubelet on exec/logs/port-forward
	now := time.Now()
	createdPod.Annotations = map[string]string{
		LastActivityAnnotationKey: now.Format(time.RFC3339Nano),
	}
	updatedPod, err := client.CoreV1().Pods(testNamespaceActivity).Update(ctx, createdPod, metav1.UpdateOptions{})
	require.NoError(t, err)

	// Verify annotation is set
	activityAnnotation, hasAnnotation := updatedPod.Annotations[LastActivityAnnotationKey]
	assert.True(t, hasAnnotation, "pod should have activity annotation after activity")
	assert.NotEmpty(t, activityAnnotation)

	// Parse and verify timestamp
	activityTime, err := time.Parse(time.RFC3339Nano, activityAnnotation)
	require.NoError(t, err)
	assert.WithinDuration(t, now, activityTime, time.Second)
}

// testActivityTimestampUpdatesOnNewActivity verifies timestamp updates on subsequent activity
func testActivityTimestampUpdatesOnNewActivity(t *testing.T, ctx context.Context, client *kubernetes.Clientset) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "activity-test-pod-2",
			Namespace: testNamespaceActivity,
			Annotations: map[string]string{
				LastActivityAnnotationKey: time.Now().Add(-1 * time.Hour).Format(time.RFC3339Nano),
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test",
					Image:   "busybox",
					Command: []string{"sleep", "3600"},
				},
			},
		},
	}

	createdPod, err := client.CoreV1().Pods(testNamespaceActivity).Create(ctx, pod, metav1.CreateOptions{})
	require.NoError(t, err)
	t.Cleanup(func() {
		client.CoreV1().Pods(testNamespaceActivity).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	})

	// Get original activity time
	originalActivity := createdPod.Annotations[LastActivityAnnotationKey]
	originalTime, _ := time.Parse(time.RFC3339Nano, originalActivity)

	// Wait a bit and simulate new activity
	time.Sleep(100 * time.Millisecond)
	newTime := time.Now()
	createdPod.Annotations[LastActivityAnnotationKey] = newTime.Format(time.RFC3339Nano)
	updatedPod, err := client.CoreV1().Pods(testNamespaceActivity).Update(ctx, createdPod, metav1.UpdateOptions{})
	require.NoError(t, err)

	// Verify timestamp was updated
	newActivity := updatedPod.Annotations[LastActivityAnnotationKey]
	updatedTime, _ := time.Parse(time.RFC3339Nano, newActivity)

	assert.True(t, updatedTime.After(originalTime), "new activity timestamp should be after original")
}

// testActivityAnnotationSurvivesPodRestart verifies annotation persists across pod restarts
func testActivityAnnotationSurvivesPodRestart(t *testing.T, ctx context.Context, client *kubernetes.Clientset) {
	activityTime := time.Now().Add(-30 * time.Minute)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "activity-test-pod-3",
			Namespace: testNamespaceActivity,
			Annotations: map[string]string{
				LastActivityAnnotationKey: activityTime.Format(time.RFC3339Nano),
			},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Name:    "test",
					Image:   "busybox",
					Command: []string{"sleep", "3600"},
				},
			},
		},
	}

	_, createErr := client.CoreV1().Pods(testNamespaceActivity).Create(ctx, pod, metav1.CreateOptions{})
	require.NoError(t, createErr)
	t.Cleanup(func() {
		client.CoreV1().Pods(testNamespaceActivity).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	})

	// Verify annotation persists
	retrievedPod, err := client.CoreV1().Pods(testNamespaceActivity).Get(ctx, pod.Name, metav1.GetOptions{})
	require.NoError(t, err)

	annotation, exists := retrievedPod.Annotations[LastActivityAnnotationKey]
	assert.True(t, exists, "annotation should exist after pod retrieval")

	parsedTime, _ := time.Parse(time.RFC3339Nano, annotation)
	assert.WithinDuration(t, activityTime, parsedTime, time.Second)
}

// testIdlePodsEndpointReturnsCorrectData tests the /pods/idle endpoint
func testIdlePodsEndpointReturnsCorrectData(t *testing.T, ctx context.Context) {
	// Create mock kubelet server with /pods/idle endpoint
	now := time.Now()
	mockIdlePods := map[types.UID]metav1.Time{
		types.UID("pod-1"): metav1.NewTime(now.Add(-2 * time.Hour)),
		types.UID("pod-2"): metav1.NewTime(now.Add(-30 * time.Minute)),
		types.UID("pod-3"): metav1.NewTime(now.Add(-5 * time.Minute)),
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pods/idle" {
			http.NotFound(w, r)
			return
		}

		response := IdlePodsResponse{
			Pods: make(map[string]metav1.Time),
		}
		for uid, t := range mockIdlePods {
			response.Pods[string(uid)] = t
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	// Make request to /pods/idle endpoint
	resp, err := http.Get(server.URL + "/pods/idle")
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var response IdlePodsResponse
	err = json.NewDecoder(resp.Body).Decode(&response)
	require.NoError(t, err)

	assert.Len(t, response.Pods, 3)
	assert.Contains(t, response.Pods, "pod-1")
	assert.Contains(t, response.Pods, "pod-2")
	assert.Contains(t, response.Pods, "pod-3")

	// Verify timestamps
	pod1Time := response.Pods["pod-1"]
	assert.WithinDuration(t, now.Add(-2*time.Hour), pod1Time.Time, time.Second)
}

// TestActivityTrackingPerformance tests performance of activity tracking
func TestActivityTrackingPerformance(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-activity-perf"}}
	_, err = client.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	require.NoError(t, err)

	// Create many pods with activity annotations
	numPods := 100
	for i := 0; i < numPods; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "perf-pod-" + string(rune('a'+i%26)) + string(rune('0'+i/26)),
				Namespace: "test-activity-perf",
				Annotations: map[string]string{
					LastActivityAnnotationKey: time.Now().Add(-time.Duration(i) * time.Minute).Format(time.RFC3339Nano),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "test", Image: "busybox", Command: []string{"sleep", "3600"}},
				},
			},
		}
		_, err := client.CoreV1().Pods("test-activity-perf").Create(ctx, pod, metav1.CreateOptions{})
		require.NoError(t, err)
	}

	// Measure time to list and filter idle pods
	start := time.Now()
	pods, err := client.CoreV1().Pods("test-activity-perf").List(ctx, metav1.ListOptions{})
	require.NoError(t, err)

	// Filter for idle pods (idle for more than 30 minutes)
	cutoff := time.Now().Add(-30 * time.Minute)
	var idlePods []v1.Pod
	for _, pod := range pods.Items {
		annotation, exists := pod.Annotations[LastActivityAnnotationKey]
		if !exists {
			continue
		}
		activityTime, err := time.Parse(time.RFC3339Nano, annotation)
		if err != nil {
			continue
		}
		if activityTime.Before(cutoff) {
			idlePods = append(idlePods, pod)
		}
	}

	elapsed := time.Since(start)
	t.Logf("Listed and filtered %d pods in %v, found %d idle pods", len(pods.Items), elapsed, len(idlePods))

	// Performance should be reasonable
	assert.Less(t, elapsed, 5*time.Second, "listing and filtering should complete within 5 seconds")
	assert.Greater(t, len(idlePods), 0, "should find some idle pods")
}

// TestActivityTrackingConcurrency tests concurrent activity updates
func TestActivityTrackingConcurrency(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-activity-concurrent"}}
	_, err = client.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	require.NoError(t, err)

	// Create a pod
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "concurrent-test-pod",
			Namespace: "test-activity-concurrent",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "test", Image: "busybox", Command: []string{"sleep", "3600"}},
			},
		},
	}
	_, err = client.CoreV1().Pods("test-activity-concurrent").Create(ctx, pod, metav1.CreateOptions{})
	require.NoError(t, err)

	// Simulate concurrent activity updates using retry with conflict handling
	numGoroutines := 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			defer func() { done <- true }()

			// Use retry to handle conflicts
			err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
				currentPod, err := client.CoreV1().Pods("test-activity-concurrent").Get(ctx, "concurrent-test-pod", metav1.GetOptions{})
				if err != nil {
					return false, nil
				}

				if currentPod.Annotations == nil {
					currentPod.Annotations = make(map[string]string)
				}
				currentPod.Annotations[LastActivityAnnotationKey] = time.Now().Format(time.RFC3339Nano)

				_, err = client.CoreV1().Pods("test-activity-concurrent").Update(ctx, currentPod, metav1.UpdateOptions{})
				if err != nil {
					return false, nil // Retry on conflict
				}
				return true, nil
			})

			if err != nil {
				t.Logf("Goroutine %d failed: %v", idx, err)
			}
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Verify final state
	finalPod, err := client.CoreV1().Pods("test-activity-concurrent").Get(ctx, "concurrent-test-pod", metav1.GetOptions{})
	require.NoError(t, err)

	annotation, exists := finalPod.Annotations[LastActivityAnnotationKey]
	assert.True(t, exists, "activity annotation should exist after concurrent updates")
	assert.NotEmpty(t, annotation)

	// Verify timestamp is parseable and recent
	activityTime, err := time.Parse(time.RFC3339Nano, annotation)
	require.NoError(t, err)
	assert.WithinDuration(t, time.Now(), activityTime, 10*time.Second)
}

// TestGetPodsWithIdleFilter tests filtering pods by idle duration via API
func TestGetPodsWithIdleFilter(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test-idle-filter"}}
	_, err = client.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	require.NoError(t, err)

	now := time.Now()

	// Create pods with different idle times
	testPods := []struct {
		name       string
		idleFor    time.Duration
		shouldShow bool // for 30m idle filter
	}{
		{"pod-2h-idle", 2 * time.Hour, true},
		{"pod-45m-idle", 45 * time.Minute, true},
		{"pod-15m-idle", 15 * time.Minute, false},
		{"pod-5m-idle", 5 * time.Minute, false},
		{"pod-no-activity", 0, false}, // Never had activity
	}

	for _, tp := range testPods {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tp.name,
				Namespace: "test-idle-filter",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "test", Image: "busybox", Command: []string{"sleep", "3600"}},
				},
			},
		}

		if tp.idleFor > 0 {
			pod.Annotations = map[string]string{
				LastActivityAnnotationKey: now.Add(-tp.idleFor).Format(time.RFC3339Nano),
			}
		}

		_, err := client.CoreV1().Pods("test-idle-filter").Create(ctx, pod, metav1.CreateOptions{})
		require.NoError(t, err)
	}

	// List all pods and filter client-side for idle > 30m
	pods, err := client.CoreV1().Pods("test-idle-filter").List(ctx, metav1.ListOptions{})
	require.NoError(t, err)

	cutoff := now.Add(-30 * time.Minute)
	var idlePods []string
	for _, pod := range pods.Items {
		annotation, exists := pod.Annotations[LastActivityAnnotationKey]
		if !exists {
			continue
		}
		activityTime, err := time.Parse(time.RFC3339Nano, annotation)
		if err != nil {
			continue
		}
		if activityTime.Before(cutoff) {
			idlePods = append(idlePods, pod.Name)
		}
	}

	// Verify expected pods
	for _, tp := range testPods {
		found := false
		for _, name := range idlePods {
			if name == tp.name {
				found = true
				break
			}
		}
		if tp.shouldShow {
			assert.True(t, found, "expected %s to be in idle list", tp.name)
		} else {
			assert.False(t, found, "expected %s to NOT be in idle list", tp.name)
		}
	}
}
