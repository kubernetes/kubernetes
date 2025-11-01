/*
Copyright 2020 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/fieldmaskpb"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
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
}

func TestPodServerFieldMasks(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"},
		Spec:       v1.PodSpec{NodeName: "node1"},
		Status:     v1.PodStatus{Phase: v1.PodRunning},
	}
	server.OnPodAdded(pod1)
	t.Run("ListPodsWithFieldMask", func(t *testing.T) {
		fieldMask := &fieldmaskpb.FieldMask{Paths: []string{"metadata.name", "spec.nodeName"}}
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, strings.Join(fieldMask.Paths, ",")))
		resp, err := server.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
		require.NoError(t, err)
		require.Len(t, resp.Pods, 1)
		maskedPod := &v1.Pod{}
		err = maskedPod.Unmarshal(resp.Pods[0])
		require.NoError(t, err)
		require.Equal(t, "pod1", maskedPod.Name)
		assert.Equal(t, "node1", maskedPod.Spec.NodeName)
		assert.Empty(t, maskedPod.Namespace)
		assert.Empty(t, maskedPod.Status.Phase)
	})
	t.Run("GetPodWithFieldMask", func(t *testing.T) {
		fieldMask := &fieldmaskpb.FieldMask{Paths: []string{"metadata.namespace", "status.phase"}}
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, strings.Join(fieldMask.Paths, ",")))
		resp, err := server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
		require.NoError(t, err)
		maskedPod := &v1.Pod{}
		err = maskedPod.Unmarshal(resp.Pod)
		require.NoError(t, err)
		require.Empty(t, maskedPod.Name)
		assert.Equal(t, "ns1", maskedPod.Namespace)
		assert.Empty(t, maskedPod.Spec.NodeName)
		assert.Equal(t, v1.PodRunning, maskedPod.Status.Phase)
	})
	t.Run("ListPodsWithEmptyFieldMask", func(t *testing.T) {
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, ""))
		resp, err := server.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
		require.NoError(t, err)
		require.Len(t, resp.Pods, 1)
		podOut := &v1.Pod{}
		err = podOut.Unmarshal(resp.Pods[0])
		require.NoError(t, err)
		require.Equal(t, pod1, podOut)
	})
	t.Run("GetPodWithNilFieldMask", func(t *testing.T) {
		ctx := context.Background()
		resp, err := server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
		require.NoError(t, err)
		podOut := &v1.Pod{}
		err = podOut.Unmarshal(resp.Pod)
		require.NoError(t, err)
		require.Equal(t, pod1, podOut)
	})
}

func getAllPaths(t reflect.Type, prefix string, paths map[string]bool) {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return
	}
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		jsonTag := field.Tag.Get("json")
		parts := strings.Split(jsonTag, ",")
		fieldName := parts[0]
		if len(parts) > 1 && parts[1] == "inline" {
			getAllPaths(field.Type, prefix, paths)
			continue
		}
		if fieldName == "" || fieldName == "-" {
			continue
		}
		path := fieldName
		if prefix != "" {
			path = prefix + "." + path
		}
		paths[path] = true
		fieldType := field.Type
		if fieldType.Kind() == reflect.Ptr {
			fieldType = fieldType.Elem()
		}
		if fieldType.Kind() == reflect.Struct {
			if !strings.Contains(fieldType.PkgPath(), "k8s.io/apimachinery/pkg/apis/meta/v1") ||
				(fieldType.Name() != "Time" && fieldType.Name() != "Duration" && fieldType.Name() != "MicroTime") {
				getAllPaths(fieldType, path, paths)
			}
		}
	}
}

func TestGetPodWithVariousFieldMasks(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	terminationGracePeriodSeconds := int64(30)
	activeDeadlineSeconds := int64(600)
	v1Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-pod",
			Namespace:         "test-ns",
			UID:               "test-uid",
			Labels:            map[string]string{"app": "test"},
			Annotations:       map[string]string{"foo": "bar"},
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1.PodSpec{
			NodeName:                      "test-node",
			RestartPolicy:                 v1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: &terminationGracePeriodSeconds,
			ActiveDeadlineSeconds:         &activeDeadlineSeconds,
			DNSPolicy:                     v1.DNSClusterFirst,
			NodeSelector:                  map[string]string{"disktype": "ssd"},
			Containers: []v1.Container{
				{
					Name:    "test-container",
					Image:   "test-image",
					Command: []string{"/bin/sh", "-c", "echo hello"},
					Args:    []string{"--verbose"},
					Ports: []v1.ContainerPort{
						{ContainerPort: 80, Protocol: v1.ProtocolTCP},
					},
					Env: []v1.EnvVar{
						{Name: "ENV_VAR", Value: "some-value"},
					},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "test-volume",
							MountPath: "/test-data",
						},
					},
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/healthz",
								Port:   intstr.FromInt(8080),
								Host:   "localhost",
								Scheme: v1.URISchemeHTTP,
								HTTPHeaders: []v1.HTTPHeader{
									{Name: "X-Custom-Header", Value: "health-check"},
								},
							},
						},
						InitialDelaySeconds: 15,
						TimeoutSeconds:      1,
						PeriodSeconds:       20,
						SuccessThreshold:    1,
						FailureThreshold:    3,
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "test-volume",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase:    v1.PodRunning,
			HostIP:   "1.2.3.4",
			PodIP:    "5.6.7.8",
			QOSClass: v1.PodQOSGuaranteed,
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:        "test-container",
					Ready:       true,
					ContainerID: "docker://abc",
					LastTerminationState: v1.ContainerState{
						Terminated: &v1.ContainerStateTerminated{
							ExitCode:   1,
							Reason:     "Error",
							Message:    "The container failed.",
							FinishedAt: metav1.Now(),
						},
					},
				},
			},
		},
	}
	server.OnPodAdded(v1Pod)
	fullResp, err := server.GetPod(context.Background(), &podsv1alpha1.GetPodRequest{PodUID: "test-uid"})
	require.NoError(t, err)
	fullAlphaPod := &v1.Pod{}
	err = fullAlphaPod.Unmarshal(fullResp.Pod)
	require.NoError(t, err)
	pathsMap := make(map[string]bool)
	getAllPaths(reflect.TypeOf(*v1Pod), "", pathsMap)
	var paths []string
	for p := range pathsMap {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, path := range paths {
		if strings.HasPrefix(path, "metadata.managedFields") || path == "apiVersion" || path == "kind" {
			continue
		}
		t.Run(path, func(t *testing.T) {
			ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, path))
			resp, err := server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "test-uid"})
			require.NoError(t, err)
			maskedPodFromAPI := &v1.Pod{}
			err = maskedPodFromAPI.Unmarshal(resp.Pod)
			require.NoError(t, err)
			fieldMask := &fieldmaskpb.FieldMask{Paths: []string{path}}
			expectedMaskedPod := &v1.Pod{}
			err = podsapi.ApplyFieldMask(fieldMask, fullAlphaPod, expectedMaskedPod)
			require.NoError(t, err)
			require.Equal(t, expectedMaskedPod, maskedPodFromAPI)
		})
	}
}

func TestListPodsWithVariousFieldMasks(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	terminationGracePeriodSeconds := int64(30)
	activeDeadlineSeconds := int64(600)
	v1Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-pod",
			Namespace:         "test-ns",
			UID:               "test-uid",
			Labels:            map[string]string{"app": "test"},
			Annotations:       map[string]string{"foo": "bar"},
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1.PodSpec{
			NodeName:                      "test-node",
			RestartPolicy:                 v1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: &terminationGracePeriodSeconds,
			ActiveDeadlineSeconds:         &activeDeadlineSeconds,
			DNSPolicy:                     v1.DNSClusterFirst,
			NodeSelector:                  map[string]string{"disktype": "ssd"},
			Containers: []v1.Container{
				{
					Name:    "test-container",
					Image:   "test-image",
					Command: []string{"/bin/sh", "-c", "echo hello"},
					Args:    []string{"--verbose"},
					Ports: []v1.ContainerPort{
						{ContainerPort: 80, Protocol: v1.ProtocolTCP},
					},
					Env: []v1.EnvVar{
						{Name: "ENV_VAR", Value: "some-value"},
					},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("50m"),
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "test-volume",
							MountPath: "/test-data",
						},
					},
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/healthz",
								Port:   intstr.FromInt(8080),
								Host:   "localhost",
								Scheme: v1.URISchemeHTTP,
								HTTPHeaders: []v1.HTTPHeader{
									{Name: "X-Custom-Header", Value: "health-check"},
								},
							},
						},
						InitialDelaySeconds: 15,
						TimeoutSeconds:      1,
						PeriodSeconds:       20,
						SuccessThreshold:    1,
						FailureThreshold:    3,
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "test-volume",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase:    v1.PodRunning,
			HostIP:   "1.2.3.4",
			PodIP:    "5.6.7.8",
			QOSClass: v1.PodQOSGuaranteed,
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:        "test-container",
					Ready:       true,
					ContainerID: "docker://abc",
					LastTerminationState: v1.ContainerState{
						Terminated: &v1.ContainerStateTerminated{
							ExitCode:   1,
							Reason:     "Error",
							Message:    "The container failed.",
							FinishedAt: metav1.Now(),
						},
					},
				},
			},
		},
	}
	server.OnPodAdded(v1Pod)
	fullResp, err := server.ListPods(context.Background(), &podsv1alpha1.ListPodsRequest{})
	require.NoError(t, err)
	require.Len(t, fullResp.Pods, 1)
	fullAlphaPod := &v1.Pod{}
	err = fullAlphaPod.Unmarshal(fullResp.Pods[0])
	require.NoError(t, err)
	pathsMap := make(map[string]bool)
	getAllPaths(reflect.TypeOf(*v1Pod), "", pathsMap)
	var paths []string
	for p := range pathsMap {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, path := range paths {
		if strings.HasPrefix(path, "metadata.managedFields") || path == "apiVersion" || path == "kind" {
			continue
		}
		t.Run(path, func(t *testing.T) {
			ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, path))
			resp, err := server.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
			require.NoError(t, err)
			require.Len(t, resp.Pods, 1)
			maskedPodFromAPI := &v1.Pod{}
			err = maskedPodFromAPI.Unmarshal(resp.Pods[0])
			require.NoError(t, err)
			fieldMask := &fieldmaskpb.FieldMask{Paths: []string{path}}
			expectedMaskedPod := &v1.Pod{}
			err = podsapi.ApplyFieldMask(fieldMask, fullAlphaPod, expectedMaskedPod)
			require.NoError(t, err)
			require.Equal(t, expectedMaskedPod, maskedPodFromAPI)
		})
	}
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

	t.Run("InvalidFieldMaskReturnsError", func(t *testing.T) {
		broadcaster := podsapi.NewBroadcaster()
		server := podsapi.NewPodsServerForTest(broadcaster)
		pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
		server.OnPodAdded(pod1)

		invalidFieldMask := "invalid.field.mask"
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, invalidFieldMask))

		_, err := server.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
		if err == nil {
			t.Fatal("ListPods should have returned an error")
		}
		assert.Contains(t, err.Error(), "failed to apply field mask path")

		_, err = server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
		if err == nil {
			t.Fatal("GetPod should have returned an error")
		}
		assert.Contains(t, err.Error(), "failed to apply field mask path")

		stream := &mockWatchPodsServer{ctx: ctx, t: t}
		err = server.WatchPods(&podsv1alpha1.WatchPodsRequest{}, stream)
		if err == nil {
			t.Fatal("WatchPods should have returned an error")
		}
		assert.Contains(t, err.Error(), "invalid field mask")
	})
}

// mockWatchPodsServer is a mock implementation of the WatchPodsServer interface for testing.
type mockWatchPodsServer struct {
	ctx context.Context
	t   *testing.T
}

func (s *mockWatchPodsServer) Send(event *podsv1alpha1.WatchPodsEvent) error {
	s.t.Helper()
	return nil
}

func (s *mockWatchPodsServer) SetHeader(md metadata.MD) error {
	s.t.Helper()
	return nil
}

func (s *mockWatchPodsServer) SendHeader(md metadata.MD) error {
	s.t.Helper()
	return nil
}

func (s *mockWatchPodsServer) SetTrailer(md metadata.MD) {
	s.t.Helper()
}

func (s *mockWatchPodsServer) Context() context.Context {
	s.t.Helper()
	if s.ctx == nil {
		return context.Background()
	}
	return s.ctx
}

func (s *mockWatchPodsServer) SendMsg(m interface{}) error {
	s.t.Helper()
	return nil
}

func (s *mockWatchPodsServer) RecvMsg(m interface{}) error {
	s.t.Helper()
	return nil
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
