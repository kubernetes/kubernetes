package pods_test

import (
	"context"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/fieldmaskpb"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	podsapi "k8s.io/kubernetes/pkg/kubelet/apis/pods"
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
	assert.NoError(t, err)
	assert.Len(t, resp.Pods, 2)
	assert.Equal(t, "pod1", resp.Pods[0].Name)
	assert.Equal(t, "pod2", resp.Pods[1].Name)
}

func TestGetPod(t *testing.T) {
	broadcaster := podsapi.NewBroadcaster()
	server := podsapi.NewPodsServerForTest(broadcaster)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod1-uid", Name: "pod1", Namespace: "ns1"}}
	status1 := v1.PodStatus{Phase: v1.PodRunning}
	server.OnPodAdded(pod1)
	server.OnPodStatusUpdated(pod1, status1)
	resp, err := server.GetPod(context.Background(), &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
	assert.NoError(t, err)
	assert.Equal(t, "pod1", resp.Pod.Name)
	assert.Equal(t, v1.PodRunning, resp.Pod.Status.Phase)
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
		assert.NoError(t, err)
		assert.Len(t, resp.Pods, 1)
		maskedPod := resp.Pods[0]
		assert.Equal(t, "pod1", maskedPod.Name)
		assert.Equal(t, "node1", maskedPod.Spec.NodeName)
		assert.Empty(t, maskedPod.Namespace)
		assert.Empty(t, maskedPod.Status.Phase)
	})
	t.Run("GetPodWithFieldMask", func(t *testing.T) {
		fieldMask := &fieldmaskpb.FieldMask{Paths: []string{"metadata.namespace", "status.phase"}}
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, strings.Join(fieldMask.Paths, ",")))
		resp, err := server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
		assert.NoError(t, err)
		maskedPod := resp.Pod
		assert.Empty(t, maskedPod.Name)
		assert.Equal(t, "ns1", maskedPod.Namespace)
		assert.Empty(t, maskedPod.Spec.NodeName)
		assert.Equal(t, v1.PodRunning, maskedPod.Status.Phase)
	})
	t.Run("ListPodsWithEmptyFieldMask", func(t *testing.T) {
		ctx := metadata.NewIncomingContext(context.Background(), metadata.Pairs(podsapi.FieldMaskMetadataKey, ""))
		resp, err := server.ListPods(ctx, &podsv1alpha1.ListPodsRequest{})
		assert.NoError(t, err)
		assert.Len(t, resp.Pods, 1)
		assert.Equal(t, pod1, resp.Pods[0])
	})
	t.Run("GetPodWithNilFieldMask", func(t *testing.T) {
		ctx := context.Background()
		resp, err := server.GetPod(ctx, &podsv1alpha1.GetPodRequest{PodUID: "pod1-uid"})
		assert.NoError(t, err)
		assert.Equal(t, pod1, resp.Pod)
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
			if !(strings.Contains(fieldType.PkgPath(), "k8s.io/apimachinery/pkg/apis/meta/v1") &&
				(fieldType.Name() == "Time" || fieldType.Name() == "Duration" || fieldType.Name() == "MicroTime")) {
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
	assert.NoError(t, err)
	fullAlphaPod := fullResp.Pod
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
			assert.NoError(t, err)
			maskedPodFromAPI := resp.Pod
			fieldMask := &fieldmaskpb.FieldMask{Paths: []string{path}}
			expectedMaskedPod := &v1.Pod{}
			err = podsapi.ApplyFieldMask(fieldMask, fullAlphaPod, expectedMaskedPod)
			assert.NoError(t, err)
			assert.Equal(t, expectedMaskedPod, maskedPodFromAPI)
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
	assert.NoError(t, err)
	assert.Len(t, fullResp.Pods, 1)
	fullAlphaPod := fullResp.Pods[0]
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
			assert.NoError(t, err)
			assert.Len(t, resp.Pods, 1)
			maskedPodFromAPI := resp.Pods[0]
			fieldMask := &fieldmaskpb.FieldMask{Paths: []string{path}}
			expectedMaskedPod := &v1.Pod{}
			err = podsapi.ApplyFieldMask(fieldMask, fullAlphaPod, expectedMaskedPod)
			assert.NoError(t, err)
			assert.Equal(t, expectedMaskedPod, maskedPodFromAPI)
		})
	}
}
