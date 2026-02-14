/*
Copyright 2025 The Kubernetes Authors.

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

package lifecycle

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func TestExtractConfigMapRefs(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected []string
	}{
		{
			name: "configmap volume",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: "my-config"},
								},
							},
						},
					},
				},
			},
			expected: []string{"my-config"},
		},
		{
			name: "configmap envFrom",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "main",
						EnvFrom: []corev1.EnvFromSource{{
							ConfigMapRef: &corev1.ConfigMapEnvSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: "env-config"},
							},
						}},
					}},
				},
			},
			expected: []string{"env-config"},
		},
		{
			name: "configmap keyRef in env",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "main",
						Env: []corev1.EnvVar{{
							Name: "MY_VAR",
							ValueFrom: &corev1.EnvVarSource{
								ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{Name: "key-config"},
									Key:                  "some-key",
								},
							},
						}},
					}},
				},
			},
			expected: []string{"key-config"},
		},
		{
			name: "init container configmap",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{{
						Name: "init",
						EnvFrom: []corev1.EnvFromSource{{
							ConfigMapRef: &corev1.ConfigMapEnvSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: "init-config"},
							},
						}},
					}},
				},
			},
			expected: []string{"init-config"},
		},
		{
			name: "no configmaps",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "main"}},
				},
			},
			expected: []string{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := &dataCollector{}
			refs := c.extractConfigMapRefs(tc.pod)

			if len(refs) != len(tc.expected) {
				t.Errorf("expected %d refs, got %d", len(tc.expected), len(refs))
				return
			}

			// Check all expected refs are present (order may vary)
			refSet := make(map[string]bool)
			for _, r := range refs {
				refSet[r] = true
			}
			for _, exp := range tc.expected {
				if !refSet[exp] {
					t.Errorf("expected ref %q not found", exp)
				}
			}
		})
	}
}

func TestExtractSecretRefs(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected []string
	}{
		{
			name: "secret volume",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "secret-vol",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName: "my-secret",
								},
							},
						},
					},
				},
			},
			expected: []string{"my-secret"},
		},
		{
			name: "secret envFrom",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "main",
						EnvFrom: []corev1.EnvFromSource{{
							SecretRef: &corev1.SecretEnvSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: "env-secret"},
							},
						}},
					}},
				},
			},
			expected: []string{"env-secret"},
		},
		{
			name: "secret keyRef in env",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "main",
						Env: []corev1.EnvVar{{
							Name: "SECRET_VAR",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{Name: "key-secret"},
									Key:                  "some-key",
								},
							},
						}},
					}},
				},
			},
			expected: []string{"key-secret"},
		},
		{
			name: "image pull secret",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					ImagePullSecrets: []corev1.LocalObjectReference{
						{Name: "pull-secret"},
					},
				},
			},
			expected: []string{"pull-secret"},
		},
		{
			name: "multiple secrets",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "secret-vol",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{SecretName: "vol-secret"},
							},
						},
					},
					ImagePullSecrets: []corev1.LocalObjectReference{
						{Name: "pull-secret"},
					},
				},
			},
			expected: []string{"vol-secret", "pull-secret"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := &dataCollector{}
			refs := c.extractSecretRefs(tc.pod)

			if len(refs) != len(tc.expected) {
				t.Errorf("expected %d refs, got %d", len(tc.expected), len(refs))
				return
			}

			refSet := make(map[string]bool)
			for _, r := range refs {
				refSet[r] = true
			}
			for _, exp := range tc.expected {
				if !refSet[exp] {
					t.Errorf("expected ref %q not found", exp)
				}
			}
		})
	}
}

func TestCollect(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodPending,
		},
	}

	fakeClient := fake.NewSimpleClientset(pod)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if data.Pod == nil {
		t.Error("expected pod to be collected")
	}

	if data.Pod.Name != "test-pod" {
		t.Errorf("expected pod name 'test-pod', got %s", data.Pod.Name)
	}
}

func TestCollectNotFound(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	_, err := collector.Collect(ctx, "test-ns", "nonexistent-pod")
	if err == nil {
		t.Error("expected error for nonexistent pod")
	}
}

func TestCollectWithNode(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			NodeName:   "node-1",
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Status:     corev1.NodeStatus{Phase: corev1.NodeRunning},
	}

	fakeClient := fake.NewSimpleClientset(pod, node)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if data.Node == nil {
		t.Error("expected node to be collected")
	}

	if data.Node.Name != "node-1" {
		t.Errorf("expected node name 'node-1', got %s", data.Node.Name)
	}
}

func TestCollectWithPVC(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
			Volumes: []corev1.Volume{
				{
					Name: "data",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
							ClaimName: "my-pvc",
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-pvc",
			Namespace: "test-ns",
		},
		Status: corev1.PersistentVolumeClaimStatus{
			Phase: corev1.ClaimBound,
		},
	}

	fakeClient := fake.NewSimpleClientset(pod, pvc)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(data.PVCs) != 1 {
		t.Errorf("expected 1 PVC, got %d", len(data.PVCs))
	}

	if data.PVCs["my-pvc"] == nil {
		t.Error("expected PVC 'my-pvc' to be collected")
	}
}

func TestCollectWithConfigMaps(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  "main",
				Image: "nginx",
				EnvFrom: []corev1.EnvFromSource{{
					ConfigMapRef: &corev1.ConfigMapEnvSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: "app-config"},
					},
				}},
			}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "app-config",
			Namespace: "test-ns",
		},
		Data: map[string]string{"key": "value"},
	}

	fakeClient := fake.NewSimpleClientset(pod, cm)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(data.ConfigMaps) != 1 {
		t.Errorf("expected 1 ConfigMap, got %d", len(data.ConfigMaps))
	}

	if data.ConfigMaps["app-config"] == nil {
		t.Error("expected ConfigMap 'app-config' to be collected")
	}
}

func TestCollectWithMissingConfigMap(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  "main",
				Image: "nginx",
				EnvFrom: []corev1.EnvFromSource{{
					ConfigMapRef: &corev1.ConfigMapEnvSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: "missing-config"},
					},
				}},
			}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	// Note: ConfigMap "missing-config" is not created
	fakeClient := fake.NewSimpleClientset(pod)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The collector should store nil for missing configmaps
	if _, exists := data.ConfigMaps["missing-config"]; !exists {
		t.Error("expected ConfigMap entry for 'missing-config' (even if nil)")
	}

	if data.ConfigMaps["missing-config"] != nil {
		t.Error("expected ConfigMap 'missing-config' to be nil")
	}
}

func TestCollectWithSecrets(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			ImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "registry-secret"},
			},
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "registry-secret",
			Namespace: "test-ns",
		},
	}

	fakeClient := fake.NewSimpleClientset(pod, secret)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !data.Secrets["registry-secret"] {
		t.Error("expected secret 'registry-secret' to exist (true)")
	}
}

func TestCollectWithMissingSecret(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			ImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "missing-secret"},
			},
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	// Note: Secret "missing-secret" is not created
	fakeClient := fake.NewSimpleClientset(pod)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if data.Secrets["missing-secret"] {
		t.Error("expected secret 'missing-secret' to be false (not exist)")
	}
}

func TestCollectWithServiceAccount(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: "my-sa",
			Containers:         []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-sa",
			Namespace: "test-ns",
		},
	}

	fakeClient := fake.NewSimpleClientset(pod, sa)

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if data.ServiceAccount == nil {
		t.Error("expected ServiceAccount to be collected")
	}

	if data.ServiceAccount.Name != "my-sa" {
		t.Errorf("expected ServiceAccount name 'my-sa', got %s", data.ServiceAccount.Name)
	}
}

func TestCollectEvents(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
			UID:       "test-uid",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}

	event := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-event",
			Namespace: "test-ns",
		},
		InvolvedObject: corev1.ObjectReference{
			Name:      "test-pod",
			Namespace: "test-ns",
		},
		Reason:  "FailedScheduling",
		Message: "0/3 nodes are available",
		Type:    "Warning",
	}

	fakeClient := fake.NewSimpleClientset(pod, event)

	// Add reactor to handle field selector filtering for events
	fakeClient.PrependReactor("list", "events", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, &corev1.EventList{
			Items: []corev1.Event{*event},
		}, nil
	})

	collector := NewDataCollector(fakeClient)
	ctx := context.Background()

	data, err := collector.Collect(ctx, "test-ns", "test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if data.Events == nil {
		t.Error("expected events to be collected")
	}

	if len(data.Events.Items) == 0 {
		t.Error("expected at least one event")
	}
}
