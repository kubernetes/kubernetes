/*
Copyright 2022 The Kubernetes Authors.

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

package fake

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1ac "k8s.io/client-go/applyconfigurations/core/v1"
)

func TestNewSimpleClientset(t *testing.T) {
	client := NewSimpleClientset()

	_, err := client.CoreV1().Pods("default").Create(context.Background(), &v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "default",
		},
	}, meta_v1.CreateOptions{})
	require.NoError(t, err)

	_, err = client.CoreV1().Pods("default").Create(context.Background(), &v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      "pod-2",
			Namespace: "default",
		},
	}, meta_v1.CreateOptions{})
	require.NoError(t, err)

	// Verify we have 2 pods before eviction
	pods, err := client.CoreV1().Pods("default").List(context.Background(), meta_v1.ListOptions{})
	require.NoError(t, err)
	require.Len(t, pods.Items, 2, "expected 2 pods before eviction")

	err = client.CoreV1().Pods("default").EvictV1(context.Background(), &policy.Eviction{
		ObjectMeta: meta_v1.ObjectMeta{
			Name: "pod-2",
		},
	})
	require.NoError(t, err)

	// Verify pod-2 was deleted after eviction
	pods, err = client.CoreV1().Pods("default").List(context.Background(), meta_v1.ListOptions{})
	require.NoError(t, err)
	require.Len(t, pods.Items, 1, "expected 1 pod after eviction")
	require.Equal(t, "pod-1", pods.Items[0].Name, "expected pod-1 to remain after evicting pod-2")
}

func TestEvictionDeletesPod(t *testing.T) {
	client := NewSimpleClientset()
	ctx := context.Background()
	namespace := "default"
	podName := "eviction-test-pod"

	// Create a pod
	_, err := client.CoreV1().Pods(namespace).Create(ctx, &v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{Name: podName, Namespace: namespace},
	}, meta_v1.CreateOptions{})
	require.NoError(t, err, "pod creation should succeed")

	// Verify pod exists
	pods, err := client.CoreV1().Pods(namespace).List(ctx, meta_v1.ListOptions{})
	require.NoError(t, err)
	require.Len(t, pods.Items, 1, "pod should exist after creation")

	// Evict the pod using PolicyV1
	err = client.PolicyV1().Evictions(namespace).Evict(ctx, &policy.Eviction{
		TypeMeta:      meta_v1.TypeMeta{APIVersion: "policy/v1", Kind: "Eviction"},
		ObjectMeta:    meta_v1.ObjectMeta{Name: podName, Namespace: namespace},
		DeleteOptions: &meta_v1.DeleteOptions{},
	})
	require.NoError(t, err, "eviction should succeed")

	// Verify pod was deleted
	pods, err = client.CoreV1().Pods(namespace).List(ctx, meta_v1.ListOptions{})
	require.NoError(t, err)
	require.Len(t, pods.Items, 0, "pod should be deleted after eviction")
}

func TestManagedFieldClientset(t *testing.T) {
	client := NewClientset()
	name := "pod-1"
	namespace := "default"
	cm, err := client.CoreV1().ConfigMaps("default").Create(context.Background(),
		&v1.ConfigMap{
			ObjectMeta: meta_v1.ObjectMeta{Name: name, Namespace: namespace},
			Data:       map[string]string{"k0": "v0"},
		}, meta_v1.CreateOptions{FieldManager: "test-manager-0"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k0": "v0"}, cm.Data)

	// Apply with test-manager-1
	// Expect data to be shared with initial create
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v1"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-1"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k0": "v0", "k1": "v1"}, cm.Data)

	// Apply conflicting with test-manager-2, expect apply to fail
	_, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "xyz"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2"})
	require.Error(t, err)
	require.Equal(t, "Apply failed with 1 conflict: conflict with \"test-manager-1\": .data.k1", err.Error())

	// Apply with test-manager-2
	// Expect data to be shared with initial create and test-manager-1
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k2": "v2"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k0": "v0", "k1": "v1", "k2": "v2"}, cm.Data)

	// Apply with test-manager-1
	// Expect owned data to be updated
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v101"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-1"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k0": "v0", "k1": "v101", "k2": "v2"}, cm.Data)

	// Force apply with test-manager-2
	// Expect data owned by test-manager-1 to be updated, expect data already owned but not in apply configuration to be removed
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v202"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2", Force: true})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k0": "v0", "k1": "v202"}, cm.Data)

	// Update with test-manager-1 to perform a force update of the entire resource
	cm, err = client.CoreV1().ConfigMaps("default").Update(context.Background(),
		&v1.ConfigMap{
			TypeMeta: meta_v1.TypeMeta{
				APIVersion: "v1",
				Kind:       "ConfigMap",
			},
			ObjectMeta: meta_v1.ObjectMeta{
				Name:      name,
				Namespace: namespace,
			},
			Data: map[string]string{
				"k99": "v99",
			},
		}, meta_v1.UpdateOptions{FieldManager: "test-manager-0"})
	require.NoError(t, err)
	require.Equal(t, map[string]string{"k99": "v99"}, cm.Data)
}
