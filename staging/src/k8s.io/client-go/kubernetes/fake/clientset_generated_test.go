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
	"github.com/stretchr/testify/assert"
	"testing"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1ac "k8s.io/client-go/applyconfigurations/core/v1"
)

func TestNewSimpleClientset(t *testing.T) {
	client := NewSimpleClientset()
	client.CoreV1().Pods("default").Create(context.Background(), &v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      "pod-1",
			Namespace: "default",
		},
	}, meta_v1.CreateOptions{})
	client.CoreV1().Pods("default").Create(context.Background(), &v1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      "pod-2",
			Namespace: "default",
		},
	}, meta_v1.CreateOptions{})
	err := client.CoreV1().Pods("default").EvictV1(context.Background(), &policy.Eviction{
		ObjectMeta: meta_v1.ObjectMeta{
			Name: "pod-2",
		},
	})

	if err != nil {
		t.Errorf("TestNewSimpleClientset() res = %v", err.Error())
	}

	pods, err := client.CoreV1().Pods("default").List(context.Background(), meta_v1.ListOptions{})
	// err: item[0]: can't assign or convert v1beta1.Eviction into v1.Pod
	if err != nil {
		t.Errorf("TestNewSimpleClientset() res = %v", err.Error())
	} else {
		t.Logf("TestNewSimpleClientset() res = %v", pods)
	}
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
	if err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k0": "v0"}, cm.Data)

	// Apply with test-manager-1
	// Expect data to be shared with initial create
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v1"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-1"})
	if err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k0": "v0", "k1": "v1"}, cm.Data)

	// Apply conflicting with test-manager-2, expect apply to fail
	_, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "xyz"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2"})
	if assert.Error(t, err) {
		assert.Equal(t, "Apply failed with 1 conflict: conflict with \"test-manager-1\": .data.k1", err.Error())
	}

	// Apply with test-manager-2
	// Expect data to be shared with initial create and test-manager-1
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k2": "v2"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2"})
	if err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k0": "v0", "k1": "v1", "k2": "v2"}, cm.Data)

	// Apply with test-manager-1
	// Expect owned data to be updated
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v101"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-1"})
	if err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k0": "v0", "k1": "v101", "k2": "v2"}, cm.Data)

	// Force apply with test-manager-2
	// Expect data owned by test-manager-1 to be updated, expect data already owned but not in apply configuration to be removed
	cm, err = client.CoreV1().ConfigMaps("default").Apply(context.Background(),
		v1ac.ConfigMap(name, namespace).WithData(map[string]string{"k1": "v202"}),
		meta_v1.ApplyOptions{FieldManager: "test-manager-2", Force: true})
	if err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k0": "v0", "k1": "v202"}, cm.Data)

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
	if err != nil {
		t.Errorf("Failed to update pod: %v", err)
	}
	assert.Equal(t, map[string]string{"k99": "v99"}, cm.Data)
}
