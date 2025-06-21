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

package podautoscaler

import (
	"context"
	"testing"
	"time"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestOwnerReferencesFilter_Filter(t *testing.T) {
	// Create a fake dynamic client and REST mapper for testing
	dynamicClient, mapper, monitor := setupTestEnv()

	// Create test deployment
	deployment := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "test-deployment",
				"namespace": "default",
				"uid":       "deployment-uid",
			},
		},
	}

	// Create test ReplicaSet owned by deployment
	replicaSet := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "ReplicaSet",
			"metadata": map[string]interface{}{
				"name":      "test-replicaset",
				"namespace": "default",
				"uid":       "replicaset-uid",
				"ownerReferences": []interface{}{
					map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"name":       "test-deployment",
						"uid":        "deployment-uid",
					},
				},
			},
		},
	}

	_, err := dynamicClient.
		Resource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}).
		Namespace("default").
		Create(context.TODO(), deployment, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test deployment: %v", err)
	}

	_, err = dynamicClient.
		Resource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "replicasets"}).
		Namespace("default").
		Create(context.TODO(), replicaSet, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create test replicaset: %v", err)
	}

	controllerCache := NewControllerCache(dynamicClient, mapper, 15*time.Minute, monitor)

	tests := []struct {
		name           string
		pods           []*v1.Pod
		scaleTargetRef *autoscalingv2.CrossVersionObjectReference
		wantFiltered   int
		wantUnfiltered int
		wantErr        bool
	}{
		{
			name: "direct ownership match",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "apps/v1",
								Kind:       "Deployment",
								Name:       "test-deployment",
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "apps/v1",
								Kind:       "Deployment",
								Name:       "test-other-deployment",
							},
						},
					},
				},
			},
			scaleTargetRef: &autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			wantFiltered:   1,
			wantUnfiltered: 1,
			wantErr:        false,
		},
		{
			name: "traditional deployment setup (pod->replicaSet->deployment)",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "apps/v1",
								Kind:       "ReplicaSet",
								Name:       "test-replicaset",
								UID:        "replicaset-uid",
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "default",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "apps/v1",
								Kind:       "ReplicaSet",
								Name:       "other-replicaset",
								UID:        "other-replicaset-uid",
							},
						},
					},
				},
			},
			scaleTargetRef: &autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			wantFiltered:   1,
			wantUnfiltered: 1,
			wantErr:        false,
		},
		{
			name:           "no pods",
			pods:           []*v1.Pod{},
			scaleTargetRef: &autoscalingv2.CrossVersionObjectReference{},
			wantFiltered:   0,
			wantUnfiltered: 0,
			wantErr:        false,
		},
		{
			name: "no owner references",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "default",
					},
				},
			},
			scaleTargetRef: &autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			wantFiltered:   0,
			wantUnfiltered: 1,
			wantErr:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &OwnerReferencesFilter{
				filterOptions: FilterOptions{
					ScaleTargetRef: tt.scaleTargetRef,
				},
				Cache:         controllerCache,
				dynamicClient: dynamicClient,
				RESTMapper:    mapper,
			}

			filtered, unfiltered, err := f.Filter(tt.pods)

			if (err != nil) != tt.wantErr {
				t.Errorf("Filter() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if len(filtered) != tt.wantFiltered {
				t.Errorf("Filter() filtered = %v, want %v", len(filtered), tt.wantFiltered)
			}

			if len(unfiltered) != tt.wantUnfiltered {
				t.Errorf("Filter() unfiltered = %v, want %v", len(unfiltered), tt.wantUnfiltered)
			}
		})
	}
}

func TestOwnerReferencesFilter_isTargetMatch(t *testing.T) {
	tests := []struct {
		name      string
		obj       *unstructured.Unstructured
		targetRef autoscalingv2.CrossVersionObjectReference
		want      bool
	}{
		{
			name: "matching target",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name": "test-deployment",
					},
				},
			},
			targetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			want: true,
		},
		{
			name: "non-matching target",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name": "different-deployment",
					},
				},
			},
			targetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			want: false,
		},
		{
			name: "non-matching target - different kind",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "ReplicaSet",
					"metadata": map[string]interface{}{
						"name": "some-replicaset",
					},
				},
			},
			targetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &OwnerReferencesFilter{}
			if got := f.isTargetMatch(tt.obj, tt.targetRef); got != tt.want {
				t.Errorf("isTargetMatch() = %v, want %v", got, tt.want)
			}
		})
	}
}