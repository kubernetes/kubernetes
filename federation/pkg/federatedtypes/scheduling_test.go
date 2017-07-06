/*
Copyright 2017 The Kubernetes Authors.

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

package federatedtypes

import (
	"fmt"
	"testing"
	"time"

	apiv1 "k8s.io/api/core/v1"
	extensionsv1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestClusterReplicaState(t *testing.T) {
	uncalledPodsGetter := func(clusterName string, obj pkgruntime.Object) (*apiv1.PodList, error) {
		t.Fatal("podsGetter should not be called when workload objects are all ready.")
		return nil, nil
	}

	podsByReplicaSet := make(map[pkgruntime.Object][]*apiv1.Pod)
	podsGetter := func(clusterName string, obj pkgruntime.Object) (*apiv1.PodList, error) {
		pods, ok := podsByReplicaSet[obj]
		if !ok {
			t.Fatalf("No pods found in test data for replica set %v", obj)
			return nil, fmt.Errorf("Not found")
		}
		var podListPods []apiv1.Pod
		for _, pod := range pods {
			podListPods = append(podListPods, *pod)
		}
		return &apiv1.PodList{Items: podListPods}, nil
	}

	readyCondition := apiv1.PodCondition{Type: apiv1.PodReady}
	unschedulableCondition := apiv1.PodCondition{
		Type:               apiv1.PodScheduled,
		Status:             apiv1.ConditionFalse,
		Reason:             apiv1.PodReasonUnschedulable,
		LastTransitionTime: metav1.NewTime(time.Now().Add(-1 * time.Hour)),
	}

	one := int64(1)
	two := int64(2)

	tests := map[string]struct {
		rs1Replicas                   int32
		rs2Replicas                   int32
		rs1ReadyReplicas              int32
		rs2ReadyReplicas              int32
		podsGetter                    func(clusterName string, obj pkgruntime.Object) (*apiv1.PodList, error)
		pod1Phase                     apiv1.PodPhase
		pod1Condition                 apiv1.PodCondition
		pod2Phase                     apiv1.PodPhase
		pod2Condition                 apiv1.PodCondition
		cluster1Replicas              *int64
		cluster2Replicas              *int64
		cluster1UnschedulableReplicas *int64
		cluster2UnschedulableReplicas *int64
	}{
		"All replica sets have an equal number of requested and ready replicas.": {rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 2, rs2ReadyReplicas: 2, podsGetter: uncalledPodsGetter, cluster1Replicas: &two, cluster2Replicas: &two},
		"One replica set has a pending schedulable pod":                          {rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 1, rs2ReadyReplicas: 2, podsGetter: podsGetter, pod1Phase: apiv1.PodRunning, pod1Condition: readyCondition, pod2Phase: apiv1.PodPending, cluster1Replicas: &one, cluster2Replicas: &two},
		"One replica set has an unschedulable pod":                               {rs1Replicas: 2, rs2Replicas: 2, rs1ReadyReplicas: 1, rs2ReadyReplicas: 2, podsGetter: podsGetter, pod1Phase: apiv1.PodRunning, pod1Condition: readyCondition, pod2Phase: apiv1.PodPending, pod2Condition: unschedulableCondition, cluster1Replicas: &one, cluster2Replicas: &two, cluster1UnschedulableReplicas: &one},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			clusters := []string{"one", "two"}
			replicaSetsByCluster := make(map[string]*extensionsv1.ReplicaSet)
			replicaSetGetter := func(clusterName string, key string) (interface{}, bool, error) {
				rs, ok := replicaSetsByCluster[clusterName]
				if !ok {
					t.Fatalf("No replica set found in test data for %v", clusterName)
					return nil, false, fmt.Errorf("Not found")
				}
				return rs, true, nil
			}
			rs1 := newReplicaSetWithReplicas("one", tt.rs1Replicas)
			rs2 := newReplicaSetWithReplicas("two", tt.rs2Replicas)
			rs1.Spec.Replicas = &tt.rs1Replicas
			rs2.Spec.Replicas = &tt.rs2Replicas
			rs1.Status.ReadyReplicas = tt.rs1ReadyReplicas
			rs2.Status.ReadyReplicas = tt.rs2ReadyReplicas

			replicaSetsByCluster["one"] = rs1
			replicaSetsByCluster["two"] = rs2

			pod1 := newPod("one")
			pod2 := newPod("two")
			podThree := newPod("three")
			podFour := newPod("four")

			pod1.Status.Phase = tt.pod1Phase
			pod2.Status.Phase = tt.pod2Phase
			pod1.Status.Conditions = []apiv1.PodCondition{tt.pod1Condition}
			pod2.Status.Conditions = []apiv1.PodCondition{tt.pod2Condition}

			podsByReplicaSet[rs1] = []*apiv1.Pod{pod1, pod2}
			podsByReplicaSet[rs2] = []*apiv1.Pod{podThree, podFour}

			current, estimatedCapacity, err := clustersReplicaState(clusters, "", replicaSetGetter, tt.podsGetter)

			assert.Nil(t, err)

			wantedCurrent := make(map[string]int64)
			if tt.cluster1Replicas != nil {
				wantedCurrent["one"] = *tt.cluster1Replicas
			}
			if tt.cluster2Replicas != nil {
				wantedCurrent["two"] = *tt.cluster2Replicas
			}
			assert.Equal(t, wantedCurrent, current)

			wantedEstimatedCapacity := make(map[string]int64)
			if tt.cluster1UnschedulableReplicas != nil {
				wantedEstimatedCapacity["one"] = *tt.cluster1UnschedulableReplicas
			}
			if tt.cluster2UnschedulableReplicas != nil {
				wantedEstimatedCapacity["two"] = *tt.cluster2UnschedulableReplicas
			}
			assert.Equal(t, wantedEstimatedCapacity, estimatedCapacity)
		})
	}
}

func newReplicaSetWithReplicas(name string, replicas int32) *extensionsv1.ReplicaSet {
	return &extensionsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/replicasets/name",
		},
		Spec: extensionsv1.ReplicaSetSpec{
			Replicas: &replicas,
		},
	}
}

func newPod(name string) *apiv1.Pod {
	return &apiv1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
	}
}
