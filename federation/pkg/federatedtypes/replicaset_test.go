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

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"

	"github.com/stretchr/testify/assert"
)

const (
	pods        = "pods"
	replicasets = "replicasets"
	k8s1        = "k8s-1"
	k8s2        = "k8s-2"
)

func TestParseFederationReplicaSetReference(t *testing.T) {
	successPrefs := []string{
		`{"rebalance": true,
		  "clusters": {
		    "k8s-1": {"minReplicas": 10, "maxReplicas": 20, "weight": 2},
		    "*": {"weight": 1}
		}}`,
	}
	failedPrefes := []string{
		`{`, // bad json
	}

	rs := newReplicaSetWithReplicas("rs-1", 100)
	accessor, _ := meta.Accessor(rs)
	anno := accessor.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		accessor.SetAnnotations(anno)
	}
	for _, prefString := range successPrefs {
		anno[FedReplicaSetPreferencesAnnotation] = prefString
		pref, err := parseFederationReplicaSetReference(rs)
		assert.NotNil(t, pref)
		assert.Nil(t, err)
	}
	for _, prefString := range failedPrefes {
		anno[FedReplicaSetPreferencesAnnotation] = prefString
		pref, err := parseFederationReplicaSetReference(rs)
		assert.Nil(t, pref)
		assert.NotNil(t, err)
	}
}

func TestClusterReplicaState(t *testing.T) {
	replicaSetsByCluster := make(map[string]*extensionsv1.ReplicaSet)
	replicaSetGetter := func(clusterName string, key string) (interface{}, bool, error) {
		rs, ok := replicaSetsByCluster[clusterName]
		if !ok {
			t.Fatalf("No replica set found in test data for %v", clusterName)
			return nil, false, fmt.Errorf("Not found")
		}
		return rs, true, nil
	}

	podsGetter := func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error) {
		t.Fatal("podsGetter should not be called when replica sets are all ready.")
		return nil, nil
	}

	clusters := []string{"one", "two"}
	rs1 := newReplicaSetWithReplicas("one", 2)
	rs2 := newReplicaSetWithReplicas("two", 2)

	replicaSetsByCluster["one"] = rs1
	replicaSetsByCluster["two"] = rs2

	// Test the happy case: all pods ready
	rs1.Status.ReadyReplicas = *rs1.Spec.Replicas
	rs2.Status.ReadyReplicas = *rs2.Spec.Replicas

	current, estimatedCapacity, err := clustersReplicaState(clusters, "", replicaSetGetter, podsGetter)

	assert.Nil(t, err)
	assert.Equal(t, map[string]int64{"one": 2, "two": 2}, current)
	assert.Empty(t, estimatedCapacity)

	// Set up the podsGetter for tests that require it.
	podsByReplicaSet := make(map[*extensionsv1.ReplicaSet][]*apiv1.Pod)
	podsGetter = func(clusterName string, replicaSet *extensionsv1.ReplicaSet) (*apiv1.PodList, error) {
		pods, ok := podsByReplicaSet[replicaSet]
		if !ok {
			t.Fatalf("No pods found in test data for replica set named %v", replicaSet.Name)
			return nil, fmt.Errorf("Not found")
		}
		var podListPods []apiv1.Pod
		for _, pod := range pods {
			podListPods = append(podListPods, *pod)
		}
		return &apiv1.PodList{Items: podListPods}, nil
	}

	podOne := newPod("one")
	podTwo := newPod("two")
	podThree := newPod("three")
	podFour := newPod("four")

	podsByReplicaSet[rs1] = []*apiv1.Pod{podOne, podTwo}
	podsByReplicaSet[rs2] = []*apiv1.Pod{podThree, podFour}

	// Test schedulable ready replicas without any unschedulable.
	rs1.Status.ReadyReplicas = 1
	podOne.Status.Phase = apiv1.PodRunning
	podOne.Status.Conditions = []apiv1.PodCondition{apiv1.PodCondition{Type: apiv1.PodReady}}
	podTwo.Status.Phase = apiv1.PodPending

	current, estimatedCapacity, err = clustersReplicaState(clusters, "", replicaSetGetter, podsGetter)

	assert.Nil(t, err)
	assert.Equal(t, map[string]int64{"one": 1, "two": 2}, current)
	assert.Empty(t, estimatedCapacity)

	// Test schedule with unschedulable replicas.
	rs1.Status.ReadyReplicas = 1
	podOne.Status.Phase = apiv1.PodRunning
	podOne.Status.Conditions = []apiv1.PodCondition{apiv1.PodCondition{Type: apiv1.PodReady}}
	podTwo.Status.Phase = apiv1.PodPending
	podTwo.Status.Conditions = []apiv1.PodCondition{apiv1.PodCondition{
		Type:               apiv1.PodScheduled,
		Status:             apiv1.ConditionFalse,
		Reason:             apiv1.PodReasonUnschedulable,
		LastTransitionTime: metav1.NewTime(time.Now().Add(-1 * time.Hour)),
	}}

	current, estimatedCapacity, err = clustersReplicaState(clusters, "", replicaSetGetter, podsGetter)

	assert.Nil(t, err)
	assert.Equal(t, map[string]int64{"one": 1, "two": 2}, current)
	assert.Equal(t, map[string]int64{"one": 1}, estimatedCapacity)
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
