/*
Copyright 2016 The Kubernetes Authors.

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

package podanalyzer

import (
	"testing"
	"time"

	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
)

func TestAnalyze(t *testing.T) {
	now := time.Now()
	replicaSet := newReplicaSet(map[string]string{"A": "B"})
	replicaSet2 := newReplicaSet(map[string]string{"C": "D"})
	podRunning := newPod("p1", replicaSet,
		api_v1.PodStatus{
			Phase: api_v1.PodRunning,
			Conditions: []api_v1.PodCondition{
				{
					Type:   api_v1.PodReady,
					Status: api_v1.ConditionTrue,
				},
			},
		})
	podUnschedulable := newPod("pU", replicaSet,
		api_v1.PodStatus{
			Phase: api_v1.PodPending,
			Conditions: []api_v1.PodCondition{
				{
					Type:               api_v1.PodScheduled,
					Status:             api_v1.ConditionFalse,
					Reason:             "Unschedulable",
					LastTransitionTime: metav1.Time{Time: now.Add(-10 * time.Minute)},
				},
			},
		})
	podOther := newPod("pO", replicaSet,
		api_v1.PodStatus{
			Phase:      api_v1.PodPending,
			Conditions: []api_v1.PodCondition{},
		})
	podOtherRS := newPod("pO", replicaSet2,
		api_v1.PodStatus{
			Phase:      api_v1.PodPending,
			Conditions: []api_v1.PodCondition{},
		})

	federatedObjects := []util.FederatedObject{
		{ClusterName: "c1", Object: podRunning},
		{ClusterName: "c1", Object: podRunning},
		{ClusterName: "c1", Object: podRunning},
		{ClusterName: "c1", Object: podUnschedulable},
		{ClusterName: "c1", Object: podUnschedulable},
		{ClusterName: "c2", Object: podOther},
		{ClusterName: "c2", Object: podOtherRS},
	}

	raport, err := AnalysePods(replicaSet.Spec.Selector, federatedObjects, now)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(raport))
	c1Raport := raport["c1"]
	c2Raport := raport["c2"]
	assert.Equal(t, PodAnalysisResult{
		Total:           5,
		RunningAndReady: 3,
		Unschedulable:   2,
	}, c1Raport)
	assert.Equal(t, PodAnalysisResult{
		Total:           1,
		RunningAndReady: 0,
		Unschedulable:   0,
	}, c2Raport)
}

func newReplicaSet(selectorMap map[string]string) *v1beta1.ReplicaSet {
	replicas := int32(3)
	rs := &v1beta1.ReplicaSet{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      "foobar",
			Namespace: "default",
		},
		Spec: v1beta1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: selectorMap},
		},
	}
	return rs
}

func newPod(name string, rs *v1beta1.ReplicaSet, status api_v1.PodStatus) *api_v1.Pod {
	return &api_v1.Pod{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      name,
			Namespace: rs.Namespace,
			Labels:    rs.Spec.Selector.MatchLabels,
		},
		Status: status,
	}
}
