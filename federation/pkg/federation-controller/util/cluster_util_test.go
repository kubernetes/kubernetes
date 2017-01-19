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

package util

import (
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"testing"
)

func TestSendToCluster(t *testing.T) {

	cluster1 := &federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "cluster1",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
		Status: federationapi.ClusterStatus{
			Conditions: []federationapi.ClusterCondition{
				{Type: federationapi.ClusterReady, Status: apiv1.ConditionTrue},
			},
		},
	}

	cluster2 := &federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "cluster2",
			Annotations: map[string]string{},
			Labels:      map[string]string{},
		},
		Status: federationapi.ClusterStatus{
			Conditions: []federationapi.ClusterCondition{
				{Type: federationapi.ClusterReady, Status: apiv1.ConditionTrue},
			},
		},
	}

	cluster1.ObjectMeta.Labels["location"] = "europe"
	cluster1.ObjectMeta.Labels["environment"] = "prod"
	cluster2.ObjectMeta.Labels["location"] = "europe"
	cluster2.ObjectMeta.Labels["environment"] = "test"

	configmap1 := &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "ns",
			SelfLink:  "/api/v1/namespaces/ns/configmaps/test-configmap",
			Annotations: map[string]string{
				"A": "B",
				federationapi.FederationClusterSelector: "[{\"key\": \"location\", \"operator\": \"in\", \"values\": [\"europe\"]}, {\"key\": \"environment\", \"operator\": \"==\", \"values\": [\"prod\"]}]",
			},
		},
		Data: map[string]string{
			"A": "ala ma kota",
			"B": "quick brown fox",
		},
	}

	assert.True(t, SendToCluster(cluster1, configmap1.ObjectMeta))
	assert.False(t, SendToCluster(cluster2, configmap1.ObjectMeta))
}
