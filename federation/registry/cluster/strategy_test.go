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

package cluster

import (
	"testing"

	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	apitesting "k8s.io/kubernetes/pkg/api/testing"

	// install all api groups for testing
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func validNewCluster() *federation.Cluster {
	return &federation.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "4",
			Labels: map[string]string{
				"name": "foo",
			},
		},
		Spec: federation.ClusterSpec{
			ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: "localhost:8888",
				},
			},
		},
		Status: federation.ClusterStatus{
			Conditions: []federation.ClusterCondition{
				{Type: federation.ClusterReady, Status: api.ConditionTrue},
			},
		},
	}
}

func invalidNewCluster() *federation.Cluster {
	// Create a cluster with empty ServerAddressByClientCIDRs (which is a required field).
	return &federation.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo2",
			ResourceVersion: "5",
		},
		Status: federation.ClusterStatus{
			Conditions: []federation.ClusterCondition{
				{Type: federation.ClusterReady, Status: api.ConditionFalse},
			},
		},
	}
}

func TestClusterStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("Cluster should not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Cluster should not allow create on update")
	}

	cluster := validNewCluster()
	Strategy.PrepareForCreate(ctx, cluster)
	if len(cluster.Status.Conditions) != 0 {
		t.Errorf("Cluster should not allow setting conditions on create")
	}
	errs := Strategy.Validate(ctx, cluster)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	invalidCluster := invalidNewCluster()
	Strategy.PrepareForUpdate(ctx, invalidCluster, cluster)
	if reflect.DeepEqual(invalidCluster.Spec, cluster.Spec) ||
		!reflect.DeepEqual(invalidCluster.Status, cluster.Status) {
		t.Error("Only spec is expected being changed")
	}
	errs = Strategy.ValidateUpdate(ctx, invalidCluster, cluster)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if cluster.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestClusterStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if StatusStrategy.NamespaceScoped() {
		t.Errorf("Cluster should not be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Cluster should not allow create on update")
	}

	cluster := validNewCluster()
	invalidCluster := invalidNewCluster()
	StatusStrategy.PrepareForUpdate(ctx, cluster, invalidCluster)
	if !reflect.DeepEqual(invalidCluster.Spec, cluster.Spec) ||
		reflect.DeepEqual(invalidCluster.Status, cluster.Status) {
		t.Logf("== cluster.Spec: %v\n", cluster.Spec)
		t.Logf("== cluster.Status: %v\n", cluster.Status)
		t.Logf("== invalidCluster.Spec: %v\n", cluster.Spec)
		t.Logf("== invalidCluster.Spec: %v\n", cluster.Status)
		t.Error("Only spec is expected being changed")
	}
	errs := Strategy.ValidateUpdate(ctx, invalidCluster, cluster)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if cluster.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestMatchCluster(t *testing.T) {
	testFieldMap := map[bool][]fields.Set{
		true: {
			{"metadata.name": "foo"},
		},
		false: {
			{"foo": "bar"},
		},
	}

	for expectedResult, fieldSet := range testFieldMap {
		for _, field := range fieldSet {
			m := MatchCluster(labels.Everything(), field.AsSelector())
			_, matchesSingle := m.MatchesSingle()
			if e, a := expectedResult, matchesSingle; e != a {
				t.Errorf("%+v: expected %v, got %v", fieldSet, e, a)
			}
		}
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		api.Registry.GroupOrDie(federation.GroupName).GroupVersion.String(),
		"Cluster",
		ClusterToSelectableFields(&federation.Cluster{}),
		nil,
	)
}
