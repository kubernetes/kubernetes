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

package planner

import (
	"testing"

	fedapi "k8s.io/kubernetes/federation/apis/federation"

	"github.com/stretchr/testify/assert"
)

func doCheck(t *testing.T, pref map[string]fedapi.ClusterReplicaSetPreferences, replicas int64, clusters []string, expected map[string]int64) {
	planer := NewPlanner(&fedapi.FederatedReplicaSetPreferences{
		Clusters: pref,
	})
	plan, overflow := planer.Plan(replicas, clusters, map[string]int64{}, map[string]int64{}, "")
	assert.EqualValues(t, expected, plan)
	assert.Equal(t, 0, len(overflow))
}

func doCheckWithExisting(t *testing.T, pref map[string]fedapi.ClusterReplicaSetPreferences, replicas int64, clusters []string,
	existing map[string]int64, expected map[string]int64) {
	planer := NewPlanner(&fedapi.FederatedReplicaSetPreferences{
		Clusters: pref,
	})
	plan, overflow := planer.Plan(replicas, clusters, existing, map[string]int64{}, "")
	assert.Equal(t, 0, len(overflow))
	assert.EqualValues(t, expected, plan)
}

func doCheckWithExistingAndCapacity(t *testing.T, rebalance bool, pref map[string]fedapi.ClusterReplicaSetPreferences, replicas int64, clusters []string,
	existing map[string]int64,
	capacity map[string]int64,
	expected map[string]int64,
	expectedOverflow map[string]int64) {
	planer := NewPlanner(&fedapi.FederatedReplicaSetPreferences{
		Rebalance: rebalance,
		Clusters:  pref,
	})
	plan, overflow := planer.Plan(replicas, clusters, existing, capacity, "")
	assert.EqualValues(t, expected, plan)
	assert.Equal(t, expectedOverflow, overflow)
}

func pint(val int64) *int64 {
	return &val
}

func TestEqual(t *testing.T) {
	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B", "C"},
		// hash dependent
		map[string]int64{"A": 16, "B": 17, "C": 17})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 25, "B": 25})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A", "B"},
		// hash dependent
		map[string]int64{"A": 0, "B": 1})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A", "B", "C", "D"},
		// hash dependent
		map[string]int64{"A": 0, "B": 0, "C": 0, "D": 1})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A"},
		map[string]int64{"A": 1})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{},
		map[string]int64{})
}

func TestEqualWithExisting(t *testing.T) {
	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"C": 30},
		map[string]int64{"A": 10, "B": 10, "C": 30})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 30},
		map[string]int64{"A": 30, "B": 20})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 0, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 1, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 4, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 5, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 6, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		15, []string{"A", "B"},
		map[string]int64{"A": 7, "B": 8},
		map[string]int64{"A": 7, "B": 8})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		500000, []string{"A", "B"},
		map[string]int64{"A": 300000},
		map[string]int64{"A": 300000, "B": 200000})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 10},
		map[string]int64{"A": 25, "B": 25})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 10, "B": 70},
		// hash dependent
		// TODO: Should be 10:40, update algorithm. Issue: #31816
		map[string]int64{"A": 0, "B": 50})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A", "B"},
		map[string]int64{"A": 30},
		map[string]int64{"A": 1, "B": 0})

	doCheckWithExisting(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 10, "B": 20},
		map[string]int64{"A": 25, "B": 25})
}

func TestWithExistingAndCapacity(t *testing.T) {
	// desired without capacity: map[string]int64{"A": 17, "B": 17, "C": 16})
	doCheckWithExistingAndCapacity(t, true, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{},
		map[string]int64{"C": 10},
		map[string]int64{"A": 20, "B": 20, "C": 10},
		map[string]int64{"C": 7})

	// desired B:50 C:0
	doCheckWithExistingAndCapacity(t, true, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000},
		"B": {Weight: 1}},
		50, []string{"B", "C"},
		map[string]int64{},
		map[string]int64{"B": 10},
		map[string]int64{"B": 10, "C": 0},
		map[string]int64{"B": 40},
	)

	// desired A:20 B:40
	doCheckWithExistingAndCapacity(t, true, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 1},
		"B": {Weight: 2}},
		60, []string{"A", "B", "C"},
		map[string]int64{},
		map[string]int64{"B": 10},
		map[string]int64{"A": 50, "B": 10, "C": 0},
		map[string]int64{"B": 30})

	// map[string]int64{"A": 10, "B": 30, "C": 21, "D": 10})
	doCheckWithExistingAndCapacity(t, true, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(21)},
		"D": {Weight: 1, MaxReplicas: pint(10)}},
		71, []string{"A", "B", "C", "D"},
		map[string]int64{},
		map[string]int64{"C": 10},
		map[string]int64{"A": 10, "B": 41, "C": 10, "D": 10},
		map[string]int64{"C": 11},
	)

	// desired A:20 B:20
	doCheckWithExistingAndCapacity(t, false, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 1},
		"B": {Weight: 1}},
		60, []string{"A", "B", "C"},
		map[string]int64{},
		map[string]int64{"A": 10, "B": 10},
		map[string]int64{"A": 10, "B": 10, "C": 0},
		map[string]int64{"A": 20, "B": 20})

	// desired A:10 B:50 although A:50 B:10 is fuly acceptable because rebalance = false
	doCheckWithExistingAndCapacity(t, false, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 1},
		"B": {Weight: 5}},
		60, []string{"A", "B", "C"},
		map[string]int64{},
		map[string]int64{"B": 10},
		map[string]int64{"A": 50, "B": 10, "C": 0},
		map[string]int64{})

	doCheckWithExistingAndCapacity(t, false, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 0}},
		50, []string{"A", "B", "C"},
		map[string]int64{},
		map[string]int64{"B": 10},
		map[string]int64{"A": 20, "B": 10, "C": 20},
		map[string]int64{})

	// Actually we would like to have extra 20 in B but 15 is also good.
	doCheckWithExistingAndCapacity(t, true, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 1}},
		60, []string{"A", "B"},
		map[string]int64{},
		map[string]int64{"B": 10},
		map[string]int64{"A": 50, "B": 10},
		map[string]int64{"B": 15})
}

func TestMin(t *testing.T) {
	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 2, Weight: 0}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 2, "B": 2, "C": 2})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 0}},
		50, []string{"A", "B", "C"},
		// hash dependant.
		map[string]int64{"A": 10, "B": 20, "C": 20})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 0},
		"A": {MinReplicas: 100, Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 50, "B": 0, "C": 0})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 10, Weight: 1, MaxReplicas: pint(12)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 12, "B": 12, "C": 12})
}

func TestMax(t *testing.T) {
	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 1, MaxReplicas: pint(2)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 2, "B": 2, "C": 2})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"*": {Weight: 0, MaxReplicas: pint(2)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 0, "B": 0, "C": 0})
}

func TestWeight(t *testing.T) {
	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 1},
		"B": {Weight: 2}},
		60, []string{"A", "B", "C"},
		map[string]int64{"A": 20, "B": 40, "C": 0})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000},
		"B": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 50, "B": 0, "C": 0})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000},
		"B": {Weight: 1}},
		50, []string{"B", "C"},
		map[string]int64{"B": 50, "C": 0})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 10, "B": 20, "C": 20})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(10)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 10, "B": 30, "C": 10})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(21)},
		"D": {Weight: 1, MaxReplicas: pint(10)}},
		71, []string{"A", "B", "C", "D"},
		map[string]int64{"A": 10, "B": 30, "C": 21, "D": 10})

	doCheck(t, map[string]fedapi.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(21)},
		"D": {Weight: 1, MaxReplicas: pint(10)},
		"E": {Weight: 1}},
		91, []string{"A", "B", "C", "D", "E"},
		map[string]int64{"A": 10, "B": 25, "C": 21, "D": 10, "E": 25})
}
