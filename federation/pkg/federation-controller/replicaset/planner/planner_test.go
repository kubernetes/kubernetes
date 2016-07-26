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

package planer

import (
	"testing"

	fed_api "k8s.io/kubernetes/federation/apis/federation"

	"github.com/stretchr/testify/assert"
)

func doCheck(t *testing.T, pref map[string]fed_api.ClusterReplicaSetPreferences, replicas int64, clusters []string, expected map[string]int64) {
	planer := NewPlanner(&fed_api.FederatedReplicaSetPreferences{
		Clusters: pref,
	})
	plan := planer.Plan(replicas, clusters)
	assert.EqualValues(t, expected, plan)
}

func pint(val int64) *int64 {
	return &val
}

func TestEqual(t *testing.T) {
	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 17, "B": 17, "C": 16})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		50, []string{"A", "B"},
		map[string]int64{"A": 25, "B": 25})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A", "B"},
		map[string]int64{"A": 1, "B": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A", "B", "C", "D"},
		map[string]int64{"A": 1, "B": 0, "C": 0, "D": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{"A"},
		map[string]int64{"A": 1})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1}},
		1, []string{},
		map[string]int64{})
}

func TestMin(t *testing.T) {
	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 2, Weight: 0}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 2, "B": 2, "C": 2})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 0}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 20, "B": 20, "C": 10})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 20, Weight: 0},
		"A": {MinReplicas: 100, Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 50, "B": 0, "C": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {MinReplicas: 10, Weight: 1, MaxReplicas: pint(12)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 12, "B": 12, "C": 12})
}

func TestMax(t *testing.T) {
	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 1, MaxReplicas: pint(2)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 2, "B": 2, "C": 2})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"*": {Weight: 0, MaxReplicas: pint(2)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 0, "B": 0, "C": 0})
}

func TestWeight(t *testing.T) {

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 1},
		"B": {Weight: 2}},
		60, []string{"A", "B", "C"},
		map[string]int64{"A": 20, "B": 40, "C": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000},
		"B": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 50, "B": 0, "C": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000},
		"B": {Weight: 1}},
		50, []string{"B", "C"},
		map[string]int64{"B": 50, "C": 0})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 10, "B": 20, "C": 20})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(10)}},
		50, []string{"A", "B", "C"},
		map[string]int64{"A": 10, "B": 30, "C": 10})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(21)},
		"D": {Weight: 1, MaxReplicas: pint(10)}},
		71, []string{"A", "B", "C", "D"},
		map[string]int64{"A": 10, "B": 30, "C": 21, "D": 10})

	doCheck(t, map[string]fed_api.ClusterReplicaSetPreferences{
		"A": {Weight: 10000, MaxReplicas: pint(10)},
		"B": {Weight: 1},
		"C": {Weight: 1, MaxReplicas: pint(21)},
		"D": {Weight: 1, MaxReplicas: pint(10)},
		"E": {Weight: 1}},
		91, []string{"A", "B", "C", "D", "E"},
		map[string]int64{"A": 10, "B": 25, "C": 21, "D": 10, "E": 25})
}
