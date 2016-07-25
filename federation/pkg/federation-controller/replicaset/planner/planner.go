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
	"sort"

	fed_api "k8s.io/kubernetes/federation/apis/federation"
)

// Planner decides how many out of the given replicas should be placed in each of the
// federated clusters.
type Planner struct {
	preferences *fed_api.FederatedReplicaSetPreferences
}

type namedClusterReplicaSetPreferences struct {
	clusterName string
	fed_api.ClusterReplicaSetPreferences
}

type byWeight []*namedClusterReplicaSetPreferences

func (a byWeight) Len() int      { return len(a) }
func (a byWeight) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// Preferences are sorted according by decreasing weight and increasing clusterName.
func (a byWeight) Less(i, j int) bool {
	return (a[i].Weight > a[j].Weight) || (a[i].Weight == a[j].Weight && a[i].clusterName < a[j].clusterName)
}

func NewPlanner(preferences *fed_api.FederatedReplicaSetPreferences) *Planner {
	return &Planner{
		preferences: preferences,
	}
}

// Distribute the desired number of replicas among the given cluster according to the planner preferences.
// The function tries its best to assign each cluster the prefered number of replicas, however if
// sum of MinReplicas for all cluster is bigger thant replicasToDistribute then some cluster will not
// have all of the replicas assigned. In such case a cluster with higher weight has priority over
// cluster with lower weight (or with lexicographically smaller name in case of draw).
func (p *Planner) Plan(replicasToDistribute int64, availableClusters []string) map[string]int64 {
	preferences := make([]*namedClusterReplicaSetPreferences, 0, len(availableClusters))
	plan := make(map[string]int64, len(preferences))

	named := func(name string, pref fed_api.ClusterReplicaSetPreferences) *namedClusterReplicaSetPreferences {
		return &namedClusterReplicaSetPreferences{
			clusterName:                  name,
			ClusterReplicaSetPreferences: pref,
		}
	}

	for _, cluster := range availableClusters {
		if localRSP, found := p.preferences.Clusters[cluster]; found {
			preferences = append(preferences, named(cluster, localRSP))
		} else {
			if localRSP, found := p.preferences.Clusters["*"]; found {
				preferences = append(preferences, named(cluster, localRSP))
			} else {
				plan[cluster] = int64(0)
			}
		}
	}
	sort.Sort(byWeight(preferences))

	remainingReplicas := replicasToDistribute

	// Assign each cluster the minimum number of replicas it requested.
	for _, preference := range preferences {
		min := minInt64(preference.MinReplicas, remainingReplicas)
		remainingReplicas -= min
		plan[preference.clusterName] = min
	}

	modified := true

	// It is possible single pass of the loop is not enough to distribue all replicas among clusters due
	// to weight, max and rounding corner cases. In such case we iterate until either
	// there is no replicas or no cluster gets any more replicas or the number
	// of attempts is less than available cluster count. Every loop either distributes all remainingReplicas
	// or maxes out at least one cluster.
	// TODO: This algorithm is O(clusterCount^2). When needed use sweep-like algorithm for O(n log n).
	for trial := 0; trial < len(availableClusters) && modified && remainingReplicas > 0; trial++ {
		modified = false
		weightSum := int64(0)
		for _, preference := range preferences {
			weightSum += preference.Weight
		}
		newPreferences := make([]*namedClusterReplicaSetPreferences, 0, len(preferences))

		distributeInThisLoop := remainingReplicas
		for _, preference := range preferences {
			if weightSum > 0 {
				start := plan[preference.clusterName]
				// Distribute the remaining replicas, rounding fractions always up.
				extra := (distributeInThisLoop*preference.Weight + weightSum - 1) / weightSum
				extra = minInt64(extra, remainingReplicas)
				// In total there should be the amount that was there at start plus whatever is due
				// in this iteration
				total := start + extra

				// Check if we don't overflow the cluster, and if yes don't consider this cluster
				// in any of the following iterations.
				if preference.MaxReplicas != nil && total > *preference.MaxReplicas {
					total = *preference.MaxReplicas
				} else {
					newPreferences = append(newPreferences, preference)
				}

				// Only total-start replicas were actually taken.
				remainingReplicas -= (total - start)
				plan[preference.clusterName] = total

				// Something extra got scheduled on this cluster.
				if total > start {
					modified = true
				}
			} else {
				break
			}
		}
		preferences = newPreferences
	}

	return plan
}

func minInt64(a int64, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
