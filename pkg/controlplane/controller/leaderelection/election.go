/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"fmt"
	"time"

	"github.com/blang/semver/v4"
	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	"k8s.io/klog/v2"
)

func pickBestLeaderOldestEmulationVersion(candidates []*v1alpha1.LeaseCandidate) *v1alpha1.LeaseCandidate {
	var electee *v1alpha1.LeaseCandidate
	for _, c := range candidates {
		if !validLeaseCandidateForOldestEmulationVersion(c) {
			continue
		}
		if electee == nil || compare(electee, c) > 0 {
			electee = c
		}
	}
	if electee == nil {
		klog.Infof("pickBestLeader: none found")
	} else {
		klog.Infof("pickBestLeader: %s %s", electee.Namespace, electee.Name)
	}
	return electee
}

func shouldReelect(candidates []*v1alpha1.LeaseCandidate, currentLeader *v1alpha1.LeaseCandidate) bool {
	klog.Infof("shouldReelect for candidates: %+v", candidates)
	pickedLeader := pickBestLeaderOldestEmulationVersion(candidates)
	if pickedLeader == nil {
		return false
	}
	return compare(currentLeader, pickedLeader) > 0
}

// topologicalSortWithOneRoot has a caveat that there may only be one root (indegree=0) node in a valid ordering.
func topologicalSortWithOneRoot(graph map[v1.CoordinatedLeaseStrategy][]v1.CoordinatedLeaseStrategy) []v1.CoordinatedLeaseStrategy {
	inDegree := make(map[v1.CoordinatedLeaseStrategy]int)
	for node := range graph {
		inDegree[node] = 0
	}
	for _, neighbors := range graph {
		for _, neighbor := range neighbors {
			inDegree[neighbor]++
		}
	}

	var queue []v1.CoordinatedLeaseStrategy
	for vertex, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, vertex)
		}
	}

	// If multiple nodes have indegree of 0, multiple strategies are non-superceding and is a conflict.
	if len(queue) > 1 {
		return nil
	}

	var sorted []v1.CoordinatedLeaseStrategy
	for len(queue) > 0 {
		vertex := queue[0]
		queue = queue[1:]
		sorted = append(sorted, vertex)

		for _, neighbor := range graph[vertex] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	if len(sorted) != len(graph) {
		fmt.Printf("%s", (sorted))
		return nil // Cycle detected
	}

	return sorted
}

func pickBestStrategy(candidates []*v1alpha1.LeaseCandidate) (v1.CoordinatedLeaseStrategy, error) {
	graph := make(map[v1.CoordinatedLeaseStrategy][]v1.CoordinatedLeaseStrategy)
	nilStrategy := v1.CoordinatedLeaseStrategy("")
	for _, c := range candidates {
		for i := range len(c.Spec.PreferredStrategies) - 1 {
			graph[c.Spec.PreferredStrategies[i]] = append(graph[c.Spec.PreferredStrategies[i]], c.Spec.PreferredStrategies[i+1])
		}
		if _, ok := graph[c.Spec.PreferredStrategies[len(c.Spec.PreferredStrategies)-1]]; !ok {
			graph[c.Spec.PreferredStrategies[len(c.Spec.PreferredStrategies)-1]] = []v1.CoordinatedLeaseStrategy{}
		}
	}

	sorted := topologicalSortWithOneRoot(graph)
	if sorted == nil {
		return nilStrategy, fmt.Errorf("invalid strategy")
	}

	return sorted[0], nil
}

func validLeaseCandidateForOldestEmulationVersion(l *v1alpha1.LeaseCandidate) bool {
	_, err := semver.ParseTolerant(l.Spec.EmulationVersion)
	if err != nil {
		return false
	}
	_, err = semver.ParseTolerant(l.Spec.BinaryVersion)
	return err == nil
}

func getEmulationVersion(l *v1alpha1.LeaseCandidate) semver.Version {
	value := l.Spec.EmulationVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

func getBinaryVersion(l *v1alpha1.LeaseCandidate) semver.Version {
	value := l.Spec.BinaryVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

// -1: lhs better, 1: rhs better
func compare(lhs, rhs *v1alpha1.LeaseCandidate) int {
	lhsVersion := getEmulationVersion(lhs)
	rhsVersion := getEmulationVersion(rhs)
	result := lhsVersion.Compare(rhsVersion)
	if result == 0 {
		lhsVersion := getBinaryVersion(lhs)
		rhsVersion := getBinaryVersion(rhs)
		result = lhsVersion.Compare(rhsVersion)
	}
	if result == 0 {
		if lhs.CreationTimestamp.After(rhs.CreationTimestamp.Time) {
			return 1
		}
		return -1
	}
	return result
}

func isLeaseExpired(lease *v1.Lease) bool {
	currentTime := time.Now()
	return lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime)
}

func isLeaseCandidateExpired(lease *v1alpha1.LeaseCandidate) bool {
	currentTime := time.Now()
	return lease.Spec.RenewTime == nil ||
		lease.Spec.RenewTime.Add(leaseCandidateValidDuration).Before(currentTime)
}
