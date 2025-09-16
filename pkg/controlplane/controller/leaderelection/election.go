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
	v1beta1 "k8s.io/api/coordination/v1beta1"
	"k8s.io/utils/clock"
)

func pickBestLeaderOldestEmulationVersion(candidates []*v1beta1.LeaseCandidate) *v1beta1.LeaseCandidate {
	var electee *v1beta1.LeaseCandidate
	for _, c := range candidates {
		if !validLeaseCandidateForOldestEmulationVersion(c) {
			continue
		}
		if electee == nil || compare(electee, c) > 0 {
			electee = c
		}
	}
	return electee
}

func pickBestStrategy(candidates []*v1beta1.LeaseCandidate) (v1.CoordinatedLeaseStrategy, error) {
	nilStrategy := v1.CoordinatedLeaseStrategy("")
	if len(candidates) == 0 {
		return nilStrategy, fmt.Errorf("no candidates")
	}
	candidateName := candidates[0].Name
	strategy := candidates[0].Spec.Strategy
	highestBV := getBinaryVersionOrZero(candidates[0])

	for _, c := range candidates[1:] {
		binVersion := getBinaryVersionOrZero(c)
		result := highestBV.Compare(binVersion)
		if result < 0 {
			strategy = c.Spec.Strategy
			highestBV = binVersion
			candidateName = c.Name
		} else if result == 0 && c.Spec.Strategy != strategy {
			return nilStrategy, fmt.Errorf("candidates %q, %q at same binary version but received differing strategies %s, %s", candidateName, c.Name, strategy, c.Spec.Strategy)
		}
	}
	return strategy, nil
}

func validLeaseCandidateForOldestEmulationVersion(l *v1beta1.LeaseCandidate) bool {
	_, err := semver.ParseTolerant(l.Spec.EmulationVersion)
	if err != nil {
		return false
	}
	_, err = semver.ParseTolerant(l.Spec.BinaryVersion)
	return err == nil
}

func getEmulationVersionOrZero(l *v1beta1.LeaseCandidate) semver.Version {
	value := l.Spec.EmulationVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

func getBinaryVersionOrZero(l *v1beta1.LeaseCandidate) semver.Version {
	value := l.Spec.BinaryVersion
	v, err := semver.ParseTolerant(value)
	if err != nil {
		return semver.Version{}
	}
	return v
}

// -1: lhs better, 1: rhs better
func compare(lhs, rhs *v1beta1.LeaseCandidate) int {
	l := getEmulationVersionOrZero(lhs)
	r := getEmulationVersionOrZero(rhs)
	result := l.Compare(r)
	if result == 0 {
		l := getBinaryVersionOrZero(lhs)
		r := getBinaryVersionOrZero(rhs)
		result = l.Compare(r)
	}
	if result == 0 {
		if lhs.CreationTimestamp.After(rhs.CreationTimestamp.Time) {
			return 1
		}
		return -1
	}
	return result
}

func isLeaseExpired(clock clock.Clock, lease *v1.Lease) bool {
	currentTime := clock.Now()
	return lease.Spec.RenewTime == nil ||
		lease.Spec.LeaseDurationSeconds == nil ||
		lease.Spec.RenewTime.Add(time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second).Before(currentTime)
}

func isLeaseCandidateExpired(clock clock.Clock, lease *v1beta1.LeaseCandidate) bool {
	currentTime := clock.Now()
	return lease.Spec.RenewTime == nil ||
		lease.Spec.RenewTime.Add(leaseCandidateValidDuration).Before(currentTime)
}
