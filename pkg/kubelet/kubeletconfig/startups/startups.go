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

package startups

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation"
)

const (
	// we allow one extra startup to account for the startup necessary to update configuration
	maxStartups = validation.MaxCrashLoopThreshold + 1
)

// Tracker tracks Kubelet startups in a storage layer
type Tracker interface {
	// Initialize sets up the storage layer
	Initialize() error
	// RecordStartup records the current time as a Kubelet startup
	RecordStartup() error
	// StartupsSince returns the number of Kubelet startus recorded since `t`
	StartupsSince(t time.Time) (int32, error)
}

func startupsSince(ls []string, start time.Time) (int32, error) {
	// since the list is append-only we only need to count the number of timestamps since `t`
	startups := int32(0)
	for _, stamp := range ls {
		t, err := time.Parse(time.RFC3339, stamp)
		if err != nil {
			return 0, fmt.Errorf("failed to parse timestamp while counting startups, error: %v", err)
		}
		if t.After(start) {
			startups++
		}
	}
	return startups, nil
}

func recordStartup(ls []string) []string {
	// record current time
	now := time.Now()
	stamp := now.Format(time.RFC3339) // use RFC3339 time format
	ls = append(ls, stamp)

	// rotate the slice if necessary
	if len(ls) > maxStartups {
		ls = ls[1:]
	}

	// return the new slice
	return ls
}
