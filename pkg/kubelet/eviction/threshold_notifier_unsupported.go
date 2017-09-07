// +build !linux

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

package eviction

import "fmt"

// NewMemCGThresholdNotifier sends notifications when a cgroup threshold
// is crossed (in either direction) for a given cgroup attribute
func NewMemCGThresholdNotifier(path, attribute, threshold, description string, handler thresholdNotifierHandlerFunc) (ThresholdNotifier, error) {
	return nil, fmt.Errorf("threshold notification not supported")
}
