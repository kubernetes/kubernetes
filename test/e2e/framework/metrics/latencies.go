/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"time"
)

const (
	// SingleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	// TODO: client should not apply this timeout to Watch calls. Increased from 30s until that is fixed.
	SingleCallTimeout = 5 * time.Minute
)

// PodLatencyData encapsulates pod startup latency information.
type PodLatencyData struct {
	// Name of the pod
	Name string
	// Node this pod was running on
	Node string
	// Latency information related to pod startuptime
	Latency time.Duration
}

// LatencySlice is an array of PodLatencyData which encapsulates pod startup latency information.
type LatencySlice []PodLatencyData

func (a LatencySlice) Len() int           { return len(a) }
func (a LatencySlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LatencySlice) Less(i, j int) bool { return a[i].Latency < a[j].Latency }
