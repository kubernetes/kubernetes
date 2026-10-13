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

package statefulset

import (
	"time"
)

const (
	// StatefulSetPoll is a poll interval for StatefulSet tests
	StatefulSetPoll = 10 * time.Second
	// StatefulSetTimeout is a timeout interval for StatefulSet operations
	StatefulSetTimeout = 10 * time.Minute
	// StatefulPodTimeout is a timeout for stateful pods to change state
	StatefulPodTimeout = 5 * time.Minute
)
