/*
Copyright The Kubernetes Authors.

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

package robustness

import (
	"context"

	"k8s.io/client-go/util/workqueue"
)

// FaultInjectingWorkQueue wraps workqueue.RateLimitingInterface to control reconciler trigger events.
type FaultInjectingWorkQueue struct {
	workqueue.RateLimitingInterface
	registry *FaultRegistry
	name     string
}

// NewFaultInjectingWorkQueue creates a wrapped RateLimitingInterface.
func NewFaultInjectingWorkQueue(realQueue workqueue.RateLimitingInterface, registry *FaultRegistry, name string) workqueue.RateLimitingInterface {
	return &FaultInjectingWorkQueue{
		RateLimitingInterface: realQueue,
		registry:              registry,
		name:                  name,
	}
}

func (q *FaultInjectingWorkQueue) Get() (item interface{}, shutdown bool) {
	// Runs any blocking faults registered on this queue's get operation.
	q.registry.ResolveQueue(context.Background(), QueueFacts{Queue: q.name, Op: "get"})
	return q.RateLimitingInterface.Get()
}

func (q *FaultInjectingWorkQueue) Add(item interface{}) {
	q.registry.ResolveQueue(context.Background(), QueueFacts{Queue: q.name, Op: "add"})
	q.RateLimitingInterface.Add(item)
}
