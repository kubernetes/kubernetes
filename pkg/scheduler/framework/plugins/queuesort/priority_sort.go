/*
Copyright 2020 The Kubernetes Authors.

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

package queuesort

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = names.PrioritySort

// PrioritySort is a plugin that implements Priority based sorting.
type PrioritySort struct{}

var _ fwk.QueueSortPlugin = &PrioritySort{}

// Name returns name of the plugin.
func (pl *PrioritySort) Name() string {
	return Name
}

// Less is the function used by the activeQ heap algorithm to sort entities.
// It sorts entities based on their priority. When priorities are equal, it uses
// the entity timestamp.
func (pl *PrioritySort) Less(entity1, entity2 fwk.QueuedEntityInfo) bool {
	p1 := entity1.GetPriority()
	p2 := entity2.GetPriority()
	return (p1 > p2) || (p1 == p2 && entity1.GetTimestamp().Before(entity2.GetTimestamp()))
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
	return &PrioritySort{}, nil
}
