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

package queuesort

import (
	"k8s.io/apimachinery/pkg/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// defaultQueueSort is the default queue sort plugin.
type defaultQueueSort struct{}

var _ = framework.QueueSortPlugin(defaultQueueSort{})

// Name is the name of the plugin used in Registry and configurations.
const Name = "default-queue-sort-plugin"

// Name returns name of the plugin. It is used in logs, etc.
func (dq defaultQueueSort) Name() string {
	return Name
}

// Less is used to sort pods in the scheduling queue.
// It sorts pods based on their priority. When priorities are equal, it uses PodInfo.timestamp.
func (dq defaultQueueSort) Less(pInfo1 *framework.PodInfo, pInfo2 *framework.PodInfo) bool {
	prio1 := util.GetPodPriority(pInfo1.Pod)
	prio2 := util.GetPodPriority(pInfo2.Pod)
	return (prio1 > prio2) || (prio1 == prio2 && pInfo1.Timestamp.Before(pInfo2.Timestamp))
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &defaultQueueSort{}, nil
}
