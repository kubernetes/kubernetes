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

package clusterautoscalerintegrationtest

import (
	scheduler_listers "k8s.io/kubernetes/pkg/scheduler/listers"
)

// DelegatingSchedulerSharedLister is an implementation of scheduler.SharedLister which
// passes logic to delegate. Delegate can be updated.
type DelegatingSchedulerSharedLister struct {
	delegate scheduler_listers.SharedLister
}

// Pods returns a PodLister
func (lister *DelegatingSchedulerSharedLister) Pods() scheduler_listers.PodLister {
	return lister.delegate.Pods()
}

// NodeInfos returns a NodeInfoLister.
func (lister *DelegatingSchedulerSharedLister) NodeInfos() scheduler_listers.NodeInfoLister {
	return lister.delegate.NodeInfos()
}

// UpdateDelegate updates the delegate
func (lister *DelegatingSchedulerSharedLister) UpdateDelegate(delegate scheduler_listers.SharedLister) {
	lister.delegate = delegate
}

