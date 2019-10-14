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

package volumezone

import (
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// VolumeZone is a plugin that checks volume zone
type VolumeZone struct {
	predicate predicates.FitPredicate
}

var _ = framework.FilterPlugin(&VolumeZone{})

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "VolumeZone"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeZone) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *VolumeZone) Filter(_ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	// metadata is not needed
	_, reasons, err := pl.predicate(pod, nil, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// New initializes a new plugin and returns it.
func New(pvInfo predicates.PersistentVolumeInfo, pvcInfo predicates.PersistentVolumeClaimInfo, classInfo predicates.StorageClassInfo) framework.Plugin {
	return &VolumeZone{
		predicate: predicates.NewVolumeZonePredicate(pvInfo, pvcInfo, classInfo),
	}
}
