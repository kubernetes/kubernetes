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

package debugger

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// Debugger is a plugin that dumps node infos
type Debugger struct {
	handle framework.FrameworkHandle
}

var _ framework.PreFilterPlugin = &Debugger{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "Debugger"

	// ErrReason for node affinity/selector not matching.
	ErrReason = "node(s) didn't match node selector"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *Debugger) Name() string {
	return Name
}

func resourceListToString(resourceList v1.ResourceList) string {
	var requested []string
	for name, quantity := range resourceList {
		requested = append(requested, fmt.Sprintf("%v=%v", name, quantity.String()))
	}
	return fmt.Sprintf("%v", requested)
}

func (pl *Debugger) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) *framework.Status {
	// Dump the node cache with current resource consumption (e.g. cpu, memory, pods)
	nodeInfos, err := pl.handle.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return framework.NewStatus(framework.Error, "nodeInfos not found")
	}
	for _, nodeInfo := range nodeInfos {
		node := nodeInfo.Node()
		klog.InfoS(
			"Dumping node infos",
			"node", node.Name,
			"requested", resourceListToString(nodeInfo.Requested.ResourceList()),
		)
	}
	return nil
}

func (pl *Debugger) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &Debugger{handle: h}, nil
}
