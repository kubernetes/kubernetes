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

package nodepreferavoidpods

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	v1helper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// NodePreferAvoidPods is a plugin that priorities nodes according to the node annotation
// "scheduler.alpha.kubernetes.io/preferAvoidPods".
type NodePreferAvoidPods struct {
	handle framework.Handle
}

var _ framework.ScorePlugin = &NodePreferAvoidPods{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "NodePreferAvoidPods"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodePreferAvoidPods) Name() string {
	return Name
}

// Score invoked at the score extension point.
func (pl *NodePreferAvoidPods) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, "node not found")
	}

	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef != nil {
		// Ignore pods that are owned by other controller than ReplicationController
		// or ReplicaSet.
		if controllerRef.Kind != "ReplicationController" && controllerRef.Kind != "ReplicaSet" {
			controllerRef = nil
		}
	}
	if controllerRef == nil {
		return framework.MaxNodeScore, nil
	}

	avoids, err := v1helper.GetAvoidPodsFromNodeAnnotations(node.Annotations)
	if err != nil {
		// If we cannot get annotation, assume it's schedulable there.
		return framework.MaxNodeScore, nil
	}
	for i := range avoids.PreferAvoidPods {
		avoid := &avoids.PreferAvoidPods[i]
		if avoid.PodSignature.PodController.Kind == controllerRef.Kind && avoid.PodSignature.PodController.UID == controllerRef.UID {
			return 0, nil
		}
	}
	return framework.MaxNodeScore, nil
}

// ScoreExtensions of the Score plugin.
func (pl *NodePreferAvoidPods) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, h framework.Handle) (framework.Plugin, error) {
	return &NodePreferAvoidPods{handle: h}, nil
}
