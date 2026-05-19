/*
Copyright 2025 The Kubernetes Authors.

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

package nodedeclaredfeatures

import (
	"context"
	"fmt"
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/version"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.NodeDeclaredFeatures
	// preFilterStateKey is the key in CycleState used to store the pod's feature requirements.
	preFilterStateKey fwk.StateKey = "PreFilter" + Name
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	reqs ndf.FeatureSet
}

// Clone implements StateData.
func (s *preFilterState) Clone() fwk.StateData {
	return s
}

// NodeDeclaredFeatures is a plugin that checks if a node has all the features required by a pod.
type NodeDeclaredFeatures struct {
	ndfFramework *ndf.Framework
	version      *versionutil.Version
	enabled      bool
}

var _ fwk.PreFilterPlugin = &NodeDeclaredFeatures{}
var _ fwk.FilterPlugin = &NodeDeclaredFeatures{}
var _ fwk.EnqueueExtensions = &NodeDeclaredFeatures{}
var _ fwk.SignPlugin = &NodeDeclaredFeatures{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeDeclaredFeatures) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(ctx context.Context, plArgs runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	if !fts.EnableNodeDeclaredFeatures {
		// Disabled, won't do anything.
		return &NodeDeclaredFeatures{}, nil
	}
	ndfFramework, err := ndf.New(features.AllFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to create node feature framework: %w", err)
	}
	ver, err := versionutil.Parse(version.Get().String())
	if err != nil {
		return nil, fmt.Errorf("failed to parse version: %w", err)
	}
	return &NodeDeclaredFeatures{ndfFramework: ndfFramework, version: ver, enabled: true}, nil
}

// PreFilter checks if the pod has any feature requirements.
func (pl *NodeDeclaredFeatures) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !pl.enabled {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	// Pod status is not updated yet, we just pass the spec to node declared features library
	podInfo := &ndf.PodInfo{Spec: &pod.Spec}
	reqs, err := pl.ndfFramework.InferForPodScheduling(podInfo, pl.version)
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	if reqs.Len() == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	cycleState.Write(preFilterStateKey, &preFilterState{reqs: reqs})
	return nil, nil
}

// PreFilterExtensions returns pre-filter extensions, pod add and remove.
func (pl *NodeDeclaredFeatures) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

// Filter checks if the node has the required features.
func (pl *NodeDeclaredFeatures) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !pl.enabled {
		return nil
	}
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}
	result, err := ndf.MatchNodeFeatureSet(s.reqs, nodeInfo.GetNodeDeclaredFeatures())
	if err != nil {
		return fwk.AsStatus(err)
	}
	if !result.IsMatch {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node declared features check failed - unsatisfied requirements: %s", strings.Join(result.UnsatisfiedRequirements, ", ")))
	}
	return nil
}

func (pl *NodeDeclaredFeatures) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	podInfo := &ndf.PodInfo{Spec: &pod.Spec}
	fs, err := pl.ndfFramework.InferForPodScheduling(podInfo, pl.version)
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	featuresList := sets.List(fs.Set)
	return []fwk.SignFragment{
		{Key: fwk.FeaturesSignerName, Value: featuresList},
	}, nil
}

// EventsToRegister returns events that may make a pod schedulable. It is required for the EnqueueExtensions interface.
func (pl *NodeDeclaredFeatures) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	if !pl.enabled {
		return nil, nil
	}
	return []fwk.ClusterEventWithHint{
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.UpdateNodeDeclaredFeature},
			QueueingHintFn: pl.isSchedulableAfterNodeChange,
		},
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Update},
			QueueingHintFn: pl.isSchedulableAfterPodUpdate,
		},
	}, nil
}

func getPreFilterState(cycleState fwk.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		return nil, fmt.Errorf("error reading %q from cycle-state: %w", preFilterStateKey, err)
	}
	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("invalid PreFilter state, got type %T, expected %T", c, &preFilterState{})
	}
	return s, nil
}

func (pl *NodeDeclaredFeatures) isSchedulableAfterPodUpdate(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldPod, newPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}
	// If the pod that was updated is not the target pod, then we don't need to re-evaluate it.
	if pod.UID != newPod.UID {
		logger.V(5).Info("the update event is not for targetPod, skipping queueing", "pod", klog.KObj(newPod))
		return fwk.QueueSkip, nil
	}
	oldPodInfo := &ndf.PodInfo{Spec: &oldPod.Spec}
	newPodInfo := &ndf.PodInfo{Spec: &newPod.Spec}
	oldReqs, err := pl.ndfFramework.InferForPodScheduling(oldPodInfo, pl.version)
	if err != nil {
		logger.Error(err, "failed to infer old pod feature requirements", "pod", klog.KObj(pod))
		return fwk.Queue, err
	}
	newReqs, err := pl.ndfFramework.InferForPodScheduling(newPodInfo, pl.version)
	if err != nil {
		logger.Error(err, "failed to infer new pod feature requirements", "pod", klog.KObj(pod))
		return fwk.Queue, err
	}
	if newReqs.Equal(oldReqs) {
		logger.V(5).Info("pod feature requirements didn't change, skipping queueing", "pod", klog.KObj(newPod))
		return fwk.QueueSkip, nil
	}
	logger.V(4).Info("pod feature requirements changed, queueing", "pod", klog.KObj(pod))
	return fwk.Queue, nil
}

func (pl *NodeDeclaredFeatures) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldNode, newNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}
	if oldNode != nil && slices.Equal(oldNode.Status.DeclaredFeatures, newNode.Status.DeclaredFeatures) {
		logger.V(5).Info("node's declared features didn't change, skipping queueing", "pod", klog.KObj(pod), "node", klog.KObj(newNode))
		return fwk.QueueSkip, nil
	}
	logger.V(4).Info("Node declared features updated, queueing", "pod", klog.KObj(pod), "node", klog.KObj(newNode))
	return fwk.Queue, nil
}
