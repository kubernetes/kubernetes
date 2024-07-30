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

package interpodaffinity

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = names.InterPodAffinity

var _ framework.PreFilterPlugin = &InterPodAffinity{}
var _ framework.FilterPlugin = &InterPodAffinity{}
var _ framework.PreScorePlugin = &InterPodAffinity{}
var _ framework.ScorePlugin = &InterPodAffinity{}
var _ framework.EnqueueExtensions = &InterPodAffinity{}

// InterPodAffinity is a plugin that checks inter pod affinity
type InterPodAffinity struct {
	parallelizer parallelize.Parallelizer
	args         config.InterPodAffinityArgs
	sharedLister framework.SharedLister
	nsLister     listersv1.NamespaceLister
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *InterPodAffinity) Name() string {
	return Name
}

// EventsToRegister returns the possible events that may make a failed Pod
// schedulable
func (pl *InterPodAffinity) EventsToRegister(_ context.Context) ([]framework.ClusterEventWithHint, error) {
	return []framework.ClusterEventWithHint{
		// All ActionType includes the following events:
		// - Delete. An unschedulable Pod may fail due to violating an existing Pod's anti-affinity constraints,
		// deleting an existing Pod may make it schedulable.
		// - UpdatePodLabel. Updating on an existing Pod's labels (e.g., removal) may make
		// an unschedulable Pod schedulable.
		// - Add. An unschedulable Pod may fail due to violating pod-affinity constraints,
		// adding an assigned Pod may make it schedulable.
		//
		// A note about UpdateNodeTaint event:
		// NodeAdd QueueingHint isn't always called because of the internal feature called preCheck.
		// As a common problematic scenario,
		// when a node is added but not ready, NodeAdd event is filtered out by preCheck and doesn't arrive.
		// In such cases, this plugin may miss some events that actually make pods schedulable.
		// As a workaround, we add UpdateNodeTaint event to catch the case.
		// We can remove UpdateNodeTaint when we remove the preCheck feature.
		// See: https://github.com/kubernetes/kubernetes/issues/110175
		{Event: framework.ClusterEvent{Resource: framework.Pod, ActionType: framework.Add | framework.UpdatePodLabel | framework.Delete}, QueueingHintFn: pl.isSchedulableAfterPodChange},
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeLabel | framework.UpdateNodeTaint}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
	}, nil
}

// New initializes a new plugin and returns it.
func New(_ context.Context, plArgs runtime.Object, h framework.Handle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	args, err := getArgs(plArgs)
	if err != nil {
		return nil, err
	}
	if err := validation.ValidateInterPodAffinityArgs(nil, &args); err != nil {
		return nil, err
	}
	pl := &InterPodAffinity{
		parallelizer: h.Parallelizer(),
		args:         args,
		sharedLister: h.SnapshotSharedLister(),
		nsLister:     h.SharedInformerFactory().Core().V1().Namespaces().Lister(),
	}

	return pl, nil
}

func getArgs(obj runtime.Object) (config.InterPodAffinityArgs, error) {
	ptr, ok := obj.(*config.InterPodAffinityArgs)
	if !ok {
		return config.InterPodAffinityArgs{}, fmt.Errorf("want args to be of type InterPodAffinityArgs, got %T", obj)
	}
	return *ptr, nil
}

// Updates Namespaces with the set of namespaces identified by NamespaceSelector.
// If successful, NamespaceSelector is set to nil.
// The assumption is that the term is for an incoming pod, in which case
// namespaceSelector is either unrolled into Namespaces (and so the selector
// is set to Nothing()) or is Empty(), which means match everything. Therefore,
// there when matching against this term, there is no need to lookup the existing
// pod's namespace labels to match them against term's namespaceSelector explicitly.
func (pl *InterPodAffinity) mergeAffinityTermNamespacesIfNotEmpty(at *framework.AffinityTerm) error {
	if at.NamespaceSelector.Empty() {
		return nil
	}
	ns, err := pl.nsLister.List(at.NamespaceSelector)
	if err != nil {
		return err
	}
	for _, n := range ns {
		at.Namespaces.Insert(n.Name)
	}
	at.NamespaceSelector = labels.Nothing()
	return nil
}

// GetNamespaceLabelsSnapshot returns a snapshot of the labels associated with
// the namespace.
func GetNamespaceLabelsSnapshot(logger klog.Logger, ns string, nsLister listersv1.NamespaceLister) (nsLabels labels.Set) {
	podNS, err := nsLister.Get(ns)
	if err == nil {
		// Create and return snapshot of the labels.
		return labels.Merge(podNS.Labels, nil)
	}
	logger.V(3).Info("getting namespace, assuming empty set of namespace labels", "namespace", ns, "err", err)
	return
}

func (pl *InterPodAffinity) isSchedulableAfterPodChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	originalPod, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}
	if (modifiedPod != nil && modifiedPod.Spec.NodeName == "") || (originalPod != nil && originalPod.Spec.NodeName == "") {
		logger.V(5).Info("the added/updated/deleted pod is unscheduled, so it doesn't make the target pod schedulable",
			"pod", klog.KObj(pod), "originalPod", klog.KObj(originalPod), "modifiedPod", klog.KObj(modifiedPod))
		return framework.QueueSkip, nil
	}

	terms, err := framework.GetAffinityTerms(pod, framework.GetPodAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		return framework.Queue, err
	}

	antiTerms, err := framework.GetAffinityTerms(pod, framework.GetPodAntiAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		return framework.Queue, err
	}

	// Pod is updated. Return Queue when the updated pod matching the target pod's affinity or not matching anti-affinity.
	// Note that, we don't need to check each affinity individually when the Pod has more than one affinity
	// because the current PodAffinity looks for a **single** existing pod that can satisfy **all** the terms of inter-pod affinity of an incoming pod.
	if modifiedPod != nil && originalPod != nil {
		if !podMatchesAllAffinityTerms(terms, originalPod) && podMatchesAllAffinityTerms(terms, modifiedPod) {
			logger.V(5).Info("a scheduled pod was updated to match the target pod's affinity, and the pod may be schedulable now",
				"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
			return framework.Queue, nil
		}
		if podMatchesAllAffinityTerms(antiTerms, originalPod) && !podMatchesAllAffinityTerms(antiTerms, modifiedPod) {
			logger.V(5).Info("a scheduled pod was updated not to match the target pod's anti affinity, and the pod may be schedulable now",
				"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
			return framework.Queue, nil
		}
		logger.V(5).Info("a scheduled pod was updated but it doesn't match the target pod's affinity or does match the target pod's anti-affinity",
			"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return framework.QueueSkip, nil
	}

	// Pod is added. Return Queue when the added pod matching the target pod's affinity.
	if modifiedPod != nil {
		if podMatchesAllAffinityTerms(terms, modifiedPod) {
			logger.V(5).Info("a scheduled pod was added and it matches the target pod's affinity",
				"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
			return framework.Queue, nil
		}
		logger.V(5).Info("a scheduled pod was added and it doesn't match the target pod's affinity",
			"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return framework.QueueSkip, nil
	}

	// Pod is deleted. Return Queue when the deleted pod matching the target pod's anti-affinity.
	if !podMatchesAllAffinityTerms(antiTerms, originalPod) {
		logger.V(5).Info("a scheduled pod was deleted but it doesn't match the target pod's anti-affinity",
			"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
		return framework.QueueSkip, nil
	}
	logger.V(5).Info("a scheduled pod was deleted and it matches the target pod's anti-affinity. The pod may be schedulable now",
		"pod", klog.KObj(pod), "modifiedPod", klog.KObj(modifiedPod))
	return framework.Queue, nil
}

func (pl *InterPodAffinity) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	_, modifiedNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	terms, err := framework.GetAffinityTerms(pod, framework.GetPodAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		return framework.Queue, err
	}

	for _, term := range terms {
		if _, ok := modifiedNode.Labels[term.TopologyKey]; ok {
			logger.V(5).Info("a node with matched pod affinity topologyKey was added/updated and it may make pod schedulable",
				"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
			return framework.Queue, err
		}
	}

	antiTerms, err := framework.GetAffinityTerms(pod, framework.GetPodAntiAffinityTerms(pod.Spec.Affinity))
	if err != nil {
		return framework.Queue, err
	}

	for _, term := range antiTerms {
		if _, ok := modifiedNode.Labels[term.TopologyKey]; ok {
			logger.V(5).Info("a node with matched pod anti-affinity topologyKey was added/updated and it may make pod schedulable",
				"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
			return framework.Queue, err
		}
	}
	logger.V(5).Info("a node is added/updated but doesn't have any topologyKey which matches pod affinity/anti-affinity",
		"pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
	return framework.QueueSkip, nil
}
