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
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
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
func (pl *InterPodAffinity) EventsToRegister() []framework.ClusterEvent {
	return []framework.ClusterEvent{
		// All ActionType includes the following events:
		// - Delete. An unschedulable Pod may fail due to violating an existing Pod's anti-affinity constraints,
		// deleting an existing Pod may make it schedulable.
		// - Update. Updating on an existing Pod's labels (e.g., removal) may make
		// an unschedulable Pod schedulable.
		// - Add. An unschedulable Pod may fail due to violating pod-affinity constraints,
		// adding an assigned Pod may make it schedulable.
		{Resource: framework.Pod, ActionType: framework.All},
		{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeLabel},
	}
}

// New initializes a new plugin and returns it.
func New(plArgs runtime.Object, h framework.Handle) (framework.Plugin, error) {
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
func GetNamespaceLabelsSnapshot(ns string, nsLister listersv1.NamespaceLister) (nsLabels labels.Set) {
	podNS, err := nsLister.Get(ns)
	if err == nil {
		// Create and return snapshot of the labels.
		return labels.Merge(podNS.Labels, nil)
	}
	klog.V(3).InfoS("getting namespace, assuming empty set of namespace labels", "namespace", ns, "err", err)
	return
}
