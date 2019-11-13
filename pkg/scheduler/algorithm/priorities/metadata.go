/*
Copyright 2016 The Kubernetes Authors.

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

package priorities

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// MetadataFactory is a factory to produce PriorityMetadata.
type MetadataFactory struct {
	serviceLister         corelisters.ServiceLister
	controllerLister      corelisters.ReplicationControllerLister
	replicaSetLister      appslisters.ReplicaSetLister
	statefulSetLister     appslisters.StatefulSetLister
	hardPodAffinityWeight int32
}

// NewMetadataFactory creates a MetadataFactory.
func NewMetadataFactory(
	serviceLister corelisters.ServiceLister,
	controllerLister corelisters.ReplicationControllerLister,
	replicaSetLister appslisters.ReplicaSetLister,
	statefulSetLister appslisters.StatefulSetLister,
	hardPodAffinityWeight int32,
) MetadataProducer {
	factory := &MetadataFactory{
		serviceLister:         serviceLister,
		controllerLister:      controllerLister,
		replicaSetLister:      replicaSetLister,
		statefulSetLister:     statefulSetLister,
		hardPodAffinityWeight: hardPodAffinityWeight,
	}
	return factory.PriorityMetadata
}

// priorityMetadata is a type that is passed as metadata for priority functions
type priorityMetadata struct {
	podLimits               *schedulernodeinfo.Resource
	podTolerations          []v1.Toleration
	affinity                *v1.Affinity
	podSelector             labels.Selector
	controllerRef           *metav1.OwnerReference
	podFirstServiceSelector labels.Selector
	totalNumNodes           int
	podTopologySpreadMap    *podTopologySpreadMap
	topologyScore           topologyPairToScore
}

// PriorityMetadata is a MetadataProducer.  Node info can be nil.
func (pmf *MetadataFactory) PriorityMetadata(
	pod *v1.Pod,
	filteredNodes []*v1.Node,
	sharedLister schedulerlisters.SharedLister,
) interface{} {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	totalNumNodes := 0
	var allNodes []*schedulernodeinfo.NodeInfo
	if sharedLister != nil {
		if l, err := sharedLister.NodeInfos().List(); err == nil {
			totalNumNodes = len(l)
			allNodes = l
		}
	}
	tpSpreadMap, err := buildPodTopologySpreadMap(pod, filteredNodes, allNodes)
	if err != nil {
		klog.Errorf("Error building podTopologySpreadMap: %v", err)
		return nil
	}
	return &priorityMetadata{
		podLimits:               getResourceLimits(pod),
		podTolerations:          getAllTolerationPreferNoSchedule(pod.Spec.Tolerations),
		affinity:                pod.Spec.Affinity,
		podSelector:             getSelector(pod, pmf.serviceLister, pmf.controllerLister, pmf.replicaSetLister, pmf.statefulSetLister),
		controllerRef:           metav1.GetControllerOf(pod),
		podFirstServiceSelector: getFirstServiceSelector(pod, pmf.serviceLister),
		totalNumNodes:           totalNumNodes,
		podTopologySpreadMap:    tpSpreadMap,
		topologyScore:           buildTopologyPairToScore(pod, sharedLister, filteredNodes, pmf.hardPodAffinityWeight),
	}
}

// getFirstServiceSelector returns one selector of services the given pod.
func getFirstServiceSelector(pod *v1.Pod, sl corelisters.ServiceLister) (firstServiceSelector labels.Selector) {
	if services, err := sl.GetPodServices(pod); err == nil && len(services) > 0 {
		return labels.SelectorFromSet(services[0].Spec.Selector)
	}
	return nil
}

// getSelector returns a selector for the services, RCs, RSs, and SSs matching the given pod.
func getSelector(pod *v1.Pod, sl corelisters.ServiceLister, cl corelisters.ReplicationControllerLister, rsl appslisters.ReplicaSetLister, ssl appslisters.StatefulSetLister) labels.Selector {
	labelSet := make(labels.Set)
	// Since services, RCs, RSs and SSs match the pod, they won't have conflicting
	// labels. Merging is safe.

	if services, err := sl.GetPodServices(pod); err == nil {
		for _, service := range services {
			labelSet = labels.Merge(labelSet, service.Spec.Selector)
		}
	}

	if rcs, err := cl.GetPodControllers(pod); err == nil {
		for _, rc := range rcs {
			labelSet = labels.Merge(labelSet, rc.Spec.Selector)
		}
	}

	selector := labels.NewSelector()
	if len(labelSet) != 0 {
		selector = labelSet.AsSelector()
	}

	if rss, err := rsl.GetPodReplicaSets(pod); err == nil {
		for _, rs := range rss {
			if other, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
				if r, ok := other.Requirements(); ok {
					selector = selector.Add(r...)
				}
			}
		}
	}

	if sss, err := ssl.GetPodStatefulSets(pod); err == nil {
		for _, ss := range sss {
			if other, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector); err == nil {
				if r, ok := other.Requirements(); ok {
					selector = selector.Add(r...)
				}
			}
		}
	}

	return selector
}
