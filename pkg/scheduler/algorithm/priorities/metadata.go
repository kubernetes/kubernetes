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
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
)

// MetadataFactory is a factory to produce PriorityMetadata.
type MetadataFactory struct {
	serviceLister     corelisters.ServiceLister
	controllerLister  corelisters.ReplicationControllerLister
	replicaSetLister  appslisters.ReplicaSetLister
	statefulSetLister appslisters.StatefulSetLister
}

// NewMetadataFactory creates a MetadataFactory.
func NewMetadataFactory(
	serviceLister corelisters.ServiceLister,
	controllerLister corelisters.ReplicationControllerLister,
	replicaSetLister appslisters.ReplicaSetLister,
	statefulSetLister appslisters.StatefulSetLister,
) MetadataProducer {
	factory := &MetadataFactory{
		serviceLister:     serviceLister,
		controllerLister:  controllerLister,
		replicaSetLister:  replicaSetLister,
		statefulSetLister: statefulSetLister,
	}
	return factory.PriorityMetadata
}

// priorityMetadata is a type that is passed as metadata for priority functions
type priorityMetadata struct{}

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
	return &priorityMetadata{}
}
