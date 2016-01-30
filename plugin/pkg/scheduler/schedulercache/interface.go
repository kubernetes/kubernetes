/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package schedulercache

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

// Cache collects pods' information and provides node-level aggregated information.
// It's intended for generic scheduler to do efficient lookup.
// Cache's operations are pod centric. It incrementally updates itself based on pod event.
// Pod events are sent via network. We don't have guaranteed delivery of all events.
// Thus, we organized the state machine flow of a pod's events as followed.
//
// State Machine of a pod's events in scheduler's cache:
//
//                                                +-------+
//                                                |       |
//                                                |       | Update
//           Assume                Add            +       |
// Initial +--------> Assumed +------------+---> Added <--+
//                       +                 |       +
//                       |                 |       |
//                       |             Add |       | Remove
//                       |                 |       |
//                       |                 +       |
//                       +-------------> Expired   +----> Deleted
//                          expire
//
// Note that an assumed pod would be expired. Because if we haven't received Add event
// notifying us that it's scheduled, there might be some problems and we shouldn't assume
// the pod scheduled anymore.
//
// Note that "Initial", "Expired", and "Deleted" pods do not actually exist in cache.
// Based on existing use cases, we are making following assumptions:
// - No same pod would be assumed twice
// - If a pod wasn't added before, it wouldn't be removed or updated.
// - Both "Expired" and "Deleted" are valid end states. An expired pod could never
//   be added if it missed all events due to network disconnection.
type Cache interface {
	// AssumePodIfBindSucceed assumes a pod to be scheduled if binding the pod succeeded.
	// If so, The pod's information is aggregated into designated node.
	// Note that between bind and assume, there might be race that other events like Add, Remove
	// would jump in. Thus we need to combine the two as a whole.
	// We are passing the bind function and let the cache to take care of concurrency.
	// The implementation might decide the policy to expire pod before being confirmed (receiving Add event).
	// After expiration, its information would be subtracted.
	AssumePodIfBindSucceed(pod *api.Pod, bind func() bool) error

	// AddPod will confirms a pod if it's assumed, or adds back if it's expired.
	// If added back, the pod's information would be added again.
	AddPod(pod *api.Pod) error

	// UpdatePod removes oldPod's information and adds newPod's information.
	UpdatePod(oldPod, newPod *api.Pod) error

	// RemovePod removes a pod. The pod's information would be subtracted from assigned node.
	RemovePod(pod *api.Pod) error

	// GetNodeNameToInfoMap returns a map of node names to node info. The node info contains
	// aggregated information of pods scheduled (including assumed to be) on this node.
	GetNodeNameToInfoMap() map[string]*NodeInfo

	// List lists all pods added (including assumed) in this cache
	List(labels.Selector) []*api.Pod
}

// PodLister is a clone of algorithm.PodLister. There is important cycle issue if we use that one.
// TODO: move algorithm.PodLister into a separate standalone package.
type PodLister interface {
	List(labels.Selector) ([]*api.Pod, error)
}

// CacheToPodLister make a Cache have the List method required by algorithm.PodLister
type CacheToPodLister struct {
	Cache Cache
}

func (c2p *CacheToPodLister) List(selector labels.Selector) ([]*api.Pod, error) {
	return c2p.Cache.List(selector), nil
}

// SystemModeler can help scheduler produce a model of the system that
// anticipates reality. For example, if scheduler has pods A and B both
// using hostPort 80, when it binds A to machine M it should not bind B
// to machine M in the time when it hasn't observed the binding of A
// take effect yet.
//
// Since the model is only an optimization, it's expected to handle
// any errors itself without sending them back to the scheduler.
type SystemModeler interface {
	// AssumePod assumes that the given pod exists in the system.
	// The assumtion should last until the system confirms the
	// assumtion or disconfirms it.
	AssumePod(pod *api.Pod)
	// ForgetPod removes a pod assumtion. (It won't make the model
	// show the absence of the given pod if the pod is in the scheduled
	// pods list!)
	ForgetPod(pod *api.Pod)
	ForgetPodByKey(key string)

	// For serializing calls to Assume/ForgetPod: imagine you want to add
	// a pod if and only if a bind succeeds, but also remove a pod if it is deleted.
	// TODO: if SystemModeler begins modeling things other than pods, this
	// should probably be parameterized or specialized for pods.
	LockedAction(f func())
}
