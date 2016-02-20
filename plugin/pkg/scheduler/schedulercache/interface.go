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
// Cache's operations are pod centric. It incrementally updates itself based on pod events.
// Pod events are sent via network. We don't have guaranteed delivery of all events:
// We use Reflector to list and watch from remote.
// Reflector might be slow and do a relist, which would lead to missing events.
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
//                          Expire
//
// Note that an assumed pod can expire, because if we haven't received Add event notifying us
// for a while, there might be some problems and we shouldn't keep the pod in cache anymore.
//
// Note that "Initial", "Expired", and "Deleted" pods do not actually exist in cache.
// Based on existing use cases, we are making the following assumptions:
// - No pod would be assumed twice
// - If a pod wasn't added, it wouldn't be removed or updated.
// - Both "Expired" and "Deleted" are valid end states. In case of some problems, e.g. network issue,
//   a pod might have changed its state (e.g. added and deleted) without delivering notification to the cache.
type Cache interface {
	// AssumePodIfBindSucceed assumes a pod to be scheduled if binding the pod succeeded.
	// If binding return true, the pod's information is aggregated into designated node.
	// Note that both binding and assuming are done as one atomic operation from cache's view.
	// No other events like Add would happen in between binding and assuming.
	// We are passing the binding function and let implementation take care of concurrency control details.
	// The implementation also decides the policy to expire pod before being confirmed (receiving Add event).
	// After expiration, its information would be subtracted.
	AssumePodIfBindSucceed(pod *api.Pod, bind func() bool) error

	// AddPod either confirms a pod if it's assumed, or adds it back if it's expired.
	// If added back, the pod's information would be added again.
	AddPod(pod *api.Pod) error

	// UpdatePod removes oldPod's information and adds newPod's information.
	UpdatePod(oldPod, newPod *api.Pod) error

	// RemovePod removes a pod. The pod's information would be subtracted from assigned node.
	RemovePod(pod *api.Pod) error

	// GetNodeNameToInfoMap returns a map of node names to node info. The node info contains
	// aggregated information of pods scheduled (including assumed to be) on this node.
	GetNodeNameToInfoMap() (map[string]*NodeInfo, error)

	// List lists all cached pods (including assumed ones).
	List(labels.Selector) ([]*api.Pod, error)
}
