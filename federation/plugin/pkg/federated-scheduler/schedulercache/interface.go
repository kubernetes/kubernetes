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
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// Cache collects replicaSets' information and provides cluster-level aggregated information.
// It's intended for generic federated-scheduler to do efficient lookup.
// Cache's operations are replicaSet centric. It does incremental updates based on replicaSet events.
// ReplicaSet events are sent via network. We don't have guaranteed delivery of all events:
// We use Reflector to list and watch from remote.
// Reflector might be slow and do a relist, which would lead to missing events.
//
// State Machine of a replicaSet's events in federated-scheduler's cache:
//
//
//   +-------------------------------------------+  +----+
//   |                            Add            |  |    |
//   |                                           |  |    | Update
//   +      Assume                Add            v  v    |
//Initial +--------> Assumed +------------+---> Added <--+
//                      +                 |       +
//                      |                 |       |
//                      |             Add |       | Remove
//                      |                 |       |
//                      |                 +       |
//                      +-------------> Expired   +----> Deleted
//                          Expire
//
//
// Note that an assumed replicaSet can expire, because if we haven't received Add event notifying us
// for a while, there might be some problems and we shouldn't keep the replicaSet in cache anymore.
//
// Note that "Initial", "Expired", and "Deleted" replicaSets do not actually exist in cache.
// Based on existing use cases, we are making the following assumptions:
// - No replicaSet would be assumed twice
// - If a replicaSet wasn't added, it wouldn't be removed or updated.
// - Both "Expired" and "Deleted" are valid end states. In case of some problems, e.g. network issue,
//   a replicaSet might have changed its state (e.g. added and deleted) without delivering notification to the cache.
type Cache interface {


	// AssumeSubRSIfBindSucceed assumes a replicaSet to be scheduled if binding the replicaSet succeeded.
	// If binding return true, the replicaSet's information is aggregated into designated cluster.
	// Note that both binding and assuming are done as one atomic operation from cache's view.
	// No other events like Add would happen in between binding and assuming.
	// We are passing the binding function and let implementation take care of concurrency control details.
	// The implementation also decides the policy to expire replicaSet before being confirmed (receiving Add event).
	// After expiration, its information would be subtracted.
	AssumeReplicaSet(replicaSet *v1beta1.ReplicaSet) error

	// AddSubRS either confirms a replicaSet if it's assumed, or adds it back if it's expired.
	// If added back, the replicaSet's information would be added again.
	AddReplicaSet(replicaSet *v1beta1.ReplicaSet) error

	// UpdateSubRS removes oldReplicaSet's information and adds newReplicaSet's information.
	UpdateReplicaSet(oldReplicaSet, newReplicaSet *v1beta1.ReplicaSet) error

	// RemoveSubRS removes a replicaSet. The replicaSet's information would be subtracted from assigned cluster.
	RemoveReplicaSet(replicaSet *v1beta1.ReplicaSet) error

	// GetClusterNameToInfoMap returns a map of cluster names to cluster info. The cluster info contains
	// aggregated information of replicaSets scheduled (including assumed to be) on this cluster.
	GetClusterNameToInfoMap() (map[string]*ClusterInfo, error)

	// List lists all cached replicaSets (including assumed ones).
	List() ([]*v1beta1.ReplicaSet, error)
}
