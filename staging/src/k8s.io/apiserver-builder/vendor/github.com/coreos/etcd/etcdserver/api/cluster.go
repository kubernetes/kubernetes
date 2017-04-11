// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package api

import (
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/pkg/types"

	"github.com/coreos/go-semver/semver"
)

// Cluster is an interface representing a collection of members in one etcd cluster.
type Cluster interface {
	// ID returns the cluster ID
	ID() types.ID
	// ClientURLs returns an aggregate set of all URLs on which this
	// cluster is listening for client requests
	ClientURLs() []string
	// Members returns a slice of members sorted by their ID
	Members() []*membership.Member
	// Member retrieves a particular member based on ID, or nil if the
	// member does not exist in the cluster
	Member(id types.ID) *membership.Member
	// IsIDRemoved checks whether the given ID has been removed from this
	// cluster at some point in the past
	IsIDRemoved(id types.ID) bool
	// Version is the cluster-wide minimum major.minor version.
	Version() *semver.Version
}
