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

package dnsprovider

import (
	"reflect"

	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Interface is an abstract, pluggable interface for DNS providers.
type Interface interface {
	// Zones returns the provider's Zones interface, or false if not supported.
	Zones() (Zones, bool)
}

type Zones interface {
	// List returns the managed Zones, or an error if the list operation failed.
	List() ([]Zone, error)
	// Add creates and returns a new managed zone, or an error if the operation failed
	Add(Zone) (Zone, error)
	// Remove deletes a managed zone, or returns an error if the operation failed.
	Remove(Zone) error
	// New allocates a new Zone, which can then be passed to Add()
	// Arguments are as per the Zone interface below.
	New(name string) (Zone, error)
}

type Zone interface {
	// Name returns the name of the zone, e.g. "example.com"
	Name() string
	// ID returns the unique provider identifier for the zone
	ID() string
	// ResourceRecordsets returns the provider's ResourceRecordSets interface, or false if not supported.
	ResourceRecordSets() (ResourceRecordSets, bool)
}

type ResourceRecordSets interface {
	// List returns the ResourceRecordSets of the Zone, or an error if the list operation failed.
	List() ([]ResourceRecordSet, error)
	// Get returns the ResourceRecordSet with the name in the Zone. if the named resource record set does not exist, but no error occurred, the returned set, and error, are both nil.
	Get(name string) (ResourceRecordSet, error)
	// New allocates a new ResourceRecordSet, which can then be passed to ResourceRecordChangeset Add() or Remove()
	// Arguments are as per the ResourceRecordSet interface below.
	New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) ResourceRecordSet
	// StartChangeset begins a new batch operation of changes against the Zone
	StartChangeset() ResourceRecordChangeset
}

// ResourceRecordChangeset accumulates a set of changes, that can then be applied with Apply
type ResourceRecordChangeset interface {
	// Add adds the creation of a ResourceRecordSet in the Zone to the changeset
	Add(ResourceRecordSet) ResourceRecordChangeset
	// Remove adds the removal of a ResourceRecordSet in the Zone to the changeset
	// The supplied ResourceRecordSet must match one of the existing recordsets (obtained via List()) exactly.
	Remove(ResourceRecordSet) ResourceRecordChangeset
	// Apply applies the accumulated operations to the Zone.
	Apply() error
}

type ResourceRecordSet interface {
	// Name returns the name of the ResourceRecordSet, e.g. "www.example.com".
	Name() string
	// Rrdatas returns the Resource Record Datas of the record set.
	Rrdatas() []string
	// Ttl returns the time-to-live of the record set, in seconds.
	Ttl() int64
	// Type returns the type of the record set (A, CNAME, SRV, etc)
	Type() rrstype.RrsType
}

/* ResourceRecordSetsEquivalent compares two ResourceRecordSets for semantic equivalence.
   Go's equality operator doesn't work the way we want it to in this case,
   hence the need for this function.
   More specifically (from the Go spec):
   "Two struct values are equal if their corresponding non-blank fields are equal."
   In our case, there may be some private internal member variables that may not be not equal,
   but we want the two structs to be considered equivalent anyway, if the fields exposed
   via their interfaces are equal.
*/
func ResourceRecordSetsEquivalent(r1, r2 ResourceRecordSet) bool {
	if r1.Name() == r2.Name() && reflect.DeepEqual(r1.Rrdatas(), r2.Rrdatas()) && r1.Ttl() == r2.Ttl() && r1.Type() == r2.Type() {
		return true
	}
	return false
}
