/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

import "k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"

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
	// ResourceRecordsets returns the provider's ResourceRecordSets interface, or false if not supported.
	ResourceRecordSets() (ResourceRecordSets, bool)
}

type ResourceRecordSets interface {
	// List returns the ResourceRecordSets of the Zone, or an error if the list operation failed.
	List() ([]ResourceRecordSet, error)
	// Add adds and returns a ResourceRecordSet of the Zone, or an error if the add operation failed.
	Add(ResourceRecordSet) (ResourceRecordSet, error)
	// Remove removes a ResourceRecordSet from the Zone, or an error if the remove operation failed.
	// The supplied ResourceRecordSet must match one of the existing zones (obtained via List()) exactly.
	Remove(ResourceRecordSet) error
	// New allocates a new ResourceRecordSet, which can then be passed to Add() or Remove()
	// Arguments are as per the ResourceRecordSet interface below.
	New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) ResourceRecordSet
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
