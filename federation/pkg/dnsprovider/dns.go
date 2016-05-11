/*
Copyright 2014 The Kubernetes Authors All rights reserved.
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
	Zones() (Zones, bool)
}

type Zones interface {
	List() ([]Zone, error)
}

type Zone interface {
	Name() string
	ResourceRecordSets() (ResourceRecordSets, bool)
}

type ResourceRecordSets interface {
	List() ([]ResourceRecordSet, error)
	Add(ResourceRecordSet) (ResourceRecordSet, error)
	Remove(ResourceRecordSet) error
	New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) ResourceRecordSet
}

type ResourceRecordSet interface {
	Name() string // e.g. "www.example.com"
	Rrdatas() []string
	Ttl() int64
	Type() rrstype.RrsType
}
