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

package azuredns

import (
	"github.com/Azure/azure-sdk-for-go/arm/dns"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adherence
var _ dnsprovider.Zone = &Zone{}

// Zone struct that also implements the federation Zone interface to access
// Azure DNS.
// It also allows navigation of the DNS hierarchy via ResourceRecordSet -> ResourceRecordSets -> Zone -> Zones
type Zone struct {
	impl  *dns.Zone
	zones *Zones
}

// Name is the implementation of Interface's Name method
func (zone *Zone) Name() string {
	return *zone.impl.Name
}

// ID is the implementation of the interfaces's ID method
func (zone *Zone) ID() string {
	// AWS unit tests want this to be the same. Keeping this the same
	return *zone.impl.Name
}

// ResourceRecordSets is the implementation of the interfaces ResourceRecordSets method
func (zone *Zone) ResourceRecordSets() (dnsprovider.ResourceRecordSets, bool) {
	return &ResourceRecordSets{zone}, true
}
