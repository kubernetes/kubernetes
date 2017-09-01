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

package clouddns

import (
	"strconv"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adherence
var _ dnsprovider.Zone = &Zone{}

type Zone struct {
	impl  interfaces.ManagedZone
	zones *Zones
}

func (zone *Zone) Name() string {
	return zone.impl.DnsName()
}

func (zone *Zone) ID() string {
	return strconv.FormatUint(zone.impl.Id(), 10)
}

func (zone *Zone) ResourceRecordSets() (dnsprovider.ResourceRecordSets, bool) {
	return &ResourceRecordSets{zone, zone.zones.interface_.service.ResourceRecordSets()}, true
}

func (zone Zone) project() string {
	return zone.zones.project()
}
