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

package dnspod

import (
	"strconv"

	dns "github.com/decker502/dnspod-go"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adeherence
var _ dnsprovider.Zone = &Zone{}

type Zone struct {
	domain dns.Domain
	zones  *Zones
}

func (zone *Zone) Name() string {
	return zone.domain.Name
}

func (zone *Zone) ID() string {
	return strconv.Itoa(zone.domain.ID)
}

func (zone *Zone) ResourceRecordSets() (dnsprovider.ResourceRecordSets, bool) {
	return &ResourceRecordSets{zone, zone.zones.client}, true
}
