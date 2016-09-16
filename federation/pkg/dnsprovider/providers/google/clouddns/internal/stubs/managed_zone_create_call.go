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

package stubs

import (
	"fmt"

	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adeherence
var _ interfaces.ManagedZonesCreateCall = ManagedZonesCreateCall{}

type ManagedZonesCreateCall struct {
	Error       *error // Use to override response for testing
	Service     *ManagedZonesService
	Project     string
	ManagedZone interfaces.ManagedZone
}

func (call ManagedZonesCreateCall) Do(opts ...googleapi.CallOption) (interfaces.ManagedZone, error) {
	if call.Error != nil {
		return nil, *call.Error
	}
	if call.Service.Impl[call.Project][call.ManagedZone.DnsName()] != nil {
		return nil, fmt.Errorf("Error - attempt to create duplicate zone %s in project %s.",
			call.ManagedZone.DnsName(), call.Project)
	}
	if call.Service.Impl == nil {
		call.Service.Impl = map[string]map[string]interfaces.ManagedZone{}
	}
	if call.Service.Impl[call.Project] == nil {
		call.Service.Impl[call.Project] = map[string]interfaces.ManagedZone{}
	}
	call.Service.Impl[call.Project][call.ManagedZone.DnsName()] = call.ManagedZone
	return call.ManagedZone, nil
}
