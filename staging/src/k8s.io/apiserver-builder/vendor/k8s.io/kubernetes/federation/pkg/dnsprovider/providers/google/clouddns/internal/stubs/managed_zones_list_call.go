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

// Compile time check for interface adherence
var _ interfaces.ManagedZonesListCall = &ManagedZonesListCall{}

type ManagedZonesListCall struct {
	Service  *ManagedZonesService
	Project  string
	Response *interfaces.ManagedZonesListResponse // Use this to override response if required
	Error    *error                               // Use this to override response if required
	DnsName_ string
}

func (call *ManagedZonesListCall) Do(opts ...googleapi.CallOption) (interfaces.ManagedZonesListResponse, error) {
	if call.Response != nil {
		return *call.Response, *call.Error
	} else {
		proj, projectFound := call.Service.Impl[call.Project]
		if !projectFound {
			return nil, fmt.Errorf("Project %s not found.", call.Project)
		}
		if call.DnsName_ != "" {
			return &ManagedZonesListResponse{[]interfaces.ManagedZone{proj[call.DnsName_]}}, nil
		}
		list := []interfaces.ManagedZone{}
		for _, zone := range proj {
			list = append(list, zone)
		}
		return &ManagedZonesListResponse{list}, nil
	}
}

func (call *ManagedZonesListCall) DnsName(dnsName string) interfaces.ManagedZonesListCall {
	call.DnsName_ = dnsName
	return call
}
