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

	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adherence
var _ interfaces.ResourceRecordSetsService = &ResourceRecordSetsService{}

type ResourceRecordSetsService struct {
	Service  *Service
	ListCall interfaces.ResourceRecordSetsListCall // Use to override response if required for testing
}

func (s ResourceRecordSetsService) List(project string, managedZone string) interfaces.ResourceRecordSetsListCall {
	if s.ListCall != nil {
		return s.ListCall
	}
	p := s.Service.ManagedZones_.Impl[project]
	if p == nil {
		return &ResourceRecordSetsListCall{Err_: fmt.Errorf("Project not found: %s", project)}
	}
	z := s.Service.ManagedZones_.Impl[project][managedZone]
	if z == nil {
		return &ResourceRecordSetsListCall{
			Err_: fmt.Errorf("Zone %s not found in project %s", managedZone, project),
		}
	}
	zone := s.Service.ManagedZones_.Impl[project][managedZone].(*ManagedZone)
	response := &ResourceRecordSetsListResponse{}
	for _, set := range zone.Rrsets {
		response.impl = append(response.impl, set)
	}
	return &ResourceRecordSetsListCall{Response_: response}
}

func (service ResourceRecordSetsService) NewResourceRecordSet(name string, rrdatas []string, ttl int64, type_ rrstype.RrsType) interfaces.ResourceRecordSet {
	rrset := ResourceRecordSet{Name_: name, Rrdatas_: rrdatas, Ttl_: ttl, Type_: string(type_)}
	return rrset
}
