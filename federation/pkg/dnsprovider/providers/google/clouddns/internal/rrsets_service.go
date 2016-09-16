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

package internal

import (
	dns "google.golang.org/api/dns/v1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adeherence
var _ interfaces.ResourceRecordSetsService = &ResourceRecordSetsService{}

type ResourceRecordSetsService struct {
	impl *dns.ResourceRecordSetsService
}

func (service ResourceRecordSetsService) List(project string, managedZone string) interfaces.ResourceRecordSetsListCall {
	return &ResourceRecordSetsListCall{service.impl.List(project, managedZone)}
}

func (service ResourceRecordSetsService) NewResourceRecordSet(name string, rrdatas []string, ttl int64, type_ rrstype.RrsType) interfaces.ResourceRecordSet {
	rrset := dns.ResourceRecordSet{Name: name, Rrdatas: rrdatas, Ttl: ttl, Type: string(type_)}
	return &ResourceRecordSet{&rrset}
}
