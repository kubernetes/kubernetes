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

import "k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"

// Compile time check for interface adeherence
var _ interfaces.ManagedZonesService = &ManagedZonesService{}

type ManagedZonesService struct {
	Impl map[string]map[string]interfaces.ManagedZone
}

func (m *ManagedZonesService) Create(project string, managedzone interfaces.ManagedZone) interfaces.ManagedZonesCreateCall {
	return &ManagedZonesCreateCall{nil, m, project, managedzone.(*ManagedZone)}
}

func (m *ManagedZonesService) Delete(project string, managedZone string) interfaces.ManagedZonesDeleteCall {
	return &ManagedZonesDeleteCall{m, project, managedZone, nil}
}

func (m *ManagedZonesService) Get(project string, managedZone string) interfaces.ManagedZonesGetCall {
	return &ManagedZonesGetCall{m, project, managedZone, nil, nil, ""}
}

func (m *ManagedZonesService) List(project string) interfaces.ManagedZonesListCall {
	return &ManagedZonesListCall{m, project, nil, nil, ""}
}

func (m *ManagedZonesService) NewManagedZone(dnsName string) interfaces.ManagedZone {
	return &ManagedZone{Service: m, Name_: dnsName}
}
