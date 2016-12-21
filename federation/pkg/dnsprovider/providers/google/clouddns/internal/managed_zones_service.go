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
	"strings"

	dns "google.golang.org/api/dns/v1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
	"k8s.io/kubernetes/pkg/util/uuid"
)

// Compile time check for interface adherence
var _ interfaces.ManagedZonesService = &ManagedZonesService{}

type ManagedZonesService struct{ impl *dns.ManagedZonesService }

func (m *ManagedZonesService) Create(project string, managedzone interfaces.ManagedZone) interfaces.ManagedZonesCreateCall {
	return &ManagedZonesCreateCall{m.impl.Create(project, managedzone.(*ManagedZone).impl)}
}

func (m *ManagedZonesService) Delete(project, managedZone string) interfaces.ManagedZonesDeleteCall {
	return &ManagedZonesDeleteCall{m.impl.Delete(project, managedZone)}
}

func (m *ManagedZonesService) Get(project, managedZone string) interfaces.ManagedZonesGetCall {
	return &ManagedZonesGetCall{m.impl.Get(project, managedZone)}
}

func (m *ManagedZonesService) List(project string) interfaces.ManagedZonesListCall {
	return &ManagedZonesListCall{m.impl.List(project)}
}

func (m *ManagedZonesService) NewManagedZone(dnsName string) interfaces.ManagedZone {
	name := "x" + strings.Replace(string(uuid.NewUUID()), "-", "", -1)[0:30] // Unique name, strip out the "-" chars to shorten it, start with a lower case alpha, and truncate to Cloud DNS 32 character limit
	return &ManagedZone{impl: &dns.ManagedZone{Name: name, Description: "Kubernetes Federated Service", DnsName: dnsName}}
}
