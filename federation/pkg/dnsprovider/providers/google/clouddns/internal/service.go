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
)

// Compile time check for interface adeherence
var _ interfaces.Service = &Service{}

type Service struct {
	impl *dns.Service
}

func NewService(service *dns.Service) *Service {
	return &Service{service}
}

func (s *Service) Changes() interfaces.ChangesService {
	return &ChangesService{s.impl.Changes}
}

func (s *Service) ManagedZones() interfaces.ManagedZonesService {
	return &ManagedZonesService{s.impl.ManagedZones}
}

func (s *Service) Projects() interfaces.ProjectsService {
	return &ProjectsService{s.impl.Projects}
}

func (s *Service) ResourceRecordSets() interfaces.ResourceRecordSetsService {
	return &ResourceRecordSetsService{s.impl.ResourceRecordSets}
}
