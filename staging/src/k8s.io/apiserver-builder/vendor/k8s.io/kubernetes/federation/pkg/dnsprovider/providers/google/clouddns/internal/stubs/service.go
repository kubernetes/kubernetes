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
var _ interfaces.Service = &Service{}

type Service struct {
	Changes_      *ChangesService
	ManagedZones_ *ManagedZonesService
	Projects_     *ProjectsService
	Rrsets_       *ResourceRecordSetsService
}

func NewService() *Service {
	s := &Service{}
	s.Changes_ = &ChangesService{s}
	s.ManagedZones_ = &ManagedZonesService{}
	s.Projects_ = &ProjectsService{}
	s.Rrsets_ = &ResourceRecordSetsService{s, nil}
	return s
}

func (s *Service) Changes() interfaces.ChangesService {
	return s.Changes_
}

func (s *Service) ManagedZones() interfaces.ManagedZonesService {
	return s.ManagedZones_
}

func (s *Service) Projects() interfaces.ProjectsService {
	return s.Projects_
}

func (s *Service) ResourceRecordSets() interfaces.ResourceRecordSetsService {
	return s.Rrsets_
}
