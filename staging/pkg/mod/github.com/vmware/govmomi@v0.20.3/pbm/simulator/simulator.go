/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"reflect"
	"time"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/pbm"
	"github.com/vmware/govmomi/pbm/methods"
	"github.com/vmware/govmomi/pbm/types"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/vim25/soap"
	vim "github.com/vmware/govmomi/vim25/types"
)

var content = types.PbmServiceInstanceContent{
	AboutInfo: types.PbmAboutInfo{
		Name:         "PBM",
		Version:      "2.0",
		InstanceUuid: "df09f335-be97-4f33-8c27-315faaaad6fc",
	},
	SessionManager:            vim.ManagedObjectReference{Type: "PbmSessionManager", Value: "SessionManager"},
	CapabilityMetadataManager: vim.ManagedObjectReference{Type: "PbmCapabilityMetadataManager", Value: "CapabilityMetadataManager"},
	ProfileManager:            vim.ManagedObjectReference{Type: "PbmProfileProfileManager", Value: "ProfileManager"},
	ComplianceManager:         vim.ManagedObjectReference{Type: "PbmComplianceManager", Value: "complianceManager"},
	PlacementSolver:           vim.ManagedObjectReference{Type: "PbmPlacementSolver", Value: "placementSolver"},
	ReplicationManager:        &vim.ManagedObjectReference{Type: "PbmReplicationManager", Value: "ReplicationManager"},
}

func New() *simulator.Registry {
	r := simulator.NewRegistry()
	r.Namespace = pbm.Namespace
	r.Path = pbm.Path

	r.Put(&ServiceInstance{
		ManagedObjectReference: pbm.ServiceInstance,
		Content:                content,
	})

	r.Put(&ProfileManager{
		ManagedObjectReference: content.ProfileManager,
	})

	r.Put(&PlacementSolver{
		ManagedObjectReference: content.PlacementSolver,
	})

	// TODO: vim25/xml typeToString() does not have an option to include namespace prefix.
	// workaround by adding type without the prefix for now.
	vim.Add("PbmCapabilityProfile", reflect.TypeOf((*types.PbmCapabilityProfile)(nil)).Elem())

	return r
}

type ServiceInstance struct {
	vim.ManagedObjectReference

	Content types.PbmServiceInstanceContent
}

func (s *ServiceInstance) PbmRetrieveServiceContent(_ *types.PbmRetrieveServiceContent) soap.HasFault {
	return &methods.PbmRetrieveServiceContentBody{
		Res: &types.PbmRetrieveServiceContentResponse{
			Returnval: s.Content,
		},
	}
}

type ProfileManager struct {
	vim.ManagedObjectReference
}

func (m *ProfileManager) PbmQueryProfile(req *types.PbmQueryProfile) soap.HasFault {
	body := new(methods.PbmQueryProfileBody)
	body.Res = new(types.PbmQueryProfileResponse)

	for i := range profiles {
		b, ok := profiles[i].(types.BasePbmCapabilityProfile)
		if !ok {
			continue
		}
		p := b.GetPbmCapabilityProfile()

		if p.ResourceType != req.ResourceType {
			continue
		}

		if req.ProfileCategory != "" {
			if p.ProfileCategory != req.ProfileCategory {
				continue
			}
		}

		body.Res.Returnval = append(body.Res.Returnval, types.PbmProfileId{
			UniqueId: p.ProfileId.UniqueId,
		})
	}

	return body
}

func (m *ProfileManager) PbmRetrieveContent(req *types.PbmRetrieveContent) soap.HasFault {
	body := new(methods.PbmRetrieveContentBody)
	if len(req.ProfileIds) == 0 {
		body.Fault_ = simulator.Fault("", &vim.InvalidArgument{InvalidProperty: "profileIds"})
		return body
	}
	body.Res = new(types.PbmRetrieveContentResponse)

	for _, p := range profiles {
		id := p.GetPbmProfile().ProfileId

		for _, rid := range req.ProfileIds {
			if id == rid {
				body.Res.Returnval = append(body.Res.Returnval, p)
			}
		}
	}

	return body
}

func (m *ProfileManager) PbmCreate(ctx *simulator.Context, req *types.PbmCreate) soap.HasFault {
	body := new(methods.PbmCreateBody)
	body.Res = new(types.PbmCreateResponse)

	profile := &types.PbmCapabilityProfile{
		PbmProfile: types.PbmProfile{
			ProfileId: types.PbmProfileId{
				UniqueId: uuid.New().String(),
			},
			Name:            req.CreateSpec.Name,
			Description:     req.CreateSpec.Description,
			CreationTime:    time.Now(),
			CreatedBy:       ctx.Session.UserName,
			LastUpdatedTime: time.Now(),
			LastUpdatedBy:   ctx.Session.UserName,
		},
		ProfileCategory:          req.CreateSpec.Category,
		ResourceType:             req.CreateSpec.ResourceType,
		Constraints:              req.CreateSpec.Constraints,
		GenerationId:             0,
		IsDefault:                false,
		SystemCreatedProfileType: "",
		LineOfService:            "",
	}

	profiles = append(profiles, profile)
	body.Res.Returnval.UniqueId = profile.PbmProfile.ProfileId.UniqueId

	return body
}

func (m *ProfileManager) PbmDelete(req *types.PbmDelete) soap.HasFault {
	body := new(methods.PbmDeleteBody)

	for _, id := range req.ProfileId {
		for i, p := range profiles {
			pid := p.GetPbmProfile().ProfileId

			if id == pid {
				profiles = append(profiles[:i], profiles[i+1:]...)
				break
			}
		}
	}

	body.Res = new(types.PbmDeleteResponse)

	return body
}

type PlacementSolver struct {
	vim.ManagedObjectReference
}

func (m *PlacementSolver) PbmCheckRequirements(req *types.PbmCheckRequirements) soap.HasFault {
	body := new(methods.PbmCheckRequirementsBody)
	body.Res = new(types.PbmCheckRequirementsResponse)

	for _, ds := range simulator.Map.All("Datastore") {
		// TODO: filter
		ref := ds.Reference()
		body.Res.Returnval = append(body.Res.Returnval, types.PbmPlacementCompatibilityResult{
			Hub: types.PbmPlacementHub{
				HubType: ref.Type,
				HubId:   ref.Value,
			},
			MatchingResources: nil,
			HowMany:           0,
			Utilization:       nil,
			Warning:           nil,
			Error:             nil,
		})
	}

	return body
}
