/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package pbm

import (
	"context"
	"fmt"

	"github.com/vmware/govmomi/pbm/methods"
	"github.com/vmware/govmomi/pbm/types"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	vim "github.com/vmware/govmomi/vim25/types"
)

const (
	Namespace = "pbm"
	Path      = "/pbm" + vim25.Path
)

var (
	ServiceInstance = vim.ManagedObjectReference{
		Type:  "PbmServiceInstance",
		Value: "ServiceInstance",
	}
)

type Client struct {
	*soap.Client

	ServiceContent types.PbmServiceInstanceContent
}

func NewClient(ctx context.Context, c *vim25.Client) (*Client, error) {
	sc := c.Client.NewServiceClient(Path, Namespace)

	req := types.PbmRetrieveServiceContent{
		This: ServiceInstance,
	}

	res, err := methods.PbmRetrieveServiceContent(ctx, sc, &req)
	if err != nil {
		return nil, err
	}

	return &Client{sc, res.Returnval}, nil
}

func (c *Client) QueryProfile(ctx context.Context, rtype types.PbmProfileResourceType, category string) ([]types.PbmProfileId, error) {
	req := types.PbmQueryProfile{
		This:            c.ServiceContent.ProfileManager,
		ResourceType:    rtype,
		ProfileCategory: category,
	}

	res, err := methods.PbmQueryProfile(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) RetrieveContent(ctx context.Context, ids []types.PbmProfileId) ([]types.BasePbmProfile, error) {
	req := types.PbmRetrieveContent{
		This:       c.ServiceContent.ProfileManager,
		ProfileIds: ids,
	}

	res, err := methods.PbmRetrieveContent(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

type PlacementCompatibilityResult []types.PbmPlacementCompatibilityResult

func (c *Client) CheckRequirements(ctx context.Context, hubs []types.PbmPlacementHub, ref *types.PbmServerObjectRef, preq []types.BasePbmPlacementRequirement) (PlacementCompatibilityResult, error) {
	req := types.PbmCheckRequirements{
		This:                        c.ServiceContent.PlacementSolver,
		HubsToSearch:                hubs,
		PlacementSubjectRef:         ref,
		PlacementSubjectRequirement: preq,
	}

	res, err := methods.PbmCheckRequirements(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (l PlacementCompatibilityResult) CompatibleDatastores() []types.PbmPlacementHub {
	var compatibleDatastores []types.PbmPlacementHub

	for _, res := range l {
		if len(res.Error) == 0 {
			compatibleDatastores = append(compatibleDatastores, res.Hub)
		}
	}
	return compatibleDatastores
}

func (l PlacementCompatibilityResult) NonCompatibleDatastores() []types.PbmPlacementHub {
	var nonCompatibleDatastores []types.PbmPlacementHub

	for _, res := range l {
		if len(res.Error) > 0 {
			nonCompatibleDatastores = append(nonCompatibleDatastores, res.Hub)
		}
	}
	return nonCompatibleDatastores
}

func (c *Client) CreateProfile(ctx context.Context, capabilityProfileCreateSpec types.PbmCapabilityProfileCreateSpec) (*types.PbmProfileId, error) {
	req := types.PbmCreate{
		This:       c.ServiceContent.ProfileManager,
		CreateSpec: capabilityProfileCreateSpec,
	}

	res, err := methods.PbmCreate(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (c *Client) UpdateProfile(ctx context.Context, id types.PbmProfileId, updateSpec types.PbmCapabilityProfileUpdateSpec) error {
	req := types.PbmUpdate{
		This:       c.ServiceContent.ProfileManager,
		ProfileId:  id,
		UpdateSpec: updateSpec,
	}

	_, err := methods.PbmUpdate(ctx, c, &req)
	if err != nil {
		return err
	}

	return nil
}

func (c *Client) DeleteProfile(ctx context.Context, ids []types.PbmProfileId) ([]types.PbmProfileOperationOutcome, error) {
	req := types.PbmDelete{
		This:      c.ServiceContent.ProfileManager,
		ProfileId: ids,
	}

	res, err := methods.PbmDelete(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) QueryAssociatedEntity(ctx context.Context, id types.PbmProfileId, entityType string) ([]types.PbmServerObjectRef, error) {
	req := types.PbmQueryAssociatedEntity{
		This:       c.ServiceContent.ProfileManager,
		Profile:    id,
		EntityType: entityType,
	}

	res, err := methods.PbmQueryAssociatedEntity(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) QueryAssociatedEntities(ctx context.Context, ids []types.PbmProfileId) ([]types.PbmQueryProfileResult, error) {
	req := types.PbmQueryAssociatedEntities{
		This:     c.ServiceContent.ProfileManager,
		Profiles: ids,
	}

	res, err := methods.PbmQueryAssociatedEntities(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) ProfileIDByName(ctx context.Context, profileName string) (string, error) {
	resourceType := types.PbmProfileResourceType{
		ResourceType: string(types.PbmProfileResourceTypeEnumSTORAGE),
	}
	category := types.PbmProfileCategoryEnumREQUIREMENT
	ids, err := c.QueryProfile(ctx, resourceType, string(category))
	if err != nil {
		return "", err
	}

	profiles, err := c.RetrieveContent(ctx, ids)
	if err != nil {
		return "", err
	}

	for i := range profiles {
		profile := profiles[i].GetPbmProfile()
		if profile.Name == profileName {
			return profile.ProfileId.UniqueId, nil
		}
	}
	return "", fmt.Errorf("no pbm profile found with name: %q", profileName)
}
