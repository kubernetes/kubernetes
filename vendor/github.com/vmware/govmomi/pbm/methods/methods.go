/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

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

package methods

import (
	"context"

	"github.com/vmware/govmomi/pbm/types"
	"github.com/vmware/govmomi/vim25/soap"
)

type PbmAssignDefaultRequirementProfileBody struct {
	Req    *types.PbmAssignDefaultRequirementProfile         `xml:"urn:pbm PbmAssignDefaultRequirementProfile,omitempty"`
	Res    *types.PbmAssignDefaultRequirementProfileResponse `xml:"urn:pbm PbmAssignDefaultRequirementProfileResponse,omitempty"`
	Fault_ *soap.Fault                                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmAssignDefaultRequirementProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmAssignDefaultRequirementProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmAssignDefaultRequirementProfile) (*types.PbmAssignDefaultRequirementProfileResponse, error) {
	var reqBody, resBody PbmAssignDefaultRequirementProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCheckCompatibilityBody struct {
	Req    *types.PbmCheckCompatibility         `xml:"urn:pbm PbmCheckCompatibility,omitempty"`
	Res    *types.PbmCheckCompatibilityResponse `xml:"urn:pbm PbmCheckCompatibilityResponse,omitempty"`
	Fault_ *soap.Fault                          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCheckCompatibilityBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCheckCompatibility(ctx context.Context, r soap.RoundTripper, req *types.PbmCheckCompatibility) (*types.PbmCheckCompatibilityResponse, error) {
	var reqBody, resBody PbmCheckCompatibilityBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCheckCompatibilityWithSpecBody struct {
	Req    *types.PbmCheckCompatibilityWithSpec         `xml:"urn:pbm PbmCheckCompatibilityWithSpec,omitempty"`
	Res    *types.PbmCheckCompatibilityWithSpecResponse `xml:"urn:pbm PbmCheckCompatibilityWithSpecResponse,omitempty"`
	Fault_ *soap.Fault                                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCheckCompatibilityWithSpecBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCheckCompatibilityWithSpec(ctx context.Context, r soap.RoundTripper, req *types.PbmCheckCompatibilityWithSpec) (*types.PbmCheckCompatibilityWithSpecResponse, error) {
	var reqBody, resBody PbmCheckCompatibilityWithSpecBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCheckComplianceBody struct {
	Req    *types.PbmCheckCompliance         `xml:"urn:pbm PbmCheckCompliance,omitempty"`
	Res    *types.PbmCheckComplianceResponse `xml:"urn:pbm PbmCheckComplianceResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCheckComplianceBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCheckCompliance(ctx context.Context, r soap.RoundTripper, req *types.PbmCheckCompliance) (*types.PbmCheckComplianceResponse, error) {
	var reqBody, resBody PbmCheckComplianceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCheckRequirementsBody struct {
	Req    *types.PbmCheckRequirements         `xml:"urn:pbm PbmCheckRequirements,omitempty"`
	Res    *types.PbmCheckRequirementsResponse `xml:"urn:pbm PbmCheckRequirementsResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCheckRequirementsBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCheckRequirements(ctx context.Context, r soap.RoundTripper, req *types.PbmCheckRequirements) (*types.PbmCheckRequirementsResponse, error) {
	var reqBody, resBody PbmCheckRequirementsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCheckRollupComplianceBody struct {
	Req    *types.PbmCheckRollupCompliance         `xml:"urn:pbm PbmCheckRollupCompliance,omitempty"`
	Res    *types.PbmCheckRollupComplianceResponse `xml:"urn:pbm PbmCheckRollupComplianceResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCheckRollupComplianceBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCheckRollupCompliance(ctx context.Context, r soap.RoundTripper, req *types.PbmCheckRollupCompliance) (*types.PbmCheckRollupComplianceResponse, error) {
	var reqBody, resBody PbmCheckRollupComplianceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmCreateBody struct {
	Req    *types.PbmCreate         `xml:"urn:pbm PbmCreate,omitempty"`
	Res    *types.PbmCreateResponse `xml:"urn:pbm PbmCreateResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmCreateBody) Fault() *soap.Fault { return b.Fault_ }

func PbmCreate(ctx context.Context, r soap.RoundTripper, req *types.PbmCreate) (*types.PbmCreateResponse, error) {
	var reqBody, resBody PbmCreateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmDeleteBody struct {
	Req    *types.PbmDelete         `xml:"urn:pbm PbmDelete,omitempty"`
	Res    *types.PbmDeleteResponse `xml:"urn:pbm PbmDeleteResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmDeleteBody) Fault() *soap.Fault { return b.Fault_ }

func PbmDelete(ctx context.Context, r soap.RoundTripper, req *types.PbmDelete) (*types.PbmDeleteResponse, error) {
	var reqBody, resBody PbmDeleteBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchCapabilityMetadataBody struct {
	Req    *types.PbmFetchCapabilityMetadata         `xml:"urn:pbm PbmFetchCapabilityMetadata,omitempty"`
	Res    *types.PbmFetchCapabilityMetadataResponse `xml:"urn:pbm PbmFetchCapabilityMetadataResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchCapabilityMetadataBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchCapabilityMetadata(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchCapabilityMetadata) (*types.PbmFetchCapabilityMetadataResponse, error) {
	var reqBody, resBody PbmFetchCapabilityMetadataBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchCapabilitySchemaBody struct {
	Req    *types.PbmFetchCapabilitySchema         `xml:"urn:pbm PbmFetchCapabilitySchema,omitempty"`
	Res    *types.PbmFetchCapabilitySchemaResponse `xml:"urn:pbm PbmFetchCapabilitySchemaResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchCapabilitySchemaBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchCapabilitySchema(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchCapabilitySchema) (*types.PbmFetchCapabilitySchemaResponse, error) {
	var reqBody, resBody PbmFetchCapabilitySchemaBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchComplianceResultBody struct {
	Req    *types.PbmFetchComplianceResult         `xml:"urn:pbm PbmFetchComplianceResult,omitempty"`
	Res    *types.PbmFetchComplianceResultResponse `xml:"urn:pbm PbmFetchComplianceResultResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchComplianceResultBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchComplianceResult(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchComplianceResult) (*types.PbmFetchComplianceResultResponse, error) {
	var reqBody, resBody PbmFetchComplianceResultBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchResourceTypeBody struct {
	Req    *types.PbmFetchResourceType         `xml:"urn:pbm PbmFetchResourceType,omitempty"`
	Res    *types.PbmFetchResourceTypeResponse `xml:"urn:pbm PbmFetchResourceTypeResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchResourceTypeBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchResourceType(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchResourceType) (*types.PbmFetchResourceTypeResponse, error) {
	var reqBody, resBody PbmFetchResourceTypeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchRollupComplianceResultBody struct {
	Req    *types.PbmFetchRollupComplianceResult         `xml:"urn:pbm PbmFetchRollupComplianceResult,omitempty"`
	Res    *types.PbmFetchRollupComplianceResultResponse `xml:"urn:pbm PbmFetchRollupComplianceResultResponse,omitempty"`
	Fault_ *soap.Fault                                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchRollupComplianceResultBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchRollupComplianceResult(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchRollupComplianceResult) (*types.PbmFetchRollupComplianceResultResponse, error) {
	var reqBody, resBody PbmFetchRollupComplianceResultBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFetchVendorInfoBody struct {
	Req    *types.PbmFetchVendorInfo         `xml:"urn:pbm PbmFetchVendorInfo,omitempty"`
	Res    *types.PbmFetchVendorInfoResponse `xml:"urn:pbm PbmFetchVendorInfoResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFetchVendorInfoBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFetchVendorInfo(ctx context.Context, r soap.RoundTripper, req *types.PbmFetchVendorInfo) (*types.PbmFetchVendorInfoResponse, error) {
	var reqBody, resBody PbmFetchVendorInfoBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmFindApplicableDefaultProfileBody struct {
	Req    *types.PbmFindApplicableDefaultProfile         `xml:"urn:pbm PbmFindApplicableDefaultProfile,omitempty"`
	Res    *types.PbmFindApplicableDefaultProfileResponse `xml:"urn:pbm PbmFindApplicableDefaultProfileResponse,omitempty"`
	Fault_ *soap.Fault                                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmFindApplicableDefaultProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmFindApplicableDefaultProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmFindApplicableDefaultProfile) (*types.PbmFindApplicableDefaultProfileResponse, error) {
	var reqBody, resBody PbmFindApplicableDefaultProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryAssociatedEntitiesBody struct {
	Req    *types.PbmQueryAssociatedEntities         `xml:"urn:pbm PbmQueryAssociatedEntities,omitempty"`
	Res    *types.PbmQueryAssociatedEntitiesResponse `xml:"urn:pbm PbmQueryAssociatedEntitiesResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryAssociatedEntitiesBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryAssociatedEntities(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryAssociatedEntities) (*types.PbmQueryAssociatedEntitiesResponse, error) {
	var reqBody, resBody PbmQueryAssociatedEntitiesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryAssociatedEntityBody struct {
	Req    *types.PbmQueryAssociatedEntity         `xml:"urn:pbm PbmQueryAssociatedEntity,omitempty"`
	Res    *types.PbmQueryAssociatedEntityResponse `xml:"urn:pbm PbmQueryAssociatedEntityResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryAssociatedEntityBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryAssociatedEntity(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryAssociatedEntity) (*types.PbmQueryAssociatedEntityResponse, error) {
	var reqBody, resBody PbmQueryAssociatedEntityBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryAssociatedProfileBody struct {
	Req    *types.PbmQueryAssociatedProfile         `xml:"urn:pbm PbmQueryAssociatedProfile,omitempty"`
	Res    *types.PbmQueryAssociatedProfileResponse `xml:"urn:pbm PbmQueryAssociatedProfileResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryAssociatedProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryAssociatedProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryAssociatedProfile) (*types.PbmQueryAssociatedProfileResponse, error) {
	var reqBody, resBody PbmQueryAssociatedProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryAssociatedProfilesBody struct {
	Req    *types.PbmQueryAssociatedProfiles         `xml:"urn:pbm PbmQueryAssociatedProfiles,omitempty"`
	Res    *types.PbmQueryAssociatedProfilesResponse `xml:"urn:pbm PbmQueryAssociatedProfilesResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryAssociatedProfilesBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryAssociatedProfiles(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryAssociatedProfiles) (*types.PbmQueryAssociatedProfilesResponse, error) {
	var reqBody, resBody PbmQueryAssociatedProfilesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryByRollupComplianceStatusBody struct {
	Req    *types.PbmQueryByRollupComplianceStatus         `xml:"urn:pbm PbmQueryByRollupComplianceStatus,omitempty"`
	Res    *types.PbmQueryByRollupComplianceStatusResponse `xml:"urn:pbm PbmQueryByRollupComplianceStatusResponse,omitempty"`
	Fault_ *soap.Fault                                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryByRollupComplianceStatusBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryByRollupComplianceStatus(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryByRollupComplianceStatus) (*types.PbmQueryByRollupComplianceStatusResponse, error) {
	var reqBody, resBody PbmQueryByRollupComplianceStatusBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryDefaultRequirementProfileBody struct {
	Req    *types.PbmQueryDefaultRequirementProfile         `xml:"urn:pbm PbmQueryDefaultRequirementProfile,omitempty"`
	Res    *types.PbmQueryDefaultRequirementProfileResponse `xml:"urn:pbm PbmQueryDefaultRequirementProfileResponse,omitempty"`
	Fault_ *soap.Fault                                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryDefaultRequirementProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryDefaultRequirementProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryDefaultRequirementProfile) (*types.PbmQueryDefaultRequirementProfileResponse, error) {
	var reqBody, resBody PbmQueryDefaultRequirementProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryDefaultRequirementProfilesBody struct {
	Req    *types.PbmQueryDefaultRequirementProfiles         `xml:"urn:pbm PbmQueryDefaultRequirementProfiles,omitempty"`
	Res    *types.PbmQueryDefaultRequirementProfilesResponse `xml:"urn:pbm PbmQueryDefaultRequirementProfilesResponse,omitempty"`
	Fault_ *soap.Fault                                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryDefaultRequirementProfilesBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryDefaultRequirementProfiles(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryDefaultRequirementProfiles) (*types.PbmQueryDefaultRequirementProfilesResponse, error) {
	var reqBody, resBody PbmQueryDefaultRequirementProfilesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryMatchingHubBody struct {
	Req    *types.PbmQueryMatchingHub         `xml:"urn:pbm PbmQueryMatchingHub,omitempty"`
	Res    *types.PbmQueryMatchingHubResponse `xml:"urn:pbm PbmQueryMatchingHubResponse,omitempty"`
	Fault_ *soap.Fault                        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryMatchingHubBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryMatchingHub(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryMatchingHub) (*types.PbmQueryMatchingHubResponse, error) {
	var reqBody, resBody PbmQueryMatchingHubBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryMatchingHubWithSpecBody struct {
	Req    *types.PbmQueryMatchingHubWithSpec         `xml:"urn:pbm PbmQueryMatchingHubWithSpec,omitempty"`
	Res    *types.PbmQueryMatchingHubWithSpecResponse `xml:"urn:pbm PbmQueryMatchingHubWithSpecResponse,omitempty"`
	Fault_ *soap.Fault                                `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryMatchingHubWithSpecBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryMatchingHubWithSpec(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryMatchingHubWithSpec) (*types.PbmQueryMatchingHubWithSpecResponse, error) {
	var reqBody, resBody PbmQueryMatchingHubWithSpecBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryProfileBody struct {
	Req    *types.PbmQueryProfile         `xml:"urn:pbm PbmQueryProfile,omitempty"`
	Res    *types.PbmQueryProfileResponse `xml:"urn:pbm PbmQueryProfileResponse,omitempty"`
	Fault_ *soap.Fault                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryProfile) (*types.PbmQueryProfileResponse, error) {
	var reqBody, resBody PbmQueryProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQueryReplicationGroupsBody struct {
	Req    *types.PbmQueryReplicationGroups         `xml:"urn:pbm PbmQueryReplicationGroups,omitempty"`
	Res    *types.PbmQueryReplicationGroupsResponse `xml:"urn:pbm PbmQueryReplicationGroupsResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQueryReplicationGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQueryReplicationGroups(ctx context.Context, r soap.RoundTripper, req *types.PbmQueryReplicationGroups) (*types.PbmQueryReplicationGroupsResponse, error) {
	var reqBody, resBody PbmQueryReplicationGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmQuerySpaceStatsForStorageContainerBody struct {
	Req    *types.PbmQuerySpaceStatsForStorageContainer         `xml:"urn:pbm PbmQuerySpaceStatsForStorageContainer,omitempty"`
	Res    *types.PbmQuerySpaceStatsForStorageContainerResponse `xml:"urn:pbm PbmQuerySpaceStatsForStorageContainerResponse,omitempty"`
	Fault_ *soap.Fault                                          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmQuerySpaceStatsForStorageContainerBody) Fault() *soap.Fault { return b.Fault_ }

func PbmQuerySpaceStatsForStorageContainer(ctx context.Context, r soap.RoundTripper, req *types.PbmQuerySpaceStatsForStorageContainer) (*types.PbmQuerySpaceStatsForStorageContainerResponse, error) {
	var reqBody, resBody PbmQuerySpaceStatsForStorageContainerBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmResetDefaultRequirementProfileBody struct {
	Req    *types.PbmResetDefaultRequirementProfile         `xml:"urn:pbm PbmResetDefaultRequirementProfile,omitempty"`
	Res    *types.PbmResetDefaultRequirementProfileResponse `xml:"urn:pbm PbmResetDefaultRequirementProfileResponse,omitempty"`
	Fault_ *soap.Fault                                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmResetDefaultRequirementProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmResetDefaultRequirementProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmResetDefaultRequirementProfile) (*types.PbmResetDefaultRequirementProfileResponse, error) {
	var reqBody, resBody PbmResetDefaultRequirementProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmResetVSanDefaultProfileBody struct {
	Req    *types.PbmResetVSanDefaultProfile         `xml:"urn:pbm PbmResetVSanDefaultProfile,omitempty"`
	Res    *types.PbmResetVSanDefaultProfileResponse `xml:"urn:pbm PbmResetVSanDefaultProfileResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmResetVSanDefaultProfileBody) Fault() *soap.Fault { return b.Fault_ }

func PbmResetVSanDefaultProfile(ctx context.Context, r soap.RoundTripper, req *types.PbmResetVSanDefaultProfile) (*types.PbmResetVSanDefaultProfileResponse, error) {
	var reqBody, resBody PbmResetVSanDefaultProfileBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmRetrieveContentBody struct {
	Req    *types.PbmRetrieveContent         `xml:"urn:pbm PbmRetrieveContent,omitempty"`
	Res    *types.PbmRetrieveContentResponse `xml:"urn:pbm PbmRetrieveContentResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmRetrieveContentBody) Fault() *soap.Fault { return b.Fault_ }

func PbmRetrieveContent(ctx context.Context, r soap.RoundTripper, req *types.PbmRetrieveContent) (*types.PbmRetrieveContentResponse, error) {
	var reqBody, resBody PbmRetrieveContentBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmRetrieveServiceContentBody struct {
	Req    *types.PbmRetrieveServiceContent         `xml:"urn:pbm PbmRetrieveServiceContent,omitempty"`
	Res    *types.PbmRetrieveServiceContentResponse `xml:"urn:pbm PbmRetrieveServiceContentResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmRetrieveServiceContentBody) Fault() *soap.Fault { return b.Fault_ }

func PbmRetrieveServiceContent(ctx context.Context, r soap.RoundTripper, req *types.PbmRetrieveServiceContent) (*types.PbmRetrieveServiceContentResponse, error) {
	var reqBody, resBody PbmRetrieveServiceContentBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type PbmUpdateBody struct {
	Req    *types.PbmUpdate         `xml:"urn:pbm PbmUpdate,omitempty"`
	Res    *types.PbmUpdateResponse `xml:"urn:pbm PbmUpdateResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *PbmUpdateBody) Fault() *soap.Fault { return b.Fault_ }

func PbmUpdate(ctx context.Context, r soap.RoundTripper, req *types.PbmUpdate) (*types.PbmUpdateResponse, error) {
	var reqBody, resBody PbmUpdateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}
