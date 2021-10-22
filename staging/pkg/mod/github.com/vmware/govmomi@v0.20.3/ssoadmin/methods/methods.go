/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/ssoadmin/types"
	"github.com/vmware/govmomi/vim25/soap"
)

type AddCertificateBody struct {
	Req    *types.AddCertificate         `xml:"urn:sso AddCertificate,omitempty"`
	Res    *types.AddCertificateResponse `xml:"urn:sso AddCertificateResponse,omitempty"`
	Fault_ *soap.Fault                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddCertificateBody) Fault() *soap.Fault { return b.Fault_ }

func AddCertificate(ctx context.Context, r soap.RoundTripper, req *types.AddCertificate) (*types.AddCertificateResponse, error) {
	var reqBody, resBody AddCertificateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type AddExternalDomainBody struct {
	Req    *types.AddExternalDomain         `xml:"urn:sso AddExternalDomain,omitempty"`
	Res    *types.AddExternalDomainResponse `xml:"urn:sso AddExternalDomainResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddExternalDomainBody) Fault() *soap.Fault { return b.Fault_ }

func AddExternalDomain(ctx context.Context, r soap.RoundTripper, req *types.AddExternalDomain) (*types.AddExternalDomainResponse, error) {
	var reqBody, resBody AddExternalDomainBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type AddGroupToLocalGroupBody struct {
	Req    *types.AddGroupToLocalGroup         `xml:"urn:sso AddGroupToLocalGroup,omitempty"`
	Res    *types.AddGroupToLocalGroupResponse `xml:"urn:sso AddGroupToLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddGroupToLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func AddGroupToLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.AddGroupToLocalGroup) (*types.AddGroupToLocalGroupResponse, error) {
	var reqBody, resBody AddGroupToLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type AddGroupsToLocalGroupBody struct {
	Req    *types.AddGroupsToLocalGroup         `xml:"urn:sso AddGroupsToLocalGroup,omitempty"`
	Res    *types.AddGroupsToLocalGroupResponse `xml:"urn:sso AddGroupsToLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddGroupsToLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func AddGroupsToLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.AddGroupsToLocalGroup) (*types.AddGroupsToLocalGroupResponse, error) {
	var reqBody, resBody AddGroupsToLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type AddUserToLocalGroupBody struct {
	Req    *types.AddUserToLocalGroup         `xml:"urn:sso AddUserToLocalGroup,omitempty"`
	Res    *types.AddUserToLocalGroupResponse `xml:"urn:sso AddUserToLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddUserToLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func AddUserToLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.AddUserToLocalGroup) (*types.AddUserToLocalGroupResponse, error) {
	var reqBody, resBody AddUserToLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type AddUsersToLocalGroupBody struct {
	Req    *types.AddUsersToLocalGroup         `xml:"urn:sso AddUsersToLocalGroup,omitempty"`
	Res    *types.AddUsersToLocalGroupResponse `xml:"urn:sso AddUsersToLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *AddUsersToLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func AddUsersToLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.AddUsersToLocalGroup) (*types.AddUsersToLocalGroupResponse, error) {
	var reqBody, resBody AddUsersToLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type CreateLocalGroupBody struct {
	Req    *types.CreateLocalGroup         `xml:"urn:sso CreateLocalGroup,omitempty"`
	Res    *types.CreateLocalGroupResponse `xml:"urn:sso CreateLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *CreateLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func CreateLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.CreateLocalGroup) (*types.CreateLocalGroupResponse, error) {
	var reqBody, resBody CreateLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type CreateLocalPersonUserBody struct {
	Req    *types.CreateLocalPersonUser         `xml:"urn:sso CreateLocalPersonUser,omitempty"`
	Res    *types.CreateLocalPersonUserResponse `xml:"urn:sso CreateLocalPersonUserResponse,omitempty"`
	Fault_ *soap.Fault                          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *CreateLocalPersonUserBody) Fault() *soap.Fault { return b.Fault_ }

func CreateLocalPersonUser(ctx context.Context, r soap.RoundTripper, req *types.CreateLocalPersonUser) (*types.CreateLocalPersonUserResponse, error) {
	var reqBody, resBody CreateLocalPersonUserBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type CreateLocalSolutionUserBody struct {
	Req    *types.CreateLocalSolutionUser         `xml:"urn:sso CreateLocalSolutionUser,omitempty"`
	Res    *types.CreateLocalSolutionUserResponse `xml:"urn:sso CreateLocalSolutionUserResponse,omitempty"`
	Fault_ *soap.Fault                            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *CreateLocalSolutionUserBody) Fault() *soap.Fault { return b.Fault_ }

func CreateLocalSolutionUser(ctx context.Context, r soap.RoundTripper, req *types.CreateLocalSolutionUser) (*types.CreateLocalSolutionUserResponse, error) {
	var reqBody, resBody CreateLocalSolutionUserBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DeleteCertificateBody struct {
	Req    *types.DeleteCertificate         `xml:"urn:sso DeleteCertificate,omitempty"`
	Res    *types.DeleteCertificateResponse `xml:"urn:sso DeleteCertificateResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *DeleteCertificateBody) Fault() *soap.Fault { return b.Fault_ }

func DeleteCertificate(ctx context.Context, r soap.RoundTripper, req *types.DeleteCertificate) (*types.DeleteCertificateResponse, error) {
	var reqBody, resBody DeleteCertificateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DeleteDomainBody struct {
	Req    *types.DeleteDomain         `xml:"urn:sso DeleteDomain,omitempty"`
	Res    *types.DeleteDomainResponse `xml:"urn:sso DeleteDomainResponse,omitempty"`
	Fault_ *soap.Fault                 `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *DeleteDomainBody) Fault() *soap.Fault { return b.Fault_ }

func DeleteDomain(ctx context.Context, r soap.RoundTripper, req *types.DeleteDomain) (*types.DeleteDomainResponse, error) {
	var reqBody, resBody DeleteDomainBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DeleteLocalPrincipalBody struct {
	Req    *types.DeleteLocalPrincipal         `xml:"urn:sso DeleteLocalPrincipal,omitempty"`
	Res    *types.DeleteLocalPrincipalResponse `xml:"urn:sso DeleteLocalPrincipalResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *DeleteLocalPrincipalBody) Fault() *soap.Fault { return b.Fault_ }

func DeleteLocalPrincipal(ctx context.Context, r soap.RoundTripper, req *types.DeleteLocalPrincipal) (*types.DeleteLocalPrincipalResponse, error) {
	var reqBody, resBody DeleteLocalPrincipalBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type DisableUserAccountBody struct {
	Req    *types.DisableUserAccount         `xml:"urn:sso DisableUserAccount,omitempty"`
	Res    *types.DisableUserAccountResponse `xml:"urn:sso DisableUserAccountResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *DisableUserAccountBody) Fault() *soap.Fault { return b.Fault_ }

func DisableUserAccount(ctx context.Context, r soap.RoundTripper, req *types.DisableUserAccount) (*types.DisableUserAccountResponse, error) {
	var reqBody, resBody DisableUserAccountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type EnableUserAccountBody struct {
	Req    *types.EnableUserAccount         `xml:"urn:sso EnableUserAccount,omitempty"`
	Res    *types.EnableUserAccountResponse `xml:"urn:sso EnableUserAccountResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *EnableUserAccountBody) Fault() *soap.Fault { return b.Fault_ }

func EnableUserAccount(ctx context.Context, r soap.RoundTripper, req *types.EnableUserAccount) (*types.EnableUserAccountResponse, error) {
	var reqBody, resBody EnableUserAccountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindBody struct {
	Req    *types.Find         `xml:"urn:sso Find,omitempty"`
	Res    *types.FindResponse `xml:"urn:sso FindResponse,omitempty"`
	Fault_ *soap.Fault         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindBody) Fault() *soap.Fault { return b.Fault_ }

func Find(ctx context.Context, r soap.RoundTripper, req *types.Find) (*types.FindResponse, error) {
	var reqBody, resBody FindBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindAllParentGroupsBody struct {
	Req    *types.FindAllParentGroups         `xml:"urn:sso FindAllParentGroups,omitempty"`
	Res    *types.FindAllParentGroupsResponse `xml:"urn:sso FindAllParentGroupsResponse,omitempty"`
	Fault_ *soap.Fault                        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindAllParentGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func FindAllParentGroups(ctx context.Context, r soap.RoundTripper, req *types.FindAllParentGroups) (*types.FindAllParentGroupsResponse, error) {
	var reqBody, resBody FindAllParentGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindCertificateBody struct {
	Req    *types.FindCertificate         `xml:"urn:sso FindCertificate,omitempty"`
	Res    *types.FindCertificateResponse `xml:"urn:sso FindCertificateResponse,omitempty"`
	Fault_ *soap.Fault                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindCertificateBody) Fault() *soap.Fault { return b.Fault_ }

func FindCertificate(ctx context.Context, r soap.RoundTripper, req *types.FindCertificate) (*types.FindCertificateResponse, error) {
	var reqBody, resBody FindCertificateBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindDirectParentGroupsBody struct {
	Req    *types.FindDirectParentGroups         `xml:"urn:sso FindDirectParentGroups,omitempty"`
	Res    *types.FindDirectParentGroupsResponse `xml:"urn:sso FindDirectParentGroupsResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindDirectParentGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func FindDirectParentGroups(ctx context.Context, r soap.RoundTripper, req *types.FindDirectParentGroups) (*types.FindDirectParentGroupsResponse, error) {
	var reqBody, resBody FindDirectParentGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindDisabledPersonUsersBody struct {
	Req    *types.FindDisabledPersonUsers         `xml:"urn:sso FindDisabledPersonUsers,omitempty"`
	Res    *types.FindDisabledPersonUsersResponse `xml:"urn:sso FindDisabledPersonUsersResponse,omitempty"`
	Fault_ *soap.Fault                            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindDisabledPersonUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindDisabledPersonUsers(ctx context.Context, r soap.RoundTripper, req *types.FindDisabledPersonUsers) (*types.FindDisabledPersonUsersResponse, error) {
	var reqBody, resBody FindDisabledPersonUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindDisabledSolutionUsersBody struct {
	Req    *types.FindDisabledSolutionUsers         `xml:"urn:sso FindDisabledSolutionUsers,omitempty"`
	Res    *types.FindDisabledSolutionUsersResponse `xml:"urn:sso FindDisabledSolutionUsersResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindDisabledSolutionUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindDisabledSolutionUsers(ctx context.Context, r soap.RoundTripper, req *types.FindDisabledSolutionUsers) (*types.FindDisabledSolutionUsersResponse, error) {
	var reqBody, resBody FindDisabledSolutionUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindExternalDomainBody struct {
	Req    *types.FindExternalDomain         `xml:"urn:sso FindExternalDomain,omitempty"`
	Res    *types.FindExternalDomainResponse `xml:"urn:sso FindExternalDomainResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindExternalDomainBody) Fault() *soap.Fault { return b.Fault_ }

func FindExternalDomain(ctx context.Context, r soap.RoundTripper, req *types.FindExternalDomain) (*types.FindExternalDomainResponse, error) {
	var reqBody, resBody FindExternalDomainBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindGroupBody struct {
	Req    *types.FindGroup         `xml:"urn:sso FindGroup,omitempty"`
	Res    *types.FindGroupResponse `xml:"urn:sso FindGroupResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindGroupBody) Fault() *soap.Fault { return b.Fault_ }

func FindGroup(ctx context.Context, r soap.RoundTripper, req *types.FindGroup) (*types.FindGroupResponse, error) {
	var reqBody, resBody FindGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindGroupsBody struct {
	Req    *types.FindGroups         `xml:"urn:sso FindGroups,omitempty"`
	Res    *types.FindGroupsResponse `xml:"urn:sso FindGroupsResponse,omitempty"`
	Fault_ *soap.Fault               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func FindGroups(ctx context.Context, r soap.RoundTripper, req *types.FindGroups) (*types.FindGroupsResponse, error) {
	var reqBody, resBody FindGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindGroupsInGroupBody struct {
	Req    *types.FindGroupsInGroup         `xml:"urn:sso FindGroupsInGroup,omitempty"`
	Res    *types.FindGroupsInGroupResponse `xml:"urn:sso FindGroupsInGroupResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindGroupsInGroupBody) Fault() *soap.Fault { return b.Fault_ }

func FindGroupsInGroup(ctx context.Context, r soap.RoundTripper, req *types.FindGroupsInGroup) (*types.FindGroupsInGroupResponse, error) {
	var reqBody, resBody FindGroupsInGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindLockedUsersBody struct {
	Req    *types.FindLockedUsers         `xml:"urn:sso FindLockedUsers,omitempty"`
	Res    *types.FindLockedUsersResponse `xml:"urn:sso FindLockedUsersResponse,omitempty"`
	Fault_ *soap.Fault                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindLockedUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindLockedUsers(ctx context.Context, r soap.RoundTripper, req *types.FindLockedUsers) (*types.FindLockedUsersResponse, error) {
	var reqBody, resBody FindLockedUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindNestedParentGroupsBody struct {
	Req    *types.FindNestedParentGroups         `xml:"urn:sso FindNestedParentGroups,omitempty"`
	Res    *types.FindNestedParentGroupsResponse `xml:"urn:sso FindNestedParentGroupsResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindNestedParentGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func FindNestedParentGroups(ctx context.Context, r soap.RoundTripper, req *types.FindNestedParentGroups) (*types.FindNestedParentGroupsResponse, error) {
	var reqBody, resBody FindNestedParentGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindParentGroupsBody struct {
	Req    *types.FindParentGroups         `xml:"urn:sso FindParentGroups,omitempty"`
	Res    *types.FindParentGroupsResponse `xml:"urn:sso FindParentGroupsResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindParentGroupsBody) Fault() *soap.Fault { return b.Fault_ }

func FindParentGroups(ctx context.Context, r soap.RoundTripper, req *types.FindParentGroups) (*types.FindParentGroupsResponse, error) {
	var reqBody, resBody FindParentGroupsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindPersonUserBody struct {
	Req    *types.FindPersonUser         `xml:"urn:sso FindPersonUser,omitempty"`
	Res    *types.FindPersonUserResponse `xml:"urn:sso FindPersonUserResponse,omitempty"`
	Fault_ *soap.Fault                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindPersonUserBody) Fault() *soap.Fault { return b.Fault_ }

func FindPersonUser(ctx context.Context, r soap.RoundTripper, req *types.FindPersonUser) (*types.FindPersonUserResponse, error) {
	var reqBody, resBody FindPersonUserBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindPersonUsersBody struct {
	Req    *types.FindPersonUsers         `xml:"urn:sso FindPersonUsers,omitempty"`
	Res    *types.FindPersonUsersResponse `xml:"urn:sso FindPersonUsersResponse,omitempty"`
	Fault_ *soap.Fault                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindPersonUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindPersonUsers(ctx context.Context, r soap.RoundTripper, req *types.FindPersonUsers) (*types.FindPersonUsersResponse, error) {
	var reqBody, resBody FindPersonUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindPersonUsersInGroupBody struct {
	Req    *types.FindPersonUsersInGroup         `xml:"urn:sso FindPersonUsersInGroup,omitempty"`
	Res    *types.FindPersonUsersInGroupResponse `xml:"urn:sso FindPersonUsersInGroupResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindPersonUsersInGroupBody) Fault() *soap.Fault { return b.Fault_ }

func FindPersonUsersInGroup(ctx context.Context, r soap.RoundTripper, req *types.FindPersonUsersInGroup) (*types.FindPersonUsersInGroupResponse, error) {
	var reqBody, resBody FindPersonUsersInGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindSolutionUserBody struct {
	Req    *types.FindSolutionUser         `xml:"urn:sso FindSolutionUser,omitempty"`
	Res    *types.FindSolutionUserResponse `xml:"urn:sso FindSolutionUserResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindSolutionUserBody) Fault() *soap.Fault { return b.Fault_ }

func FindSolutionUser(ctx context.Context, r soap.RoundTripper, req *types.FindSolutionUser) (*types.FindSolutionUserResponse, error) {
	var reqBody, resBody FindSolutionUserBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindSolutionUsersBody struct {
	Req    *types.FindSolutionUsers         `xml:"urn:sso FindSolutionUsers,omitempty"`
	Res    *types.FindSolutionUsersResponse `xml:"urn:sso FindSolutionUsersResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindSolutionUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindSolutionUsers(ctx context.Context, r soap.RoundTripper, req *types.FindSolutionUsers) (*types.FindSolutionUsersResponse, error) {
	var reqBody, resBody FindSolutionUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindSolutionUsersInGroupBody struct {
	Req    *types.FindSolutionUsersInGroup         `xml:"urn:sso FindSolutionUsersInGroup,omitempty"`
	Res    *types.FindSolutionUsersInGroupResponse `xml:"urn:sso FindSolutionUsersInGroupResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindSolutionUsersInGroupBody) Fault() *soap.Fault { return b.Fault_ }

func FindSolutionUsersInGroup(ctx context.Context, r soap.RoundTripper, req *types.FindSolutionUsersInGroup) (*types.FindSolutionUsersInGroupResponse, error) {
	var reqBody, resBody FindSolutionUsersInGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindUserBody struct {
	Req    *types.FindUser         `xml:"urn:sso FindUser,omitempty"`
	Res    *types.FindUserResponse `xml:"urn:sso FindUserResponse,omitempty"`
	Fault_ *soap.Fault             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindUserBody) Fault() *soap.Fault { return b.Fault_ }

func FindUser(ctx context.Context, r soap.RoundTripper, req *types.FindUser) (*types.FindUserResponse, error) {
	var reqBody, resBody FindUserBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindUsersBody struct {
	Req    *types.FindUsers         `xml:"urn:sso FindUsers,omitempty"`
	Res    *types.FindUsersResponse `xml:"urn:sso FindUsersResponse,omitempty"`
	Fault_ *soap.Fault              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindUsersBody) Fault() *soap.Fault { return b.Fault_ }

func FindUsers(ctx context.Context, r soap.RoundTripper, req *types.FindUsers) (*types.FindUsersResponse, error) {
	var reqBody, resBody FindUsersBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type FindUsersInGroupBody struct {
	Req    *types.FindUsersInGroup         `xml:"urn:sso FindUsersInGroup,omitempty"`
	Res    *types.FindUsersInGroupResponse `xml:"urn:sso FindUsersInGroupResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *FindUsersInGroupBody) Fault() *soap.Fault { return b.Fault_ }

func FindUsersInGroup(ctx context.Context, r soap.RoundTripper, req *types.FindUsersInGroup) (*types.FindUsersInGroupResponse, error) {
	var reqBody, resBody FindUsersInGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetAllCertificatesBody struct {
	Req    *types.GetAllCertificates         `xml:"urn:sso GetAllCertificates,omitempty"`
	Res    *types.GetAllCertificatesResponse `xml:"urn:sso GetAllCertificatesResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetAllCertificatesBody) Fault() *soap.Fault { return b.Fault_ }

func GetAllCertificates(ctx context.Context, r soap.RoundTripper, req *types.GetAllCertificates) (*types.GetAllCertificatesResponse, error) {
	var reqBody, resBody GetAllCertificatesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetClockToleranceBody struct {
	Req    *types.GetClockTolerance         `xml:"urn:sso GetClockTolerance,omitempty"`
	Res    *types.GetClockToleranceResponse `xml:"urn:sso GetClockToleranceResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetClockToleranceBody) Fault() *soap.Fault { return b.Fault_ }

func GetClockTolerance(ctx context.Context, r soap.RoundTripper, req *types.GetClockTolerance) (*types.GetClockToleranceResponse, error) {
	var reqBody, resBody GetClockToleranceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetDelegationCountBody struct {
	Req    *types.GetDelegationCount         `xml:"urn:sso GetDelegationCount,omitempty"`
	Res    *types.GetDelegationCountResponse `xml:"urn:sso GetDelegationCountResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetDelegationCountBody) Fault() *soap.Fault { return b.Fault_ }

func GetDelegationCount(ctx context.Context, r soap.RoundTripper, req *types.GetDelegationCount) (*types.GetDelegationCountResponse, error) {
	var reqBody, resBody GetDelegationCountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetDomainsBody struct {
	Req    *types.GetDomains         `xml:"urn:sso GetDomains,omitempty"`
	Res    *types.GetDomainsResponse `xml:"urn:sso GetDomainsResponse,omitempty"`
	Fault_ *soap.Fault               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetDomainsBody) Fault() *soap.Fault { return b.Fault_ }

func GetDomains(ctx context.Context, r soap.RoundTripper, req *types.GetDomains) (*types.GetDomainsResponse, error) {
	var reqBody, resBody GetDomainsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetIssuersCertificatesBody struct {
	Req    *types.GetIssuersCertificates         `xml:"urn:sso GetIssuersCertificates,omitempty"`
	Res    *types.GetIssuersCertificatesResponse `xml:"urn:sso GetIssuersCertificatesResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetIssuersCertificatesBody) Fault() *soap.Fault { return b.Fault_ }

func GetIssuersCertificates(ctx context.Context, r soap.RoundTripper, req *types.GetIssuersCertificates) (*types.GetIssuersCertificatesResponse, error) {
	var reqBody, resBody GetIssuersCertificatesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetKnownCertificateChainsBody struct {
	Req    *types.GetKnownCertificateChains         `xml:"urn:sso GetKnownCertificateChains,omitempty"`
	Res    *types.GetKnownCertificateChainsResponse `xml:"urn:sso GetKnownCertificateChainsResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetKnownCertificateChainsBody) Fault() *soap.Fault { return b.Fault_ }

func GetKnownCertificateChains(ctx context.Context, r soap.RoundTripper, req *types.GetKnownCertificateChains) (*types.GetKnownCertificateChainsResponse, error) {
	var reqBody, resBody GetKnownCertificateChainsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetLocalPasswordPolicyBody struct {
	Req    *types.GetLocalPasswordPolicy         `xml:"urn:sso GetLocalPasswordPolicy,omitempty"`
	Res    *types.GetLocalPasswordPolicyResponse `xml:"urn:sso GetLocalPasswordPolicyResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetLocalPasswordPolicyBody) Fault() *soap.Fault { return b.Fault_ }

func GetLocalPasswordPolicy(ctx context.Context, r soap.RoundTripper, req *types.GetLocalPasswordPolicy) (*types.GetLocalPasswordPolicyResponse, error) {
	var reqBody, resBody GetLocalPasswordPolicyBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetLockoutPolicyBody struct {
	Req    *types.GetLockoutPolicy         `xml:"urn:sso GetLockoutPolicy,omitempty"`
	Res    *types.GetLockoutPolicyResponse `xml:"urn:sso GetLockoutPolicyResponse,omitempty"`
	Fault_ *soap.Fault                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetLockoutPolicyBody) Fault() *soap.Fault { return b.Fault_ }

func GetLockoutPolicy(ctx context.Context, r soap.RoundTripper, req *types.GetLockoutPolicy) (*types.GetLockoutPolicyResponse, error) {
	var reqBody, resBody GetLockoutPolicyBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetMaximumBearerTokenLifetimeBody struct {
	Req    *types.GetMaximumBearerTokenLifetime         `xml:"urn:sso GetMaximumBearerTokenLifetime,omitempty"`
	Res    *types.GetMaximumBearerTokenLifetimeResponse `xml:"urn:sso GetMaximumBearerTokenLifetimeResponse,omitempty"`
	Fault_ *soap.Fault                                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetMaximumBearerTokenLifetimeBody) Fault() *soap.Fault { return b.Fault_ }

func GetMaximumBearerTokenLifetime(ctx context.Context, r soap.RoundTripper, req *types.GetMaximumBearerTokenLifetime) (*types.GetMaximumBearerTokenLifetimeResponse, error) {
	var reqBody, resBody GetMaximumBearerTokenLifetimeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetMaximumHoKTokenLifetimeBody struct {
	Req    *types.GetMaximumHoKTokenLifetime         `xml:"urn:sso GetMaximumHoKTokenLifetime,omitempty"`
	Res    *types.GetMaximumHoKTokenLifetimeResponse `xml:"urn:sso GetMaximumHoKTokenLifetimeResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetMaximumHoKTokenLifetimeBody) Fault() *soap.Fault { return b.Fault_ }

func GetMaximumHoKTokenLifetime(ctx context.Context, r soap.RoundTripper, req *types.GetMaximumHoKTokenLifetime) (*types.GetMaximumHoKTokenLifetimeResponse, error) {
	var reqBody, resBody GetMaximumHoKTokenLifetimeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetPasswordExpirationConfigurationBody struct {
	Req    *types.GetPasswordExpirationConfiguration         `xml:"urn:sso GetPasswordExpirationConfiguration,omitempty"`
	Res    *types.GetPasswordExpirationConfigurationResponse `xml:"urn:sso GetPasswordExpirationConfigurationResponse,omitempty"`
	Fault_ *soap.Fault                                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetPasswordExpirationConfigurationBody) Fault() *soap.Fault { return b.Fault_ }

func GetPasswordExpirationConfiguration(ctx context.Context, r soap.RoundTripper, req *types.GetPasswordExpirationConfiguration) (*types.GetPasswordExpirationConfigurationResponse, error) {
	var reqBody, resBody GetPasswordExpirationConfigurationBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetRenewCountBody struct {
	Req    *types.GetRenewCount         `xml:"urn:sso GetRenewCount,omitempty"`
	Res    *types.GetRenewCountResponse `xml:"urn:sso GetRenewCountResponse,omitempty"`
	Fault_ *soap.Fault                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetRenewCountBody) Fault() *soap.Fault { return b.Fault_ }

func GetRenewCount(ctx context.Context, r soap.RoundTripper, req *types.GetRenewCount) (*types.GetRenewCountResponse, error) {
	var reqBody, resBody GetRenewCountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetSmtpConfigurationBody struct {
	Req    *types.GetSmtpConfiguration         `xml:"urn:sso GetSmtpConfiguration,omitempty"`
	Res    *types.GetSmtpConfigurationResponse `xml:"urn:sso GetSmtpConfigurationResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetSmtpConfigurationBody) Fault() *soap.Fault { return b.Fault_ }

func GetSmtpConfiguration(ctx context.Context, r soap.RoundTripper, req *types.GetSmtpConfiguration) (*types.GetSmtpConfigurationResponse, error) {
	var reqBody, resBody GetSmtpConfigurationBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetSslCertificateManagerBody struct {
	Req    *types.GetSslCertificateManager         `xml:"urn:sso GetSslCertificateManager,omitempty"`
	Res    *types.GetSslCertificateManagerResponse `xml:"urn:sso GetSslCertificateManagerResponse,omitempty"`
	Fault_ *soap.Fault                             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetSslCertificateManagerBody) Fault() *soap.Fault { return b.Fault_ }

func GetSslCertificateManager(ctx context.Context, r soap.RoundTripper, req *types.GetSslCertificateManager) (*types.GetSslCertificateManagerResponse, error) {
	var reqBody, resBody GetSslCertificateManagerBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetSystemDomainNameBody struct {
	Req    *types.GetSystemDomainName         `xml:"urn:sso GetSystemDomainName,omitempty"`
	Res    *types.GetSystemDomainNameResponse `xml:"urn:sso GetSystemDomainNameResponse,omitempty"`
	Fault_ *soap.Fault                        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetSystemDomainNameBody) Fault() *soap.Fault { return b.Fault_ }

func GetSystemDomainName(ctx context.Context, r soap.RoundTripper, req *types.GetSystemDomainName) (*types.GetSystemDomainNameResponse, error) {
	var reqBody, resBody GetSystemDomainNameBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type GetTrustedCertificatesBody struct {
	Req    *types.GetTrustedCertificates         `xml:"urn:sso GetTrustedCertificates,omitempty"`
	Res    *types.GetTrustedCertificatesResponse `xml:"urn:sso GetTrustedCertificatesResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *GetTrustedCertificatesBody) Fault() *soap.Fault { return b.Fault_ }

func GetTrustedCertificates(ctx context.Context, r soap.RoundTripper, req *types.GetTrustedCertificates) (*types.GetTrustedCertificatesResponse, error) {
	var reqBody, resBody GetTrustedCertificatesBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type HasAdministratorRoleBody struct {
	Req    *types.HasAdministratorRole         `xml:"urn:sso HasAdministratorRole,omitempty"`
	Res    *types.HasAdministratorRoleResponse `xml:"urn:sso HasAdministratorRoleResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *HasAdministratorRoleBody) Fault() *soap.Fault { return b.Fault_ }

func HasAdministratorRole(ctx context.Context, r soap.RoundTripper, req *types.HasAdministratorRole) (*types.HasAdministratorRoleResponse, error) {
	var reqBody, resBody HasAdministratorRoleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type HasRegularUserRoleBody struct {
	Req    *types.HasRegularUserRole         `xml:"urn:sso HasRegularUserRole,omitempty"`
	Res    *types.HasRegularUserRoleResponse `xml:"urn:sso HasRegularUserRoleResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *HasRegularUserRoleBody) Fault() *soap.Fault { return b.Fault_ }

func HasRegularUserRole(ctx context.Context, r soap.RoundTripper, req *types.HasRegularUserRole) (*types.HasRegularUserRoleResponse, error) {
	var reqBody, resBody HasRegularUserRoleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type IsMemberOfGroupBody struct {
	Req    *types.IsMemberOfGroup         `xml:"urn:sso IsMemberOfGroup,omitempty"`
	Res    *types.IsMemberOfGroupResponse `xml:"urn:sso IsMemberOfGroupResponse,omitempty"`
	Fault_ *soap.Fault                    `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *IsMemberOfGroupBody) Fault() *soap.Fault { return b.Fault_ }

func IsMemberOfGroup(ctx context.Context, r soap.RoundTripper, req *types.IsMemberOfGroup) (*types.IsMemberOfGroupResponse, error) {
	var reqBody, resBody IsMemberOfGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type LoginBody struct {
	Req    *types.Login         `xml:"urn:sso Login,omitempty"`
	Res    *types.LoginResponse `xml:"urn:sso LoginResponse,omitempty"`
	Fault_ *soap.Fault          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *LoginBody) Fault() *soap.Fault { return b.Fault_ }

func Login(ctx context.Context, r soap.RoundTripper, req *types.Login) (*types.LoginResponse, error) {
	var reqBody, resBody LoginBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type LogoutBody struct {
	Req    *types.Logout         `xml:"urn:sso Logout,omitempty"`
	Res    *types.LogoutResponse `xml:"urn:sso LogoutResponse,omitempty"`
	Fault_ *soap.Fault           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *LogoutBody) Fault() *soap.Fault { return b.Fault_ }

func Logout(ctx context.Context, r soap.RoundTripper, req *types.Logout) (*types.LogoutResponse, error) {
	var reqBody, resBody LogoutBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ProbeConnectivityBody struct {
	Req    *types.ProbeConnectivity         `xml:"urn:sso ProbeConnectivity,omitempty"`
	Res    *types.ProbeConnectivityResponse `xml:"urn:sso ProbeConnectivityResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *ProbeConnectivityBody) Fault() *soap.Fault { return b.Fault_ }

func ProbeConnectivity(ctx context.Context, r soap.RoundTripper, req *types.ProbeConnectivity) (*types.ProbeConnectivityResponse, error) {
	var reqBody, resBody ProbeConnectivityBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RemoveFromLocalGroupBody struct {
	Req    *types.RemoveFromLocalGroup         `xml:"urn:sso RemoveFromLocalGroup,omitempty"`
	Res    *types.RemoveFromLocalGroupResponse `xml:"urn:sso RemoveFromLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RemoveFromLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func RemoveFromLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.RemoveFromLocalGroup) (*types.RemoveFromLocalGroupResponse, error) {
	var reqBody, resBody RemoveFromLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type RemovePrincipalsFromLocalGroupBody struct {
	Req    *types.RemovePrincipalsFromLocalGroup         `xml:"urn:sso RemovePrincipalsFromLocalGroup,omitempty"`
	Res    *types.RemovePrincipalsFromLocalGroupResponse `xml:"urn:sso RemovePrincipalsFromLocalGroupResponse,omitempty"`
	Fault_ *soap.Fault                                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *RemovePrincipalsFromLocalGroupBody) Fault() *soap.Fault { return b.Fault_ }

func RemovePrincipalsFromLocalGroup(ctx context.Context, r soap.RoundTripper, req *types.RemovePrincipalsFromLocalGroup) (*types.RemovePrincipalsFromLocalGroupResponse, error) {
	var reqBody, resBody RemovePrincipalsFromLocalGroupBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ResetLocalPersonUserPasswordBody struct {
	Req    *types.ResetLocalPersonUserPassword         `xml:"urn:sso ResetLocalPersonUserPassword,omitempty"`
	Res    *types.ResetLocalPersonUserPasswordResponse `xml:"urn:sso ResetLocalPersonUserPasswordResponse,omitempty"`
	Fault_ *soap.Fault                                 `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *ResetLocalPersonUserPasswordBody) Fault() *soap.Fault { return b.Fault_ }

func ResetLocalPersonUserPassword(ctx context.Context, r soap.RoundTripper, req *types.ResetLocalPersonUserPassword) (*types.ResetLocalPersonUserPasswordResponse, error) {
	var reqBody, resBody ResetLocalPersonUserPasswordBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ResetLocalUserPasswordBody struct {
	Req    *types.ResetLocalUserPassword         `xml:"urn:sso ResetLocalUserPassword,omitempty"`
	Res    *types.ResetLocalUserPasswordResponse `xml:"urn:sso ResetLocalUserPasswordResponse,omitempty"`
	Fault_ *soap.Fault                           `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *ResetLocalUserPasswordBody) Fault() *soap.Fault { return b.Fault_ }

func ResetLocalUserPassword(ctx context.Context, r soap.RoundTripper, req *types.ResetLocalUserPassword) (*types.ResetLocalUserPasswordResponse, error) {
	var reqBody, resBody ResetLocalUserPasswordBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type ResetSelfLocalPersonUserPasswordBody struct {
	Req    *types.ResetSelfLocalPersonUserPassword         `xml:"urn:sso ResetSelfLocalPersonUserPassword,omitempty"`
	Res    *types.ResetSelfLocalPersonUserPasswordResponse `xml:"urn:sso ResetSelfLocalPersonUserPasswordResponse,omitempty"`
	Fault_ *soap.Fault                                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *ResetSelfLocalPersonUserPasswordBody) Fault() *soap.Fault { return b.Fault_ }

func ResetSelfLocalPersonUserPassword(ctx context.Context, r soap.RoundTripper, req *types.ResetSelfLocalPersonUserPassword) (*types.ResetSelfLocalPersonUserPasswordResponse, error) {
	var reqBody, resBody ResetSelfLocalPersonUserPasswordBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SendMailBody struct {
	Req    *types.SendMail         `xml:"urn:sso SendMail,omitempty"`
	Res    *types.SendMailResponse `xml:"urn:sso SendMailResponse,omitempty"`
	Fault_ *soap.Fault             `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SendMailBody) Fault() *soap.Fault { return b.Fault_ }

func SendMail(ctx context.Context, r soap.RoundTripper, req *types.SendMail) (*types.SendMailResponse, error) {
	var reqBody, resBody SendMailBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetClockToleranceBody struct {
	Req    *types.SetClockTolerance         `xml:"urn:sso SetClockTolerance,omitempty"`
	Res    *types.SetClockToleranceResponse `xml:"urn:sso SetClockToleranceResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetClockToleranceBody) Fault() *soap.Fault { return b.Fault_ }

func SetClockTolerance(ctx context.Context, r soap.RoundTripper, req *types.SetClockTolerance) (*types.SetClockToleranceResponse, error) {
	var reqBody, resBody SetClockToleranceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetDelegationCountBody struct {
	Req    *types.SetDelegationCount         `xml:"urn:sso SetDelegationCount,omitempty"`
	Res    *types.SetDelegationCountResponse `xml:"urn:sso SetDelegationCountResponse,omitempty"`
	Fault_ *soap.Fault                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetDelegationCountBody) Fault() *soap.Fault { return b.Fault_ }

func SetDelegationCount(ctx context.Context, r soap.RoundTripper, req *types.SetDelegationCount) (*types.SetDelegationCountResponse, error) {
	var reqBody, resBody SetDelegationCountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetMaximumBearerTokenLifetimeBody struct {
	Req    *types.SetMaximumBearerTokenLifetime         `xml:"urn:sso SetMaximumBearerTokenLifetime,omitempty"`
	Res    *types.SetMaximumBearerTokenLifetimeResponse `xml:"urn:sso SetMaximumBearerTokenLifetimeResponse,omitempty"`
	Fault_ *soap.Fault                                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetMaximumBearerTokenLifetimeBody) Fault() *soap.Fault { return b.Fault_ }

func SetMaximumBearerTokenLifetime(ctx context.Context, r soap.RoundTripper, req *types.SetMaximumBearerTokenLifetime) (*types.SetMaximumBearerTokenLifetimeResponse, error) {
	var reqBody, resBody SetMaximumBearerTokenLifetimeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetMaximumHoKTokenLifetimeBody struct {
	Req    *types.SetMaximumHoKTokenLifetime         `xml:"urn:sso SetMaximumHoKTokenLifetime,omitempty"`
	Res    *types.SetMaximumHoKTokenLifetimeResponse `xml:"urn:sso SetMaximumHoKTokenLifetimeResponse,omitempty"`
	Fault_ *soap.Fault                               `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetMaximumHoKTokenLifetimeBody) Fault() *soap.Fault { return b.Fault_ }

func SetMaximumHoKTokenLifetime(ctx context.Context, r soap.RoundTripper, req *types.SetMaximumHoKTokenLifetime) (*types.SetMaximumHoKTokenLifetimeResponse, error) {
	var reqBody, resBody SetMaximumHoKTokenLifetimeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetNewSignerIdentityBody struct {
	Req    *types.SetNewSignerIdentity         `xml:"urn:sso SetNewSignerIdentity,omitempty"`
	Res    *types.SetNewSignerIdentityResponse `xml:"urn:sso SetNewSignerIdentityResponse,omitempty"`
	Fault_ *soap.Fault                         `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetNewSignerIdentityBody) Fault() *soap.Fault { return b.Fault_ }

func SetNewSignerIdentity(ctx context.Context, r soap.RoundTripper, req *types.SetNewSignerIdentity) (*types.SetNewSignerIdentityResponse, error) {
	var reqBody, resBody SetNewSignerIdentityBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetRenewCountBody struct {
	Req    *types.SetRenewCount         `xml:"urn:sso SetRenewCount,omitempty"`
	Res    *types.SetRenewCountResponse `xml:"urn:sso SetRenewCountResponse,omitempty"`
	Fault_ *soap.Fault                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetRenewCountBody) Fault() *soap.Fault { return b.Fault_ }

func SetRenewCount(ctx context.Context, r soap.RoundTripper, req *types.SetRenewCount) (*types.SetRenewCountResponse, error) {
	var reqBody, resBody SetRenewCountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetRoleBody struct {
	Req    *types.SetRole         `xml:"urn:sso SetRole,omitempty"`
	Res    *types.SetRoleResponse `xml:"urn:sso SetRoleResponse,omitempty"`
	Fault_ *soap.Fault            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetRoleBody) Fault() *soap.Fault { return b.Fault_ }

func SetRole(ctx context.Context, r soap.RoundTripper, req *types.SetRole) (*types.SetRoleResponse, error) {
	var reqBody, resBody SetRoleBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SetSignerIdentityBody struct {
	Req    *types.SetSignerIdentity         `xml:"urn:sso SetSignerIdentity,omitempty"`
	Res    *types.SetSignerIdentityResponse `xml:"urn:sso SetSignerIdentityResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SetSignerIdentityBody) Fault() *soap.Fault { return b.Fault_ }

func SetSignerIdentity(ctx context.Context, r soap.RoundTripper, req *types.SetSignerIdentity) (*types.SetSignerIdentityResponse, error) {
	var reqBody, resBody SetSignerIdentityBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SsoAdminServiceInstanceBody struct {
	Req    *types.SsoAdminServiceInstance         `xml:"urn:sso SsoAdminServiceInstance,omitempty"`
	Res    *types.SsoAdminServiceInstanceResponse `xml:"urn:sso SsoAdminServiceInstanceResponse,omitempty"`
	Fault_ *soap.Fault                            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SsoAdminServiceInstanceBody) Fault() *soap.Fault { return b.Fault_ }

func SsoAdminServiceInstance(ctx context.Context, r soap.RoundTripper, req *types.SsoAdminServiceInstance) (*types.SsoAdminServiceInstanceResponse, error) {
	var reqBody, resBody SsoAdminServiceInstanceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type SsoGroupcheckServiceInstanceBody struct {
	Req    *types.SsoGroupcheckServiceInstance         `xml:"urn:sso SsoGroupcheckServiceInstance,omitempty"`
	Res    *types.SsoGroupcheckServiceInstanceResponse `xml:"urn:sso SsoGroupcheckServiceInstanceResponse,omitempty"`
	Fault_ *soap.Fault                                 `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *SsoGroupcheckServiceInstanceBody) Fault() *soap.Fault { return b.Fault_ }

func SsoGroupcheckServiceInstance(ctx context.Context, r soap.RoundTripper, req *types.SsoGroupcheckServiceInstance) (*types.SsoGroupcheckServiceInstanceResponse, error) {
	var reqBody, resBody SsoGroupcheckServiceInstanceBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UnlockUserAccountBody struct {
	Req    *types.UnlockUserAccount         `xml:"urn:sso UnlockUserAccount,omitempty"`
	Res    *types.UnlockUserAccountResponse `xml:"urn:sso UnlockUserAccountResponse,omitempty"`
	Fault_ *soap.Fault                      `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UnlockUserAccountBody) Fault() *soap.Fault { return b.Fault_ }

func UnlockUserAccount(ctx context.Context, r soap.RoundTripper, req *types.UnlockUserAccount) (*types.UnlockUserAccountResponse, error) {
	var reqBody, resBody UnlockUserAccountBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateExternalDomainAuthnTypeBody struct {
	Req    *types.UpdateExternalDomainAuthnType         `xml:"urn:sso UpdateExternalDomainAuthnType,omitempty"`
	Res    *types.UpdateExternalDomainAuthnTypeResponse `xml:"urn:sso UpdateExternalDomainAuthnTypeResponse,omitempty"`
	Fault_ *soap.Fault                                  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateExternalDomainAuthnTypeBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateExternalDomainAuthnType(ctx context.Context, r soap.RoundTripper, req *types.UpdateExternalDomainAuthnType) (*types.UpdateExternalDomainAuthnTypeResponse, error) {
	var reqBody, resBody UpdateExternalDomainAuthnTypeBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateExternalDomainDetailsBody struct {
	Req    *types.UpdateExternalDomainDetails         `xml:"urn:sso UpdateExternalDomainDetails,omitempty"`
	Res    *types.UpdateExternalDomainDetailsResponse `xml:"urn:sso UpdateExternalDomainDetailsResponse,omitempty"`
	Fault_ *soap.Fault                                `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateExternalDomainDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateExternalDomainDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateExternalDomainDetails) (*types.UpdateExternalDomainDetailsResponse, error) {
	var reqBody, resBody UpdateExternalDomainDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateLocalGroupDetailsBody struct {
	Req    *types.UpdateLocalGroupDetails         `xml:"urn:sso UpdateLocalGroupDetails,omitempty"`
	Res    *types.UpdateLocalGroupDetailsResponse `xml:"urn:sso UpdateLocalGroupDetailsResponse,omitempty"`
	Fault_ *soap.Fault                            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateLocalGroupDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateLocalGroupDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateLocalGroupDetails) (*types.UpdateLocalGroupDetailsResponse, error) {
	var reqBody, resBody UpdateLocalGroupDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateLocalPasswordPolicyBody struct {
	Req    *types.UpdateLocalPasswordPolicy         `xml:"urn:sso UpdateLocalPasswordPolicy,omitempty"`
	Res    *types.UpdateLocalPasswordPolicyResponse `xml:"urn:sso UpdateLocalPasswordPolicyResponse,omitempty"`
	Fault_ *soap.Fault                              `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateLocalPasswordPolicyBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateLocalPasswordPolicy(ctx context.Context, r soap.RoundTripper, req *types.UpdateLocalPasswordPolicy) (*types.UpdateLocalPasswordPolicyResponse, error) {
	var reqBody, resBody UpdateLocalPasswordPolicyBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateLocalPersonUserDetailsBody struct {
	Req    *types.UpdateLocalPersonUserDetails         `xml:"urn:sso UpdateLocalPersonUserDetails,omitempty"`
	Res    *types.UpdateLocalPersonUserDetailsResponse `xml:"urn:sso UpdateLocalPersonUserDetailsResponse,omitempty"`
	Fault_ *soap.Fault                                 `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateLocalPersonUserDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateLocalPersonUserDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateLocalPersonUserDetails) (*types.UpdateLocalPersonUserDetailsResponse, error) {
	var reqBody, resBody UpdateLocalPersonUserDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateLocalSolutionUserDetailsBody struct {
	Req    *types.UpdateLocalSolutionUserDetails         `xml:"urn:sso UpdateLocalSolutionUserDetails,omitempty"`
	Res    *types.UpdateLocalSolutionUserDetailsResponse `xml:"urn:sso UpdateLocalSolutionUserDetailsResponse,omitempty"`
	Fault_ *soap.Fault                                   `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateLocalSolutionUserDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateLocalSolutionUserDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateLocalSolutionUserDetails) (*types.UpdateLocalSolutionUserDetailsResponse, error) {
	var reqBody, resBody UpdateLocalSolutionUserDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateLockoutPolicyBody struct {
	Req    *types.UpdateLockoutPolicy         `xml:"urn:sso UpdateLockoutPolicy,omitempty"`
	Res    *types.UpdateLockoutPolicyResponse `xml:"urn:sso UpdateLockoutPolicyResponse,omitempty"`
	Fault_ *soap.Fault                        `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateLockoutPolicyBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateLockoutPolicy(ctx context.Context, r soap.RoundTripper, req *types.UpdateLockoutPolicy) (*types.UpdateLockoutPolicyResponse, error) {
	var reqBody, resBody UpdateLockoutPolicyBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdatePasswordExpirationConfigurationBody struct {
	Req    *types.UpdatePasswordExpirationConfiguration         `xml:"urn:sso UpdatePasswordExpirationConfiguration,omitempty"`
	Res    *types.UpdatePasswordExpirationConfigurationResponse `xml:"urn:sso UpdatePasswordExpirationConfigurationResponse,omitempty"`
	Fault_ *soap.Fault                                          `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdatePasswordExpirationConfigurationBody) Fault() *soap.Fault { return b.Fault_ }

func UpdatePasswordExpirationConfiguration(ctx context.Context, r soap.RoundTripper, req *types.UpdatePasswordExpirationConfiguration) (*types.UpdatePasswordExpirationConfigurationResponse, error) {
	var reqBody, resBody UpdatePasswordExpirationConfigurationBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateSelfLocalPersonUserDetailsBody struct {
	Req    *types.UpdateSelfLocalPersonUserDetails         `xml:"urn:sso UpdateSelfLocalPersonUserDetails,omitempty"`
	Res    *types.UpdateSelfLocalPersonUserDetailsResponse `xml:"urn:sso UpdateSelfLocalPersonUserDetailsResponse,omitempty"`
	Fault_ *soap.Fault                                     `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateSelfLocalPersonUserDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateSelfLocalPersonUserDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateSelfLocalPersonUserDetails) (*types.UpdateSelfLocalPersonUserDetailsResponse, error) {
	var reqBody, resBody UpdateSelfLocalPersonUserDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateSelfLocalSolutionUserDetailsBody struct {
	Req    *types.UpdateSelfLocalSolutionUserDetails         `xml:"urn:sso UpdateSelfLocalSolutionUserDetails,omitempty"`
	Res    *types.UpdateSelfLocalSolutionUserDetailsResponse `xml:"urn:sso UpdateSelfLocalSolutionUserDetailsResponse,omitempty"`
	Fault_ *soap.Fault                                       `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateSelfLocalSolutionUserDetailsBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateSelfLocalSolutionUserDetails(ctx context.Context, r soap.RoundTripper, req *types.UpdateSelfLocalSolutionUserDetails) (*types.UpdateSelfLocalSolutionUserDetailsResponse, error) {
	var reqBody, resBody UpdateSelfLocalSolutionUserDetailsBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}

type UpdateSmtpConfigurationBody struct {
	Req    *types.UpdateSmtpConfiguration         `xml:"urn:sso UpdateSmtpConfiguration,omitempty"`
	Res    *types.UpdateSmtpConfigurationResponse `xml:"urn:sso UpdateSmtpConfigurationResponse,omitempty"`
	Fault_ *soap.Fault                            `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *UpdateSmtpConfigurationBody) Fault() *soap.Fault { return b.Fault_ }

func UpdateSmtpConfiguration(ctx context.Context, r soap.RoundTripper, req *types.UpdateSmtpConfiguration) (*types.UpdateSmtpConfigurationResponse, error) {
	var reqBody, resBody UpdateSmtpConfigurationBody

	reqBody.Req = req

	if err := r.RoundTrip(ctx, &reqBody, &resBody); err != nil {
		return nil, err
	}

	return resBody.Res, nil
}
