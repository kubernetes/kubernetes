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

package ssoadmin

import (
	"context"
	"path"
	"reflect"
	"strings"

	"github.com/vmware/govmomi/lookup"
	ltypes "github.com/vmware/govmomi/lookup/types"
	"github.com/vmware/govmomi/ssoadmin/methods"
	"github.com/vmware/govmomi/ssoadmin/types"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	vim "github.com/vmware/govmomi/vim25/types"
)

const (
	Namespace = "sso"
	Version   = "version2"
	Path      = "/sso-adminserver" + vim25.Path
)

var (
	ServiceInstance = vim.ManagedObjectReference{
		Type:  "SsoAdminServiceInstance",
		Value: "SsoAdminServiceInstance",
	}
)

type Client struct {
	*soap.Client

	ServiceContent types.AdminServiceContent
	GroupCheck     types.GroupcheckServiceContent
	Domain         string
	Limit          int32
}

func init() {
	// Fault types are not in the ssoadmin.wsdl
	vim.Add("SsoFaultNotAuthenticated", reflect.TypeOf((*vim.NotAuthenticated)(nil)).Elem())
	vim.Add("SsoFaultNoPermission", reflect.TypeOf((*vim.NoPermission)(nil)).Elem())
	vim.Add("SsoFaultInvalidCredentials", reflect.TypeOf((*vim.InvalidLogin)(nil)).Elem())
	vim.Add("SsoAdminFaultDuplicateSolutionCertificateFaultFault", reflect.TypeOf((*vim.InvalidArgument)(nil)).Elem())
}

func NewClient(ctx context.Context, c *vim25.Client) (*Client, error) {
	filter := &ltypes.LookupServiceRegistrationFilter{
		ServiceType: &ltypes.LookupServiceRegistrationServiceType{
			Product: "com.vmware.cis",
			Type:    "sso:admin",
		},
		EndpointType: &ltypes.LookupServiceRegistrationEndpointType{
			Protocol: "vmomi",
			Type:     "com.vmware.cis.cs.identity.admin",
		},
	}

	url := lookup.EndpointURL(ctx, c, Path, filter)
	sc := c.Client.NewServiceClient(url, Namespace)
	sc.Version = Version

	admin := &Client{
		Client: sc,
		Domain: "vsphere.local", // Default
		Limit:  100,
	}
	if url != Path {
		admin.Domain = path.Base(url)
	}

	{
		req := types.SsoAdminServiceInstance{
			This: ServiceInstance,
		}

		res, err := methods.SsoAdminServiceInstance(ctx, sc, &req)
		if err != nil {
			return nil, err
		}

		admin.ServiceContent = res.Returnval
	}

	{
		req := types.SsoGroupcheckServiceInstance{
			This: vim.ManagedObjectReference{
				Type: "SsoGroupcheckServiceInstance", Value: "ServiceInstance",
			},
		}

		res, err := methods.SsoGroupcheckServiceInstance(ctx, sc, &req)
		if err != nil {
			return nil, err
		}

		admin.GroupCheck = res.Returnval
	}

	return admin, nil
}

func (c *Client) parseID(name string) types.PrincipalId {
	p := strings.SplitN(name, "@", 2)
	id := types.PrincipalId{Name: p[0]}
	if len(p) == 2 {
		id.Domain = p[1]
	} else {
		id.Domain = c.Domain
	}
	return id
}

func (c *Client) CreateSolutionUser(ctx context.Context, name string, details types.AdminSolutionDetails) error {
	req := types.CreateLocalSolutionUser{
		This:        c.ServiceContent.PrincipalManagementService,
		UserName:    name,
		UserDetails: details,
	}

	_, err := methods.CreateLocalSolutionUser(ctx, c, &req)
	return err
}

func (c *Client) UpdateSolutionUser(ctx context.Context, name string, details types.AdminSolutionDetails) error {
	req := types.UpdateLocalSolutionUserDetails{
		This:        c.ServiceContent.PrincipalManagementService,
		UserName:    name,
		UserDetails: details,
	}

	_, err := methods.UpdateLocalSolutionUserDetails(ctx, c, &req)
	return err
}

func (c *Client) DeletePrincipal(ctx context.Context, name string) error {
	req := types.DeleteLocalPrincipal{
		This:          c.ServiceContent.PrincipalManagementService,
		PrincipalName: name,
	}

	_, err := methods.DeleteLocalPrincipal(ctx, c, &req)
	return err
}

func (c *Client) AddUsersToGroup(ctx context.Context, groupName string, userIDs ...types.PrincipalId) error {
	req := types.AddUsersToLocalGroup{
		This:      c.ServiceContent.PrincipalManagementService,
		GroupName: groupName,
		UserIds:   userIDs,
	}

	_, err := methods.AddUsersToLocalGroup(ctx, c, &req)
	return err
}

func (c *Client) CreateGroup(ctx context.Context, name string, details types.AdminGroupDetails) error {
	req := types.CreateLocalGroup{
		This:         c.ServiceContent.PrincipalManagementService,
		GroupName:    name,
		GroupDetails: details,
	}

	_, err := methods.CreateLocalGroup(ctx, c, &req)
	return err
}

func (c *Client) CreatePersonUser(ctx context.Context, name string, details types.AdminPersonDetails, password string) error {
	req := types.CreateLocalPersonUser{
		This:        c.ServiceContent.PrincipalManagementService,
		UserName:    name,
		UserDetails: details,
		Password:    password,
	}

	_, err := methods.CreateLocalPersonUser(ctx, c, &req)
	return err
}

func (c *Client) UpdatePersonUser(ctx context.Context, name string, details types.AdminPersonDetails) error {
	req := types.UpdateLocalPersonUserDetails{
		This:        c.ServiceContent.PrincipalManagementService,
		UserName:    name,
		UserDetails: details,
	}

	_, err := methods.UpdateLocalPersonUserDetails(ctx, c, &req)
	return err
}

func (c *Client) ResetPersonPassword(ctx context.Context, name string, password string) error {
	req := types.ResetLocalPersonUserPassword{
		This:        c.ServiceContent.PrincipalManagementService,
		UserName:    name,
		NewPassword: password,
	}

	_, err := methods.ResetLocalPersonUserPassword(ctx, c, &req)
	return err
}

func (c *Client) FindSolutionUser(ctx context.Context, name string) (*types.AdminSolutionUser, error) {
	req := types.FindSolutionUser{
		This:     c.ServiceContent.PrincipalDiscoveryService,
		UserName: name,
	}

	res, err := methods.FindSolutionUser(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) FindPersonUser(ctx context.Context, name string) (*types.AdminPersonUser, error) {
	req := types.FindPersonUser{
		This:   c.ServiceContent.PrincipalDiscoveryService,
		UserId: c.parseID(name),
	}

	res, err := methods.FindPersonUser(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) FindUser(ctx context.Context, name string) (*types.AdminUser, error) {
	req := types.FindUser{
		This:   c.ServiceContent.PrincipalDiscoveryService,
		UserId: c.parseID(name),
	}

	res, err := methods.FindUser(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) FindSolutionUsers(ctx context.Context, search string) ([]types.AdminSolutionUser, error) {
	req := types.FindSolutionUsers{
		This:         c.ServiceContent.PrincipalDiscoveryService,
		SearchString: search,
		Limit:        c.Limit,
	}

	res, err := methods.FindSolutionUsers(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) FindPersonUsers(ctx context.Context, search string) ([]types.AdminPersonUser, error) {
	req := types.FindPersonUsers{
		This: c.ServiceContent.PrincipalDiscoveryService,
		Criteria: types.AdminPrincipalDiscoveryServiceSearchCriteria{
			Domain:       c.Domain,
			SearchString: search,
		},
		Limit: c.Limit,
	}

	res, err := methods.FindPersonUsers(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (c *Client) FindParentGroups(ctx context.Context, id types.PrincipalId, groups ...types.PrincipalId) ([]types.PrincipalId, error) {
	if len(groups) == 0 {
		req := types.FindAllParentGroups{
			This:   c.GroupCheck.GroupCheckService,
			UserId: id,
		}
		res, err := methods.FindAllParentGroups(ctx, c, &req)
		if err != nil {
			return nil, err
		}
		return res.Returnval, nil
	}

	return nil, nil
}

func (c *Client) Login(ctx context.Context) error {
	req := types.Login{
		This: c.ServiceContent.SessionManager,
	}

	_, err := methods.Login(ctx, c, &req)
	return err
}

func (c *Client) Logout(ctx context.Context) error {
	req := types.Logout{
		This: c.ServiceContent.SessionManager,
	}

	_, err := methods.Logout(ctx, c, &req)
	return err
}

func (c *Client) SetRole(ctx context.Context, id types.PrincipalId, role string) (bool, error) {
	req := types.SetRole{
		This:   c.ServiceContent.RoleManagementService,
		UserId: id,
		Role:   role,
	}

	res, err := methods.SetRole(ctx, c, &req)
	if err != nil {
		return false, err
	}

	return res.Returnval, nil
}

func (c *Client) GrantWSTrustRole(ctx context.Context, id types.PrincipalId, role string) (bool, error) {
	req := types.GrantWSTrustRole{
		This:   c.ServiceContent.RoleManagementService,
		UserId: id,
		Role:   role,
	}

	res, err := methods.GrantWSTrustRole(ctx, c, &req)
	if err != nil {
		return false, err
	}

	return res.Returnval, nil
}

func (c *Client) RevokeWSTrustRole(ctx context.Context, id types.PrincipalId, role string) (bool, error) {
	req := types.RevokeWSTrustRole{
		This:   c.ServiceContent.RoleManagementService,
		UserId: id,
		Role:   role,
	}

	res, err := methods.RevokeWSTrustRole(ctx, c, &req)
	if err != nil {
		return false, err
	}

	return res.Returnval, nil
}
