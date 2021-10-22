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

package types

import (
	"net/url"
	"reflect"

	"github.com/vmware/govmomi/vim25/types"
)

type AddCertificate AddCertificateRequestType

func init() {
	types.Add("sso:AddCertificate", reflect.TypeOf((*AddCertificate)(nil)).Elem())
}

type AddCertificateRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	Certificate string                       `xml:"certificate"`
}

func init() {
	types.Add("sso:AddCertificateRequestType", reflect.TypeOf((*AddCertificateRequestType)(nil)).Elem())
}

type AddCertificateResponse struct {
	Returnval bool `xml:"returnval"`
}

type AddExternalDomain AddExternalDomainRequestType

func init() {
	types.Add("sso:AddExternalDomain", reflect.TypeOf((*AddExternalDomain)(nil)).Elem())
}

type AddExternalDomainRequestType struct {
	This               types.ManagedObjectReference                           `xml:"_this"`
	ServerType         string                                                 `xml:"serverType"`
	DomainName         string                                                 `xml:"domainName"`
	DomainAlias        string                                                 `xml:"domainAlias,omitempty"`
	Details            AdminExternalDomainDetails                             `xml:"details"`
	AuthenticationType string                                                 `xml:"authenticationType"`
	AuthnCredentials   *AdminDomainManagementServiceAuthenticationCredentails `xml:"authnCredentials,omitempty"`
}

func init() {
	types.Add("sso:AddExternalDomainRequestType", reflect.TypeOf((*AddExternalDomainRequestType)(nil)).Elem())
}

type AddExternalDomainResponse struct {
}

type AddGroupToLocalGroup AddGroupToLocalGroupRequestType

func init() {
	types.Add("sso:AddGroupToLocalGroup", reflect.TypeOf((*AddGroupToLocalGroup)(nil)).Elem())
}

type AddGroupToLocalGroupRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	GroupId   PrincipalId                  `xml:"groupId"`
	GroupName string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:AddGroupToLocalGroupRequestType", reflect.TypeOf((*AddGroupToLocalGroupRequestType)(nil)).Elem())
}

type AddGroupToLocalGroupResponse struct {
	Returnval bool `xml:"returnval"`
}

type AddGroupsToLocalGroup AddGroupsToLocalGroupRequestType

func init() {
	types.Add("sso:AddGroupsToLocalGroup", reflect.TypeOf((*AddGroupsToLocalGroup)(nil)).Elem())
}

type AddGroupsToLocalGroupRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	GroupIds  []PrincipalId                `xml:"groupIds"`
	GroupName string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:AddGroupsToLocalGroupRequestType", reflect.TypeOf((*AddGroupsToLocalGroupRequestType)(nil)).Elem())
}

type AddGroupsToLocalGroupResponse struct {
	Returnval []bool `xml:"returnval"`
}

type AddUserToLocalGroup AddUserToLocalGroupRequestType

func init() {
	types.Add("sso:AddUserToLocalGroup", reflect.TypeOf((*AddUserToLocalGroup)(nil)).Elem())
}

type AddUserToLocalGroupRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	UserId    PrincipalId                  `xml:"userId"`
	GroupName string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:AddUserToLocalGroupRequestType", reflect.TypeOf((*AddUserToLocalGroupRequestType)(nil)).Elem())
}

type AddUserToLocalGroupResponse struct {
	Returnval bool `xml:"returnval"`
}

type AddUsersToLocalGroup AddUsersToLocalGroupRequestType

func init() {
	types.Add("sso:AddUsersToLocalGroup", reflect.TypeOf((*AddUsersToLocalGroup)(nil)).Elem())
}

type AddUsersToLocalGroupRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	UserIds   []PrincipalId                `xml:"userIds"`
	GroupName string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:AddUsersToLocalGroupRequestType", reflect.TypeOf((*AddUsersToLocalGroupRequestType)(nil)).Elem())
}

type AddUsersToLocalGroupResponse struct {
	Returnval []bool `xml:"returnval"`
}

type AdminConfigurationManagementServiceCertificateChain struct {
	types.DynamicData

	Certificates []string `xml:"certificates"`
}

func init() {
	types.Add("sso:AdminConfigurationManagementServiceCertificateChain", reflect.TypeOf((*AdminConfigurationManagementServiceCertificateChain)(nil)).Elem())
}

type AdminDomainManagementServiceAuthenticationCredentails struct {
	types.DynamicData

	Username string `xml:"username"`
	Password string `xml:"password"`
}

func init() {
	types.Add("sso:AdminDomainManagementServiceAuthenticationCredentails", reflect.TypeOf((*AdminDomainManagementServiceAuthenticationCredentails)(nil)).Elem())
}

type AdminDomains struct {
	types.DynamicData

	ExternalDomains  []AdminExternalDomain `xml:"externalDomains"`
	SystemDomainName string                `xml:"systemDomainName"`
}

func init() {
	types.Add("sso:AdminDomains", reflect.TypeOf((*AdminDomains)(nil)).Elem())
}

type AdminExternalDomain struct {
	types.DynamicData

	Type                  string                                   `xml:"type"`
	Name                  string                                   `xml:"name"`
	Alias                 string                                   `xml:"alias,omitempty"`
	Details               AdminExternalDomainDetails               `xml:"details"`
	AuthenticationDetails AdminExternalDomainAuthenticationDetails `xml:"authenticationDetails"`
}

func init() {
	types.Add("sso:AdminExternalDomain", reflect.TypeOf((*AdminExternalDomain)(nil)).Elem())
}

type AdminExternalDomainAuthenticationDetails struct {
	types.DynamicData

	AuthenticationType string `xml:"authenticationType"`
	Username           string `xml:"username,omitempty"`
}

func init() {
	types.Add("sso:AdminExternalDomainAuthenticationDetails", reflect.TypeOf((*AdminExternalDomainAuthenticationDetails)(nil)).Elem())
}

type AdminExternalDomainDetails struct {
	types.DynamicData

	FriendlyName         string  `xml:"friendlyName"`
	UserBaseDn           string  `xml:"userBaseDn"`
	GroupBaseDn          string  `xml:"groupBaseDn"`
	PrimaryUrl           url.URL `xml:"primaryUrl"`
	FailoverUrl          url.URL `xml:"failoverUrl,omitempty"`
	SearchTimeoutSeconds int32   `xml:"searchTimeoutSeconds"`
}

func init() {
	types.Add("sso:AdminExternalDomainDetails", reflect.TypeOf((*AdminExternalDomainDetails)(nil)).Elem())
}

type AdminGroup struct {
	types.DynamicData

	Id      PrincipalId       `xml:"id"`
	Alias   *PrincipalId      `xml:"alias,omitempty"`
	Details AdminGroupDetails `xml:"details"`
}

func init() {
	types.Add("sso:AdminGroup", reflect.TypeOf((*AdminGroup)(nil)).Elem())
}

type AdminGroupDetails struct {
	types.DynamicData

	Description string `xml:"description,omitempty"`
}

func init() {
	types.Add("sso:AdminGroupDetails", reflect.TypeOf((*AdminGroupDetails)(nil)).Elem())
}

type AdminLockoutPolicy struct {
	types.DynamicData

	Description              string `xml:"description"`
	MaxFailedAttempts        int32  `xml:"maxFailedAttempts"`
	FailedAttemptIntervalSec int64  `xml:"failedAttemptIntervalSec"`
	AutoUnlockIntervalSec    int64  `xml:"autoUnlockIntervalSec"`
}

func init() {
	types.Add("sso:AdminLockoutPolicy", reflect.TypeOf((*AdminLockoutPolicy)(nil)).Elem())
}

type AdminMailContent struct {
	types.DynamicData

	From    string `xml:"from"`
	To      string `xml:"to"`
	Subject string `xml:"subject"`
	Content string `xml:"content"`
}

func init() {
	types.Add("sso:AdminMailContent", reflect.TypeOf((*AdminMailContent)(nil)).Elem())
}

type AdminPasswordExpirationConfig struct {
	types.DynamicData

	EmailNotificationEnabled bool    `xml:"emailNotificationEnabled"`
	EmailFrom                string  `xml:"emailFrom,omitempty"`
	EmailSubject             string  `xml:"emailSubject,omitempty"`
	NotificationDays         []int32 `xml:"notificationDays,omitempty"`
}

func init() {
	types.Add("sso:AdminPasswordExpirationConfig", reflect.TypeOf((*AdminPasswordExpirationConfig)(nil)).Elem())
}

type AdminPasswordFormat struct {
	types.DynamicData

	LengthRestriction              AdminPasswordFormatLengthRestriction     `xml:"lengthRestriction"`
	AlphabeticRestriction          AdminPasswordFormatAlphabeticRestriction `xml:"alphabeticRestriction"`
	MinNumericCount                int32                                    `xml:"minNumericCount"`
	MinSpecialCharCount            int32                                    `xml:"minSpecialCharCount"`
	MaxIdenticalAdjacentCharacters int32                                    `xml:"maxIdenticalAdjacentCharacters"`
}

func init() {
	types.Add("sso:AdminPasswordFormat", reflect.TypeOf((*AdminPasswordFormat)(nil)).Elem())
}

type AdminPasswordFormatAlphabeticRestriction struct {
	types.DynamicData

	MinAlphabeticCount int32 `xml:"minAlphabeticCount"`
	MinUppercaseCount  int32 `xml:"minUppercaseCount"`
	MinLowercaseCount  int32 `xml:"minLowercaseCount"`
}

func init() {
	types.Add("sso:AdminPasswordFormatAlphabeticRestriction", reflect.TypeOf((*AdminPasswordFormatAlphabeticRestriction)(nil)).Elem())
}

type AdminPasswordFormatLengthRestriction struct {
	types.DynamicData

	MinLength int32 `xml:"minLength"`
	MaxLength int32 `xml:"maxLength"`
}

func init() {
	types.Add("sso:AdminPasswordFormatLengthRestriction", reflect.TypeOf((*AdminPasswordFormatLengthRestriction)(nil)).Elem())
}

type AdminPasswordPolicy struct {
	types.DynamicData

	Description                      string              `xml:"description"`
	ProhibitedPreviousPasswordsCount int32               `xml:"prohibitedPreviousPasswordsCount"`
	PasswordFormat                   AdminPasswordFormat `xml:"passwordFormat"`
	PasswordLifetimeDays             int32               `xml:"passwordLifetimeDays,omitempty"`
}

func init() {
	types.Add("sso:AdminPasswordPolicy", reflect.TypeOf((*AdminPasswordPolicy)(nil)).Elem())
}

type AdminPersonDetails struct {
	types.DynamicData

	Description  string `xml:"description,omitempty"`
	EmailAddress string `xml:"emailAddress,omitempty"`
	FirstName    string `xml:"firstName,omitempty"`
	LastName     string `xml:"lastName,omitempty"`
}

func init() {
	types.Add("sso:AdminPersonDetails", reflect.TypeOf((*AdminPersonDetails)(nil)).Elem())
}

type AdminPersonUser struct {
	types.DynamicData

	Id       PrincipalId        `xml:"id"`
	Alias    *PrincipalId       `xml:"alias,omitempty"`
	Details  AdminPersonDetails `xml:"details"`
	Disabled bool               `xml:"disabled"`
	Locked   bool               `xml:"locked"`
}

func init() {
	types.Add("sso:AdminPersonUser", reflect.TypeOf((*AdminPersonUser)(nil)).Elem())
}

type AdminPrincipalDiscoveryServiceSearchCriteria struct {
	types.DynamicData

	SearchString string `xml:"searchString"`
	Domain       string `xml:"domain"`
}

func init() {
	types.Add("sso:AdminPrincipalDiscoveryServiceSearchCriteria", reflect.TypeOf((*AdminPrincipalDiscoveryServiceSearchCriteria)(nil)).Elem())
}

type AdminPrincipalDiscoveryServiceSearchResult struct {
	types.DynamicData

	PersonUsers   []AdminPersonUser   `xml:"personUsers,omitempty"`
	SolutionUsers []AdminSolutionUser `xml:"solutionUsers,omitempty"`
	Groups        []AdminGroup        `xml:"groups,omitempty"`
}

func init() {
	types.Add("sso:AdminPrincipalDiscoveryServiceSearchResult", reflect.TypeOf((*AdminPrincipalDiscoveryServiceSearchResult)(nil)).Elem())
}

type AdminServiceContent struct {
	types.DynamicData

	SessionManager                  types.ManagedObjectReference `xml:"sessionManager"`
	ConfigurationManagementService  types.ManagedObjectReference `xml:"configurationManagementService"`
	SmtpManagementService           types.ManagedObjectReference `xml:"smtpManagementService"`
	PrincipalDiscoveryService       types.ManagedObjectReference `xml:"principalDiscoveryService"`
	PrincipalManagementService      types.ManagedObjectReference `xml:"principalManagementService"`
	RoleManagementService           types.ManagedObjectReference `xml:"roleManagementService"`
	PasswordPolicyService           types.ManagedObjectReference `xml:"passwordPolicyService"`
	LockoutPolicyService            types.ManagedObjectReference `xml:"lockoutPolicyService"`
	DomainManagementService         types.ManagedObjectReference `xml:"domainManagementService"`
	IdentitySourceManagementService types.ManagedObjectReference `xml:"identitySourceManagementService"`
	SystemManagementService         types.ManagedObjectReference `xml:"systemManagementService"`
	ComputerManagementService       types.ManagedObjectReference `xml:"computerManagementService"`
	SsoHealthManagementService      types.ManagedObjectReference `xml:"ssoHealthManagementService"`
	DeploymentInformationService    types.ManagedObjectReference `xml:"deploymentInformationService"`
	ReplicationService              types.ManagedObjectReference `xml:"replicationService"`
}

func init() {
	types.Add("sso:AdminServiceContent", reflect.TypeOf((*AdminServiceContent)(nil)).Elem())
}

type AdminSmtpConfig struct {
	types.DynamicData

	Host         string `xml:"host,omitempty"`
	Port         int32  `xml:"port,omitempty"`
	Authenticate *bool  `xml:"authenticate"`
	User         string `xml:"user,omitempty"`
	Password     string `xml:"password,omitempty"`
}

func init() {
	types.Add("sso:AdminSmtpConfig", reflect.TypeOf((*AdminSmtpConfig)(nil)).Elem())
}

type AdminSolutionDetails struct {
	types.DynamicData

	Description string `xml:"description,omitempty"`
	Certificate string `xml:"certificate"`
}

func init() {
	types.Add("sso:AdminSolutionDetails", reflect.TypeOf((*AdminSolutionDetails)(nil)).Elem())
}

type AdminSolutionUser struct {
	types.DynamicData

	Id       PrincipalId          `xml:"id"`
	Alias    *PrincipalId         `xml:"alias,omitempty"`
	Details  AdminSolutionDetails `xml:"details"`
	Disabled bool                 `xml:"disabled"`
}

func init() {
	types.Add("sso:AdminSolutionUser", reflect.TypeOf((*AdminSolutionUser)(nil)).Elem())
}

type AdminUser struct {
	types.DynamicData

	Id          PrincipalId  `xml:"id"`
	Alias       *PrincipalId `xml:"alias,omitempty"`
	Kind        string       `xml:"kind"`
	Description string       `xml:"description,omitempty"`
}

func init() {
	types.Add("sso:AdminUser", reflect.TypeOf((*AdminUser)(nil)).Elem())
}

type CreateLocalGroup CreateLocalGroupRequestType

func init() {
	types.Add("sso:CreateLocalGroup", reflect.TypeOf((*CreateLocalGroup)(nil)).Elem())
}

type CreateLocalGroupRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupName    string                       `xml:"groupName"`
	GroupDetails AdminGroupDetails            `xml:"groupDetails"`
}

func init() {
	types.Add("sso:CreateLocalGroupRequestType", reflect.TypeOf((*CreateLocalGroupRequestType)(nil)).Elem())
}

type CreateLocalGroupResponse struct {
}

type CreateLocalPersonUser CreateLocalPersonUserRequestType

func init() {
	types.Add("sso:CreateLocalPersonUser", reflect.TypeOf((*CreateLocalPersonUser)(nil)).Elem())
}

type CreateLocalPersonUserRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserName    string                       `xml:"userName"`
	UserDetails AdminPersonDetails           `xml:"userDetails"`
	Password    string                       `xml:"password"`
}

func init() {
	types.Add("sso:CreateLocalPersonUserRequestType", reflect.TypeOf((*CreateLocalPersonUserRequestType)(nil)).Elem())
}

type CreateLocalPersonUserResponse struct {
}

type CreateLocalSolutionUser CreateLocalSolutionUserRequestType

func init() {
	types.Add("sso:CreateLocalSolutionUser", reflect.TypeOf((*CreateLocalSolutionUser)(nil)).Elem())
}

type CreateLocalSolutionUserRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserName    string                       `xml:"userName"`
	UserDetails AdminSolutionDetails         `xml:"userDetails"`
}

func init() {
	types.Add("sso:CreateLocalSolutionUserRequestType", reflect.TypeOf((*CreateLocalSolutionUserRequestType)(nil)).Elem())
}

type CreateLocalSolutionUserResponse struct {
}

type DeleteCertificate DeleteCertificateRequestType

func init() {
	types.Add("sso:DeleteCertificate", reflect.TypeOf((*DeleteCertificate)(nil)).Elem())
}

type DeleteCertificateRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	Fingerprint string                       `xml:"fingerprint"`
}

func init() {
	types.Add("sso:DeleteCertificateRequestType", reflect.TypeOf((*DeleteCertificateRequestType)(nil)).Elem())
}

type DeleteCertificateResponse struct {
	Returnval bool `xml:"returnval"`
}

type DeleteDomain DeleteDomainRequestType

func init() {
	types.Add("sso:DeleteDomain", reflect.TypeOf((*DeleteDomain)(nil)).Elem())
}

type DeleteDomainRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
	Name string                       `xml:"name"`
}

func init() {
	types.Add("sso:DeleteDomainRequestType", reflect.TypeOf((*DeleteDomainRequestType)(nil)).Elem())
}

type DeleteDomainResponse struct {
}

type DeleteLocalPrincipal DeleteLocalPrincipalRequestType

func init() {
	types.Add("sso:DeleteLocalPrincipal", reflect.TypeOf((*DeleteLocalPrincipal)(nil)).Elem())
}

type DeleteLocalPrincipalRequestType struct {
	This          types.ManagedObjectReference `xml:"_this"`
	PrincipalName string                       `xml:"principalName"`
}

func init() {
	types.Add("sso:DeleteLocalPrincipalRequestType", reflect.TypeOf((*DeleteLocalPrincipalRequestType)(nil)).Elem())
}

type DeleteLocalPrincipalResponse struct {
}

type DisableUserAccount DisableUserAccountRequestType

func init() {
	types.Add("sso:DisableUserAccount", reflect.TypeOf((*DisableUserAccount)(nil)).Elem())
}

type DisableUserAccountRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:DisableUserAccountRequestType", reflect.TypeOf((*DisableUserAccountRequestType)(nil)).Elem())
}

type DisableUserAccountResponse struct {
	Returnval bool `xml:"returnval"`
}

type EnableUserAccount EnableUserAccountRequestType

func init() {
	types.Add("sso:EnableUserAccount", reflect.TypeOf((*EnableUserAccount)(nil)).Elem())
}

type EnableUserAccountRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:EnableUserAccountRequestType", reflect.TypeOf((*EnableUserAccountRequestType)(nil)).Elem())
}

type EnableUserAccountResponse struct {
	Returnval bool `xml:"returnval"`
}

type Find FindRequestType

func init() {
	types.Add("sso:Find", reflect.TypeOf((*Find)(nil)).Elem())
}

type FindAllParentGroups FindAllParentGroupsRequestType

func init() {
	types.Add("sso:FindAllParentGroups", reflect.TypeOf((*FindAllParentGroups)(nil)).Elem())
}

type FindAllParentGroupsRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:FindAllParentGroupsRequestType", reflect.TypeOf((*FindAllParentGroupsRequestType)(nil)).Elem())
}

type FindAllParentGroupsResponse struct {
	Returnval []PrincipalId `xml:"returnval,omitempty"`
}

type FindCertificate FindCertificateRequestType

func init() {
	types.Add("sso:FindCertificate", reflect.TypeOf((*FindCertificate)(nil)).Elem())
}

type FindCertificateRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	Fingerprint string                       `xml:"fingerprint"`
}

func init() {
	types.Add("sso:FindCertificateRequestType", reflect.TypeOf((*FindCertificateRequestType)(nil)).Elem())
}

type FindCertificateResponse struct {
	Returnval string `xml:"returnval,omitempty"`
}

type FindDirectParentGroups FindDirectParentGroupsRequestType

func init() {
	types.Add("sso:FindDirectParentGroups", reflect.TypeOf((*FindDirectParentGroups)(nil)).Elem())
}

type FindDirectParentGroupsRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	PrincipalId PrincipalId                  `xml:"principalId"`
}

func init() {
	types.Add("sso:FindDirectParentGroupsRequestType", reflect.TypeOf((*FindDirectParentGroupsRequestType)(nil)).Elem())
}

type FindDirectParentGroupsResponse struct {
	Returnval []AdminGroup `xml:"returnval,omitempty"`
}

type FindDisabledPersonUsers FindDisabledPersonUsersRequestType

func init() {
	types.Add("sso:FindDisabledPersonUsers", reflect.TypeOf((*FindDisabledPersonUsers)(nil)).Elem())
}

type FindDisabledPersonUsersRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	SearchStr string                       `xml:"searchStr"`
	Limit     int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindDisabledPersonUsersRequestType", reflect.TypeOf((*FindDisabledPersonUsersRequestType)(nil)).Elem())
}

type FindDisabledPersonUsersResponse struct {
	Returnval []AdminPersonUser `xml:"returnval,omitempty"`
}

type FindDisabledSolutionUsers FindDisabledSolutionUsersRequestType

func init() {
	types.Add("sso:FindDisabledSolutionUsers", reflect.TypeOf((*FindDisabledSolutionUsers)(nil)).Elem())
}

type FindDisabledSolutionUsersRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	SearchStr string                       `xml:"searchStr"`
}

func init() {
	types.Add("sso:FindDisabledSolutionUsersRequestType", reflect.TypeOf((*FindDisabledSolutionUsersRequestType)(nil)).Elem())
}

type FindDisabledSolutionUsersResponse struct {
	Returnval []AdminSolutionUser `xml:"returnval,omitempty"`
}

type FindExternalDomain FindExternalDomainRequestType

func init() {
	types.Add("sso:FindExternalDomain", reflect.TypeOf((*FindExternalDomain)(nil)).Elem())
}

type FindExternalDomainRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Filter string                       `xml:"filter"`
}

func init() {
	types.Add("sso:FindExternalDomainRequestType", reflect.TypeOf((*FindExternalDomainRequestType)(nil)).Elem())
}

type FindExternalDomainResponse struct {
	Returnval *AdminExternalDomain `xml:"returnval,omitempty"`
}

type FindGroup FindGroupRequestType

func init() {
	types.Add("sso:FindGroup", reflect.TypeOf((*FindGroup)(nil)).Elem())
}

type FindGroupRequestType struct {
	This    types.ManagedObjectReference `xml:"_this"`
	GroupId PrincipalId                  `xml:"groupId"`
}

func init() {
	types.Add("sso:FindGroupRequestType", reflect.TypeOf((*FindGroupRequestType)(nil)).Elem())
}

type FindGroupResponse struct {
	Returnval *AdminGroup `xml:"returnval,omitempty"`
}

type FindGroups FindGroupsRequestType

func init() {
	types.Add("sso:FindGroups", reflect.TypeOf((*FindGroups)(nil)).Elem())
}

type FindGroupsInGroup FindGroupsInGroupRequestType

func init() {
	types.Add("sso:FindGroupsInGroup", reflect.TypeOf((*FindGroupsInGroup)(nil)).Elem())
}

type FindGroupsInGroupRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupId      PrincipalId                  `xml:"groupId"`
	SearchString string                       `xml:"searchString"`
	Limit        int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindGroupsInGroupRequestType", reflect.TypeOf((*FindGroupsInGroupRequestType)(nil)).Elem())
}

type FindGroupsInGroupResponse struct {
	Returnval []AdminGroup `xml:"returnval,omitempty"`
}

type FindGroupsRequestType struct {
	This     types.ManagedObjectReference                 `xml:"_this"`
	Criteria AdminPrincipalDiscoveryServiceSearchCriteria `xml:"criteria"`
	Limit    int32                                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindGroupsRequestType", reflect.TypeOf((*FindGroupsRequestType)(nil)).Elem())
}

type FindGroupsResponse struct {
	Returnval []AdminGroup `xml:"returnval,omitempty"`
}

type FindLockedUsers FindLockedUsersRequestType

func init() {
	types.Add("sso:FindLockedUsers", reflect.TypeOf((*FindLockedUsers)(nil)).Elem())
}

type FindLockedUsersRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	SearchStr string                       `xml:"searchStr"`
	Limit     int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindLockedUsersRequestType", reflect.TypeOf((*FindLockedUsersRequestType)(nil)).Elem())
}

type FindLockedUsersResponse struct {
	Returnval []AdminPersonUser `xml:"returnval,omitempty"`
}

type FindNestedParentGroups FindNestedParentGroupsRequestType

func init() {
	types.Add("sso:FindNestedParentGroups", reflect.TypeOf((*FindNestedParentGroups)(nil)).Elem())
}

type FindNestedParentGroupsRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:FindNestedParentGroupsRequestType", reflect.TypeOf((*FindNestedParentGroupsRequestType)(nil)).Elem())
}

type FindNestedParentGroupsResponse struct {
	Returnval []AdminGroup `xml:"returnval,omitempty"`
}

type FindParentGroups FindParentGroupsRequestType

func init() {
	types.Add("sso:FindParentGroups", reflect.TypeOf((*FindParentGroups)(nil)).Elem())
}

type FindParentGroupsRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	UserId    PrincipalId                  `xml:"userId"`
	GroupList []PrincipalId                `xml:"groupList,omitempty"`
}

func init() {
	types.Add("sso:FindParentGroupsRequestType", reflect.TypeOf((*FindParentGroupsRequestType)(nil)).Elem())
}

type FindParentGroupsResponse struct {
	Returnval []PrincipalId `xml:"returnval,omitempty"`
}

type FindPersonUser FindPersonUserRequestType

func init() {
	types.Add("sso:FindPersonUser", reflect.TypeOf((*FindPersonUser)(nil)).Elem())
}

type FindPersonUserRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:FindPersonUserRequestType", reflect.TypeOf((*FindPersonUserRequestType)(nil)).Elem())
}

type FindPersonUserResponse struct {
	Returnval *AdminPersonUser `xml:"returnval,omitempty"`
}

type FindPersonUsers FindPersonUsersRequestType

func init() {
	types.Add("sso:FindPersonUsers", reflect.TypeOf((*FindPersonUsers)(nil)).Elem())
}

type FindPersonUsersInGroup FindPersonUsersInGroupRequestType

func init() {
	types.Add("sso:FindPersonUsersInGroup", reflect.TypeOf((*FindPersonUsersInGroup)(nil)).Elem())
}

type FindPersonUsersInGroupRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupId      PrincipalId                  `xml:"groupId"`
	SearchString string                       `xml:"searchString"`
	Limit        int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindPersonUsersInGroupRequestType", reflect.TypeOf((*FindPersonUsersInGroupRequestType)(nil)).Elem())
}

type FindPersonUsersInGroupResponse struct {
	Returnval []AdminPersonUser `xml:"returnval,omitempty"`
}

type FindPersonUsersRequestType struct {
	This     types.ManagedObjectReference                 `xml:"_this"`
	Criteria AdminPrincipalDiscoveryServiceSearchCriteria `xml:"criteria"`
	Limit    int32                                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindPersonUsersRequestType", reflect.TypeOf((*FindPersonUsersRequestType)(nil)).Elem())
}

type FindPersonUsersResponse struct {
	Returnval []AdminPersonUser `xml:"returnval,omitempty"`
}

type FindRequestType struct {
	This     types.ManagedObjectReference                 `xml:"_this"`
	Criteria AdminPrincipalDiscoveryServiceSearchCriteria `xml:"criteria"`
	Limit    int32                                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindRequestType", reflect.TypeOf((*FindRequestType)(nil)).Elem())
}

type FindResponse struct {
	Returnval AdminPrincipalDiscoveryServiceSearchResult `xml:"returnval"`
}

type FindSolutionUser FindSolutionUserRequestType

func init() {
	types.Add("sso:FindSolutionUser", reflect.TypeOf((*FindSolutionUser)(nil)).Elem())
}

type FindSolutionUserRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	UserName string                       `xml:"userName"`
}

func init() {
	types.Add("sso:FindSolutionUserRequestType", reflect.TypeOf((*FindSolutionUserRequestType)(nil)).Elem())
}

type FindSolutionUserResponse struct {
	Returnval *AdminSolutionUser `xml:"returnval,omitempty"`
}

type FindSolutionUsers FindSolutionUsersRequestType

func init() {
	types.Add("sso:FindSolutionUsers", reflect.TypeOf((*FindSolutionUsers)(nil)).Elem())
}

type FindSolutionUsersInGroup FindSolutionUsersInGroupRequestType

func init() {
	types.Add("sso:FindSolutionUsersInGroup", reflect.TypeOf((*FindSolutionUsersInGroup)(nil)).Elem())
}

type FindSolutionUsersInGroupRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupName    string                       `xml:"groupName"`
	SearchString string                       `xml:"searchString"`
	Limit        int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindSolutionUsersInGroupRequestType", reflect.TypeOf((*FindSolutionUsersInGroupRequestType)(nil)).Elem())
}

type FindSolutionUsersInGroupResponse struct {
	Returnval []AdminSolutionUser `xml:"returnval,omitempty"`
}

type FindSolutionUsersRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	SearchString string                       `xml:"searchString"`
	Limit        int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindSolutionUsersRequestType", reflect.TypeOf((*FindSolutionUsersRequestType)(nil)).Elem())
}

type FindSolutionUsersResponse struct {
	Returnval []AdminSolutionUser `xml:"returnval,omitempty"`
}

type FindUser FindUserRequestType

func init() {
	types.Add("sso:FindUser", reflect.TypeOf((*FindUser)(nil)).Elem())
}

type FindUserRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:FindUserRequestType", reflect.TypeOf((*FindUserRequestType)(nil)).Elem())
}

type FindUserResponse struct {
	Returnval *AdminUser `xml:"returnval,omitempty"`
}

type FindUsers FindUsersRequestType

func init() {
	types.Add("sso:FindUsers", reflect.TypeOf((*FindUsers)(nil)).Elem())
}

type FindUsersInGroup FindUsersInGroupRequestType

func init() {
	types.Add("sso:FindUsersInGroup", reflect.TypeOf((*FindUsersInGroup)(nil)).Elem())
}

type FindUsersInGroupRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupId      PrincipalId                  `xml:"groupId"`
	SearchString string                       `xml:"searchString"`
	Limit        int32                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindUsersInGroupRequestType", reflect.TypeOf((*FindUsersInGroupRequestType)(nil)).Elem())
}

type FindUsersInGroupResponse struct {
	Returnval []AdminUser `xml:"returnval,omitempty"`
}

type FindUsersRequestType struct {
	This     types.ManagedObjectReference                 `xml:"_this"`
	Criteria AdminPrincipalDiscoveryServiceSearchCriteria `xml:"criteria"`
	Limit    int32                                        `xml:"limit"`
}

func init() {
	types.Add("sso:FindUsersRequestType", reflect.TypeOf((*FindUsersRequestType)(nil)).Elem())
}

type FindUsersResponse struct {
	Returnval []AdminUser `xml:"returnval,omitempty"`
}

type GetAllCertificates GetAllCertificatesRequestType

func init() {
	types.Add("sso:GetAllCertificates", reflect.TypeOf((*GetAllCertificates)(nil)).Elem())
}

type GetAllCertificatesRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetAllCertificatesRequestType", reflect.TypeOf((*GetAllCertificatesRequestType)(nil)).Elem())
}

type GetAllCertificatesResponse struct {
	Returnval []string `xml:"returnval,omitempty"`
}

type GetClockTolerance GetClockToleranceRequestType

func init() {
	types.Add("sso:GetClockTolerance", reflect.TypeOf((*GetClockTolerance)(nil)).Elem())
}

type GetClockToleranceRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetClockToleranceRequestType", reflect.TypeOf((*GetClockToleranceRequestType)(nil)).Elem())
}

type GetClockToleranceResponse struct {
	Returnval int64 `xml:"returnval"`
}

type GetDelegationCount GetDelegationCountRequestType

func init() {
	types.Add("sso:GetDelegationCount", reflect.TypeOf((*GetDelegationCount)(nil)).Elem())
}

type GetDelegationCountRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetDelegationCountRequestType", reflect.TypeOf((*GetDelegationCountRequestType)(nil)).Elem())
}

type GetDelegationCountResponse struct {
	Returnval int32 `xml:"returnval"`
}

type GetDomains GetDomainsRequestType

func init() {
	types.Add("sso:GetDomains", reflect.TypeOf((*GetDomains)(nil)).Elem())
}

type GetDomainsRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetDomainsRequestType", reflect.TypeOf((*GetDomainsRequestType)(nil)).Elem())
}

type GetDomainsResponse struct {
	Returnval *AdminDomains `xml:"returnval,omitempty"`
}

type GetIssuersCertificates GetIssuersCertificatesRequestType

func init() {
	types.Add("sso:GetIssuersCertificates", reflect.TypeOf((*GetIssuersCertificates)(nil)).Elem())
}

type GetIssuersCertificatesRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetIssuersCertificatesRequestType", reflect.TypeOf((*GetIssuersCertificatesRequestType)(nil)).Elem())
}

type GetIssuersCertificatesResponse struct {
	Returnval []string `xml:"returnval"`
}

type GetKnownCertificateChains GetKnownCertificateChainsRequestType

func init() {
	types.Add("sso:GetKnownCertificateChains", reflect.TypeOf((*GetKnownCertificateChains)(nil)).Elem())
}

type GetKnownCertificateChainsRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetKnownCertificateChainsRequestType", reflect.TypeOf((*GetKnownCertificateChainsRequestType)(nil)).Elem())
}

type GetKnownCertificateChainsResponse struct {
	Returnval []AdminConfigurationManagementServiceCertificateChain `xml:"returnval"`
}

type GetLocalPasswordPolicy GetLocalPasswordPolicyRequestType

func init() {
	types.Add("sso:GetLocalPasswordPolicy", reflect.TypeOf((*GetLocalPasswordPolicy)(nil)).Elem())
}

type GetLocalPasswordPolicyRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetLocalPasswordPolicyRequestType", reflect.TypeOf((*GetLocalPasswordPolicyRequestType)(nil)).Elem())
}

type GetLocalPasswordPolicyResponse struct {
	Returnval AdminPasswordPolicy `xml:"returnval"`
}

type GetLockoutPolicy GetLockoutPolicyRequestType

func init() {
	types.Add("sso:GetLockoutPolicy", reflect.TypeOf((*GetLockoutPolicy)(nil)).Elem())
}

type GetLockoutPolicyRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetLockoutPolicyRequestType", reflect.TypeOf((*GetLockoutPolicyRequestType)(nil)).Elem())
}

type GetLockoutPolicyResponse struct {
	Returnval AdminLockoutPolicy `xml:"returnval"`
}

type GetMaximumBearerTokenLifetime GetMaximumBearerTokenLifetimeRequestType

func init() {
	types.Add("sso:GetMaximumBearerTokenLifetime", reflect.TypeOf((*GetMaximumBearerTokenLifetime)(nil)).Elem())
}

type GetMaximumBearerTokenLifetimeRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetMaximumBearerTokenLifetimeRequestType", reflect.TypeOf((*GetMaximumBearerTokenLifetimeRequestType)(nil)).Elem())
}

type GetMaximumBearerTokenLifetimeResponse struct {
	Returnval int64 `xml:"returnval"`
}

type GetMaximumHoKTokenLifetime GetMaximumHoKTokenLifetimeRequestType

func init() {
	types.Add("sso:GetMaximumHoKTokenLifetime", reflect.TypeOf((*GetMaximumHoKTokenLifetime)(nil)).Elem())
}

type GetMaximumHoKTokenLifetimeRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetMaximumHoKTokenLifetimeRequestType", reflect.TypeOf((*GetMaximumHoKTokenLifetimeRequestType)(nil)).Elem())
}

type GetMaximumHoKTokenLifetimeResponse struct {
	Returnval int64 `xml:"returnval"`
}

type GetPasswordExpirationConfiguration GetPasswordExpirationConfigurationRequestType

func init() {
	types.Add("sso:GetPasswordExpirationConfiguration", reflect.TypeOf((*GetPasswordExpirationConfiguration)(nil)).Elem())
}

type GetPasswordExpirationConfigurationRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetPasswordExpirationConfigurationRequestType", reflect.TypeOf((*GetPasswordExpirationConfigurationRequestType)(nil)).Elem())
}

type GetPasswordExpirationConfigurationResponse struct {
	Returnval AdminPasswordExpirationConfig `xml:"returnval"`
}

type GetRenewCount GetRenewCountRequestType

func init() {
	types.Add("sso:GetRenewCount", reflect.TypeOf((*GetRenewCount)(nil)).Elem())
}

type GetRenewCountRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetRenewCountRequestType", reflect.TypeOf((*GetRenewCountRequestType)(nil)).Elem())
}

type GetRenewCountResponse struct {
	Returnval int32 `xml:"returnval"`
}

type GetSmtpConfiguration GetSmtpConfigurationRequestType

func init() {
	types.Add("sso:GetSmtpConfiguration", reflect.TypeOf((*GetSmtpConfiguration)(nil)).Elem())
}

type GetSmtpConfigurationRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetSmtpConfigurationRequestType", reflect.TypeOf((*GetSmtpConfigurationRequestType)(nil)).Elem())
}

type GetSmtpConfigurationResponse struct {
	Returnval AdminSmtpConfig `xml:"returnval"`
}

type GetSslCertificateManager GetSslCertificateManagerRequestType

func init() {
	types.Add("sso:GetSslCertificateManager", reflect.TypeOf((*GetSslCertificateManager)(nil)).Elem())
}

type GetSslCertificateManagerRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetSslCertificateManagerRequestType", reflect.TypeOf((*GetSslCertificateManagerRequestType)(nil)).Elem())
}

type GetSslCertificateManagerResponse struct {
	Returnval types.ManagedObjectReference `xml:"returnval"`
}

type GetSystemDomainName GetSystemDomainNameRequestType

func init() {
	types.Add("sso:GetSystemDomainName", reflect.TypeOf((*GetSystemDomainName)(nil)).Elem())
}

type GetSystemDomainNameRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetSystemDomainNameRequestType", reflect.TypeOf((*GetSystemDomainNameRequestType)(nil)).Elem())
}

type GetSystemDomainNameResponse struct {
	Returnval string `xml:"returnval"`
}

type GetTrustedCertificates GetTrustedCertificatesRequestType

func init() {
	types.Add("sso:GetTrustedCertificates", reflect.TypeOf((*GetTrustedCertificates)(nil)).Elem())
}

type GetTrustedCertificatesRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:GetTrustedCertificatesRequestType", reflect.TypeOf((*GetTrustedCertificatesRequestType)(nil)).Elem())
}

type GetTrustedCertificatesResponse struct {
	Returnval []string `xml:"returnval"`
}

type GroupcheckServiceContent struct {
	types.DynamicData

	SessionManager    types.ManagedObjectReference `xml:"sessionManager"`
	GroupCheckService types.ManagedObjectReference `xml:"groupCheckService"`
}

func init() {
	types.Add("sso:GroupcheckServiceContent", reflect.TypeOf((*GroupcheckServiceContent)(nil)).Elem())
}

type HasAdministratorRole HasAdministratorRoleRequestType

func init() {
	types.Add("sso:HasAdministratorRole", reflect.TypeOf((*HasAdministratorRole)(nil)).Elem())
}

type HasAdministratorRoleRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:HasAdministratorRoleRequestType", reflect.TypeOf((*HasAdministratorRoleRequestType)(nil)).Elem())
}

type HasAdministratorRoleResponse struct {
	Returnval bool `xml:"returnval"`
}

type HasRegularUserRole HasRegularUserRoleRequestType

func init() {
	types.Add("sso:HasRegularUserRole", reflect.TypeOf((*HasRegularUserRole)(nil)).Elem())
}

type HasRegularUserRoleRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:HasRegularUserRoleRequestType", reflect.TypeOf((*HasRegularUserRoleRequestType)(nil)).Elem())
}

type HasRegularUserRoleResponse struct {
	Returnval bool `xml:"returnval"`
}

type IsMemberOfGroup IsMemberOfGroupRequestType

func init() {
	types.Add("sso:IsMemberOfGroup", reflect.TypeOf((*IsMemberOfGroup)(nil)).Elem())
}

type IsMemberOfGroupRequestType struct {
	This    types.ManagedObjectReference `xml:"_this"`
	UserId  PrincipalId                  `xml:"userId"`
	GroupId PrincipalId                  `xml:"groupId"`
}

func init() {
	types.Add("sso:IsMemberOfGroupRequestType", reflect.TypeOf((*IsMemberOfGroupRequestType)(nil)).Elem())
}

type IsMemberOfGroupResponse struct {
	Returnval bool `xml:"returnval"`
}

type Login LoginRequestType

func init() {
	types.Add("sso:Login", reflect.TypeOf((*Login)(nil)).Elem())
}

type LoginRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:LoginRequestType", reflect.TypeOf((*LoginRequestType)(nil)).Elem())
}

type LoginResponse struct {
}

type Logout LogoutRequestType

func init() {
	types.Add("sso:Logout", reflect.TypeOf((*Logout)(nil)).Elem())
}

type LogoutRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:LogoutRequestType", reflect.TypeOf((*LogoutRequestType)(nil)).Elem())
}

type LogoutResponse struct {
}

type PrincipalId struct {
	types.DynamicData

	Name   string `xml:"name"`
	Domain string `xml:"domain"`
}

func init() {
	types.Add("sso:PrincipalId", reflect.TypeOf((*PrincipalId)(nil)).Elem())
}

type ProbeConnectivity ProbeConnectivityRequestType

func init() {
	types.Add("sso:ProbeConnectivity", reflect.TypeOf((*ProbeConnectivity)(nil)).Elem())
}

type ProbeConnectivityRequestType struct {
	This               types.ManagedObjectReference                           `xml:"_this"`
	ServiceUri         url.URL                                                `xml:"serviceUri"`
	AuthenticationType string                                                 `xml:"authenticationType"`
	AuthnCredentials   *AdminDomainManagementServiceAuthenticationCredentails `xml:"authnCredentials,omitempty"`
}

func init() {
	types.Add("sso:ProbeConnectivityRequestType", reflect.TypeOf((*ProbeConnectivityRequestType)(nil)).Elem())
}

type ProbeConnectivityResponse struct {
}

type RemoveFromLocalGroup RemoveFromLocalGroupRequestType

func init() {
	types.Add("sso:RemoveFromLocalGroup", reflect.TypeOf((*RemoveFromLocalGroup)(nil)).Elem())
}

type RemoveFromLocalGroupRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	PrincipalId PrincipalId                  `xml:"principalId"`
	GroupName   string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:RemoveFromLocalGroupRequestType", reflect.TypeOf((*RemoveFromLocalGroupRequestType)(nil)).Elem())
}

type RemoveFromLocalGroupResponse struct {
	Returnval bool `xml:"returnval"`
}

type RemovePrincipalsFromLocalGroup RemovePrincipalsFromLocalGroupRequestType

func init() {
	types.Add("sso:RemovePrincipalsFromLocalGroup", reflect.TypeOf((*RemovePrincipalsFromLocalGroup)(nil)).Elem())
}

type RemovePrincipalsFromLocalGroupRequestType struct {
	This          types.ManagedObjectReference `xml:"_this"`
	PrincipalsIds []PrincipalId                `xml:"principalsIds"`
	GroupName     string                       `xml:"groupName"`
}

func init() {
	types.Add("sso:RemovePrincipalsFromLocalGroupRequestType", reflect.TypeOf((*RemovePrincipalsFromLocalGroupRequestType)(nil)).Elem())
}

type RemovePrincipalsFromLocalGroupResponse struct {
	Returnval []bool `xml:"returnval"`
}

type ResetLocalPersonUserPassword ResetLocalPersonUserPasswordRequestType

func init() {
	types.Add("sso:ResetLocalPersonUserPassword", reflect.TypeOf((*ResetLocalPersonUserPassword)(nil)).Elem())
}

type ResetLocalPersonUserPasswordRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserName    string                       `xml:"userName"`
	NewPassword string                       `xml:"newPassword"`
}

func init() {
	types.Add("sso:ResetLocalPersonUserPasswordRequestType", reflect.TypeOf((*ResetLocalPersonUserPasswordRequestType)(nil)).Elem())
}

type ResetLocalPersonUserPasswordResponse struct {
}

type ResetLocalUserPassword ResetLocalUserPasswordRequestType

func init() {
	types.Add("sso:ResetLocalUserPassword", reflect.TypeOf((*ResetLocalUserPassword)(nil)).Elem())
}

type ResetLocalUserPasswordRequestType struct {
	This            types.ManagedObjectReference `xml:"_this"`
	Username        string                       `xml:"username"`
	CurrentPassword string                       `xml:"currentPassword"`
	NewPassword     string                       `xml:"newPassword"`
}

func init() {
	types.Add("sso:ResetLocalUserPasswordRequestType", reflect.TypeOf((*ResetLocalUserPasswordRequestType)(nil)).Elem())
}

type ResetLocalUserPasswordResponse struct {
}

type ResetSelfLocalPersonUserPassword ResetSelfLocalPersonUserPasswordRequestType

func init() {
	types.Add("sso:ResetSelfLocalPersonUserPassword", reflect.TypeOf((*ResetSelfLocalPersonUserPassword)(nil)).Elem())
}

type ResetSelfLocalPersonUserPasswordRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	NewPassword string                       `xml:"newPassword"`
}

func init() {
	types.Add("sso:ResetSelfLocalPersonUserPasswordRequestType", reflect.TypeOf((*ResetSelfLocalPersonUserPasswordRequestType)(nil)).Elem())
}

type ResetSelfLocalPersonUserPasswordResponse struct {
}

type SendMail SendMailRequestType

func init() {
	types.Add("sso:SendMail", reflect.TypeOf((*SendMail)(nil)).Elem())
}

type SendMailRequestType struct {
	This    types.ManagedObjectReference `xml:"_this"`
	Content AdminMailContent             `xml:"content"`
}

func init() {
	types.Add("sso:SendMailRequestType", reflect.TypeOf((*SendMailRequestType)(nil)).Elem())
}

type SendMailResponse struct {
}

type SetClockTolerance SetClockToleranceRequestType

func init() {
	types.Add("sso:SetClockTolerance", reflect.TypeOf((*SetClockTolerance)(nil)).Elem())
}

type SetClockToleranceRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	Milliseconds int64                        `xml:"milliseconds"`
}

func init() {
	types.Add("sso:SetClockToleranceRequestType", reflect.TypeOf((*SetClockToleranceRequestType)(nil)).Elem())
}

type SetClockToleranceResponse struct {
}

type SetDelegationCount SetDelegationCountRequestType

func init() {
	types.Add("sso:SetDelegationCount", reflect.TypeOf((*SetDelegationCount)(nil)).Elem())
}

type SetDelegationCountRequestType struct {
	This            types.ManagedObjectReference `xml:"_this"`
	DelegationCount int32                        `xml:"delegationCount"`
}

func init() {
	types.Add("sso:SetDelegationCountRequestType", reflect.TypeOf((*SetDelegationCountRequestType)(nil)).Elem())
}

type SetDelegationCountResponse struct {
}

type SetMaximumBearerTokenLifetime SetMaximumBearerTokenLifetimeRequestType

func init() {
	types.Add("sso:SetMaximumBearerTokenLifetime", reflect.TypeOf((*SetMaximumBearerTokenLifetime)(nil)).Elem())
}

type SetMaximumBearerTokenLifetimeRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	MaxLifetime int64                        `xml:"maxLifetime"`
}

func init() {
	types.Add("sso:SetMaximumBearerTokenLifetimeRequestType", reflect.TypeOf((*SetMaximumBearerTokenLifetimeRequestType)(nil)).Elem())
}

type SetMaximumBearerTokenLifetimeResponse struct {
}

type SetMaximumHoKTokenLifetime SetMaximumHoKTokenLifetimeRequestType

func init() {
	types.Add("sso:SetMaximumHoKTokenLifetime", reflect.TypeOf((*SetMaximumHoKTokenLifetime)(nil)).Elem())
}

type SetMaximumHoKTokenLifetimeRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	MaxLifetime int64                        `xml:"maxLifetime"`
}

func init() {
	types.Add("sso:SetMaximumHoKTokenLifetimeRequestType", reflect.TypeOf((*SetMaximumHoKTokenLifetimeRequestType)(nil)).Elem())
}

type SetMaximumHoKTokenLifetimeResponse struct {
}

type SetNewSignerIdentity SetNewSignerIdentityRequestType

func init() {
	types.Add("sso:SetNewSignerIdentity", reflect.TypeOf((*SetNewSignerIdentity)(nil)).Elem())
}

type SetNewSignerIdentityRequestType struct {
	This                    types.ManagedObjectReference                        `xml:"_this"`
	SigningKey              string                                              `xml:"signingKey"`
	SigningCertificateChain AdminConfigurationManagementServiceCertificateChain `xml:"signingCertificateChain"`
}

func init() {
	types.Add("sso:SetNewSignerIdentityRequestType", reflect.TypeOf((*SetNewSignerIdentityRequestType)(nil)).Elem())
}

type SetNewSignerIdentityResponse struct {
}

type SetRenewCount SetRenewCountRequestType

func init() {
	types.Add("sso:SetRenewCount", reflect.TypeOf((*SetRenewCount)(nil)).Elem())
}

type SetRenewCountRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	RenewCount int32                        `xml:"renewCount"`
}

func init() {
	types.Add("sso:SetRenewCountRequestType", reflect.TypeOf((*SetRenewCountRequestType)(nil)).Elem())
}

type SetRenewCountResponse struct {
}

type SetRole SetRoleRequestType

func init() {
	types.Add("sso:SetRole", reflect.TypeOf((*SetRole)(nil)).Elem())
}

type SetRoleRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
	Role   string                       `xml:"role"`
}

func init() {
	types.Add("sso:SetRoleRequestType", reflect.TypeOf((*SetRoleRequestType)(nil)).Elem())
}

type SetRoleResponse struct {
	Returnval bool `xml:"returnval"`
}

type SetSignerIdentity SetSignerIdentityRequestType

func init() {
	types.Add("sso:SetSignerIdentity", reflect.TypeOf((*SetSignerIdentity)(nil)).Elem())
}

type SetSignerIdentityRequestType struct {
	This                    types.ManagedObjectReference                        `xml:"_this"`
	AdminUser               PrincipalId                                         `xml:"adminUser"`
	AdminPass               string                                              `xml:"adminPass"`
	SigningKey              string                                              `xml:"signingKey"`
	SigningCertificateChain AdminConfigurationManagementServiceCertificateChain `xml:"signingCertificateChain"`
}

func init() {
	types.Add("sso:SetSignerIdentityRequestType", reflect.TypeOf((*SetSignerIdentityRequestType)(nil)).Elem())
}

type SetSignerIdentityResponse struct {
}

type SsoAdminServiceInstance SsoAdminServiceInstanceRequestType

func init() {
	types.Add("sso:SsoAdminServiceInstance", reflect.TypeOf((*SsoAdminServiceInstance)(nil)).Elem())
}

type SsoAdminServiceInstanceRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:SsoAdminServiceInstanceRequestType", reflect.TypeOf((*SsoAdminServiceInstanceRequestType)(nil)).Elem())
}

type SsoAdminServiceInstanceResponse struct {
	Returnval AdminServiceContent `xml:"returnval"`
}

type SsoGroupcheckServiceInstance SsoGroupcheckServiceInstanceRequestType

func init() {
	types.Add("sso:SsoGroupcheckServiceInstance", reflect.TypeOf((*SsoGroupcheckServiceInstance)(nil)).Elem())
}

type SsoGroupcheckServiceInstanceRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("sso:SsoGroupcheckServiceInstanceRequestType", reflect.TypeOf((*SsoGroupcheckServiceInstanceRequestType)(nil)).Elem())
}

type SsoGroupcheckServiceInstanceResponse struct {
	Returnval GroupcheckServiceContent `xml:"returnval"`
}

type UnlockUserAccount UnlockUserAccountRequestType

func init() {
	types.Add("sso:UnlockUserAccount", reflect.TypeOf((*UnlockUserAccount)(nil)).Elem())
}

type UnlockUserAccountRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
}

func init() {
	types.Add("sso:UnlockUserAccountRequestType", reflect.TypeOf((*UnlockUserAccountRequestType)(nil)).Elem())
}

type UnlockUserAccountResponse struct {
	Returnval bool `xml:"returnval"`
}

type UpdateExternalDomainAuthnType UpdateExternalDomainAuthnTypeRequestType

func init() {
	types.Add("sso:UpdateExternalDomainAuthnType", reflect.TypeOf((*UpdateExternalDomainAuthnType)(nil)).Elem())
}

type UpdateExternalDomainAuthnTypeRequestType struct {
	This             types.ManagedObjectReference                           `xml:"_this"`
	Name             string                                                 `xml:"name"`
	AuthnType        string                                                 `xml:"authnType"`
	AuthnCredentials *AdminDomainManagementServiceAuthenticationCredentails `xml:"authnCredentials,omitempty"`
}

func init() {
	types.Add("sso:UpdateExternalDomainAuthnTypeRequestType", reflect.TypeOf((*UpdateExternalDomainAuthnTypeRequestType)(nil)).Elem())
}

type UpdateExternalDomainAuthnTypeResponse struct {
}

type UpdateExternalDomainDetails UpdateExternalDomainDetailsRequestType

func init() {
	types.Add("sso:UpdateExternalDomainDetails", reflect.TypeOf((*UpdateExternalDomainDetails)(nil)).Elem())
}

type UpdateExternalDomainDetailsRequestType struct {
	This    types.ManagedObjectReference `xml:"_this"`
	Name    string                       `xml:"name"`
	Details AdminExternalDomainDetails   `xml:"details"`
}

func init() {
	types.Add("sso:UpdateExternalDomainDetailsRequestType", reflect.TypeOf((*UpdateExternalDomainDetailsRequestType)(nil)).Elem())
}

type UpdateExternalDomainDetailsResponse struct {
}

type UpdateLocalGroupDetails UpdateLocalGroupDetailsRequestType

func init() {
	types.Add("sso:UpdateLocalGroupDetails", reflect.TypeOf((*UpdateLocalGroupDetails)(nil)).Elem())
}

type UpdateLocalGroupDetailsRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	GroupName    string                       `xml:"groupName"`
	GroupDetails AdminGroupDetails            `xml:"groupDetails"`
}

func init() {
	types.Add("sso:UpdateLocalGroupDetailsRequestType", reflect.TypeOf((*UpdateLocalGroupDetailsRequestType)(nil)).Elem())
}

type UpdateLocalGroupDetailsResponse struct {
}

type UpdateLocalPasswordPolicy UpdateLocalPasswordPolicyRequestType

func init() {
	types.Add("sso:UpdateLocalPasswordPolicy", reflect.TypeOf((*UpdateLocalPasswordPolicy)(nil)).Elem())
}

type UpdateLocalPasswordPolicyRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Policy AdminPasswordPolicy          `xml:"policy"`
}

func init() {
	types.Add("sso:UpdateLocalPasswordPolicyRequestType", reflect.TypeOf((*UpdateLocalPasswordPolicyRequestType)(nil)).Elem())
}

type UpdateLocalPasswordPolicyResponse struct {
}

type UpdateLocalPersonUserDetails UpdateLocalPersonUserDetailsRequestType

func init() {
	types.Add("sso:UpdateLocalPersonUserDetails", reflect.TypeOf((*UpdateLocalPersonUserDetails)(nil)).Elem())
}

type UpdateLocalPersonUserDetailsRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserName    string                       `xml:"userName"`
	UserDetails AdminPersonDetails           `xml:"userDetails"`
}

func init() {
	types.Add("sso:UpdateLocalPersonUserDetailsRequestType", reflect.TypeOf((*UpdateLocalPersonUserDetailsRequestType)(nil)).Elem())
}

type UpdateLocalPersonUserDetailsResponse struct {
}

type UpdateLocalSolutionUserDetails UpdateLocalSolutionUserDetailsRequestType

func init() {
	types.Add("sso:UpdateLocalSolutionUserDetails", reflect.TypeOf((*UpdateLocalSolutionUserDetails)(nil)).Elem())
}

type UpdateLocalSolutionUserDetailsRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserName    string                       `xml:"userName"`
	UserDetails AdminSolutionDetails         `xml:"userDetails"`
}

func init() {
	types.Add("sso:UpdateLocalSolutionUserDetailsRequestType", reflect.TypeOf((*UpdateLocalSolutionUserDetailsRequestType)(nil)).Elem())
}

type UpdateLocalSolutionUserDetailsResponse struct {
}

type UpdateLockoutPolicy UpdateLockoutPolicyRequestType

func init() {
	types.Add("sso:UpdateLockoutPolicy", reflect.TypeOf((*UpdateLockoutPolicy)(nil)).Elem())
}

type UpdateLockoutPolicyRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Policy AdminLockoutPolicy           `xml:"policy"`
}

func init() {
	types.Add("sso:UpdateLockoutPolicyRequestType", reflect.TypeOf((*UpdateLockoutPolicyRequestType)(nil)).Elem())
}

type UpdateLockoutPolicyResponse struct {
}

type UpdatePasswordExpirationConfiguration UpdatePasswordExpirationConfigurationRequestType

func init() {
	types.Add("sso:UpdatePasswordExpirationConfiguration", reflect.TypeOf((*UpdatePasswordExpirationConfiguration)(nil)).Elem())
}

type UpdatePasswordExpirationConfigurationRequestType struct {
	This   types.ManagedObjectReference  `xml:"_this"`
	Config AdminPasswordExpirationConfig `xml:"config"`
}

func init() {
	types.Add("sso:UpdatePasswordExpirationConfigurationRequestType", reflect.TypeOf((*UpdatePasswordExpirationConfigurationRequestType)(nil)).Elem())
}

type UpdatePasswordExpirationConfigurationResponse struct {
}

type UpdateSelfLocalPersonUserDetails UpdateSelfLocalPersonUserDetailsRequestType

func init() {
	types.Add("sso:UpdateSelfLocalPersonUserDetails", reflect.TypeOf((*UpdateSelfLocalPersonUserDetails)(nil)).Elem())
}

type UpdateSelfLocalPersonUserDetailsRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserDetails AdminPersonDetails           `xml:"userDetails"`
}

func init() {
	types.Add("sso:UpdateSelfLocalPersonUserDetailsRequestType", reflect.TypeOf((*UpdateSelfLocalPersonUserDetailsRequestType)(nil)).Elem())
}

type UpdateSelfLocalPersonUserDetailsResponse struct {
}

type UpdateSelfLocalSolutionUserDetails UpdateSelfLocalSolutionUserDetailsRequestType

func init() {
	types.Add("sso:UpdateSelfLocalSolutionUserDetails", reflect.TypeOf((*UpdateSelfLocalSolutionUserDetails)(nil)).Elem())
}

type UpdateSelfLocalSolutionUserDetailsRequestType struct {
	This        types.ManagedObjectReference `xml:"_this"`
	UserDetails AdminSolutionDetails         `xml:"userDetails"`
}

func init() {
	types.Add("sso:UpdateSelfLocalSolutionUserDetailsRequestType", reflect.TypeOf((*UpdateSelfLocalSolutionUserDetailsRequestType)(nil)).Elem())
}

type UpdateSelfLocalSolutionUserDetailsResponse struct {
}

type UpdateSmtpConfiguration UpdateSmtpConfigurationRequestType

func init() {
	types.Add("sso:UpdateSmtpConfiguration", reflect.TypeOf((*UpdateSmtpConfiguration)(nil)).Elem())
}

type UpdateSmtpConfigurationRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Config AdminSmtpConfig              `xml:"config"`
}

func init() {
	types.Add("sso:UpdateSmtpConfigurationRequestType", reflect.TypeOf((*UpdateSmtpConfigurationRequestType)(nil)).Elem())
}

type UpdateSmtpConfigurationResponse struct {
}
