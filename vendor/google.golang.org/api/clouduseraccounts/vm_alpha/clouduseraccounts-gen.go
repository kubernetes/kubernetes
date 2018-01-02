// Package clouduseraccounts provides access to the Cloud User Accounts API.
//
// See https://cloud.google.com/compute/docs/access/user-accounts/api/latest/
//
// Usage example:
//
//   import "google.golang.org/api/clouduseraccounts/vm_alpha"
//   ...
//   clouduseraccountsService, err := clouduseraccounts.New(oauthHttpClient)
package clouduseraccounts // import "google.golang.org/api/clouduseraccounts/vm_alpha"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "clouduseraccounts:vm_alpha"
const apiName = "clouduseraccounts"
const apiVersion = "vm_alpha"
const basePath = "https://www.googleapis.com/clouduseraccounts/vm_alpha/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// Manage your Google Cloud User Accounts
	CloudUseraccountsScope = "https://www.googleapis.com/auth/cloud.useraccounts"

	// View your Google Cloud User Accounts
	CloudUseraccountsReadonlyScope = "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.GlobalAccountsOperations = NewGlobalAccountsOperationsService(s)
	s.Groups = NewGroupsService(s)
	s.Linux = NewLinuxService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	GlobalAccountsOperations *GlobalAccountsOperationsService

	Groups *GroupsService

	Linux *LinuxService

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewGlobalAccountsOperationsService(s *Service) *GlobalAccountsOperationsService {
	rs := &GlobalAccountsOperationsService{s: s}
	return rs
}

type GlobalAccountsOperationsService struct {
	s *Service
}

func NewGroupsService(s *Service) *GroupsService {
	rs := &GroupsService{s: s}
	return rs
}

type GroupsService struct {
	s *Service
}

func NewLinuxService(s *Service) *LinuxService {
	rs := &LinuxService{s: s}
	return rs
}

type LinuxService struct {
	s *Service
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	return rs
}

type UsersService struct {
	s *Service
}

// AuditConfig: Enables "data access" audit logging for a service and
// specifies a list of members that are log-exempted.
type AuditConfig struct {
	// ExemptedMembers: Specifies the identities that are exempted from
	// "data access" audit logging for the `service` specified above.
	// Follows the same format of Binding.members.
	ExemptedMembers []string `json:"exemptedMembers,omitempty"`

	// Service: Specifies a service that will be enabled for "data access"
	// audit logging. For example, `resourcemanager`, `storage`, `compute`.
	// `allServices` is a special value that covers all services.
	Service string `json:"service,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExemptedMembers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExemptedMembers") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuditConfig) MarshalJSON() ([]byte, error) {
	type noMethod AuditConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthorizedKeysView: A list of authorized public keys for a user
// account.
type AuthorizedKeysView struct {
	// Keys: [Output Only] The list of authorized public keys in SSH format.
	Keys []string `json:"keys,omitempty"`

	// Sudoer: [Output Only] Whether the user has the ability to elevate on
	// the instance that requested the authorized keys.
	Sudoer bool `json:"sudoer,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Keys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Keys") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthorizedKeysView) MarshalJSON() ([]byte, error) {
	type noMethod AuthorizedKeysView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Members: Specifies the identities requesting access for a Cloud
	// Platform resource. `members` can have the following values:
	//
	// * `allUsers`: A special identifier that represents anyone who is on
	// the internet; with or without a Google account.
	//
	// * `allAuthenticatedUsers`: A special identifier that represents
	// anyone who is authenticated with a Google account or a service
	// account.
	//
	// * `user:{emailid}`: An email address that represents a specific
	// Google account. For example, `alice@gmail.com` or
	// `joe@example.com`.
	//
	// * `serviceAccount:{emailid}`: An email address that represents a
	// service account. For example,
	// `my-other-app@appspot.gserviceaccount.com`.
	//
	// * `group:{emailid}`: An email address that represents a Google group.
	// For example, `admins@example.com`.
	//
	// * `domain:{domain}`: A Google Apps domain name that represents all
	// the users of that domain. For example, `google.com` or `example.com`.
	Members []string `json:"members,omitempty"`

	// Role: Role that is assigned to `members`. For example,
	// `roles/viewer`, `roles/editor`, or `roles/owner`.
	Role string `json:"role,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Members") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Binding) MarshalJSON() ([]byte, error) {
	type noMethod Binding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Condition: A condition to be met.
type Condition struct {
	// Iam: Trusted attributes supplied by the IAM system.
	//
	// Possible values:
	//   "ATTRIBUTION"
	//   "AUTHORITY"
	//   "NO_ATTR"
	Iam string `json:"iam,omitempty"`

	// Op: An operator to apply the subject with.
	//
	// Possible values:
	//   "DISCHARGED"
	//   "EQUALS"
	//   "IN"
	//   "NOT_EQUALS"
	//   "NOT_IN"
	//   "NO_OP"
	Op string `json:"op,omitempty"`

	// Svc: Trusted attributes discharged by the service.
	Svc string `json:"svc,omitempty"`

	// Sys: Trusted attributes supplied by any service that owns resources
	// and uses the IAM system for access control.
	//
	// Possible values:
	//   "IP"
	//   "NAME"
	//   "NO_ATTR"
	//   "REGION"
	//   "SERVICE"
	Sys string `json:"sys,omitempty"`

	// Value: The object of the condition. Exactly one of these must be set.
	Value string `json:"value,omitempty"`

	// Values: The objects of the condition. This is mutually exclusive with
	// 'value'.
	Values []string `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Iam") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Iam") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Condition) MarshalJSON() ([]byte, error) {
	type noMethod Condition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Group: A Group resource.
type Group struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always
	// clouduseraccounts#group for groups.
	Kind string `json:"kind,omitempty"`

	// Members: [Output Only] A list of URLs to User resources who belong to
	// the group. Users may only be members of groups in the same project.
	Members []string `json:"members,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTimestamp")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Group) MarshalJSON() ([]byte, error) {
	type noMethod Group
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GroupList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Group resources.
	Items []*Group `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// clouduseraccounts#groupList for lists of groups.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GroupList) MarshalJSON() ([]byte, error) {
	type noMethod GroupList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GroupsAddMemberRequest struct {
	// Users: Fully-qualified URLs of the User resources to add.
	Users []string `json:"users,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Users") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Users") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GroupsAddMemberRequest) MarshalJSON() ([]byte, error) {
	type noMethod GroupsAddMemberRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GroupsRemoveMemberRequest struct {
	// Users: Fully-qualified URLs of the User resources to remove.
	Users []string `json:"users,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Users") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Users") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GroupsRemoveMemberRequest) MarshalJSON() ([]byte, error) {
	type noMethod GroupsRemoveMemberRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LinuxAccountViews: A list of all Linux accounts for this project.
// This API is only used by Compute Engine virtual machines to get
// information about user accounts for a project or instance. Linux
// resources are read-only views into users and groups managed by the
// Compute Engine Accounts API.
type LinuxAccountViews struct {
	// GroupViews: [Output Only] A list of all groups within a project.
	GroupViews []*LinuxGroupView `json:"groupViews,omitempty"`

	// Kind: [Output Only] Type of the resource. Always
	// clouduseraccounts#linuxAccountViews for Linux resources.
	Kind string `json:"kind,omitempty"`

	// UserViews: [Output Only] A list of all users within a project.
	UserViews []*LinuxUserView `json:"userViews,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GroupViews") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GroupViews") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinuxAccountViews) MarshalJSON() ([]byte, error) {
	type noMethod LinuxAccountViews
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type LinuxGetAuthorizedKeysViewResponse struct {
	// Resource: [Output Only] A list of authorized public keys for a user.
	Resource *AuthorizedKeysView `json:"resource,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Resource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resource") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinuxGetAuthorizedKeysViewResponse) MarshalJSON() ([]byte, error) {
	type noMethod LinuxGetAuthorizedKeysViewResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type LinuxGetLinuxAccountViewsResponse struct {
	// Resource: [Output Only] A list of authorized user accounts and
	// groups.
	Resource *LinuxAccountViews `json:"resource,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Resource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Resource") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinuxGetLinuxAccountViewsResponse) MarshalJSON() ([]byte, error) {
	type noMethod LinuxGetLinuxAccountViewsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LinuxGroupView: A detailed view of a Linux group.
type LinuxGroupView struct {
	// Gid: [Output Only] The Group ID.
	Gid int64 `json:"gid,omitempty"`

	// GroupName: [Output Only] Group name.
	GroupName string `json:"groupName,omitempty"`

	// Members: [Output Only] List of user accounts that belong to the
	// group.
	Members []string `json:"members,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Gid") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Gid") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinuxGroupView) MarshalJSON() ([]byte, error) {
	type noMethod LinuxGroupView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LinuxUserView: A detailed view of a Linux user account.
type LinuxUserView struct {
	// Gecos: [Output Only] The GECOS (user information) entry for this
	// account.
	Gecos string `json:"gecos,omitempty"`

	// Gid: [Output Only] User's default group ID.
	Gid int64 `json:"gid,omitempty"`

	// HomeDirectory: [Output Only] The path to the home directory for this
	// account.
	HomeDirectory string `json:"homeDirectory,omitempty"`

	// Shell: [Output Only] The path to the login shell for this account.
	Shell string `json:"shell,omitempty"`

	// Uid: [Output Only] User ID.
	Uid int64 `json:"uid,omitempty"`

	// Username: [Output Only] The username of the account.
	Username string `json:"username,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Gecos") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Gecos") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinuxUserView) MarshalJSON() ([]byte, error) {
	type noMethod LinuxUserView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogConfig: Specifies what kind of log the caller must write
type LogConfig struct {
	// Counter: Counter options.
	Counter *LogConfigCounterOptions `json:"counter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Counter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Counter") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogConfig) MarshalJSON() ([]byte, error) {
	type noMethod LogConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogConfigCounterOptions: Options for counters
type LogConfigCounterOptions struct {
	// Field: The field value to attribute.
	Field string `json:"field,omitempty"`

	// Metric: The metric to update.
	Metric string `json:"metric,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Field") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Field") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogConfigCounterOptions) MarshalJSON() ([]byte, error) {
	type noMethod LogConfigCounterOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: An Operation resource, used to manage asynchronous API
// requests.
type Operation struct {
	// ClientOperationId: [Output Only] Reserved for future use.
	ClientOperationId string `json:"clientOperationId,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: [Output Only] A textual description of the operation,
	// which is set when the operation is created.
	Description string `json:"description,omitempty"`

	// EndTime: [Output Only] The time that this operation was completed.
	// This value is in RFC3339 text format.
	EndTime string `json:"endTime,omitempty"`

	// Error: [Output Only] If errors are generated during processing of the
	// operation, this field will be populated.
	Error *OperationError `json:"error,omitempty"`

	// HttpErrorMessage: [Output Only] If the operation fails, this field
	// contains the HTTP error message that was returned, such as NOT FOUND.
	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	// HttpErrorStatusCode: [Output Only] If the operation fails, this field
	// contains the HTTP error status code that was returned. For example, a
	// 404 means the resource was not found.
	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	// Id: [Output Only] The unique identifier for the resource. This
	// identifier is defined by the server.
	Id uint64 `json:"id,omitempty,string"`

	// InsertTime: [Output Only] The time that this operation was requested.
	// This value is in RFC3339 text format.
	InsertTime string `json:"insertTime,omitempty"`

	// Kind: [Output Only] Type of the resource. Always compute#operation
	// for Operation resources.
	Kind string `json:"kind,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// OperationType: [Output Only] The type of operation, such as insert,
	// update, or delete, and so on.
	OperationType string `json:"operationType,omitempty"`

	// Progress: [Output Only] An optional progress indicator that ranges
	// from 0 to 100. There is no requirement that this be linear or support
	// any granularity of operations. This should not be used to guess when
	// the operation will be complete. This number should monotonically
	// increase as the operation progresses.
	Progress int64 `json:"progress,omitempty"`

	// Region: [Output Only] The URL of the region where the operation
	// resides. Only available when performing regional operations.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: [Output Only] The time that this operation was started by
	// the server. This value is in RFC3339 text format.
	StartTime string `json:"startTime,omitempty"`

	// Status: [Output Only] The status of the operation, which can be one
	// of the following: PENDING, RUNNING, or DONE.
	//
	// Possible values:
	//   "DONE"
	//   "PENDING"
	//   "RUNNING"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output Only] An optional textual description of the
	// current status of the operation.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetId: [Output Only] The unique target ID, which identifies a
	// specific incarnation of the target resource.
	TargetId uint64 `json:"targetId,omitempty,string"`

	// TargetLink: [Output Only] The URL of the resource that the operation
	// modifies.
	TargetLink string `json:"targetLink,omitempty"`

	// User: [Output Only] User who requested the operation, for example:
	// user@example.com.
	User string `json:"user,omitempty"`

	// Warnings: [Output Only] If warning messages are generated during
	// processing of the operation, this field will be populated.
	Warnings []*OperationWarnings `json:"warnings,omitempty"`

	// Zone: [Output Only] The URL of the zone where the operation resides.
	// Only available when performing per-zone operations.
	Zone string `json:"zone,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClientOperationId")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientOperationId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationError: [Output Only] If errors are generated during
// processing of the operation, this field will be populated.
type OperationError struct {
	// Errors: [Output Only] The array of errors encountered while
	// processing this operation.
	Errors []*OperationErrorErrors `json:"errors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Errors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Errors") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationError) MarshalJSON() ([]byte, error) {
	type noMethod OperationError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OperationErrorErrors struct {
	// Code: [Output Only] The error type identifier for this error.
	Code string `json:"code,omitempty"`

	// Location: [Output Only] Indicates the field in the request that
	// caused the error. This property is optional.
	Location string `json:"location,omitempty"`

	// Message: [Output Only] An optional, human-readable error message.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationErrorErrors) MarshalJSON() ([]byte, error) {
	type noMethod OperationErrorErrors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OperationWarnings struct {
	// Code: [Output Only] A warning code, if applicable. For example,
	// Compute Engine returns NO_RESULTS_ON_PAGE if there are no results in
	// the response.
	//
	// Possible values:
	//   "CLEANUP_FAILED"
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NOT_CRITICAL_ERROR"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output Only] Metadata about this warning in key: value format.
	// For example:
	// "data": [ { "key": "scope", "value": "zones/us-east1-d" }
	Data []*OperationWarningsData `json:"data,omitempty"`

	// Message: [Output Only] A human-readable description of the warning
	// code.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationWarnings) MarshalJSON() ([]byte, error) {
	type noMethod OperationWarnings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type OperationWarningsData struct {
	// Key: [Output Only] A key that provides more detail on the warning
	// being returned. For example, for warnings where there are no results
	// in a list request for a particular zone, this key might be scope and
	// the key value might be the zone name. Other examples might be a key
	// indicating a deprecated resource and a suggested replacement, or a
	// warning about invalid network settings (for example, if an instance
	// attempts to perform IP forwarding but is not enabled for IP
	// forwarding).
	Key string `json:"key,omitempty"`

	// Value: [Output Only] A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Key") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationWarningsData) MarshalJSON() ([]byte, error) {
	type noMethod OperationWarningsData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationList: Contains a list of Operation resources.
type OperationList struct {
	// Id: [Output Only] The unique identifier for the resource. This
	// identifier is defined by the server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of Operation resources.
	Items []*Operation `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#operations for
	// Operations resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] This token allows you to get the next
	// page of results for list requests. If the number of results is larger
	// than maxResults, use the nextPageToken as a value for the query
	// parameter pageToken in the next list request. Subsequent list
	// requests will have their own nextPageToken to continue paging through
	// the results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server-defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationList) MarshalJSON() ([]byte, error) {
	type noMethod OperationList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Policy: Defines an Identity and Access Management (IAM) policy. It is
// used to specify access control policies for Cloud Platform
// resources.
//
//
//
// A `Policy` consists of a list of `bindings`. A `Binding` binds a list
// of `members` to a `role`, where the members can be user accounts,
// Google groups, Google domains, and service accounts. A `role` is a
// named list of permissions defined by IAM.
//
// **Example**
//
// { "bindings": [ { "role": "roles/owner", "members": [
// "user:mike@example.com", "group:admins@example.com",
// "domain:google.com",
// "serviceAccount:my-other-app@appspot.gserviceaccount.com", ] }, {
// "role": "roles/viewer", "members": ["user:sean@example.com"] } ]
// }
//
// For a description of IAM and its features, see the [IAM developer's
// guide](https://cloud.google.com/iam).
type Policy struct {
	// AuditConfigs: Specifies audit logging configs for "data access".
	// "data access": generally refers to data reads/writes and admin reads.
	// "admin activity": generally refers to admin writes.
	//
	// Note: `AuditConfig` doesn't apply to "admin activity", which always
	// enables audit logging.
	AuditConfigs []*AuditConfig `json:"auditConfigs,omitempty"`

	// Bindings: Associates a list of `members` to a `role`. Multiple
	// `bindings` must not be specified for the same `role`. `bindings` with
	// no members will result in an error.
	Bindings []*Binding `json:"bindings,omitempty"`

	// Etag: `etag` is used for optimistic concurrency control as a way to
	// help prevent simultaneous updates of a policy from overwriting each
	// other. It is strongly suggested that systems make use of the `etag`
	// in the read-modify-write cycle to perform policy updates in order to
	// avoid race conditions: An `etag` is returned in the response to
	// `getIamPolicy`, and systems are expected to put that etag in the
	// request to `setIamPolicy` to ensure that their change will be applied
	// to the same version of the policy.
	//
	// If no `etag` is provided in the call to `setIamPolicy`, then the
	// existing policy is overwritten blindly.
	Etag string `json:"etag,omitempty"`

	IamOwned bool `json:"iamOwned,omitempty"`

	// Rules: If more than one rule is specified, the rules are applied in
	// the following manner: - All matching LOG rules are always applied. -
	// If any DENY/DENY_WITH_LOG rule matches, permission is denied. Logging
	// will be applied if one or more matching rule requires logging. -
	// Otherwise, if any ALLOW/ALLOW_WITH_LOG rule matches, permission is
	// granted. Logging will be applied if one or more matching rule
	// requires logging. - Otherwise, if no rule applies, permission is
	// denied.
	Rules []*Rule `json:"rules,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AuditConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuditConfigs") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PublicKey: A public key for authenticating to guests.
type PublicKey struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// ExpirationTimestamp: Optional expiration timestamp. If provided, the
	// timestamp must be in RFC3339 text format. If not provided, the public
	// key never expires.
	ExpirationTimestamp string `json:"expirationTimestamp,omitempty"`

	// Fingerprint: [Output Only] The fingerprint of the key is defined by
	// RFC4716 to be the MD5 digest of the public key.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Key: Public key text in SSH format, defined by RFC4253 section 6.6.
	Key string `json:"key,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreationTimestamp")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PublicKey) MarshalJSON() ([]byte, error) {
	type noMethod PublicKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Rule: A rule to be applied in a Policy.
type Rule struct {
	// Action: Required
	//
	// Possible values:
	//   "ALLOW"
	//   "ALLOW_WITH_LOG"
	//   "DENY"
	//   "DENY_WITH_LOG"
	//   "LOG"
	//   "NO_ACTION"
	Action string `json:"action,omitempty"`

	// Conditions: Additional restrictions that must be met
	Conditions []*Condition `json:"conditions,omitempty"`

	// Description: Human-readable description of the rule.
	Description string `json:"description,omitempty"`

	// Ins: The rule matches if the PRINCIPAL/AUTHORITY_SELECTOR is in this
	// set of entries.
	Ins []string `json:"ins,omitempty"`

	// LogConfigs: The config returned to callers of
	// tech.iam.IAM.CheckPolicy for any entries that match the LOG action.
	LogConfigs []*LogConfig `json:"logConfigs,omitempty"`

	// NotIns: The rule matches if the PRINCIPAL/AUTHORITY_SELECTOR is not
	// in this set of entries.
	NotIns []string `json:"notIns,omitempty"`

	// Permissions: A permission is a string of form '..' (e.g.,
	// 'storage.buckets.list'). A value of '*' matches all permissions, and
	// a verb part of '*' (e.g., 'storage.buckets.*') matches all verbs.
	Permissions []string `json:"permissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Action") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Action") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Rule) MarshalJSON() ([]byte, error) {
	type noMethod Rule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestPermissionsRequest struct {
	// Permissions: The set of permissions to check for the 'resource'.
	// Permissions with wildcards (such as '*' or 'storage.*') are not
	// allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestPermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestPermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TestPermissionsResponse struct {
	// Permissions: A subset of `TestPermissionsRequest.permissions` that
	// the caller is allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// User: A User resource.
type User struct {
	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource;
	// provided by the client when the resource is created.
	Description string `json:"description,omitempty"`

	// Groups: [Output Only] A list of URLs to Group resources who contain
	// the user. Users are only members of groups in the same project.
	Groups []string `json:"groups,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always
	// clouduseraccounts#user for users.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the resource; provided by the client when the resource
	// is created.
	Name string `json:"name,omitempty"`

	// Owner: Email address of account's owner. This account will be
	// validated to make sure it exists. The email can belong to any domain,
	// but it must be tied to a Google account.
	Owner string `json:"owner,omitempty"`

	// PublicKeys: [Output Only] Public keys that this user may use to
	// login.
	PublicKeys []*PublicKey `json:"publicKeys,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTimestamp")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreationTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UserList struct {
	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id string `json:"id,omitempty"`

	// Items: [Output Only] A list of User resources.
	Items []*User `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always
	// clouduseraccounts#userList for lists of users.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output Only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: [Output Only] Server defined URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserList) MarshalJSON() ([]byte, error) {
	type noMethod UserList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "clouduseraccounts.globalAccountsOperations.delete":

type GlobalAccountsOperationsDeleteCall struct {
	s          *Service
	project    string
	operation  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the specified operation resource.
func (r *GlobalAccountsOperationsService) Delete(project string, operation string) *GlobalAccountsOperationsDeleteCall {
	c := &GlobalAccountsOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAccountsOperationsDeleteCall) Fields(s ...googleapi.Field) *GlobalAccountsOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GlobalAccountsOperationsDeleteCall) Context(ctx context.Context) *GlobalAccountsOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GlobalAccountsOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GlobalAccountsOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.globalAccountsOperations.delete" call.
func (c *GlobalAccountsOperationsDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified operation resource.",
	//   "httpMethod": "DELETE",
	//   "id": "clouduseraccounts.globalAccountsOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations/{operation}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.globalAccountsOperations.get":

type GlobalAccountsOperationsGetCall struct {
	s            *Service
	project      string
	operation    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the specified operation resource.
func (r *GlobalAccountsOperationsService) Get(project string, operation string) *GlobalAccountsOperationsGetCall {
	c := &GlobalAccountsOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAccountsOperationsGetCall) Fields(s ...googleapi.Field) *GlobalAccountsOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GlobalAccountsOperationsGetCall) IfNoneMatch(entityTag string) *GlobalAccountsOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GlobalAccountsOperationsGetCall) Context(ctx context.Context) *GlobalAccountsOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GlobalAccountsOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GlobalAccountsOperationsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.globalAccountsOperations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GlobalAccountsOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the specified operation resource.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.globalAccountsOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the Operations resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.globalAccountsOperations.list":

type GlobalAccountsOperationsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the list of operation resources contained within the
// specified project.
func (r *GlobalAccountsOperationsService) List(project string) *GlobalAccountsOperationsListCall {
	c := &GlobalAccountsOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must be in the format: field_name comparison_string
// literal_string.
//
// The field_name is the name of the field you want to compare. Only
// atomic field types are supported (string, number, boolean). The
// comparison_string must be either eq (equals) or ne (not equals). The
// literal_string is the string value to filter to. The literal value
// must be valid for the type of field you are filtering by (string,
// number, boolean). For string fields, the literal value is interpreted
// as a regular expression using RE2 syntax. The literal value must
// match the entire field.
//
// For example, to filter for instances that do not have a name of
// example-instance, you would use filter=name ne
// example-instance.
//
// Compute Engine Beta API Only: If you use filtering in the Beta API,
// you can also filter on nested fields. For example, you could filter
// on instances that have set the scheduling.automaticRestart field to
// true. In particular, use filtering on nested fields to take advantage
// of instance labels to organize and filter results based on label
// values.
//
// The Beta API also supports filtering on multiple expressions by
// providing each separate expression within parentheses. For example,
// (scheduling.automaticRestart eq true) (zone eq us-central1-f).
// Multiple expressions are treated as AND expressions, meaning that
// resources must match all expressions to pass the filters.
func (c *GlobalAccountsOperationsListCall) Filter(filter string) *GlobalAccountsOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results per page that should be returned. If the number of
// available results is larger than maxResults, Compute Engine returns a
// nextPageToken that can be used to get the next page of results in
// subsequent list requests.
func (c *GlobalAccountsOperationsListCall) MaxResults(maxResults int64) *GlobalAccountsOperationsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderBy sets the optional parameter "orderBy": Sorts list results by
// a certain order. By default, results are returned in alphanumerical
// order based on the resource name.
//
// You can also sort results in descending order based on the creation
// timestamp using orderBy="creationTimestamp desc". This sorts results
// based on the creationTimestamp field in reverse chronological order
// (newest result first). Use this to sort resources like operations so
// that the newest operation is returned first.
//
// Currently, only sorting by name or creationTimestamp desc is
// supported.
func (c *GlobalAccountsOperationsListCall) OrderBy(orderBy string) *GlobalAccountsOperationsListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Set pageToken to the nextPageToken returned by a
// previous list request to get the next page of results.
func (c *GlobalAccountsOperationsListCall) PageToken(pageToken string) *GlobalAccountsOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GlobalAccountsOperationsListCall) Fields(s ...googleapi.Field) *GlobalAccountsOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GlobalAccountsOperationsListCall) IfNoneMatch(entityTag string) *GlobalAccountsOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GlobalAccountsOperationsListCall) Context(ctx context.Context) *GlobalAccountsOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GlobalAccountsOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GlobalAccountsOperationsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.globalAccountsOperations.list" call.
// Exactly one of *OperationList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *OperationList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GlobalAccountsOperationsListCall) Do(opts ...googleapi.CallOption) (*OperationList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &OperationList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of operation resources contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.globalAccountsOperations.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must be in the format: field_name comparison_string literal_string.\n\nThe field_name is the name of the field you want to compare. Only atomic field types are supported (string, number, boolean). The comparison_string must be either eq (equals) or ne (not equals). The literal_string is the string value to filter to. The literal value must be valid for the type of field you are filtering by (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.\n\nFor example, to filter for instances that do not have a name of example-instance, you would use filter=name ne example-instance.\n\nCompute Engine Beta API Only: If you use filtering in the Beta API, you can also filter on nested fields. For example, you could filter on instances that have set the scheduling.automaticRestart field to true. In particular, use filtering on nested fields to take advantage of instance labels to organize and filter results based on label values.\n\nThe Beta API also supports filtering on multiple expressions by providing each separate expression within parentheses. For example, (scheduling.automaticRestart eq true) (zone eq us-central1-f). Multiple expressions are treated as AND expressions, meaning that resources must match all expressions to pass the filters.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "The maximum number of results per page that should be returned. If the number of available results is larger than maxResults, Compute Engine returns a nextPageToken that can be used to get the next page of results in subsequent list requests.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Sorts list results by a certain order. By default, results are returned in alphanumerical order based on the resource name.\n\nYou can also sort results in descending order based on the creation timestamp using orderBy=\"creationTimestamp desc\". This sorts results based on the creationTimestamp field in reverse chronological order (newest result first). Use this to sort resources like operations so that the newest operation is returned first.\n\nCurrently, only sorting by name or creationTimestamp desc is supported.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Set pageToken to the nextPageToken returned by a previous list request to get the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *GlobalAccountsOperationsListCall) Pages(ctx context.Context, f func(*OperationList) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "clouduseraccounts.groups.addMember":

type GroupsAddMemberCall struct {
	s                      *Service
	project                string
	groupName              string
	groupsaddmemberrequest *GroupsAddMemberRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// AddMember: Adds users to the specified group.
func (r *GroupsService) AddMember(project string, groupName string, groupsaddmemberrequest *GroupsAddMemberRequest) *GroupsAddMemberCall {
	c := &GroupsAddMemberCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.groupName = groupName
	c.groupsaddmemberrequest = groupsaddmemberrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsAddMemberCall) Fields(s ...googleapi.Field) *GroupsAddMemberCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsAddMemberCall) Context(ctx context.Context) *GroupsAddMemberCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsAddMemberCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsAddMemberCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groupsaddmemberrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{groupName}/addMember")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"groupName": c.groupName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.addMember" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupsAddMemberCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds users to the specified group.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.groups.addMember",
	//   "parameterOrder": [
	//     "project",
	//     "groupName"
	//   ],
	//   "parameters": {
	//     "groupName": {
	//       "description": "Name of the group for this request.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{groupName}/addMember",
	//   "request": {
	//     "$ref": "GroupsAddMemberRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.delete":

type GroupsDeleteCall struct {
	s          *Service
	project    string
	groupName  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the specified Group resource.
func (r *GroupsService) Delete(project string, groupName string) *GroupsDeleteCall {
	c := &GroupsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.groupName = groupName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsDeleteCall) Fields(s ...googleapi.Field) *GroupsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsDeleteCall) Context(ctx context.Context) *GroupsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{groupName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"groupName": c.groupName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupsDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified Group resource.",
	//   "httpMethod": "DELETE",
	//   "id": "clouduseraccounts.groups.delete",
	//   "parameterOrder": [
	//     "project",
	//     "groupName"
	//   ],
	//   "parameters": {
	//     "groupName": {
	//       "description": "Name of the Group resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{groupName}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.get":

type GroupsGetCall struct {
	s            *Service
	project      string
	groupName    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the specified Group resource.
func (r *GroupsService) Get(project string, groupName string) *GroupsGetCall {
	c := &GroupsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.groupName = groupName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsGetCall) Fields(s ...googleapi.Field) *GroupsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupsGetCall) IfNoneMatch(entityTag string) *GroupsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsGetCall) Context(ctx context.Context) *GroupsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{groupName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"groupName": c.groupName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.get" call.
// Exactly one of *Group or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Group.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsGetCall) Do(opts ...googleapi.CallOption) (*Group, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Group{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified Group resource.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.groups.get",
	//   "parameterOrder": [
	//     "project",
	//     "groupName"
	//   ],
	//   "parameters": {
	//     "groupName": {
	//       "description": "Name of the Group resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{groupName}",
	//   "response": {
	//     "$ref": "Group"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.getIamPolicy":

type GroupsGetIamPolicyCall struct {
	s            *Service
	project      string
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetIamPolicy: Gets the access control policy for a resource. May be
// empty if no such policy or resource exists.
func (r *GroupsService) GetIamPolicy(project string, resource string) *GroupsGetIamPolicyCall {
	c := &GroupsGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsGetIamPolicyCall) Fields(s ...googleapi.Field) *GroupsGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupsGetIamPolicyCall) IfNoneMatch(entityTag string) *GroupsGetIamPolicyCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsGetIamPolicyCall) Context(ctx context.Context) *GroupsGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{resource}/getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the access control policy for a resource. May be empty if no such policy or resource exists.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.groups.getIamPolicy",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{resource}/getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.insert":

type GroupsInsertCall struct {
	s          *Service
	project    string
	group      *Group
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a Group resource in the specified project using the
// data included in the request.
func (r *GroupsService) Insert(project string, group *Group) *GroupsInsertCall {
	c := &GroupsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.group = group
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsInsertCall) Fields(s ...googleapi.Field) *GroupsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsInsertCall) Context(ctx context.Context) *GroupsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.group)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupsInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a Group resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.groups.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups",
	//   "request": {
	//     "$ref": "Group"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.list":

type GroupsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the list of groups contained within the specified
// project.
func (r *GroupsService) List(project string) *GroupsListCall {
	c := &GroupsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must be in the format: field_name comparison_string
// literal_string.
//
// The field_name is the name of the field you want to compare. Only
// atomic field types are supported (string, number, boolean). The
// comparison_string must be either eq (equals) or ne (not equals). The
// literal_string is the string value to filter to. The literal value
// must be valid for the type of field you are filtering by (string,
// number, boolean). For string fields, the literal value is interpreted
// as a regular expression using RE2 syntax. The literal value must
// match the entire field.
//
// For example, to filter for instances that do not have a name of
// example-instance, you would use filter=name ne
// example-instance.
//
// Compute Engine Beta API Only: If you use filtering in the Beta API,
// you can also filter on nested fields. For example, you could filter
// on instances that have set the scheduling.automaticRestart field to
// true. In particular, use filtering on nested fields to take advantage
// of instance labels to organize and filter results based on label
// values.
//
// The Beta API also supports filtering on multiple expressions by
// providing each separate expression within parentheses. For example,
// (scheduling.automaticRestart eq true) (zone eq us-central1-f).
// Multiple expressions are treated as AND expressions, meaning that
// resources must match all expressions to pass the filters.
func (c *GroupsListCall) Filter(filter string) *GroupsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results per page that should be returned. If the number of
// available results is larger than maxResults, Compute Engine returns a
// nextPageToken that can be used to get the next page of results in
// subsequent list requests.
func (c *GroupsListCall) MaxResults(maxResults int64) *GroupsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderBy sets the optional parameter "orderBy": Sorts list results by
// a certain order. By default, results are returned in alphanumerical
// order based on the resource name.
//
// You can also sort results in descending order based on the creation
// timestamp using orderBy="creationTimestamp desc". This sorts results
// based on the creationTimestamp field in reverse chronological order
// (newest result first). Use this to sort resources like operations so
// that the newest operation is returned first.
//
// Currently, only sorting by name or creationTimestamp desc is
// supported.
func (c *GroupsListCall) OrderBy(orderBy string) *GroupsListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Set pageToken to the nextPageToken returned by a
// previous list request to get the next page of results.
func (c *GroupsListCall) PageToken(pageToken string) *GroupsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsListCall) Fields(s ...googleapi.Field) *GroupsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupsListCall) IfNoneMatch(entityTag string) *GroupsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsListCall) Context(ctx context.Context) *GroupsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.list" call.
// Exactly one of *GroupList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *GroupList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupsListCall) Do(opts ...googleapi.CallOption) (*GroupList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &GroupList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of groups contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.groups.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must be in the format: field_name comparison_string literal_string.\n\nThe field_name is the name of the field you want to compare. Only atomic field types are supported (string, number, boolean). The comparison_string must be either eq (equals) or ne (not equals). The literal_string is the string value to filter to. The literal value must be valid for the type of field you are filtering by (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.\n\nFor example, to filter for instances that do not have a name of example-instance, you would use filter=name ne example-instance.\n\nCompute Engine Beta API Only: If you use filtering in the Beta API, you can also filter on nested fields. For example, you could filter on instances that have set the scheduling.automaticRestart field to true. In particular, use filtering on nested fields to take advantage of instance labels to organize and filter results based on label values.\n\nThe Beta API also supports filtering on multiple expressions by providing each separate expression within parentheses. For example, (scheduling.automaticRestart eq true) (zone eq us-central1-f). Multiple expressions are treated as AND expressions, meaning that resources must match all expressions to pass the filters.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "The maximum number of results per page that should be returned. If the number of available results is larger than maxResults, Compute Engine returns a nextPageToken that can be used to get the next page of results in subsequent list requests.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Sorts list results by a certain order. By default, results are returned in alphanumerical order based on the resource name.\n\nYou can also sort results in descending order based on the creation timestamp using orderBy=\"creationTimestamp desc\". This sorts results based on the creationTimestamp field in reverse chronological order (newest result first). Use this to sort resources like operations so that the newest operation is returned first.\n\nCurrently, only sorting by name or creationTimestamp desc is supported.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Set pageToken to the nextPageToken returned by a previous list request to get the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups",
	//   "response": {
	//     "$ref": "GroupList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *GroupsListCall) Pages(ctx context.Context, f func(*GroupList) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "clouduseraccounts.groups.removeMember":

type GroupsRemoveMemberCall struct {
	s                         *Service
	project                   string
	groupName                 string
	groupsremovememberrequest *GroupsRemoveMemberRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// RemoveMember: Removes users from the specified group.
func (r *GroupsService) RemoveMember(project string, groupName string, groupsremovememberrequest *GroupsRemoveMemberRequest) *GroupsRemoveMemberCall {
	c := &GroupsRemoveMemberCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.groupName = groupName
	c.groupsremovememberrequest = groupsremovememberrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsRemoveMemberCall) Fields(s ...googleapi.Field) *GroupsRemoveMemberCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsRemoveMemberCall) Context(ctx context.Context) *GroupsRemoveMemberCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsRemoveMemberCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsRemoveMemberCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groupsremovememberrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{groupName}/removeMember")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"groupName": c.groupName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.removeMember" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupsRemoveMemberCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes users from the specified group.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.groups.removeMember",
	//   "parameterOrder": [
	//     "project",
	//     "groupName"
	//   ],
	//   "parameters": {
	//     "groupName": {
	//       "description": "Name of the group for this request.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{groupName}/removeMember",
	//   "request": {
	//     "$ref": "GroupsRemoveMemberRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.setIamPolicy":

type GroupsSetIamPolicyCall struct {
	s          *Service
	project    string
	resource   string
	policy     *Policy
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any existing policy.
func (r *GroupsService) SetIamPolicy(project string, resource string, policy *Policy) *GroupsSetIamPolicyCall {
	c := &GroupsSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	c.policy = policy
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsSetIamPolicyCall) Fields(s ...googleapi.Field) *GroupsSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsSetIamPolicyCall) Context(ctx context.Context) *GroupsSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.policy)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{resource}/setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the access control policy on the specified resource. Replaces any existing policy.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.groups.setIamPolicy",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{resource}/setIamPolicy",
	//   "request": {
	//     "$ref": "Policy"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.groups.testIamPermissions":

type GroupsTestIamPermissionsCall struct {
	s                      *Service
	project                string
	resource               string
	testpermissionsrequest *TestPermissionsRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
func (r *GroupsService) TestIamPermissions(project string, resource string, testpermissionsrequest *TestPermissionsRequest) *GroupsTestIamPermissionsCall {
	c := &GroupsTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	c.testpermissionsrequest = testpermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsTestIamPermissionsCall) Fields(s ...googleapi.Field) *GroupsTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsTestIamPermissionsCall) Context(ctx context.Context) *GroupsTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *GroupsTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *GroupsTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testpermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/groups/{resource}/testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.groups.testIamPermissions" call.
// Exactly one of *TestPermissionsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TestPermissionsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GroupsTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestPermissionsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TestPermissionsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns permissions that a caller has on the specified resource.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.groups.testIamPermissions",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/groups/{resource}/testIamPermissions",
	//   "request": {
	//     "$ref": "TestPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.linux.getAuthorizedKeysView":

type LinuxGetAuthorizedKeysViewCall struct {
	s          *Service
	project    string
	zone       string
	user       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// GetAuthorizedKeysView: Returns a list of authorized public keys for a
// specific user account.
func (r *LinuxService) GetAuthorizedKeysView(project string, zone string, user string, instance string) *LinuxGetAuthorizedKeysViewCall {
	c := &LinuxGetAuthorizedKeysViewCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.user = user
	c.urlParams_.Set("instance", instance)
	return c
}

// Login sets the optional parameter "login": Whether the view was
// requested as part of a user-initiated login.
func (c *LinuxGetAuthorizedKeysViewCall) Login(login bool) *LinuxGetAuthorizedKeysViewCall {
	c.urlParams_.Set("login", fmt.Sprint(login))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LinuxGetAuthorizedKeysViewCall) Fields(s ...googleapi.Field) *LinuxGetAuthorizedKeysViewCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LinuxGetAuthorizedKeysViewCall) Context(ctx context.Context) *LinuxGetAuthorizedKeysViewCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LinuxGetAuthorizedKeysViewCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LinuxGetAuthorizedKeysViewCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/authorizedKeysView/{user}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
		"user":    c.user,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.linux.getAuthorizedKeysView" call.
// Exactly one of *LinuxGetAuthorizedKeysViewResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *LinuxGetAuthorizedKeysViewResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *LinuxGetAuthorizedKeysViewCall) Do(opts ...googleapi.CallOption) (*LinuxGetAuthorizedKeysViewResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &LinuxGetAuthorizedKeysViewResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of authorized public keys for a specific user account.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.linux.getAuthorizedKeysView",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "user",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "The fully-qualified URL of the virtual machine requesting the view.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "login": {
	//       "description": "Whether the view was requested as part of a user-initiated login.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "user": {
	//       "description": "The user account for which you want to get a list of authorized public keys.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/authorizedKeysView/{user}",
	//   "response": {
	//     "$ref": "LinuxGetAuthorizedKeysViewResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.linux.getLinuxAccountViews":

type LinuxGetLinuxAccountViewsCall struct {
	s          *Service
	project    string
	zone       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// GetLinuxAccountViews: Retrieves a list of user accounts for an
// instance within a specific project.
func (r *LinuxService) GetLinuxAccountViews(project string, zone string, instance string) *LinuxGetLinuxAccountViewsCall {
	c := &LinuxGetLinuxAccountViewsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.urlParams_.Set("instance", instance)
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must be in the format: field_name comparison_string
// literal_string.
//
// The field_name is the name of the field you want to compare. Only
// atomic field types are supported (string, number, boolean). The
// comparison_string must be either eq (equals) or ne (not equals). The
// literal_string is the string value to filter to. The literal value
// must be valid for the type of field you are filtering by (string,
// number, boolean). For string fields, the literal value is interpreted
// as a regular expression using RE2 syntax. The literal value must
// match the entire field.
//
// For example, to filter for instances that do not have a name of
// example-instance, you would use filter=name ne
// example-instance.
//
// Compute Engine Beta API Only: If you use filtering in the Beta API,
// you can also filter on nested fields. For example, you could filter
// on instances that have set the scheduling.automaticRestart field to
// true. In particular, use filtering on nested fields to take advantage
// of instance labels to organize and filter results based on label
// values.
//
// The Beta API also supports filtering on multiple expressions by
// providing each separate expression within parentheses. For example,
// (scheduling.automaticRestart eq true) (zone eq us-central1-f).
// Multiple expressions are treated as AND expressions, meaning that
// resources must match all expressions to pass the filters.
func (c *LinuxGetLinuxAccountViewsCall) Filter(filter string) *LinuxGetLinuxAccountViewsCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results per page that should be returned. If the number of
// available results is larger than maxResults, Compute Engine returns a
// nextPageToken that can be used to get the next page of results in
// subsequent list requests.
func (c *LinuxGetLinuxAccountViewsCall) MaxResults(maxResults int64) *LinuxGetLinuxAccountViewsCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderBy sets the optional parameter "orderBy": Sorts list results by
// a certain order. By default, results are returned in alphanumerical
// order based on the resource name.
//
// You can also sort results in descending order based on the creation
// timestamp using orderBy="creationTimestamp desc". This sorts results
// based on the creationTimestamp field in reverse chronological order
// (newest result first). Use this to sort resources like operations so
// that the newest operation is returned first.
//
// Currently, only sorting by name or creationTimestamp desc is
// supported.
func (c *LinuxGetLinuxAccountViewsCall) OrderBy(orderBy string) *LinuxGetLinuxAccountViewsCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Set pageToken to the nextPageToken returned by a
// previous list request to get the next page of results.
func (c *LinuxGetLinuxAccountViewsCall) PageToken(pageToken string) *LinuxGetLinuxAccountViewsCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *LinuxGetLinuxAccountViewsCall) Fields(s ...googleapi.Field) *LinuxGetLinuxAccountViewsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *LinuxGetLinuxAccountViewsCall) Context(ctx context.Context) *LinuxGetLinuxAccountViewsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *LinuxGetLinuxAccountViewsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *LinuxGetLinuxAccountViewsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/linuxAccountViews")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.linux.getLinuxAccountViews" call.
// Exactly one of *LinuxGetLinuxAccountViewsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *LinuxGetLinuxAccountViewsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *LinuxGetLinuxAccountViewsCall) Do(opts ...googleapi.CallOption) (*LinuxGetLinuxAccountViewsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &LinuxGetLinuxAccountViewsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of user accounts for an instance within a specific project.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.linux.getLinuxAccountViews",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must be in the format: field_name comparison_string literal_string.\n\nThe field_name is the name of the field you want to compare. Only atomic field types are supported (string, number, boolean). The comparison_string must be either eq (equals) or ne (not equals). The literal_string is the string value to filter to. The literal value must be valid for the type of field you are filtering by (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.\n\nFor example, to filter for instances that do not have a name of example-instance, you would use filter=name ne example-instance.\n\nCompute Engine Beta API Only: If you use filtering in the Beta API, you can also filter on nested fields. For example, you could filter on instances that have set the scheduling.automaticRestart field to true. In particular, use filtering on nested fields to take advantage of instance labels to organize and filter results based on label values.\n\nThe Beta API also supports filtering on multiple expressions by providing each separate expression within parentheses. For example, (scheduling.automaticRestart eq true) (zone eq us-central1-f). Multiple expressions are treated as AND expressions, meaning that resources must match all expressions to pass the filters.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "The fully-qualified URL of the virtual machine requesting the views.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "The maximum number of results per page that should be returned. If the number of available results is larger than maxResults, Compute Engine returns a nextPageToken that can be used to get the next page of results in subsequent list requests.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Sorts list results by a certain order. By default, results are returned in alphanumerical order based on the resource name.\n\nYou can also sort results in descending order based on the creation timestamp using orderBy=\"creationTimestamp desc\". This sorts results based on the creationTimestamp field in reverse chronological order (newest result first). Use this to sort resources like operations so that the newest operation is returned first.\n\nCurrently, only sorting by name or creationTimestamp desc is supported.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Set pageToken to the nextPageToken returned by a previous list request to get the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/linuxAccountViews",
	//   "response": {
	//     "$ref": "LinuxGetLinuxAccountViewsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.users.addPublicKey":

type UsersAddPublicKeyCall struct {
	s          *Service
	project    string
	user       string
	publickey  *PublicKey
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// AddPublicKey: Adds a public key to the specified User resource with
// the data included in the request.
func (r *UsersService) AddPublicKey(project string, user string, publickey *PublicKey) *UsersAddPublicKeyCall {
	c := &UsersAddPublicKeyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.user = user
	c.publickey = publickey
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersAddPublicKeyCall) Fields(s ...googleapi.Field) *UsersAddPublicKeyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersAddPublicKeyCall) Context(ctx context.Context) *UsersAddPublicKeyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersAddPublicKeyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersAddPublicKeyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.publickey)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{user}/addPublicKey")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"user":    c.user,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.addPublicKey" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersAddPublicKeyCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a public key to the specified User resource with the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.users.addPublicKey",
	//   "parameterOrder": [
	//     "project",
	//     "user"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "user": {
	//       "description": "Name of the user for this request.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{user}/addPublicKey",
	//   "request": {
	//     "$ref": "PublicKey"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.users.delete":

type UsersDeleteCall struct {
	s          *Service
	project    string
	user       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the specified User resource.
func (r *UsersService) Delete(project string, user string) *UsersDeleteCall {
	c := &UsersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDeleteCall) Fields(s ...googleapi.Field) *UsersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDeleteCall) Context(ctx context.Context) *UsersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{user}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"user":    c.user,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified User resource.",
	//   "httpMethod": "DELETE",
	//   "id": "clouduseraccounts.users.delete",
	//   "parameterOrder": [
	//     "project",
	//     "user"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "user": {
	//       "description": "Name of the user resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{user}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.users.get":

type UsersGetCall struct {
	s            *Service
	project      string
	user         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the specified User resource.
func (r *UsersService) Get(project string, user string) *UsersGetCall {
	c := &UsersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetCall) Fields(s ...googleapi.Field) *UsersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetCall) IfNoneMatch(entityTag string) *UsersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetCall) Context(ctx context.Context) *UsersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{user}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"user":    c.user,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.get" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersGetCall) Do(opts ...googleapi.CallOption) (*User, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &User{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified User resource.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.users.get",
	//   "parameterOrder": [
	//     "project",
	//     "user"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "user": {
	//       "description": "Name of the user resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{user}",
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.users.getIamPolicy":

type UsersGetIamPolicyCall struct {
	s            *Service
	project      string
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetIamPolicy: Gets the access control policy for a resource. May be
// empty if no such policy or resource exists.
func (r *UsersService) GetIamPolicy(project string, resource string) *UsersGetIamPolicyCall {
	c := &UsersGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetIamPolicyCall) Fields(s ...googleapi.Field) *UsersGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetIamPolicyCall) IfNoneMatch(entityTag string) *UsersGetIamPolicyCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetIamPolicyCall) Context(ctx context.Context) *UsersGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{resource}/getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the access control policy for a resource. May be empty if no such policy or resource exists.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.users.getIamPolicy",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{resource}/getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.users.insert":

type UsersInsertCall struct {
	s          *Service
	project    string
	user       *User
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a User resource in the specified project using the
// data included in the request.
func (r *UsersService) Insert(project string, user *User) *UsersInsertCall {
	c := &UsersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersInsertCall) Fields(s ...googleapi.Field) *UsersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersInsertCall) Context(ctx context.Context) *UsersInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a User resource in the specified project using the data included in the request.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.users.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.users.list":

type UsersListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves a list of users contained within the specified
// project.
func (r *UsersService) List(project string) *UsersListCall {
	c := &UsersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": Sets a filter expression
// for filtering listed resources, in the form filter={expression}. Your
// {expression} must be in the format: field_name comparison_string
// literal_string.
//
// The field_name is the name of the field you want to compare. Only
// atomic field types are supported (string, number, boolean). The
// comparison_string must be either eq (equals) or ne (not equals). The
// literal_string is the string value to filter to. The literal value
// must be valid for the type of field you are filtering by (string,
// number, boolean). For string fields, the literal value is interpreted
// as a regular expression using RE2 syntax. The literal value must
// match the entire field.
//
// For example, to filter for instances that do not have a name of
// example-instance, you would use filter=name ne
// example-instance.
//
// Compute Engine Beta API Only: If you use filtering in the Beta API,
// you can also filter on nested fields. For example, you could filter
// on instances that have set the scheduling.automaticRestart field to
// true. In particular, use filtering on nested fields to take advantage
// of instance labels to organize and filter results based on label
// values.
//
// The Beta API also supports filtering on multiple expressions by
// providing each separate expression within parentheses. For example,
// (scheduling.automaticRestart eq true) (zone eq us-central1-f).
// Multiple expressions are treated as AND expressions, meaning that
// resources must match all expressions to pass the filters.
func (c *UsersListCall) Filter(filter string) *UsersListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results per page that should be returned. If the number of
// available results is larger than maxResults, Compute Engine returns a
// nextPageToken that can be used to get the next page of results in
// subsequent list requests.
func (c *UsersListCall) MaxResults(maxResults int64) *UsersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// OrderBy sets the optional parameter "orderBy": Sorts list results by
// a certain order. By default, results are returned in alphanumerical
// order based on the resource name.
//
// You can also sort results in descending order based on the creation
// timestamp using orderBy="creationTimestamp desc". This sorts results
// based on the creationTimestamp field in reverse chronological order
// (newest result first). Use this to sort resources like operations so
// that the newest operation is returned first.
//
// Currently, only sorting by name or creationTimestamp desc is
// supported.
func (c *UsersListCall) OrderBy(orderBy string) *UsersListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a page
// token to use. Set pageToken to the nextPageToken returned by a
// previous list request to get the next page of results.
func (c *UsersListCall) PageToken(pageToken string) *UsersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersListCall) Fields(s ...googleapi.Field) *UsersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersListCall) IfNoneMatch(entityTag string) *UsersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersListCall) Context(ctx context.Context) *UsersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.list" call.
// Exactly one of *UserList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *UserList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersListCall) Do(opts ...googleapi.CallOption) (*UserList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &UserList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of users contained within the specified project.",
	//   "httpMethod": "GET",
	//   "id": "clouduseraccounts.users.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Sets a filter expression for filtering listed resources, in the form filter={expression}. Your {expression} must be in the format: field_name comparison_string literal_string.\n\nThe field_name is the name of the field you want to compare. Only atomic field types are supported (string, number, boolean). The comparison_string must be either eq (equals) or ne (not equals). The literal_string is the string value to filter to. The literal value must be valid for the type of field you are filtering by (string, number, boolean). For string fields, the literal value is interpreted as a regular expression using RE2 syntax. The literal value must match the entire field.\n\nFor example, to filter for instances that do not have a name of example-instance, you would use filter=name ne example-instance.\n\nCompute Engine Beta API Only: If you use filtering in the Beta API, you can also filter on nested fields. For example, you could filter on instances that have set the scheduling.automaticRestart field to true. In particular, use filtering on nested fields to take advantage of instance labels to organize and filter results based on label values.\n\nThe Beta API also supports filtering on multiple expressions by providing each separate expression within parentheses. For example, (scheduling.automaticRestart eq true) (zone eq us-central1-f). Multiple expressions are treated as AND expressions, meaning that resources must match all expressions to pass the filters.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "The maximum number of results per page that should be returned. If the number of available results is larger than maxResults, Compute Engine returns a nextPageToken that can be used to get the next page of results in subsequent list requests.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Sorts list results by a certain order. By default, results are returned in alphanumerical order based on the resource name.\n\nYou can also sort results in descending order based on the creation timestamp using orderBy=\"creationTimestamp desc\". This sorts results based on the creationTimestamp field in reverse chronological order (newest result first). Use this to sort resources like operations so that the newest operation is returned first.\n\nCurrently, only sorting by name or creationTimestamp desc is supported.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a page token to use. Set pageToken to the nextPageToken returned by a previous list request to get the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users",
	//   "response": {
	//     "$ref": "UserList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *UsersListCall) Pages(ctx context.Context, f func(*UserList) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "clouduseraccounts.users.removePublicKey":

type UsersRemovePublicKeyCall struct {
	s          *Service
	project    string
	user       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// RemovePublicKey: Removes the specified public key from the user.
func (r *UsersService) RemovePublicKey(project string, user string, fingerprint string) *UsersRemovePublicKeyCall {
	c := &UsersRemovePublicKeyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.user = user
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersRemovePublicKeyCall) Fields(s ...googleapi.Field) *UsersRemovePublicKeyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersRemovePublicKeyCall) Context(ctx context.Context) *UsersRemovePublicKeyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersRemovePublicKeyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersRemovePublicKeyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{user}/removePublicKey")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"user":    c.user,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.removePublicKey" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersRemovePublicKeyCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Removes the specified public key from the user.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.users.removePublicKey",
	//   "parameterOrder": [
	//     "project",
	//     "user",
	//     "fingerprint"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "The fingerprint of the public key to delete. Public keys are identified by their fingerprint, which is defined by RFC4716 to be the MD5 digest of the public key.",
	//       "location": "query",
	//       "pattern": "[a-f0-9]{32}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "user": {
	//       "description": "Name of the user for this request.",
	//       "location": "path",
	//       "pattern": "[a-z][-a-z0-9_]{0,31}",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{user}/removePublicKey",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud.useraccounts"
	//   ]
	// }

}

// method id "clouduseraccounts.users.setIamPolicy":

type UsersSetIamPolicyCall struct {
	s          *Service
	project    string
	resource   string
	policy     *Policy
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any existing policy.
func (r *UsersService) SetIamPolicy(project string, resource string, policy *Policy) *UsersSetIamPolicyCall {
	c := &UsersSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	c.policy = policy
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersSetIamPolicyCall) Fields(s ...googleapi.Field) *UsersSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersSetIamPolicyCall) Context(ctx context.Context) *UsersSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.policy)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{resource}/setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the access control policy on the specified resource. Replaces any existing policy.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.users.setIamPolicy",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{resource}/setIamPolicy",
	//   "request": {
	//     "$ref": "Policy"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}

// method id "clouduseraccounts.users.testIamPermissions":

type UsersTestIamPermissionsCall struct {
	s                      *Service
	project                string
	resource               string
	testpermissionsrequest *TestPermissionsRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
func (r *UsersService) TestIamPermissions(project string, resource string, testpermissionsrequest *TestPermissionsRequest) *UsersTestIamPermissionsCall {
	c := &UsersTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.resource = resource
	c.testpermissionsrequest = testpermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersTestIamPermissionsCall) Fields(s ...googleapi.Field) *UsersTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersTestIamPermissionsCall) Context(ctx context.Context) *UsersTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testpermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/users/{resource}/testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "clouduseraccounts.users.testIamPermissions" call.
// Exactly one of *TestPermissionsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TestPermissionsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestPermissionsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TestPermissionsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns permissions that a caller has on the specified resource.",
	//   "httpMethod": "POST",
	//   "id": "clouduseraccounts.users.testIamPermissions",
	//   "parameterOrder": [
	//     "project",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "Name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9_]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/users/{resource}/testIamPermissions",
	//   "request": {
	//     "$ref": "TestPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/cloud.useraccounts",
	//     "https://www.googleapis.com/auth/cloud.useraccounts.readonly"
	//   ]
	// }

}
