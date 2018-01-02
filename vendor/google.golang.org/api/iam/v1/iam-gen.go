// Package iam provides access to the Google Identity and Access Management (IAM) API.
//
// See https://cloud.google.com/iam/
//
// Usage example:
//
//   import "google.golang.org/api/iam/v1"
//   ...
//   iamService, err := iam.New(oauthHttpClient)
package iam // import "google.golang.org/api/iam/v1"

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

const apiId = "iam:v1"
const apiName = "iam"
const apiVersion = "v1"
const basePath = "https://iam.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Organizations = NewOrganizationsService(s)
	s.Permissions = NewPermissionsService(s)
	s.Projects = NewProjectsService(s)
	s.Roles = NewRolesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Organizations *OrganizationsService

	Permissions *PermissionsService

	Projects *ProjectsService

	Roles *RolesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewOrganizationsService(s *Service) *OrganizationsService {
	rs := &OrganizationsService{s: s}
	rs.Roles = NewOrganizationsRolesService(s)
	return rs
}

type OrganizationsService struct {
	s *Service

	Roles *OrganizationsRolesService
}

func NewOrganizationsRolesService(s *Service) *OrganizationsRolesService {
	rs := &OrganizationsRolesService{s: s}
	return rs
}

type OrganizationsRolesService struct {
	s *Service
}

func NewPermissionsService(s *Service) *PermissionsService {
	rs := &PermissionsService{s: s}
	return rs
}

type PermissionsService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Roles = NewProjectsRolesService(s)
	rs.ServiceAccounts = NewProjectsServiceAccountsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Roles *ProjectsRolesService

	ServiceAccounts *ProjectsServiceAccountsService
}

func NewProjectsRolesService(s *Service) *ProjectsRolesService {
	rs := &ProjectsRolesService{s: s}
	return rs
}

type ProjectsRolesService struct {
	s *Service
}

func NewProjectsServiceAccountsService(s *Service) *ProjectsServiceAccountsService {
	rs := &ProjectsServiceAccountsService{s: s}
	rs.Keys = NewProjectsServiceAccountsKeysService(s)
	return rs
}

type ProjectsServiceAccountsService struct {
	s *Service

	Keys *ProjectsServiceAccountsKeysService
}

func NewProjectsServiceAccountsKeysService(s *Service) *ProjectsServiceAccountsKeysService {
	rs := &ProjectsServiceAccountsKeysService{s: s}
	return rs
}

type ProjectsServiceAccountsKeysService struct {
	s *Service
}

func NewRolesService(s *Service) *RolesService {
	rs := &RolesService{s: s}
	return rs
}

type RolesService struct {
	s *Service
}

// AuditData: Audit log information specific to Cloud IAM. This message
// is serialized
// as an `Any` type in the `ServiceData` message of an
// `AuditLog` message.
type AuditData struct {
	// PolicyDelta: Policy delta between the original policy and the newly
	// set policy.
	PolicyDelta *PolicyDelta `json:"policyDelta,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PolicyDelta") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PolicyDelta") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuditData) MarshalJSON() ([]byte, error) {
	type noMethod AuditData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Members: Specifies the identities requesting access for a Cloud
	// Platform resource.
	// `members` can have the following values:
	//
	// * `allUsers`: A special identifier that represents anyone who is
	//    on the internet; with or without a Google account.
	//
	// * `allAuthenticatedUsers`: A special identifier that represents
	// anyone
	//    who is authenticated with a Google account or a service
	// account.
	//
	// * `user:{emailid}`: An email address that represents a specific
	// Google
	//    account. For example, `alice@gmail.com` or `joe@example.com`.
	//
	//
	// * `serviceAccount:{emailid}`: An email address that represents a
	// service
	//    account. For example,
	// `my-other-app@appspot.gserviceaccount.com`.
	//
	// * `group:{emailid}`: An email address that represents a Google
	// group.
	//    For example, `admins@example.com`.
	//
	//
	// * `domain:{domain}`: A Google Apps domain name that represents all
	// the
	//    users of that domain. For example, `google.com` or
	// `example.com`.
	//
	//
	Members []string `json:"members,omitempty"`

	// Role: Role that is assigned to `members`.
	// For example, `roles/viewer`, `roles/editor`, or
	// `roles/owner`.
	// Required
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

// BindingDelta: One delta entry for Binding. Each individual change
// (only one member in each
// entry) to a binding will be a separate entry.
type BindingDelta struct {
	// Action: The action that was performed on a Binding.
	// Required
	//
	// Possible values:
	//   "ACTION_UNSPECIFIED" - Unspecified.
	//   "ADD" - Addition of a Binding.
	//   "REMOVE" - Removal of a Binding.
	Action string `json:"action,omitempty"`

	// Condition: The condition that is associated with this binding.
	// This field is GOOGLE_INTERNAL.
	// This field is not logged in IAM side because it's only for audit
	// logging.
	// Optional
	Condition *Expr `json:"condition,omitempty"`

	// Member: A single identity requesting access for a Cloud Platform
	// resource.
	// Follows the same format of Binding.members.
	// Required
	Member string `json:"member,omitempty"`

	// Role: Role that is assigned to `members`.
	// For example, `roles/viewer`, `roles/editor`, or
	// `roles/owner`.
	// Required
	Role string `json:"role,omitempty"`

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

func (s *BindingDelta) MarshalJSON() ([]byte, error) {
	type noMethod BindingDelta
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateRoleRequest: The request to create a new role.
type CreateRoleRequest struct {
	// Role: The Role resource to create.
	Role *Role `json:"role,omitempty"`

	// RoleId: The role id to use for this role.
	RoleId string `json:"roleId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Role") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Role") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateRoleRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateRoleRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateServiceAccountKeyRequest: The service account key create
// request.
type CreateServiceAccountKeyRequest struct {
	// KeyAlgorithm: Which type of key and algorithm to use for the key.
	// The default is currently a 2K RSA key.  However this may change in
	// the
	// future.
	//
	// Possible values:
	//   "KEY_ALG_UNSPECIFIED" - An unspecified key algorithm.
	//   "KEY_ALG_RSA_1024" - 1k RSA Key.
	//   "KEY_ALG_RSA_2048" - 2k RSA Key.
	KeyAlgorithm string `json:"keyAlgorithm,omitempty"`

	// PrivateKeyType: The output format of the private key.
	// `GOOGLE_CREDENTIALS_FILE` is the
	// default output format.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED" - Unspecified. Equivalent to
	// `TYPE_GOOGLE_CREDENTIALS_FILE`.
	//   "TYPE_PKCS12_FILE" - PKCS12 format.
	// The password for the PKCS12 file is `notasecret`.
	// For more information, see https://tools.ietf.org/html/rfc7292.
	//   "TYPE_GOOGLE_CREDENTIALS_FILE" - Google Credentials File format.
	PrivateKeyType string `json:"privateKeyType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "KeyAlgorithm") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KeyAlgorithm") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateServiceAccountKeyRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateServiceAccountKeyRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateServiceAccountRequest: The service account create request.
type CreateServiceAccountRequest struct {
	// AccountId: Required. The account id that is used to generate the
	// service account
	// email address and a stable unique id. It is unique within a
	// project,
	// must be 6-30 characters long, and match the regular
	// expression
	// `[a-z]([-a-z0-9]*[a-z0-9])` to comply with RFC1035.
	AccountId string `json:"accountId,omitempty"`

	// ServiceAccount: The ServiceAccount resource to create.
	// Currently, only the following values are user
	// assignable:
	// `display_name` .
	ServiceAccount *ServiceAccount `json:"serviceAccount,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateServiceAccountRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateServiceAccountRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated
// empty messages in your APIs. A typical example is to use it as the
// request
// or the response type of an API method. For instance:
//
//     service Foo {
//       rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
//     }
//
// The JSON representation for `Empty` is empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// Expr: Represents an expression text. Example:
//
//     title: "User account presence"
//     description: "Determines whether the request has a user account"
//     expression: "size(request.user) > 0"
type Expr struct {
	// Description: An optional description of the expression. This is a
	// longer text which
	// describes the expression, e.g. when hovered over it in a UI.
	Description string `json:"description,omitempty"`

	// Expression: Textual representation of an expression in
	// Common Expression Language syntax.
	//
	// The application context of the containing message determines
	// which
	// well-known feature set of CEL is supported.
	Expression string `json:"expression,omitempty"`

	// Location: An optional string indicating the location of the
	// expression for error
	// reporting, e.g. a file name and a position in the file.
	Location string `json:"location,omitempty"`

	// Title: An optional title for the expression, i.e. a short string
	// describing
	// its purpose. This can be used e.g. in UIs which allow to enter
	// the
	// expression.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Expr) MarshalJSON() ([]byte, error) {
	type noMethod Expr
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListRolesResponse: The response containing the roles defined under a
// resource.
type ListRolesResponse struct {
	// NextPageToken: To retrieve the next page of results,
	// set
	// `ListRolesRequest.page_token` to this value.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Roles: The Roles defined on this resource.
	Roles []*Role `json:"roles,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListRolesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListRolesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListServiceAccountKeysResponse: The service account keys list
// response.
type ListServiceAccountKeysResponse struct {
	// Keys: The public keys for the service account.
	Keys []*ServiceAccountKey `json:"keys,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *ListServiceAccountKeysResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListServiceAccountKeysResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListServiceAccountsResponse: The service account list response.
type ListServiceAccountsResponse struct {
	// Accounts: The list of matching service accounts.
	Accounts []*ServiceAccount `json:"accounts,omitempty"`

	// NextPageToken: To retrieve the next page of results,
	// set
	// ListServiceAccountsRequest.page_token
	// to this value.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Accounts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Accounts") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListServiceAccountsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListServiceAccountsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Permission: A permission which can be included by a role.
type Permission struct {
	// CustomRolesSupportLevel: The current custom role support level.
	//
	// Possible values:
	//   "SUPPORTED" - Permission is fully supported for custom role use.
	//   "TESTING" - Permission is being tested to check custom role
	// compatibility.
	//   "NOT_SUPPORTED" - Permission is not supported for custom role use.
	CustomRolesSupportLevel string `json:"customRolesSupportLevel,omitempty"`

	// Description: A brief description of what this Permission is used for.
	Description string `json:"description,omitempty"`

	// Name: The name of this Permission.
	Name string `json:"name,omitempty"`

	// OnlyInPredefinedRoles: This permission can ONLY be used in predefined
	// roles.
	OnlyInPredefinedRoles bool `json:"onlyInPredefinedRoles,omitempty"`

	// Stage: The current launch stage of the permission.
	//
	// Possible values:
	//   "ALPHA" - The permission is currently in an alpha phase.
	//   "BETA" - The permission is currently in a beta phase.
	//   "GA" - The permission is generally available.
	//   "DEPRECATED" - The permission is being deprecated.
	Stage string `json:"stage,omitempty"`

	// Title: The title of this Permission.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CustomRolesSupportLevel") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CustomRolesSupportLevel")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Permission) MarshalJSON() ([]byte, error) {
	type noMethod Permission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Policy: Defines an Identity and Access Management (IAM) policy. It is
// used to
// specify access control policies for Cloud Platform resources.
//
//
// A `Policy` consists of a list of `bindings`. A `Binding` binds a list
// of
// `members` to a `role`, where the members can be user accounts, Google
// groups,
// Google domains, and service accounts. A `role` is a named list of
// permissions
// defined by IAM.
//
// **Example**
//
//     {
//       "bindings": [
//         {
//           "role": "roles/owner",
//           "members": [
//             "user:mike@example.com",
//             "group:admins@example.com",
//             "domain:google.com",
//
// "serviceAccount:my-other-app@appspot.gserviceaccount.com",
//           ]
//         },
//         {
//           "role": "roles/viewer",
//           "members": ["user:sean@example.com"]
//         }
//       ]
//     }
//
// For a description of IAM and its features, see the
// [IAM developer's guide](https://cloud.google.com/iam).
type Policy struct {
	// Bindings: Associates a list of `members` to a `role`.
	// `bindings` with no members will result in an error.
	Bindings []*Binding `json:"bindings,omitempty"`

	// Etag: `etag` is used for optimistic concurrency control as a way to
	// help
	// prevent simultaneous updates of a policy from overwriting each
	// other.
	// It is strongly suggested that systems make use of the `etag` in
	// the
	// read-modify-write cycle to perform policy updates in order to avoid
	// race
	// conditions: An `etag` is returned in the response to `getIamPolicy`,
	// and
	// systems are expected to put that etag in the request to
	// `setIamPolicy` to
	// ensure that their change will be applied to the same version of the
	// policy.
	//
	// If no `etag` is provided in the call to `setIamPolicy`, then the
	// existing
	// policy is overwritten blindly.
	Etag string `json:"etag,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Bindings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bindings") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PolicyDelta: The difference delta between two policies.
type PolicyDelta struct {
	// BindingDeltas: The delta for Bindings between two policies.
	BindingDeltas []*BindingDelta `json:"bindingDeltas,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BindingDeltas") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BindingDeltas") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PolicyDelta) MarshalJSON() ([]byte, error) {
	type noMethod PolicyDelta
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryGrantableRolesRequest: The grantable role query request.
type QueryGrantableRolesRequest struct {
	// FullResourceName: Required. The full resource name to query from the
	// list of grantable roles.
	//
	// The name follows the Google Cloud Platform resource format.
	// For example, a Cloud Platform project with id `my-project` will be
	// named
	// `//cloudresourcemanager.googleapis.com/projects/my-project`.
	FullResourceName string `json:"fullResourceName,omitempty"`

	// PageSize: Optional limit on the number of roles to include in the
	// response.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: Optional pagination token returned in an
	// earlier
	// QueryGrantableRolesResponse.
	PageToken string `json:"pageToken,omitempty"`

	// Possible values:
	//   "BASIC" - Omits the `included_permissions` field.
	// This is the default value.
	//   "FULL" - Returns all fields.
	View string `json:"view,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FullResourceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FullResourceName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *QueryGrantableRolesRequest) MarshalJSON() ([]byte, error) {
	type noMethod QueryGrantableRolesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryGrantableRolesResponse: The grantable role query response.
type QueryGrantableRolesResponse struct {
	// NextPageToken: To retrieve the next page of results,
	// set
	// `QueryGrantableRolesRequest.page_token` to this value.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Roles: The list of matching roles.
	Roles []*Role `json:"roles,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryGrantableRolesResponse) MarshalJSON() ([]byte, error) {
	type noMethod QueryGrantableRolesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryTestablePermissionsRequest: A request to get permissions which
// can be tested on a resource.
type QueryTestablePermissionsRequest struct {
	// FullResourceName: Required. The full resource name to query from the
	// list of testable
	// permissions.
	//
	// The name follows the Google Cloud Platform resource format.
	// For example, a Cloud Platform project with id `my-project` will be
	// named
	// `//cloudresourcemanager.googleapis.com/projects/my-project`.
	FullResourceName string `json:"fullResourceName,omitempty"`

	// PageSize: Optional limit on the number of permissions to include in
	// the response.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: Optional pagination token returned in an
	// earlier
	// QueryTestablePermissionsRequest.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FullResourceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FullResourceName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *QueryTestablePermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod QueryTestablePermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryTestablePermissionsResponse: The response containing permissions
// which can be tested on a resource.
type QueryTestablePermissionsResponse struct {
	// NextPageToken: To retrieve the next page of results,
	// set
	// `QueryTestableRolesRequest.page_token` to this value.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Permissions: The Permissions testable on the requested resource.
	Permissions []*Permission `json:"permissions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryTestablePermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod QueryTestablePermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Role: A role in the Identity and Access Management API.
type Role struct {
	// Deleted: The current deleted state of the role. This field is read
	// only.
	// It will be ignored in calls to CreateRole and UpdateRole.
	Deleted bool `json:"deleted,omitempty"`

	// Description: Optional.  A human-readable description for the role.
	Description string `json:"description,omitempty"`

	// Etag: Used to perform a consistent read-modify-write.
	Etag string `json:"etag,omitempty"`

	// IncludedPermissions: The names of the permissions this role grants
	// when bound in an IAM policy.
	IncludedPermissions []string `json:"includedPermissions,omitempty"`

	// Name: The name of the role.
	//
	// When Role is used in CreateRole, the role name must not be set.
	//
	// When Role is used in output and other input such as UpdateRole, the
	// role
	// name is the complete path, e.g., roles/logging.viewer for curated
	// roles
	// and organizations/{ORGANIZATION_ID}/roles/logging.viewer for custom
	// roles.
	Name string `json:"name,omitempty"`

	// Stage: The current launch stage of the role.
	//
	// Possible values:
	//   "ALPHA" - The user has indicated this role is currently in an alpha
	// phase.
	//   "BETA" - The user has indicated this role is currently in a beta
	// phase.
	//   "GA" - The user has indicated this role is generally available.
	//   "DEPRECATED" - The user has indicated this role is being
	// deprecated.
	//   "DISABLED" - This role is disabled and will not contribute
	// permissions to any members
	// it is granted to in policies.
	//   "EAP" - The user has indicated this role is currently in an eap
	// phase.
	Stage string `json:"stage,omitempty"`

	// Title: Optional.  A human-readable title for the role.  Typically
	// this
	// is limited to 100 UTF-8 bytes.
	Title string `json:"title,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Deleted") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Role) MarshalJSON() ([]byte, error) {
	type noMethod Role
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServiceAccount: A service account in the Identity and Access
// Management API.
//
// To create a service account, specify the `project_id` and the
// `account_id`
// for the account.  The `account_id` is unique within the project, and
// is used
// to generate the service account email address and a
// stable
// `unique_id`.
//
// If the account already exists, the account's resource name is
// returned
// in util::Status's ResourceInfo.resource_name in the format
// of
// projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}. The
// caller can
// use the name in other methods to access the account.
//
// All other methods can identify the service account using the
// format
// `projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`
// .
// Using `-` as a wildcard for the project will infer the project
// from
// the account. The `account` value can be the `email` address or
// the
// `unique_id` of the service account.
type ServiceAccount struct {
	// DisplayName: Optional. A user-specified description of the service
	// account.  Must be
	// fewer than 100 UTF-8 bytes.
	DisplayName string `json:"displayName,omitempty"`

	// Email: @OutputOnly The email address of the service account.
	Email string `json:"email,omitempty"`

	// Etag: Used to perform a consistent read-modify-write.
	Etag string `json:"etag,omitempty"`

	// Name: The resource name of the service account in the following
	// format:
	// `projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}
	// `.
	//
	// Requests using `-` as a wildcard for the project will infer the
	// project
	// from the `account` and the `account` value can be the `email` address
	// or
	// the `unique_id` of the service account.
	//
	// In responses the resource name will always be in the
	// format
	// `projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`
	// .
	Name string `json:"name,omitempty"`

	// Oauth2ClientId: @OutputOnly. The OAuth2 client id for the service
	// account.
	// This is used in conjunction with the OAuth2 clientconfig API to
	// make
	// three legged OAuth2 (3LO) flows to access the data of Google users.
	Oauth2ClientId string `json:"oauth2ClientId,omitempty"`

	// ProjectId: @OutputOnly The id of the project that owns the service
	// account.
	ProjectId string `json:"projectId,omitempty"`

	// UniqueId: @OutputOnly The unique and stable id of the service
	// account.
	UniqueId string `json:"uniqueId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ServiceAccount) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ServiceAccountKey: Represents a service account key.
//
// A service account has two sets of key-pairs: user-managed,
// and
// system-managed.
//
// User-managed key-pairs can be created and deleted by users.  Users
// are
// responsible for rotating these keys periodically to ensure security
// of
// their service accounts.  Users retain the private key of these
// key-pairs,
// and Google retains ONLY the public key.
//
// System-managed key-pairs are managed automatically by Google, and
// rotated
// daily without user intervention.  The private key never leaves
// Google's
// servers to maximize security.
//
// Public keys for all service accounts are also published at the
// OAuth2
// Service Account API.
type ServiceAccountKey struct {
	// KeyAlgorithm: Specifies the algorithm (and possibly key size) for the
	// key.
	//
	// Possible values:
	//   "KEY_ALG_UNSPECIFIED" - An unspecified key algorithm.
	//   "KEY_ALG_RSA_1024" - 1k RSA Key.
	//   "KEY_ALG_RSA_2048" - 2k RSA Key.
	KeyAlgorithm string `json:"keyAlgorithm,omitempty"`

	// Name: The resource name of the service account key in the following
	// format
	// `projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}/
	// keys/{key}`.
	Name string `json:"name,omitempty"`

	// PrivateKeyData: The private key data. Only provided in
	// `CreateServiceAccountKey`
	// responses. Make sure to keep the private key data secure because
	// it
	// allows for the assertion of the service account identity.
	// When decoded, the private key data can be used to authenticate
	// with
	// Google API client libraries and with
	// <a
	// href="/sdk/gcloud/reference/auth/activate-service-account">gcloud
	// auth
	//  activate-service-account</a>.
	PrivateKeyData string `json:"privateKeyData,omitempty"`

	// PrivateKeyType: The output format for the private key.
	// Only provided in `CreateServiceAccountKey` responses, not
	// in `GetServiceAccountKey` or `ListServiceAccountKey`
	// responses.
	//
	// Google never exposes system-managed private keys, and never
	// retains
	// user-managed private keys.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED" - Unspecified. Equivalent to
	// `TYPE_GOOGLE_CREDENTIALS_FILE`.
	//   "TYPE_PKCS12_FILE" - PKCS12 format.
	// The password for the PKCS12 file is `notasecret`.
	// For more information, see https://tools.ietf.org/html/rfc7292.
	//   "TYPE_GOOGLE_CREDENTIALS_FILE" - Google Credentials File format.
	PrivateKeyType string `json:"privateKeyType,omitempty"`

	// PublicKeyData: The public key data. Only provided in
	// `GetServiceAccountKey` responses.
	PublicKeyData string `json:"publicKeyData,omitempty"`

	// ValidAfterTime: The key can be used after this timestamp.
	ValidAfterTime string `json:"validAfterTime,omitempty"`

	// ValidBeforeTime: The key can be used before this timestamp.
	ValidBeforeTime string `json:"validBeforeTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "KeyAlgorithm") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KeyAlgorithm") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ServiceAccountKey) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccountKey
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetIamPolicyRequest: Request message for `SetIamPolicy` method.
type SetIamPolicyRequest struct {
	// Policy: REQUIRED: The complete policy to be applied to the
	// `resource`. The size of
	// the policy is limited to a few 10s of KB. An empty policy is a
	// valid policy but certain Cloud Platform services (such as
	// Projects)
	// might reject them.
	Policy *Policy `json:"policy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Policy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Policy") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SetIamPolicyRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetIamPolicyRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SignBlobRequest: The service account sign blob request.
type SignBlobRequest struct {
	// BytesToSign: The bytes to sign.
	BytesToSign string `json:"bytesToSign,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BytesToSign") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BytesToSign") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SignBlobRequest) MarshalJSON() ([]byte, error) {
	type noMethod SignBlobRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SignBlobResponse: The service account sign blob response.
type SignBlobResponse struct {
	// KeyId: The id of the key used to sign the blob.
	KeyId string `json:"keyId,omitempty"`

	// Signature: The signed blob.
	Signature string `json:"signature,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "KeyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KeyId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SignBlobResponse) MarshalJSON() ([]byte, error) {
	type noMethod SignBlobResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SignJwtRequest: The service account sign JWT request.
type SignJwtRequest struct {
	// Payload: The JWT payload to sign, a JSON JWT Claim set.
	Payload string `json:"payload,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Payload") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Payload") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SignJwtRequest) MarshalJSON() ([]byte, error) {
	type noMethod SignJwtRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SignJwtResponse: The service account sign JWT response.
type SignJwtResponse struct {
	// KeyId: The id of the key used to sign the JWT.
	KeyId string `json:"keyId,omitempty"`

	// SignedJwt: The signed JWT.
	SignedJwt string `json:"signedJwt,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "KeyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "KeyId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SignJwtResponse) MarshalJSON() ([]byte, error) {
	type noMethod SignJwtResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsRequest: Request message for `TestIamPermissions`
// method.
type TestIamPermissionsRequest struct {
	// Permissions: The set of permissions to check for the `resource`.
	// Permissions with
	// wildcards (such as '*' or 'storage.*') are not allowed. For
	// more
	// information see
	// [IAM
	// Overview](https://cloud.google.com/iam/docs/overview#permissions).
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

func (s *TestIamPermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsResponse: Response message for `TestIamPermissions`
// method.
type TestIamPermissionsResponse struct {
	// Permissions: A subset of `TestPermissionsRequest.permissions` that
	// the caller is
	// allowed.
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

func (s *TestIamPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UndeleteRoleRequest: The request to undelete an existing role.
type UndeleteRoleRequest struct {
	// Etag: Used to perform a consistent read-modify-write.
	Etag string `json:"etag,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Etag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UndeleteRoleRequest) MarshalJSON() ([]byte, error) {
	type noMethod UndeleteRoleRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "iam.organizations.roles.create":

type OrganizationsRolesCreateCall struct {
	s                 *Service
	parent            string
	createrolerequest *CreateRoleRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Create: Creates a new Role.
func (r *OrganizationsRolesService) Create(parent string, createrolerequest *CreateRoleRequest) *OrganizationsRolesCreateCall {
	c := &OrganizationsRolesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createrolerequest = createrolerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesCreateCall) Fields(s ...googleapi.Field) *OrganizationsRolesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesCreateCall) Context(ctx context.Context) *OrganizationsRolesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createrolerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/roles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.create" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrganizationsRolesCreateCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Creates a new Role.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles",
	//   "httpMethod": "POST",
	//   "id": "iam.organizations.roles.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The resource name of the parent resource in one of the following formats:\n`organizations/{ORGANIZATION_ID}`\n`projects/{PROJECT_ID}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/roles",
	//   "request": {
	//     "$ref": "CreateRoleRequest"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.organizations.roles.delete":

type OrganizationsRolesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Soft deletes a role. The role is suspended and cannot be used
// to create new
// IAM Policy Bindings.
// The Role will not be included in `ListRoles()` unless `show_deleted`
// is set
// in the `ListRolesRequest`. The Role contains the deleted boolean
// set.
// Existing Bindings remains, but are inactive. The Role can be
// undeleted
// within 7 days. After 7 days the Role is deleted and all Bindings
// associated
// with the role are removed.
func (r *OrganizationsRolesService) Delete(name string) *OrganizationsRolesDeleteCall {
	c := &OrganizationsRolesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Etag sets the optional parameter "etag": Used to perform a consistent
// read-modify-write.
func (c *OrganizationsRolesDeleteCall) Etag(etag string) *OrganizationsRolesDeleteCall {
	c.urlParams_.Set("etag", etag)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesDeleteCall) Fields(s ...googleapi.Field) *OrganizationsRolesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesDeleteCall) Context(ctx context.Context) *OrganizationsRolesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.delete" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrganizationsRolesDeleteCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Soft deletes a role. The role is suspended and cannot be used to create new\nIAM Policy Bindings.\nThe Role will not be included in `ListRoles()` unless `show_deleted` is set\nin the `ListRolesRequest`. The Role contains the deleted boolean set.\nExisting Bindings remains, but are inactive. The Role can be undeleted\nwithin 7 days. After 7 days the Role is deleted and all Bindings associated\nwith the role are removed.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles/{rolesId}",
	//   "httpMethod": "DELETE",
	//   "id": "iam.organizations.roles.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "etag": {
	//       "description": "Used to perform a consistent read-modify-write.",
	//       "format": "byte",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.organizations.roles.get":

type OrganizationsRolesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Role definition.
func (r *OrganizationsRolesService) Get(name string) *OrganizationsRolesGetCall {
	c := &OrganizationsRolesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesGetCall) Fields(s ...googleapi.Field) *OrganizationsRolesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrganizationsRolesGetCall) IfNoneMatch(entityTag string) *OrganizationsRolesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesGetCall) Context(ctx context.Context) *OrganizationsRolesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.get" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrganizationsRolesGetCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Gets a Role definition.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles/{rolesId}",
	//   "httpMethod": "GET",
	//   "id": "iam.organizations.roles.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`roles/{ROLE_NAME}`\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.organizations.roles.list":

type OrganizationsRolesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the Roles defined on a resource.
func (r *OrganizationsRolesService) List(parent string) *OrganizationsRolesListCall {
	c := &OrganizationsRolesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": Optional limit on
// the number of roles to include in the response.
func (c *OrganizationsRolesListCall) PageSize(pageSize int64) *OrganizationsRolesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Optional
// pagination token returned in an earlier ListRolesResponse.
func (c *OrganizationsRolesListCall) PageToken(pageToken string) *OrganizationsRolesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Include Roles
// that have been deleted.
func (c *OrganizationsRolesListCall) ShowDeleted(showDeleted bool) *OrganizationsRolesListCall {
	c.urlParams_.Set("showDeleted", fmt.Sprint(showDeleted))
	return c
}

// View sets the optional parameter "view": Optional view for the
// returned Role objects.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *OrganizationsRolesListCall) View(view string) *OrganizationsRolesListCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesListCall) Fields(s ...googleapi.Field) *OrganizationsRolesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrganizationsRolesListCall) IfNoneMatch(entityTag string) *OrganizationsRolesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesListCall) Context(ctx context.Context) *OrganizationsRolesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/roles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.list" call.
// Exactly one of *ListRolesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListRolesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrganizationsRolesListCall) Do(opts ...googleapi.CallOption) (*ListRolesResponse, error) {
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
	ret := &ListRolesResponse{
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
	//   "description": "Lists the Roles defined on a resource.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles",
	//   "httpMethod": "GET",
	//   "id": "iam.organizations.roles.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional limit on the number of roles to include in the response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional pagination token returned in an earlier ListRolesResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The resource name of the parent resource in one of the following formats:\n`` (empty string) -- this refers to curated roles.\n`organizations/{ORGANIZATION_ID}`\n`projects/{PROJECT_ID}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Include Roles that have been deleted.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "view": {
	//       "description": "Optional view for the returned Role objects.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/roles",
	//   "response": {
	//     "$ref": "ListRolesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OrganizationsRolesListCall) Pages(ctx context.Context, f func(*ListRolesResponse) error) error {
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

// method id "iam.organizations.roles.patch":

type OrganizationsRolesPatchCall struct {
	s          *Service
	name       string
	role       *Role
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a Role definition.
func (r *OrganizationsRolesService) Patch(name string, role *Role) *OrganizationsRolesPatchCall {
	c := &OrganizationsRolesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.role = role
	return c
}

// UpdateMask sets the optional parameter "updateMask": A mask
// describing which fields in the Role have changed.
func (c *OrganizationsRolesPatchCall) UpdateMask(updateMask string) *OrganizationsRolesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesPatchCall) Fields(s ...googleapi.Field) *OrganizationsRolesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesPatchCall) Context(ctx context.Context) *OrganizationsRolesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.role)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.patch" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrganizationsRolesPatchCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Updates a Role definition.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles/{rolesId}",
	//   "httpMethod": "PATCH",
	//   "id": "iam.organizations.roles.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`roles/{ROLE_NAME}`\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "A mask describing which fields in the Role have changed.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Role"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.organizations.roles.undelete":

type OrganizationsRolesUndeleteCall struct {
	s                   *Service
	name                string
	undeleterolerequest *UndeleteRoleRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// Undelete: Undelete a Role, bringing it back in its previous state.
func (r *OrganizationsRolesService) Undelete(name string, undeleterolerequest *UndeleteRoleRequest) *OrganizationsRolesUndeleteCall {
	c := &OrganizationsRolesUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.undeleterolerequest = undeleterolerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsRolesUndeleteCall) Fields(s ...googleapi.Field) *OrganizationsRolesUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsRolesUndeleteCall) Context(ctx context.Context) *OrganizationsRolesUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsRolesUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsRolesUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.undeleterolerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.organizations.roles.undelete" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *OrganizationsRolesUndeleteCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Undelete a Role, bringing it back in its previous state.",
	//   "flatPath": "v1/organizations/{organizationsId}/roles/{rolesId}:undelete",
	//   "httpMethod": "POST",
	//   "id": "iam.organizations.roles.undelete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:undelete",
	//   "request": {
	//     "$ref": "UndeleteRoleRequest"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.permissions.queryTestablePermissions":

type PermissionsQueryTestablePermissionsCall struct {
	s                               *Service
	querytestablepermissionsrequest *QueryTestablePermissionsRequest
	urlParams_                      gensupport.URLParams
	ctx_                            context.Context
	header_                         http.Header
}

// QueryTestablePermissions: Lists the permissions testable on a
// resource.
// A permission is testable if it can be tested for an identity on a
// resource.
func (r *PermissionsService) QueryTestablePermissions(querytestablepermissionsrequest *QueryTestablePermissionsRequest) *PermissionsQueryTestablePermissionsCall {
	c := &PermissionsQueryTestablePermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.querytestablepermissionsrequest = querytestablepermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PermissionsQueryTestablePermissionsCall) Fields(s ...googleapi.Field) *PermissionsQueryTestablePermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PermissionsQueryTestablePermissionsCall) Context(ctx context.Context) *PermissionsQueryTestablePermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PermissionsQueryTestablePermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PermissionsQueryTestablePermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.querytestablepermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/permissions:queryTestablePermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.permissions.queryTestablePermissions" call.
// Exactly one of *QueryTestablePermissionsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *QueryTestablePermissionsResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PermissionsQueryTestablePermissionsCall) Do(opts ...googleapi.CallOption) (*QueryTestablePermissionsResponse, error) {
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
	ret := &QueryTestablePermissionsResponse{
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
	//   "description": "Lists the permissions testable on a resource.\nA permission is testable if it can be tested for an identity on a resource.",
	//   "flatPath": "v1/permissions:queryTestablePermissions",
	//   "httpMethod": "POST",
	//   "id": "iam.permissions.queryTestablePermissions",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/permissions:queryTestablePermissions",
	//   "request": {
	//     "$ref": "QueryTestablePermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "QueryTestablePermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PermissionsQueryTestablePermissionsCall) Pages(ctx context.Context, f func(*QueryTestablePermissionsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.querytestablepermissionsrequest.PageToken = pt }(c.querytestablepermissionsrequest.PageToken) // reset paging to original point
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
		c.querytestablepermissionsrequest.PageToken = x.NextPageToken
	}
}

// method id "iam.projects.roles.create":

type ProjectsRolesCreateCall struct {
	s                 *Service
	parent            string
	createrolerequest *CreateRoleRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Create: Creates a new Role.
func (r *ProjectsRolesService) Create(parent string, createrolerequest *CreateRoleRequest) *ProjectsRolesCreateCall {
	c := &ProjectsRolesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createrolerequest = createrolerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesCreateCall) Fields(s ...googleapi.Field) *ProjectsRolesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesCreateCall) Context(ctx context.Context) *ProjectsRolesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createrolerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/roles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.create" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRolesCreateCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Creates a new Role.",
	//   "flatPath": "v1/projects/{projectsId}/roles",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.roles.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The resource name of the parent resource in one of the following formats:\n`organizations/{ORGANIZATION_ID}`\n`projects/{PROJECT_ID}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/roles",
	//   "request": {
	//     "$ref": "CreateRoleRequest"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.roles.delete":

type ProjectsRolesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Soft deletes a role. The role is suspended and cannot be used
// to create new
// IAM Policy Bindings.
// The Role will not be included in `ListRoles()` unless `show_deleted`
// is set
// in the `ListRolesRequest`. The Role contains the deleted boolean
// set.
// Existing Bindings remains, but are inactive. The Role can be
// undeleted
// within 7 days. After 7 days the Role is deleted and all Bindings
// associated
// with the role are removed.
func (r *ProjectsRolesService) Delete(name string) *ProjectsRolesDeleteCall {
	c := &ProjectsRolesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Etag sets the optional parameter "etag": Used to perform a consistent
// read-modify-write.
func (c *ProjectsRolesDeleteCall) Etag(etag string) *ProjectsRolesDeleteCall {
	c.urlParams_.Set("etag", etag)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesDeleteCall) Fields(s ...googleapi.Field) *ProjectsRolesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesDeleteCall) Context(ctx context.Context) *ProjectsRolesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.delete" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRolesDeleteCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Soft deletes a role. The role is suspended and cannot be used to create new\nIAM Policy Bindings.\nThe Role will not be included in `ListRoles()` unless `show_deleted` is set\nin the `ListRolesRequest`. The Role contains the deleted boolean set.\nExisting Bindings remains, but are inactive. The Role can be undeleted\nwithin 7 days. After 7 days the Role is deleted and all Bindings associated\nwith the role are removed.",
	//   "flatPath": "v1/projects/{projectsId}/roles/{rolesId}",
	//   "httpMethod": "DELETE",
	//   "id": "iam.projects.roles.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "etag": {
	//       "description": "Used to perform a consistent read-modify-write.",
	//       "format": "byte",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.roles.get":

type ProjectsRolesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Role definition.
func (r *ProjectsRolesService) Get(name string) *ProjectsRolesGetCall {
	c := &ProjectsRolesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesGetCall) Fields(s ...googleapi.Field) *ProjectsRolesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRolesGetCall) IfNoneMatch(entityTag string) *ProjectsRolesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesGetCall) Context(ctx context.Context) *ProjectsRolesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.get" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRolesGetCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Gets a Role definition.",
	//   "flatPath": "v1/projects/{projectsId}/roles/{rolesId}",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.roles.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`roles/{ROLE_NAME}`\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.roles.list":

type ProjectsRolesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the Roles defined on a resource.
func (r *ProjectsRolesService) List(parent string) *ProjectsRolesListCall {
	c := &ProjectsRolesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": Optional limit on
// the number of roles to include in the response.
func (c *ProjectsRolesListCall) PageSize(pageSize int64) *ProjectsRolesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Optional
// pagination token returned in an earlier ListRolesResponse.
func (c *ProjectsRolesListCall) PageToken(pageToken string) *ProjectsRolesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Include Roles
// that have been deleted.
func (c *ProjectsRolesListCall) ShowDeleted(showDeleted bool) *ProjectsRolesListCall {
	c.urlParams_.Set("showDeleted", fmt.Sprint(showDeleted))
	return c
}

// View sets the optional parameter "view": Optional view for the
// returned Role objects.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *ProjectsRolesListCall) View(view string) *ProjectsRolesListCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesListCall) Fields(s ...googleapi.Field) *ProjectsRolesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRolesListCall) IfNoneMatch(entityTag string) *ProjectsRolesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesListCall) Context(ctx context.Context) *ProjectsRolesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/roles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.list" call.
// Exactly one of *ListRolesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListRolesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRolesListCall) Do(opts ...googleapi.CallOption) (*ListRolesResponse, error) {
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
	ret := &ListRolesResponse{
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
	//   "description": "Lists the Roles defined on a resource.",
	//   "flatPath": "v1/projects/{projectsId}/roles",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.roles.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional limit on the number of roles to include in the response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional pagination token returned in an earlier ListRolesResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The resource name of the parent resource in one of the following formats:\n`` (empty string) -- this refers to curated roles.\n`organizations/{ORGANIZATION_ID}`\n`projects/{PROJECT_ID}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Include Roles that have been deleted.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "view": {
	//       "description": "Optional view for the returned Role objects.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/roles",
	//   "response": {
	//     "$ref": "ListRolesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsRolesListCall) Pages(ctx context.Context, f func(*ListRolesResponse) error) error {
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

// method id "iam.projects.roles.patch":

type ProjectsRolesPatchCall struct {
	s          *Service
	name       string
	role       *Role
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a Role definition.
func (r *ProjectsRolesService) Patch(name string, role *Role) *ProjectsRolesPatchCall {
	c := &ProjectsRolesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.role = role
	return c
}

// UpdateMask sets the optional parameter "updateMask": A mask
// describing which fields in the Role have changed.
func (c *ProjectsRolesPatchCall) UpdateMask(updateMask string) *ProjectsRolesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesPatchCall) Fields(s ...googleapi.Field) *ProjectsRolesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesPatchCall) Context(ctx context.Context) *ProjectsRolesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.role)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.patch" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRolesPatchCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Updates a Role definition.",
	//   "flatPath": "v1/projects/{projectsId}/roles/{rolesId}",
	//   "httpMethod": "PATCH",
	//   "id": "iam.projects.roles.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`roles/{ROLE_NAME}`\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "A mask describing which fields in the Role have changed.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Role"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.roles.undelete":

type ProjectsRolesUndeleteCall struct {
	s                   *Service
	name                string
	undeleterolerequest *UndeleteRoleRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// Undelete: Undelete a Role, bringing it back in its previous state.
func (r *ProjectsRolesService) Undelete(name string, undeleterolerequest *UndeleteRoleRequest) *ProjectsRolesUndeleteCall {
	c := &ProjectsRolesUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.undeleterolerequest = undeleterolerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRolesUndeleteCall) Fields(s ...googleapi.Field) *ProjectsRolesUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRolesUndeleteCall) Context(ctx context.Context) *ProjectsRolesUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRolesUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRolesUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.undeleterolerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.roles.undelete" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRolesUndeleteCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Undelete a Role, bringing it back in its previous state.",
	//   "flatPath": "v1/projects/{projectsId}/roles/{rolesId}:undelete",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.roles.undelete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:undelete",
	//   "request": {
	//     "$ref": "UndeleteRoleRequest"
	//   },
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.create":

type ProjectsServiceAccountsCreateCall struct {
	s                           *Service
	name                        string
	createserviceaccountrequest *CreateServiceAccountRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Create: Creates a ServiceAccount
// and returns it.
func (r *ProjectsServiceAccountsService) Create(name string, createserviceaccountrequest *CreateServiceAccountRequest) *ProjectsServiceAccountsCreateCall {
	c := &ProjectsServiceAccountsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.createserviceaccountrequest = createserviceaccountrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsCreateCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsCreateCall) Context(ctx context.Context) *ProjectsServiceAccountsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createserviceaccountrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/serviceAccounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.create" call.
// Exactly one of *ServiceAccount or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ServiceAccount.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsCreateCall) Do(opts ...googleapi.CallOption) (*ServiceAccount, error) {
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
	ret := &ServiceAccount{
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
	//   "description": "Creates a ServiceAccount\nand returns it.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The resource name of the project associated with the service\naccounts, such as `projects/my-project-123`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/serviceAccounts",
	//   "request": {
	//     "$ref": "CreateServiceAccountRequest"
	//   },
	//   "response": {
	//     "$ref": "ServiceAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.delete":

type ProjectsServiceAccountsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a ServiceAccount.
func (r *ProjectsServiceAccountsService) Delete(name string) *ProjectsServiceAccountsDeleteCall {
	c := &ProjectsServiceAccountsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsDeleteCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsDeleteCall) Context(ctx context.Context) *ProjectsServiceAccountsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsServiceAccountsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes a ServiceAccount.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}",
	//   "httpMethod": "DELETE",
	//   "id": "iam.projects.serviceAccounts.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.get":

type ProjectsServiceAccountsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a ServiceAccount.
func (r *ProjectsServiceAccountsService) Get(name string) *ProjectsServiceAccountsGetCall {
	c := &ProjectsServiceAccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsGetCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsServiceAccountsGetCall) IfNoneMatch(entityTag string) *ProjectsServiceAccountsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsGetCall) Context(ctx context.Context) *ProjectsServiceAccountsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.get" call.
// Exactly one of *ServiceAccount or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ServiceAccount.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsGetCall) Do(opts ...googleapi.CallOption) (*ServiceAccount, error) {
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
	ret := &ServiceAccount{
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
	//   "description": "Gets a ServiceAccount.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.serviceAccounts.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ServiceAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.getIamPolicy":

type ProjectsServiceAccountsGetIamPolicyCall struct {
	s          *Service
	resource   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// GetIamPolicy: Returns the IAM access control policy for
// a
// ServiceAccount.
func (r *ProjectsServiceAccountsService) GetIamPolicy(resource string) *ProjectsServiceAccountsGetIamPolicyCall {
	c := &ProjectsServiceAccountsGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsGetIamPolicyCall) Context(ctx context.Context) *ProjectsServiceAccountsGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsServiceAccountsGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Returns the IAM access control policy for a\nServiceAccount.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.list":

type ProjectsServiceAccountsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists ServiceAccounts for a project.
func (r *ProjectsServiceAccountsService) List(name string) *ProjectsServiceAccountsListCall {
	c := &ProjectsServiceAccountsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// PageSize sets the optional parameter "pageSize": Optional limit on
// the number of service accounts to include in the
// response. Further accounts can subsequently be obtained by including
// the
// ListServiceAccountsResponse.next_page_token
// in a subsequent request.
func (c *ProjectsServiceAccountsListCall) PageSize(pageSize int64) *ProjectsServiceAccountsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Optional
// pagination token returned in an
// earlier
// ListServiceAccountsResponse.next_page_token.
func (c *ProjectsServiceAccountsListCall) PageToken(pageToken string) *ProjectsServiceAccountsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsListCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsServiceAccountsListCall) IfNoneMatch(entityTag string) *ProjectsServiceAccountsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsListCall) Context(ctx context.Context) *ProjectsServiceAccountsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/serviceAccounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.list" call.
// Exactly one of *ListServiceAccountsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListServiceAccountsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsListCall) Do(opts ...googleapi.CallOption) (*ListServiceAccountsResponse, error) {
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
	ret := &ListServiceAccountsResponse{
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
	//   "description": "Lists ServiceAccounts for a project.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.serviceAccounts.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required. The resource name of the project associated with the service\naccounts, such as `projects/my-project-123`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Optional limit on the number of service accounts to include in the\nresponse. Further accounts can subsequently be obtained by including the\nListServiceAccountsResponse.next_page_token\nin a subsequent request.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional pagination token returned in an earlier\nListServiceAccountsResponse.next_page_token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/serviceAccounts",
	//   "response": {
	//     "$ref": "ListServiceAccountsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsServiceAccountsListCall) Pages(ctx context.Context, f func(*ListServiceAccountsResponse) error) error {
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

// method id "iam.projects.serviceAccounts.setIamPolicy":

type ProjectsServiceAccountsSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the IAM access control policy for
// a
// ServiceAccount.
func (r *ProjectsServiceAccountsService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsServiceAccountsSetIamPolicyCall {
	c := &ProjectsServiceAccountsSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsSetIamPolicyCall) Context(ctx context.Context) *ProjectsServiceAccountsSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsServiceAccountsSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the IAM access control policy for a\nServiceAccount.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.signBlob":

type ProjectsServiceAccountsSignBlobCall struct {
	s               *Service
	name            string
	signblobrequest *SignBlobRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// SignBlob: Signs a blob using a service account's system-managed
// private key.
func (r *ProjectsServiceAccountsService) SignBlob(name string, signblobrequest *SignBlobRequest) *ProjectsServiceAccountsSignBlobCall {
	c := &ProjectsServiceAccountsSignBlobCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.signblobrequest = signblobrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsSignBlobCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsSignBlobCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsSignBlobCall) Context(ctx context.Context) *ProjectsServiceAccountsSignBlobCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsSignBlobCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsSignBlobCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.signblobrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:signBlob")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.signBlob" call.
// Exactly one of *SignBlobResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SignBlobResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsSignBlobCall) Do(opts ...googleapi.CallOption) (*SignBlobResponse, error) {
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
	ret := &SignBlobResponse{
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
	//   "description": "Signs a blob using a service account's system-managed private key.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}:signBlob",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.signBlob",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:signBlob",
	//   "request": {
	//     "$ref": "SignBlobRequest"
	//   },
	//   "response": {
	//     "$ref": "SignBlobResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.signJwt":

type ProjectsServiceAccountsSignJwtCall struct {
	s              *Service
	name           string
	signjwtrequest *SignJwtRequest
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// SignJwt: Signs a JWT using a service account's system-managed private
// key.
//
// If no expiry time (`exp`) is provided in the `SignJwtRequest`, IAM
// sets an
// an expiry time of one hour by default. If you request an expiry time
// of
// more than one hour, the request will fail.
func (r *ProjectsServiceAccountsService) SignJwt(name string, signjwtrequest *SignJwtRequest) *ProjectsServiceAccountsSignJwtCall {
	c := &ProjectsServiceAccountsSignJwtCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.signjwtrequest = signjwtrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsSignJwtCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsSignJwtCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsSignJwtCall) Context(ctx context.Context) *ProjectsServiceAccountsSignJwtCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsSignJwtCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsSignJwtCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.signjwtrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:signJwt")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.signJwt" call.
// Exactly one of *SignJwtResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *SignJwtResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsSignJwtCall) Do(opts ...googleapi.CallOption) (*SignJwtResponse, error) {
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
	ret := &SignJwtResponse{
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
	//   "description": "Signs a JWT using a service account's system-managed private key.\n\nIf no expiry time (`exp`) is provided in the `SignJwtRequest`, IAM sets an\nan expiry time of one hour by default. If you request an expiry time of\nmore than one hour, the request will fail.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}:signJwt",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.signJwt",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:signJwt",
	//   "request": {
	//     "$ref": "SignJwtRequest"
	//   },
	//   "response": {
	//     "$ref": "SignJwtResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.testIamPermissions":

type ProjectsServiceAccountsTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Tests the specified permissions against the IAM
// access control policy
// for a ServiceAccount.
func (r *ProjectsServiceAccountsService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsServiceAccountsTestIamPermissionsCall {
	c := &ProjectsServiceAccountsTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsTestIamPermissionsCall) Context(ctx context.Context) *ProjectsServiceAccountsTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	ret := &TestIamPermissionsResponse{
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
	//   "description": "Tests the specified permissions against the IAM access control policy\nfor a ServiceAccount.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.update":

type ProjectsServiceAccountsUpdateCall struct {
	s              *Service
	name           string
	serviceaccount *ServiceAccount
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Update: Updates a ServiceAccount.
//
// Currently, only the following fields are updatable:
// `display_name` .
// The `etag` is mandatory.
func (r *ProjectsServiceAccountsService) Update(name string, serviceaccount *ServiceAccount) *ProjectsServiceAccountsUpdateCall {
	c := &ProjectsServiceAccountsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.serviceaccount = serviceaccount
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsUpdateCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsUpdateCall) Context(ctx context.Context) *ProjectsServiceAccountsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.serviceaccount)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.update" call.
// Exactly one of *ServiceAccount or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ServiceAccount.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsUpdateCall) Do(opts ...googleapi.CallOption) (*ServiceAccount, error) {
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
	ret := &ServiceAccount{
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
	//   "description": "Updates a ServiceAccount.\n\nCurrently, only the following fields are updatable:\n`display_name` .\nThe `etag` is mandatory.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}",
	//   "httpMethod": "PUT",
	//   "id": "iam.projects.serviceAccounts.update",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\n\nRequests using `-` as a wildcard for the project will infer the project\nfrom the `account` and the `account` value can be the `email` address or\nthe `unique_id` of the service account.\n\nIn responses the resource name will always be in the format\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "ServiceAccount"
	//   },
	//   "response": {
	//     "$ref": "ServiceAccount"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.keys.create":

type ProjectsServiceAccountsKeysCreateCall struct {
	s                              *Service
	name                           string
	createserviceaccountkeyrequest *CreateServiceAccountKeyRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// Create: Creates a ServiceAccountKey
// and returns it.
func (r *ProjectsServiceAccountsKeysService) Create(name string, createserviceaccountkeyrequest *CreateServiceAccountKeyRequest) *ProjectsServiceAccountsKeysCreateCall {
	c := &ProjectsServiceAccountsKeysCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.createserviceaccountkeyrequest = createserviceaccountkeyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsKeysCreateCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsKeysCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsKeysCreateCall) Context(ctx context.Context) *ProjectsServiceAccountsKeysCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsKeysCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsKeysCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createserviceaccountkeyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/keys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.keys.create" call.
// Exactly one of *ServiceAccountKey or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ServiceAccountKey.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsKeysCreateCall) Do(opts ...googleapi.CallOption) (*ServiceAccountKey, error) {
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
	ret := &ServiceAccountKey{
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
	//   "description": "Creates a ServiceAccountKey\nand returns it.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys",
	//   "httpMethod": "POST",
	//   "id": "iam.projects.serviceAccounts.keys.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/keys",
	//   "request": {
	//     "$ref": "CreateServiceAccountKeyRequest"
	//   },
	//   "response": {
	//     "$ref": "ServiceAccountKey"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.keys.delete":

type ProjectsServiceAccountsKeysDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a ServiceAccountKey.
func (r *ProjectsServiceAccountsKeysService) Delete(name string) *ProjectsServiceAccountsKeysDeleteCall {
	c := &ProjectsServiceAccountsKeysDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsKeysDeleteCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsKeysDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsKeysDeleteCall) Context(ctx context.Context) *ProjectsServiceAccountsKeysDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsKeysDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsKeysDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.keys.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsServiceAccountsKeysDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes a ServiceAccountKey.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}",
	//   "httpMethod": "DELETE",
	//   "id": "iam.projects.serviceAccounts.keys.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account key in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}/keys/{key}`.\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+/keys/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.keys.get":

type ProjectsServiceAccountsKeysGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the ServiceAccountKey
// by key id.
func (r *ProjectsServiceAccountsKeysService) Get(name string) *ProjectsServiceAccountsKeysGetCall {
	c := &ProjectsServiceAccountsKeysGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// PublicKeyType sets the optional parameter "publicKeyType": The output
// format of the public key requested.
// X509_PEM is the default output format.
//
// Possible values:
//   "TYPE_NONE"
//   "TYPE_X509_PEM_FILE"
//   "TYPE_RAW_PUBLIC_KEY"
func (c *ProjectsServiceAccountsKeysGetCall) PublicKeyType(publicKeyType string) *ProjectsServiceAccountsKeysGetCall {
	c.urlParams_.Set("publicKeyType", publicKeyType)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsKeysGetCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsKeysGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsServiceAccountsKeysGetCall) IfNoneMatch(entityTag string) *ProjectsServiceAccountsKeysGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsKeysGetCall) Context(ctx context.Context) *ProjectsServiceAccountsKeysGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsKeysGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsKeysGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.keys.get" call.
// Exactly one of *ServiceAccountKey or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ServiceAccountKey.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsKeysGetCall) Do(opts ...googleapi.CallOption) (*ServiceAccountKey, error) {
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
	ret := &ServiceAccountKey{
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
	//   "description": "Gets the ServiceAccountKey\nby key id.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.serviceAccounts.keys.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the service account key in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}/keys/{key}`.\n\nUsing `-` as a wildcard for the project will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+/keys/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publicKeyType": {
	//       "description": "The output format of the public key requested.\nX509_PEM is the default output format.",
	//       "enum": [
	//         "TYPE_NONE",
	//         "TYPE_X509_PEM_FILE",
	//         "TYPE_RAW_PUBLIC_KEY"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ServiceAccountKey"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.projects.serviceAccounts.keys.list":

type ProjectsServiceAccountsKeysListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists ServiceAccountKeys.
func (r *ProjectsServiceAccountsKeysService) List(name string) *ProjectsServiceAccountsKeysListCall {
	c := &ProjectsServiceAccountsKeysListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// KeyTypes sets the optional parameter "keyTypes": Filters the types of
// keys the user wants to include in the list
// response. Duplicate key types are not allowed. If no key type
// is provided, all keys are returned.
//
// Possible values:
//   "KEY_TYPE_UNSPECIFIED"
//   "USER_MANAGED"
//   "SYSTEM_MANAGED"
func (c *ProjectsServiceAccountsKeysListCall) KeyTypes(keyTypes ...string) *ProjectsServiceAccountsKeysListCall {
	c.urlParams_.SetMulti("keyTypes", append([]string{}, keyTypes...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsServiceAccountsKeysListCall) Fields(s ...googleapi.Field) *ProjectsServiceAccountsKeysListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsServiceAccountsKeysListCall) IfNoneMatch(entityTag string) *ProjectsServiceAccountsKeysListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsServiceAccountsKeysListCall) Context(ctx context.Context) *ProjectsServiceAccountsKeysListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsServiceAccountsKeysListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsServiceAccountsKeysListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/keys")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.projects.serviceAccounts.keys.list" call.
// Exactly one of *ListServiceAccountKeysResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListServiceAccountKeysResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsServiceAccountsKeysListCall) Do(opts ...googleapi.CallOption) (*ListServiceAccountKeysResponse, error) {
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
	ret := &ListServiceAccountKeysResponse{
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
	//   "description": "Lists ServiceAccountKeys.",
	//   "flatPath": "v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys",
	//   "httpMethod": "GET",
	//   "id": "iam.projects.serviceAccounts.keys.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "keyTypes": {
	//       "description": "Filters the types of keys the user wants to include in the list\nresponse. Duplicate key types are not allowed. If no key type\nis provided, all keys are returned.",
	//       "enum": [
	//         "KEY_TYPE_UNSPECIFIED",
	//         "USER_MANAGED",
	//         "SYSTEM_MANAGED"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The resource name of the service account in the following format:\n`projects/{PROJECT_ID}/serviceAccounts/{SERVICE_ACCOUNT_EMAIL}`.\n\nUsing `-` as a wildcard for the project, will infer the project from\nthe account. The `account` value can be the `email` address or the\n`unique_id` of the service account.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/serviceAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/keys",
	//   "response": {
	//     "$ref": "ListServiceAccountKeysResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.roles.get":

type RolesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Role definition.
func (r *RolesService) Get(name string) *RolesGetCall {
	c := &RolesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RolesGetCall) Fields(s ...googleapi.Field) *RolesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RolesGetCall) IfNoneMatch(entityTag string) *RolesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RolesGetCall) Context(ctx context.Context) *RolesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RolesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RolesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.roles.get" call.
// Exactly one of *Role or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Role.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *RolesGetCall) Do(opts ...googleapi.CallOption) (*Role, error) {
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
	ret := &Role{
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
	//   "description": "Gets a Role definition.",
	//   "flatPath": "v1/roles/{rolesId}",
	//   "httpMethod": "GET",
	//   "id": "iam.roles.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the role in one of the following formats:\n`roles/{ROLE_NAME}`\n`organizations/{ORGANIZATION_ID}/roles/{ROLE_NAME}`\n`projects/{PROJECT_ID}/roles/{ROLE_NAME}`",
	//       "location": "path",
	//       "pattern": "^roles/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Role"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "iam.roles.list":

type RolesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the Roles defined on a resource.
func (r *RolesService) List() *RolesListCall {
	c := &RolesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": Optional limit on
// the number of roles to include in the response.
func (c *RolesListCall) PageSize(pageSize int64) *RolesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Optional
// pagination token returned in an earlier ListRolesResponse.
func (c *RolesListCall) PageToken(pageToken string) *RolesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Parent sets the optional parameter "parent": The resource name of the
// parent resource in one of the following formats:
// `` (empty string) -- this refers to curated
// roles.
// `organizations/{ORGANIZATION_ID}`
// `projects/{PROJECT_ID}`
func (c *RolesListCall) Parent(parent string) *RolesListCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Include Roles
// that have been deleted.
func (c *RolesListCall) ShowDeleted(showDeleted bool) *RolesListCall {
	c.urlParams_.Set("showDeleted", fmt.Sprint(showDeleted))
	return c
}

// View sets the optional parameter "view": Optional view for the
// returned Role objects.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *RolesListCall) View(view string) *RolesListCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RolesListCall) Fields(s ...googleapi.Field) *RolesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *RolesListCall) IfNoneMatch(entityTag string) *RolesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RolesListCall) Context(ctx context.Context) *RolesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RolesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RolesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/roles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.roles.list" call.
// Exactly one of *ListRolesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListRolesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RolesListCall) Do(opts ...googleapi.CallOption) (*ListRolesResponse, error) {
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
	ret := &ListRolesResponse{
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
	//   "description": "Lists the Roles defined on a resource.",
	//   "flatPath": "v1/roles",
	//   "httpMethod": "GET",
	//   "id": "iam.roles.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional limit on the number of roles to include in the response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional pagination token returned in an earlier ListRolesResponse.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The resource name of the parent resource in one of the following formats:\n`` (empty string) -- this refers to curated roles.\n`organizations/{ORGANIZATION_ID}`\n`projects/{PROJECT_ID}`",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Include Roles that have been deleted.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "view": {
	//       "description": "Optional view for the returned Role objects.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/roles",
	//   "response": {
	//     "$ref": "ListRolesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RolesListCall) Pages(ctx context.Context, f func(*ListRolesResponse) error) error {
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

// method id "iam.roles.queryGrantableRoles":

type RolesQueryGrantableRolesCall struct {
	s                          *Service
	querygrantablerolesrequest *QueryGrantableRolesRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// QueryGrantableRoles: Queries roles that can be granted on a
// particular resource.
// A role is grantable if it can be used as the role in a binding for a
// policy
// for that resource.
func (r *RolesService) QueryGrantableRoles(querygrantablerolesrequest *QueryGrantableRolesRequest) *RolesQueryGrantableRolesCall {
	c := &RolesQueryGrantableRolesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.querygrantablerolesrequest = querygrantablerolesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *RolesQueryGrantableRolesCall) Fields(s ...googleapi.Field) *RolesQueryGrantableRolesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *RolesQueryGrantableRolesCall) Context(ctx context.Context) *RolesQueryGrantableRolesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *RolesQueryGrantableRolesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *RolesQueryGrantableRolesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.querygrantablerolesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/roles:queryGrantableRoles")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "iam.roles.queryGrantableRoles" call.
// Exactly one of *QueryGrantableRolesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *QueryGrantableRolesResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *RolesQueryGrantableRolesCall) Do(opts ...googleapi.CallOption) (*QueryGrantableRolesResponse, error) {
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
	ret := &QueryGrantableRolesResponse{
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
	//   "description": "Queries roles that can be granted on a particular resource.\nA role is grantable if it can be used as the role in a binding for a policy\nfor that resource.",
	//   "flatPath": "v1/roles:queryGrantableRoles",
	//   "httpMethod": "POST",
	//   "id": "iam.roles.queryGrantableRoles",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/roles:queryGrantableRoles",
	//   "request": {
	//     "$ref": "QueryGrantableRolesRequest"
	//   },
	//   "response": {
	//     "$ref": "QueryGrantableRolesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *RolesQueryGrantableRolesCall) Pages(ctx context.Context, f func(*QueryGrantableRolesResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.querygrantablerolesrequest.PageToken = pt }(c.querygrantablerolesrequest.PageToken) // reset paging to original point
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
		c.querygrantablerolesrequest.PageToken = x.NextPageToken
	}
}
