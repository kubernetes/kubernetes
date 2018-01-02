// Package sourcerepo provides access to the Cloud Source Repositories API.
//
// See https://cloud.google.com/source-repositories/docs/apis
//
// Usage example:
//
//   import "google.golang.org/api/sourcerepo/v1"
//   ...
//   sourcerepoService, err := sourcerepo.New(oauthHttpClient)
package sourcerepo // import "google.golang.org/api/sourcerepo/v1"

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

const apiId = "sourcerepo:v1"
const apiName = "sourcerepo"
const apiVersion = "v1"
const basePath = "https://sourcerepo.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View the contents of your source code repositories
	SourceReadOnlyScope = "https://www.googleapis.com/auth/source.read_only"

	// Manage the contents of your source code repositories
	SourceReadWriteScope = "https://www.googleapis.com/auth/source.read_write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Projects = NewProjectsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Projects *ProjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Repos = NewProjectsReposService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Repos *ProjectsReposService
}

func NewProjectsReposService(s *Service) *ProjectsReposService {
	rs := &ProjectsReposService{s: s}
	return rs
}

type ProjectsReposService struct {
	s *Service
}

// AuditConfig: Specifies the audit configuration for a service.
// The configuration determines which permission types are logged, and
// what
// identities, if any, are exempted from logging.
// An AuditConfig must have one or more AuditLogConfigs.
//
// If there are AuditConfigs for both `allServices` and a specific
// service,
// the union of the two AuditConfigs is used for that service: the
// log_types
// specified in each AuditConfig are enabled, and the exempted_members
// in each
// AuditConfig are exempted.
//
// Example Policy with multiple AuditConfigs:
//
//     {
//       "audit_configs": [
//         {
//           "service": "allServices"
//           "audit_log_configs": [
//             {
//               "log_type": "DATA_READ",
//               "exempted_members": [
//                 "user:foo@gmail.com"
//               ]
//             },
//             {
//               "log_type": "DATA_WRITE",
//             },
//             {
//               "log_type": "ADMIN_READ",
//             }
//           ]
//         },
//         {
//           "service": "fooservice.googleapis.com"
//           "audit_log_configs": [
//             {
//               "log_type": "DATA_READ",
//             },
//             {
//               "log_type": "DATA_WRITE",
//               "exempted_members": [
//                 "user:bar@gmail.com"
//               ]
//             }
//           ]
//         }
//       ]
//     }
//
// For fooservice, this policy enables DATA_READ, DATA_WRITE and
// ADMIN_READ
// logging. It also exempts foo@gmail.com from DATA_READ logging,
// and
// bar@gmail.com from DATA_WRITE logging.
type AuditConfig struct {
	// AuditLogConfigs: The configuration for logging of each type of
	// permission.
	// Next ID: 4
	AuditLogConfigs []*AuditLogConfig `json:"auditLogConfigs,omitempty"`

	ExemptedMembers []string `json:"exemptedMembers,omitempty"`

	// Service: Specifies a service that will be enabled for audit
	// logging.
	// For example, `storage.googleapis.com`,
	// `cloudsql.googleapis.com`.
	// `allServices` is a special value that covers all services.
	Service string `json:"service,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuditLogConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuditLogConfigs") to
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

// AuditLogConfig: Provides the configuration for logging a type of
// permissions.
// Example:
//
//     {
//       "audit_log_configs": [
//         {
//           "log_type": "DATA_READ",
//           "exempted_members": [
//             "user:foo@gmail.com"
//           ]
//         },
//         {
//           "log_type": "DATA_WRITE",
//         }
//       ]
//     }
//
// This enables 'DATA_READ' and 'DATA_WRITE' logging, while
// exempting
// foo@gmail.com from DATA_READ logging.
type AuditLogConfig struct {
	// ExemptedMembers: Specifies the identities that do not cause logging
	// for this type of
	// permission.
	// Follows the same format of Binding.members.
	ExemptedMembers []string `json:"exemptedMembers,omitempty"`

	// LogType: The log type that this config enables.
	//
	// Possible values:
	//   "LOG_TYPE_UNSPECIFIED" - Default case. Should never be this.
	//   "ADMIN_READ" - Admin reads. Example: CloudIAM getIamPolicy
	//   "DATA_WRITE" - Data writes. Example: CloudSQL Users create
	//   "DATA_READ" - Data reads. Example: CloudSQL Users list
	LogType string `json:"logType,omitempty"`

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

func (s *AuditLogConfig) MarshalJSON() ([]byte, error) {
	type noMethod AuditLogConfig
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

// CloudAuditOptions: Write a Cloud Audit log
type CloudAuditOptions struct {
	// LogName: The log_name to populate in the Cloud Audit Record.
	//
	// Possible values:
	//   "UNSPECIFIED_LOG_NAME" - Default. Should not be used.
	//   "ADMIN_ACTIVITY" - Corresponds to
	// "cloudaudit.googleapis.com/activity"
	//   "DATA_ACCESS" - Corresponds to
	// "cloudaudit.googleapis.com/data_access"
	LogName string `json:"logName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LogName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LogName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CloudAuditOptions) MarshalJSON() ([]byte, error) {
	type noMethod CloudAuditOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Condition: A condition to be met.
type Condition struct {
	// Iam: Trusted attributes supplied by the IAM system.
	//
	// Possible values:
	//   "NO_ATTR" - Default non-attribute.
	//   "AUTHORITY" - Either principal or (if present) authority selector.
	//   "ATTRIBUTION" - The principal (even if an authority selector is
	// present), which
	// must only be used for attribution, not authorization.
	//   "APPROVER" - An approver (distinct from the requester) that has
	// authorized this
	// request.
	// When used with IN, the condition indicates that one of the
	// approvers
	// associated with the request matches the specified principal, or is
	// a
	// member of the specified group. Approvers can only grant
	// additional
	// access, and are thus only used in a strictly positive context
	// (e.g. ALLOW/IN or DENY/NOT_IN).
	// See: go/rpc-security-policy-dynamicauth.
	//   "JUSTIFICATION_TYPE" - What types of justifications have been
	// supplied with this request.
	// String values should match enum names from
	// tech.iam.JustificationType,
	// e.g. "MANUAL_STRING". It is not permitted to grant access based
	// on
	// the *absence* of a justification, so justification conditions can
	// only
	// be used in a "positive" context (e.g., ALLOW/IN or
	// DENY/NOT_IN).
	//
	// Multiple justifications, e.g., a Buganizer ID and a
	// manually-entered
	// reason, are normal and supported.
	Iam string `json:"iam,omitempty"`

	// Op: An operator to apply the subject with.
	//
	// Possible values:
	//   "NO_OP" - Default no-op.
	//   "EQUALS" - DEPRECATED. Use IN instead.
	//   "NOT_EQUALS" - DEPRECATED. Use NOT_IN instead.
	//   "IN" - The condition is true if the subject (or any element of it
	// if it is
	// a set) matches any of the supplied values.
	//   "NOT_IN" - The condition is true if the subject (or every element
	// of it if it is
	// a set) matches none of the supplied values.
	//   "DISCHARGED" - Subject is discharged
	Op string `json:"op,omitempty"`

	// Svc: Trusted attributes discharged by the service.
	Svc string `json:"svc,omitempty"`

	// Sys: Trusted attributes supplied by any service that owns resources
	// and uses
	// the IAM system for access control.
	//
	// Possible values:
	//   "NO_ATTR" - Default non-attribute type
	//   "REGION" - Region of the resource
	//   "SERVICE" - Service name
	//   "NAME" - Resource name
	//   "IP" - IP address of the caller
	Sys string `json:"sys,omitempty"`

	// Value: DEPRECATED. Use 'values' instead.
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

// CounterOptions: Options for counters
type CounterOptions struct {
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

func (s *CounterOptions) MarshalJSON() ([]byte, error) {
	type noMethod CounterOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DataAccessOptions: Write a Data Access (Gin) log
type DataAccessOptions struct {
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

// ListReposResponse: Response for ListRepos.  The size is not set in
// the returned repositories.
type ListReposResponse struct {
	// NextPageToken: If non-empty, additional repositories exist within the
	// project. These
	// can be retrieved by including this value in the next
	// ListReposRequest's
	// page_token field.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Repos: The listed repos.
	Repos []*Repo `json:"repos,omitempty"`

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

func (s *ListReposResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListReposResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogConfig: Specifies what kind of log the caller must write
type LogConfig struct {
	// CloudAudit: Cloud audit options.
	CloudAudit *CloudAuditOptions `json:"cloudAudit,omitempty"`

	// Counter: Counter options.
	Counter *CounterOptions `json:"counter,omitempty"`

	// DataAccess: Data access options.
	DataAccess *DataAccessOptions `json:"dataAccess,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CloudAudit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CloudAudit") to include in
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

// MirrorConfig: Configuration to automatically mirror a repository from
// another
// hosting service, for example GitHub or BitBucket.
type MirrorConfig struct {
	// DeployKeyId: ID of the SSH deploy key at the other hosting
	// service.
	// Removing this key from the other service would deauthorize
	// Google Cloud Source Repositories from mirroring.
	DeployKeyId string `json:"deployKeyId,omitempty"`

	// Url: URL of the main repository at the other hosting service.
	Url string `json:"url,omitempty"`

	// WebhookId: ID of the webhook listening to updates to trigger
	// mirroring.
	// Removing this webook from the other hosting service will stop
	// Google Cloud Source Repositories from receiving notifications,
	// and thereby disabling mirroring.
	WebhookId string `json:"webhookId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeployKeyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeployKeyId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MirrorConfig) MarshalJSON() ([]byte, error) {
	type noMethod MirrorConfig
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
	// AuditConfigs: Specifies cloud audit logging configuration for this
	// policy.
	AuditConfigs []*AuditConfig `json:"auditConfigs,omitempty"`

	// Bindings: Associates a list of `members` to a `role`.
	// Multiple `bindings` must not be specified for the same
	// `role`.
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

	IamOwned bool `json:"iamOwned,omitempty"`

	// Rules: If more than one rule is specified, the rules are applied in
	// the following
	// manner:
	// - All matching LOG rules are always applied.
	// - If any DENY/DENY_WITH_LOG rule matches, permission is denied.
	//   Logging will be applied if one or more matching rule requires
	// logging.
	// - Otherwise, if any ALLOW/ALLOW_WITH_LOG rule matches, permission is
	//   granted.
	//   Logging will be applied if one or more matching rule requires
	// logging.
	// - Otherwise, if no rule applies, permission is denied.
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

// Repo: A repository (or repo) is a Git repository storing versioned
// source content.
type Repo struct {
	// MirrorConfig: How this repository mirrors a repository managed by
	// another service.
	MirrorConfig *MirrorConfig `json:"mirrorConfig,omitempty"`

	// Name: Resource name of the repository, of the
	// form
	// `projects/<project>/repos/<repo>`.  The repo name may contain
	// slashes.
	// eg, `projects/myproject/repos/name/with/slash`
	Name string `json:"name,omitempty"`

	// Size: The disk usage of the repo, in bytes.
	// Only returned by GetRepo.
	Size int64 `json:"size,omitempty,string"`

	// Url: URL to clone the repository from Google Cloud Source
	// Repositories.
	Url string `json:"url,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MirrorConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MirrorConfig") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Repo) MarshalJSON() ([]byte, error) {
	type noMethod Repo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Rule: A rule to be applied in a Policy.
type Rule struct {
	// Action: Required
	//
	// Possible values:
	//   "NO_ACTION" - Default no action.
	//   "ALLOW" - Matching 'Entries' grant access.
	//   "ALLOW_WITH_LOG" - Matching 'Entries' grant access and the caller
	// promises to log
	// the request per the returned log_configs.
	//   "DENY" - Matching 'Entries' deny access.
	//   "DENY_WITH_LOG" - Matching 'Entries' deny access and the caller
	// promises to log
	// the request per the returned log_configs.
	//   "LOG" - Matching 'Entries' tell IAM.Check callers to generate logs.
	Action string `json:"action,omitempty"`

	// Conditions: Additional restrictions that must be met
	Conditions []*Condition `json:"conditions,omitempty"`

	// Description: Human-readable description of the rule.
	Description string `json:"description,omitempty"`

	// In: If one or more 'in' clauses are specified, the rule matches
	// if
	// the PRINCIPAL/AUTHORITY_SELECTOR is in at least one of these entries.
	In []string `json:"in,omitempty"`

	// LogConfig: The config returned to callers of tech.iam.IAM.CheckPolicy
	// for any entries
	// that match the LOG action.
	LogConfig []*LogConfig `json:"logConfig,omitempty"`

	// NotIn: If one or more 'not_in' clauses are specified, the rule
	// matches
	// if the PRINCIPAL/AUTHORITY_SELECTOR is in none of the entries.
	// The format for in and not_in entries is the same as for members in
	// a
	// Binding (see google/iam/v1/policy.proto).
	NotIn []string `json:"notIn,omitempty"`

	// Permissions: A permission is a string of form '<service>.<resource
	// type>.<verb>'
	// (e.g., 'storage.buckets.list'). A value of '*' matches all
	// permissions,
	// and a verb part of '*' (e.g., 'storage.buckets.*') matches all verbs.
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

// SetIamPolicyRequest: Request message for `SetIamPolicy` method.
type SetIamPolicyRequest struct {
	// Policy: REQUIRED: The complete policy to be applied to the
	// `resource`. The size of
	// the policy is limited to a few 10s of KB. An empty policy is a
	// valid policy but certain Cloud Platform services (such as
	// Projects)
	// might reject them.
	Policy *Policy `json:"policy,omitempty"`

	// UpdateMask: OPTIONAL: A FieldMask specifying which fields of the
	// policy to modify. Only
	// the fields in the mask will be modified. If no mask is provided,
	// the
	// following default mask is used:
	// paths: "bindings, etag"
	// This field is only used by Cloud IAM.
	UpdateMask string `json:"updateMask,omitempty"`

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

// method id "sourcerepo.projects.repos.create":

type ProjectsReposCreateCall struct {
	s          *Service
	parent     string
	repo       *Repo
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a repo in the given project with the given name.
//
// If the named repository already exists, `CreateRepo`
// returns
// `ALREADY_EXISTS`.
func (r *ProjectsReposService) Create(parent string, repo *Repo) *ProjectsReposCreateCall {
	c := &ProjectsReposCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.repo = repo
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposCreateCall) Fields(s ...googleapi.Field) *ProjectsReposCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposCreateCall) Context(ctx context.Context) *ProjectsReposCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.repo)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/repos")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sourcerepo.projects.repos.create" call.
// Exactly one of *Repo or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Repo.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsReposCreateCall) Do(opts ...googleapi.CallOption) (*Repo, error) {
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
	ret := &Repo{
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
	//   "description": "Creates a repo in the given project with the given name.\n\nIf the named repository already exists, `CreateRepo` returns\n`ALREADY_EXISTS`.",
	//   "flatPath": "v1/projects/{projectsId}/repos",
	//   "httpMethod": "POST",
	//   "id": "sourcerepo.projects.repos.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The project in which to create the repo. Values are of the form\n`projects/\u003cproject\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/repos",
	//   "request": {
	//     "$ref": "Repo"
	//   },
	//   "response": {
	//     "$ref": "Repo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// method id "sourcerepo.projects.repos.delete":

type ProjectsReposDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a repo.
func (r *ProjectsReposService) Delete(name string) *ProjectsReposDeleteCall {
	c := &ProjectsReposDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposDeleteCall) Fields(s ...googleapi.Field) *ProjectsReposDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposDeleteCall) Context(ctx context.Context) *ProjectsReposDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "sourcerepo.projects.repos.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReposDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a repo.",
	//   "flatPath": "v1/projects/{projectsId}/repos/{reposId}",
	//   "httpMethod": "DELETE",
	//   "id": "sourcerepo.projects.repos.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the repo to delete. Values are of the form\n`projects/\u003cproject\u003e/repos/\u003crepo\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/repos/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// method id "sourcerepo.projects.repos.get":

type ProjectsReposGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns information about a repo.
func (r *ProjectsReposService) Get(name string) *ProjectsReposGetCall {
	c := &ProjectsReposGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposGetCall) Fields(s ...googleapi.Field) *ProjectsReposGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsReposGetCall) IfNoneMatch(entityTag string) *ProjectsReposGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposGetCall) Context(ctx context.Context) *ProjectsReposGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "sourcerepo.projects.repos.get" call.
// Exactly one of *Repo or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Repo.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsReposGetCall) Do(opts ...googleapi.CallOption) (*Repo, error) {
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
	ret := &Repo{
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
	//   "description": "Returns information about a repo.",
	//   "flatPath": "v1/projects/{projectsId}/repos/{reposId}",
	//   "httpMethod": "GET",
	//   "id": "sourcerepo.projects.repos.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the requested repository. Values are of the form\n`projects/\u003cproject\u003e/repos/\u003crepo\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/repos/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Repo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// method id "sourcerepo.projects.repos.getIamPolicy":

type ProjectsReposGetIamPolicyCall struct {
	s            *Service
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetIamPolicy: Gets the access control policy for a resource.
// Returns an empty policy if the resource exists and does not have a
// policy
// set.
func (r *ProjectsReposService) GetIamPolicy(resource string) *ProjectsReposGetIamPolicyCall {
	c := &ProjectsReposGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsReposGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsReposGetIamPolicyCall) IfNoneMatch(entityTag string) *ProjectsReposGetIamPolicyCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposGetIamPolicyCall) Context(ctx context.Context) *ProjectsReposGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sourcerepo.projects.repos.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReposGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets the access control policy for a resource.\nReturns an empty policy if the resource exists and does not have a policy\nset.",
	//   "flatPath": "v1/projects/{projectsId}/repos/{reposId}:getIamPolicy",
	//   "httpMethod": "GET",
	//   "id": "sourcerepo.projects.repos.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/repos/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// method id "sourcerepo.projects.repos.list":

type ProjectsReposListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns all repos belonging to a project. The sizes of the
// repos are
// not set by ListRepos.  To get the size of a repo, use GetRepo.
func (r *ProjectsReposService) List(name string) *ProjectsReposListCall {
	c := &ProjectsReposListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// repositories to return; between 1 and 500.
// If not set or zero, defaults to 100 at the server.
func (c *ProjectsReposListCall) PageSize(pageSize int64) *ProjectsReposListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Resume listing
// repositories where a prior ListReposResponse
// left off. This is an opaque token that must be obtained from
// a recent, prior ListReposResponse's next_page_token field.
func (c *ProjectsReposListCall) PageToken(pageToken string) *ProjectsReposListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposListCall) Fields(s ...googleapi.Field) *ProjectsReposListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsReposListCall) IfNoneMatch(entityTag string) *ProjectsReposListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposListCall) Context(ctx context.Context) *ProjectsReposListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}/repos")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sourcerepo.projects.repos.list" call.
// Exactly one of *ListReposResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListReposResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsReposListCall) Do(opts ...googleapi.CallOption) (*ListReposResponse, error) {
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
	ret := &ListReposResponse{
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
	//   "description": "Returns all repos belonging to a project. The sizes of the repos are\nnot set by ListRepos.  To get the size of a repo, use GetRepo.",
	//   "flatPath": "v1/projects/{projectsId}/repos",
	//   "httpMethod": "GET",
	//   "id": "sourcerepo.projects.repos.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The project ID whose repos should be listed. Values are of the form\n`projects/\u003cproject\u003e`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of repositories to return; between 1 and 500.\nIf not set or zero, defaults to 100 at the server.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Resume listing repositories where a prior ListReposResponse\nleft off. This is an opaque token that must be obtained from\na recent, prior ListReposResponse's next_page_token field.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}/repos",
	//   "response": {
	//     "$ref": "ListReposResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsReposListCall) Pages(ctx context.Context, f func(*ListReposResponse) error) error {
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

// method id "sourcerepo.projects.repos.setIamPolicy":

type ProjectsReposSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any
// existing policy.
func (r *ProjectsReposService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsReposSetIamPolicyCall {
	c := &ProjectsReposSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsReposSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposSetIamPolicyCall) Context(ctx context.Context) *ProjectsReposSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "sourcerepo.projects.repos.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsReposSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the access control policy on the specified resource. Replaces any\nexisting policy.",
	//   "flatPath": "v1/projects/{projectsId}/repos/{reposId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "sourcerepo.projects.repos.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/repos/.+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}

// method id "sourcerepo.projects.repos.testIamPermissions":

type ProjectsReposTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
// If the resource does not exist, this will return an empty set
// of
// permissions, not a NOT_FOUND error.
func (r *ProjectsReposService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsReposTestIamPermissionsCall {
	c := &ProjectsReposTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsReposTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsReposTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsReposTestIamPermissionsCall) Context(ctx context.Context) *ProjectsReposTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsReposTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsReposTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "sourcerepo.projects.repos.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsReposTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	//   "description": "Returns permissions that a caller has on the specified resource.\nIf the resource does not exist, this will return an empty set of\npermissions, not a NOT_FOUND error.",
	//   "flatPath": "v1/projects/{projectsId}/repos/{reposId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "sourcerepo.projects.repos.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/repos/.+$",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/source.read_only",
	//     "https://www.googleapis.com/auth/source.read_write"
	//   ]
	// }

}
