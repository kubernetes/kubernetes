// Package cloudresourcemanager provides access to the Google Cloud Resource Manager API.
//
// See https://cloud.google.com/resource-manager
//
// Usage example:
//
//   import "google.golang.org/api/cloudresourcemanager/v2beta1"
//   ...
//   cloudresourcemanagerService, err := cloudresourcemanager.New(oauthHttpClient)
package cloudresourcemanager // import "google.golang.org/api/cloudresourcemanager/v2beta1"

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

const apiId = "cloudresourcemanager:v2beta1"
const apiName = "cloudresourcemanager"
const apiVersion = "v2beta1"
const basePath = "https://cloudresourcemanager.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Folders = NewFoldersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Folders *FoldersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewFoldersService(s *Service) *FoldersService {
	rs := &FoldersService{s: s}
	return rs
}

type FoldersService struct {
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

// Folder: A Folder in an Organization's resource hierarchy, used
// to
// organize that Organization's resources.
type Folder struct {
	// CreateTime: Output only. Timestamp when the Folder was created.
	// Assigned by the server.
	CreateTime string `json:"createTime,omitempty"`

	// DisplayName: The folder’s display name.
	// A folder’s display name must be unique amongst its siblings,
	// e.g.
	// no two folders with the same parent can share the same display
	// name.
	// The display name must start and end with a letter or digit, may
	// contain
	// letters, digits, spaces, hyphens and underscores and can be no
	// longer
	// than 30 characters. This is captured by the regular
	// expression:
	// [\p{L}\p{N}]({\p{L}\p{N}_- ]{0,28}[\p{L}\p{N}])?.
	DisplayName string `json:"displayName,omitempty"`

	// LifecycleState: Output only.  The lifecycle state of the
	// folder.
	// Updates to the lifecycle_state must be performed via
	// [DeleteFolder] and [UndeleteFolder].
	//
	// Possible values:
	//   "LIFECYCLE_STATE_UNSPECIFIED" - Unspecified state.
	//   "ACTIVE" - The normal and active state.
	//   "DELETE_REQUESTED" - The folder has been marked for deletion by the
	// user.
	LifecycleState string `json:"lifecycleState,omitempty"`

	// Name: Output only. The resource name of the Folder.
	// Its format is `folders/{folder_id}`, for example: "folders/1234".
	Name string `json:"name,omitempty"`

	// Parent: The Folder’s parent's resource name.
	// Updates to the folder's parent must be performed via [MoveFolders].
	Parent string `json:"parent,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Folder) MarshalJSON() ([]byte, error) {
	type noMethod Folder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FolderOperation: Metadata describing a long running folder operation
type FolderOperation struct {
	// DestinationParent: The resource name of the folder or organization we
	// are either creating
	// the folder under or moving the folder to.
	DestinationParent string `json:"destinationParent,omitempty"`

	// DisplayName: The display name of the folder.
	DisplayName string `json:"displayName,omitempty"`

	// OperationType: The type of this operation.
	//
	// Possible values:
	//   "OPERATION_TYPE_UNSPECIFIED" - Operation type not specified.
	//   "CREATE" - A create folder operation.
	//   "MOVE" - A move folder operation.
	OperationType string `json:"operationType,omitempty"`

	// SourceParent: The resource name of the folder's parent.
	// Only applicable when the operation_type is MOVE.
	SourceParent string `json:"sourceParent,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DestinationParent")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationParent") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FolderOperation) MarshalJSON() ([]byte, error) {
	type noMethod FolderOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FolderOperationError: A classification of the Folder Operation error.
type FolderOperationError struct {
	// ErrorMessageId: The type of operation error experienced.
	//
	// Possible values:
	//   "ERROR_TYPE_UNSPECIFIED" - The error type was unrecognized or
	// unspecified.
	//   "ACTIVE_FOLDER_HEIGHT_VIOLATION" - The attempted action would
	// violate the max folder depth constraint.
	//   "MAX_CHILD_FOLDERS_VIOLATION" - The attempted action would violate
	// the max child folders constraint.
	//   "FOLDER_NAME_UNIQUENESS_VIOLATION" - The attempted action would
	// violate the locally-unique folder
	// display_name constraint.
	//   "RESOURCE_DELETED_VIOLATION" - The resource being moved has been
	// deleted.
	//   "PARENT_DELETED_VIOLATION" - The resource a folder was being added
	// to has been deleted.
	//   "CYCLE_INTRODUCED_VIOLATION" - The attempted action would introduce
	// cycle in resource path.
	//   "FOLDER_BEING_MOVED_VIOLATION" - The attempted action would move a
	// folder that is already being moved.
	//   "FOLDER_TO_DELETE_NON_EMPTY_VIOLATION" - The folder the caller is
	// trying to delete contains active resources.
	//   "DELETED_FOLDER_HEIGHT_VIOLATION" - The attempted action would
	// violate the max deleted folder depth
	// constraint.
	ErrorMessageId string `json:"errorMessageId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorMessageId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ErrorMessageId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FolderOperationError) MarshalJSON() ([]byte, error) {
	type noMethod FolderOperationError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetIamPolicyRequest: Request message for `GetIamPolicy` method.
type GetIamPolicyRequest struct {
}

// ListFoldersResponse: The ListFolders response message.
type ListFoldersResponse struct {
	// Folders: A possibly paginated list of Folders that are direct
	// descendants of
	// the specified parent resource.
	Folders []*Folder `json:"folders,omitempty"`

	// NextPageToken: A pagination token returned from a previous call to
	// `ListFolders`
	// that indicates from where listing should continue.
	// This field is optional.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Folders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Folders") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListFoldersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListFoldersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MoveFolderRequest: The MoveFolder request message.
type MoveFolderRequest struct {
	// DestinationParent: The resource name of the Folder or Organization to
	// reparent
	// the folder under.
	// Must be of the form `folders/{folder_id}` or
	// `organizations/{org_id}`.
	DestinationParent string `json:"destinationParent,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DestinationParent")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationParent") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MoveFolderRequest) MarshalJSON() ([]byte, error) {
	type noMethod MoveFolderRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a
// network API call.
type Operation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation.
	// It typically
	// contains progress information and common metadata such as create
	// time.
	// Some services might not provide such metadata.  Any method that
	// returns a
	// long-running operation should document the metadata type, if any.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that
	// originally returns it. If you use the default HTTP mapping,
	// the
	// `name` should have the format of `operations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success.
	// If the original
	// method returns no data on success, such as `Delete`, the response
	// is
	// `google.protobuf.Empty`.  If the original method is
	// standard
	// `Get`/`Create`/`Update`, the response should be the resource.  For
	// other
	// methods, the response should have the type `XxxResponse`, where
	// `Xxx`
	// is the original method name.  For example, if the original method
	// name
	// is `TakeSnapshot()`, the inferred response type
	// is
	// `TakeSnapshotResponse`.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Done") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Done") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
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

// ProjectCreationStatus: A status object which is used as the
// `metadata` field for the Operation
// returned by CreateProject. It provides insight for when significant
// phases of
// Project creation have completed.
type ProjectCreationStatus struct {
	// CreateTime: Creation time of the project creation workflow.
	CreateTime string `json:"createTime,omitempty"`

	// Gettable: True if the project can be retrieved using GetProject. No
	// other operations
	// on the project are guaranteed to work until the project creation
	// is
	// complete.
	Gettable bool `json:"gettable,omitempty"`

	// Ready: True if the project creation process is complete.
	Ready bool `json:"ready,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProjectCreationStatus) MarshalJSON() ([]byte, error) {
	type noMethod ProjectCreationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchFoldersRequest: The request message for searching folders.
type SearchFoldersRequest struct {
	// PageSize: The maximum number of folders to return in the
	// response.
	// This field is optional.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: A pagination token returned from a previous call to
	// `SearchFolders`
	// that indicates from where search should continue.
	// This field is optional.
	PageToken string `json:"pageToken,omitempty"`

	// Query: Search criteria used to select the Folders to return.
	// If no search criteria is specified then all accessible folders will
	// be
	// returned.
	//
	// Query expressions can be used to restrict results based upon
	// displayName,
	// lifecycleState and parent, where the operators `=`, `NOT`, `AND` and
	// `OR`
	// can be used along with the suffix wildcard symbol `*`.
	//
	// Some example queries
	// are:
	// |Query|Description|
	// |------|-----------|
	// |displayName=Test*|Folde
	// rs whose display name starts with
	// "Test".|
	// |lifecycleState=ACTIVE|Folders whose lifecycleState is
	// ACTIVE.|
	// |parent=folders/123|Folders whose parent is
	// "folders/123".|
	// |parent=folders/123 AND lifecycleState=ACTIVE|Active folders
	// whose
	// parent is "folders/123".|
	Query string `json:"query,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PageSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PageSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchFoldersRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchFoldersRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchFoldersResponse: The response message for searching folders.
type SearchFoldersResponse struct {
	// Folders: A possibly paginated folder search results.
	// the specified parent resource.
	Folders []*Folder `json:"folders,omitempty"`

	// NextPageToken: A pagination token returned from a previous call to
	// `SearchFolders`
	// that indicates from where searching should continue.
	// This field is optional.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Folders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Folders") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchFoldersResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchFoldersResponse
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

// Status: The `Status` type defines a logical error model that is
// suitable for different
// programming environments, including REST APIs and RPC APIs. It is
// used by
// [gRPC](https://github.com/grpc). The error model is designed to
// be:
//
// - Simple to use and understand for most users
// - Flexible enough to meet unexpected needs
//
// # Overview
//
// The `Status` message contains three pieces of data: error code, error
// message,
// and error details. The error code should be an enum value
// of
// google.rpc.Code, but it may accept additional error codes if needed.
// The
// error message should be a developer-facing English message that
// helps
// developers *understand* and *resolve* the error. If a localized
// user-facing
// error message is needed, put the localized message in the error
// details or
// localize it in the client. The optional error details may contain
// arbitrary
// information about the error. There is a predefined set of error
// detail types
// in the package `google.rpc` that can be used for common error
// conditions.
//
// # Language mapping
//
// The `Status` message is the logical representation of the error
// model, but it
// is not necessarily the actual wire format. When the `Status` message
// is
// exposed in different client libraries and different wire protocols,
// it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions
// in Java, but more likely mapped to some error codes in C.
//
// # Other uses
//
// The error model and the `Status` message can be used in a variety
// of
// environments, either with or without APIs, to provide a
// consistent developer experience across different
// environments.
//
// Example uses of this error model include:
//
// - Partial errors. If a service needs to return partial errors to the
// client,
//     it may embed the `Status` in the normal response to indicate the
// partial
//     errors.
//
// - Workflow errors. A typical workflow has multiple steps. Each step
// may
//     have a `Status` message for error reporting.
//
// - Batch operations. If a client uses batch request and batch
// response, the
//     `Status` message should be used directly inside batch response,
// one for
//     each error sub-response.
//
// - Asynchronous operations. If an API call embeds asynchronous
// operation
//     results in its response, the status of those operations should
// be
//     represented directly using the `Status` message.
//
// - Logging. If some API errors are stored in logs, the message
// `Status` could
//     be used directly after any stripping needed for security/privacy
// reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details.  There is a
	// common set of
	// message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any
	// user-facing error message should be localized and sent in
	// the
	// google.rpc.Status.details field, or localized by the client.
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

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
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

// UndeleteFolderRequest: The UndeleteFolder request message.
type UndeleteFolderRequest struct {
}

// method id "cloudresourcemanager.folders.create":

type FoldersCreateCall struct {
	s          *Service
	folder     *Folder
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a Folder in the resource hierarchy.
// Returns an Operation which can be used to track the progress of
// the
// folder creation workflow.
// Upon success the Operation.response field will be populated with
// the
// created Folder.
//
// In order to succeed, the addition of this new Folder must not
// violate
// the Folder naming, height or fanout constraints.
// + The Folder's display_name must be distinct from all other Folder's
// that
// share its parent.
// + The addition of the Folder must not cause the active Folder
// hierarchy
// to exceed a height of 4. Note, the full active + deleted Folder
// hierarchy
// is allowed to reach a height of 8; this provides additional headroom
// when
// moving folders that contain deleted folders.
// + The addition of the Folder must not cause the total number of
// Folders
// under its parent to exceed 100.
//
// If the operation fails due to a folder constraint violation,
// a PreconditionFailure explaining the violation will be returned.
// If the failure occurs synchronously then the PreconditionFailure
// will be returned via the Status.details field and if it
// occurs
// asynchronously then the PreconditionFailure will be returned
// via the the Operation.error field.
//
// The caller must have `resourcemanager.folders.create` permission on
// the
// identified parent.
func (r *FoldersService) Create(folder *Folder) *FoldersCreateCall {
	c := &FoldersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.folder = folder
	return c
}

// Parent sets the optional parameter "parent": The resource name of the
// new Folder's parent.
// Must be of the form `folders/{folder_id}` or
// `organizations/{org_id}`.
func (c *FoldersCreateCall) Parent(parent string) *FoldersCreateCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersCreateCall) Fields(s ...googleapi.Field) *FoldersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersCreateCall) Context(ctx context.Context) *FoldersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.folder)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/folders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *FoldersCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a Folder in the resource hierarchy.\nReturns an Operation which can be used to track the progress of the\nfolder creation workflow.\nUpon success the Operation.response field will be populated with the\ncreated Folder.\n\nIn order to succeed, the addition of this new Folder must not violate\nthe Folder naming, height or fanout constraints.\n+ The Folder's display_name must be distinct from all other Folder's that\nshare its parent.\n+ The addition of the Folder must not cause the active Folder hierarchy\nto exceed a height of 4. Note, the full active + deleted Folder hierarchy\nis allowed to reach a height of 8; this provides additional headroom when\nmoving folders that contain deleted folders.\n+ The addition of the Folder must not cause the total number of Folders\nunder its parent to exceed 100.\n\nIf the operation fails due to a folder constraint violation,\na PreconditionFailure explaining the violation will be returned.\nIf the failure occurs synchronously then the PreconditionFailure\nwill be returned via the Status.details field and if it occurs\nasynchronously then the PreconditionFailure will be returned\nvia the the Operation.error field.\n\nThe caller must have `resourcemanager.folders.create` permission on the\nidentified parent.",
	//   "flatPath": "v2beta1/folders",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.create",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "parent": {
	//       "description": "The resource name of the new Folder's parent.\nMust be of the form `folders/{folder_id}` or `organizations/{org_id}`.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/folders",
	//   "request": {
	//     "$ref": "Folder"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.delete":

type FoldersDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Requests deletion of a Folder. The Folder is moved into
// the
// [DELETE_REQUESTED] state immediately, and is deleted approximately 30
// days
// later. This method may only be called on an empty Folder in the
// [ACTIVE]
// state, where a Folder is empty if it doesn't contain any Folders
// or
// Projects in the [ACTIVE] state.
// The caller must have `resourcemanager.folders.delete` permission on
// the
// identified folder.
func (r *FoldersService) Delete(name string) *FoldersDeleteCall {
	c := &FoldersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersDeleteCall) Fields(s ...googleapi.Field) *FoldersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersDeleteCall) Context(ctx context.Context) *FoldersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.delete" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersDeleteCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Requests deletion of a Folder. The Folder is moved into the\n[DELETE_REQUESTED] state immediately, and is deleted approximately 30 days\nlater. This method may only be called on an empty Folder in the [ACTIVE]\nstate, where a Folder is empty if it doesn't contain any Folders or\nProjects in the [ACTIVE] state.\nThe caller must have `resourcemanager.folders.delete` permission on the\nidentified folder.",
	//   "flatPath": "v2beta1/folders/{foldersId}",
	//   "httpMethod": "DELETE",
	//   "id": "cloudresourcemanager.folders.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "the resource name of the Folder to be deleted.\nMust be of the form `folders/{folder_id}`.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.get":

type FoldersGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a Folder identified by the supplied resource
// name.
// Valid Folder resource names have the format
// `folders/{folder_id}`
// (for example, `folders/1234`).
// The caller must have `resourcemanager.folders.get` permission on
// the
// identified folder.
func (r *FoldersService) Get(name string) *FoldersGetCall {
	c := &FoldersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersGetCall) Fields(s ...googleapi.Field) *FoldersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FoldersGetCall) IfNoneMatch(entityTag string) *FoldersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersGetCall) Context(ctx context.Context) *FoldersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.get" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersGetCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Retrieves a Folder identified by the supplied resource name.\nValid Folder resource names have the format `folders/{folder_id}`\n(for example, `folders/1234`).\nThe caller must have `resourcemanager.folders.get` permission on the\nidentified folder.",
	//   "flatPath": "v2beta1/folders/{foldersId}",
	//   "httpMethod": "GET",
	//   "id": "cloudresourcemanager.folders.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the Folder to retrieve.\nMust be of the form `folders/{folder_id}`.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.getIamPolicy":

type FoldersGetIamPolicyCall struct {
	s                   *Service
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for a Folder. The
// returned policy may be
// empty if no such policy or resource exists. The `resource` field
// should
// be the Folder's resource name, e.g. "folders/1234".
// The caller must have `resourcemanager.folders.getIamPolicy`
// permission
// on the identified folder.
func (r *FoldersService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *FoldersGetIamPolicyCall {
	c := &FoldersGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersGetIamPolicyCall) Fields(s ...googleapi.Field) *FoldersGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersGetIamPolicyCall) Context(ctx context.Context) *FoldersGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets the access control policy for a Folder. The returned policy may be\nempty if no such policy or resource exists. The `resource` field should\nbe the Folder's resource name, e.g. \"folders/1234\".\nThe caller must have `resourcemanager.folders.getIamPolicy` permission\non the identified folder.",
	//   "flatPath": "v2beta1/folders/{foldersId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+resource}:getIamPolicy",
	//   "request": {
	//     "$ref": "GetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.list":

type FoldersListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the Folders that are direct descendants of supplied
// parent resource.
// List provides a strongly consistent view of the Folders
// underneath
// the specified parent resource.
// List returns Folders sorted based upon the (ascending) lexical
// ordering
// of their display_name.
// The caller must have `resourcemanager.folders.list` permission on
// the
// identified parent.
func (r *FoldersService) List() *FoldersListCall {
	c := &FoldersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of Folders to return in the response.
// This field is optional.
func (c *FoldersListCall) PageSize(pageSize int64) *FoldersListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A pagination token
// returned from a previous call to `ListFolders`
// that indicates where this listing should continue from.
// This field is optional.
func (c *FoldersListCall) PageToken(pageToken string) *FoldersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Parent sets the optional parameter "parent": The resource name of the
// Organization or Folder whose Folders are
// being listed.
// Must be of the form `folders/{folder_id}` or
// `organizations/{org_id}`.
// Access to this method is controlled by checking
// the
// `resourcemanager.folders.list` permission on the `parent`.
func (c *FoldersListCall) Parent(parent string) *FoldersListCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Controls
// whether Folders in the [DELETE_REQUESTED} state should
// be returned.
func (c *FoldersListCall) ShowDeleted(showDeleted bool) *FoldersListCall {
	c.urlParams_.Set("showDeleted", fmt.Sprint(showDeleted))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersListCall) Fields(s ...googleapi.Field) *FoldersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FoldersListCall) IfNoneMatch(entityTag string) *FoldersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersListCall) Context(ctx context.Context) *FoldersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/folders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.list" call.
// Exactly one of *ListFoldersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListFoldersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *FoldersListCall) Do(opts ...googleapi.CallOption) (*ListFoldersResponse, error) {
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
	ret := &ListFoldersResponse{
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
	//   "description": "Lists the Folders that are direct descendants of supplied parent resource.\nList provides a strongly consistent view of the Folders underneath\nthe specified parent resource.\nList returns Folders sorted based upon the (ascending) lexical ordering\nof their display_name.\nThe caller must have `resourcemanager.folders.list` permission on the\nidentified parent.",
	//   "flatPath": "v2beta1/folders",
	//   "httpMethod": "GET",
	//   "id": "cloudresourcemanager.folders.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of Folders to return in the response.\nThis field is optional.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A pagination token returned from a previous call to `ListFolders`\nthat indicates where this listing should continue from.\nThis field is optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The resource name of the Organization or Folder whose Folders are\nbeing listed.\nMust be of the form `folders/{folder_id}` or `organizations/{org_id}`.\nAccess to this method is controlled by checking the\n`resourcemanager.folders.list` permission on the `parent`.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Controls whether Folders in the [DELETE_REQUESTED} state should\nbe returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "v2beta1/folders",
	//   "response": {
	//     "$ref": "ListFoldersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *FoldersListCall) Pages(ctx context.Context, f func(*ListFoldersResponse) error) error {
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

// method id "cloudresourcemanager.folders.move":

type FoldersMoveCall struct {
	s                 *Service
	name              string
	movefolderrequest *MoveFolderRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Move: Moves a Folder under a new resource parent.
// Returns an Operation which can be used to track the progress of
// the
// folder move workflow.
// Upon success the Operation.response field will be populated with
// the
// moved Folder.
// Upon failure, a FolderOperationError categorizing the failure cause
// will
// be returned - if the failure occurs synchronously then
// the
// FolderOperationError will be returned via the Status.details
// field
// and if it occurs asynchronously then the FolderOperation will be
// returned
// via the the Operation.error field.
// In addition, the Operation.metadata field will be populated with
// a
// FolderOperation message as an aid to stateless clients.
// Folder moves will be rejected if they violate either the naming,
// height
// or fanout constraints described in the [CreateFolder]
// documentation.
// The caller must have `resourcemanager.folders.move` permission on
// the
// folder's current and proposed new parent.
func (r *FoldersService) Move(name string, movefolderrequest *MoveFolderRequest) *FoldersMoveCall {
	c := &FoldersMoveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.movefolderrequest = movefolderrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersMoveCall) Fields(s ...googleapi.Field) *FoldersMoveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersMoveCall) Context(ctx context.Context) *FoldersMoveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersMoveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersMoveCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.movefolderrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}:move")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.move" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *FoldersMoveCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Moves a Folder under a new resource parent.\nReturns an Operation which can be used to track the progress of the\nfolder move workflow.\nUpon success the Operation.response field will be populated with the\nmoved Folder.\nUpon failure, a FolderOperationError categorizing the failure cause will\nbe returned - if the failure occurs synchronously then the\nFolderOperationError will be returned via the Status.details field\nand if it occurs asynchronously then the FolderOperation will be returned\nvia the the Operation.error field.\nIn addition, the Operation.metadata field will be populated with a\nFolderOperation message as an aid to stateless clients.\nFolder moves will be rejected if they violate either the naming, height\nor fanout constraints described in the [CreateFolder] documentation.\nThe caller must have `resourcemanager.folders.move` permission on the\nfolder's current and proposed new parent.",
	//   "flatPath": "v2beta1/folders/{foldersId}:move",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.move",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the Folder to move.\nMust be of the form folders/{folder_id}",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}:move",
	//   "request": {
	//     "$ref": "MoveFolderRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.patch":

type FoldersPatchCall struct {
	s          *Service
	name       string
	folder     *Folder
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a Folder, changing its display_name.
// Changes to the folder display_name will be rejected if they violate
// either
// the display_name formatting rules or naming constraints described
// in
// the [CreateFolder] documentation.
// + The Folder's display name must start and end with a letter or
// digit,
// may contain letters, digits, spaces, hyphens and underscores and can
// be
// no longer than 30 characters. This is captured by the regular
// expression:
// [\p{L}\p{N}]({\p{L}\p{N}_- ]{0,28}[\p{L}\p{N}])?.
// The caller must have `resourcemanager.folders.update` permission on
// the
// identified folder.
//
// If the update fails due to the unique name constraint then
// a
// PreconditionFailure explaining this violation will be returned
// in the Status.details field.
func (r *FoldersService) Patch(name string, folder *Folder) *FoldersPatchCall {
	c := &FoldersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.folder = folder
	return c
}

// UpdateMask sets the optional parameter "updateMask": Fields to be
// updated.
// Only the `display_name` can be updated.
func (c *FoldersPatchCall) UpdateMask(updateMask string) *FoldersPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersPatchCall) Fields(s ...googleapi.Field) *FoldersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersPatchCall) Context(ctx context.Context) *FoldersPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.folder)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.patch" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersPatchCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Updates a Folder, changing its display_name.\nChanges to the folder display_name will be rejected if they violate either\nthe display_name formatting rules or naming constraints described in\nthe [CreateFolder] documentation.\n+ The Folder's display name must start and end with a letter or digit,\nmay contain letters, digits, spaces, hyphens and underscores and can be\nno longer than 30 characters. This is captured by the regular expression:\n[\\p{L}\\p{N}]({\\p{L}\\p{N}_- ]{0,28}[\\p{L}\\p{N}])?.\nThe caller must have `resourcemanager.folders.update` permission on the\nidentified folder.\n\nIf the update fails due to the unique name constraint then a\nPreconditionFailure explaining this violation will be returned\nin the Status.details field.",
	//   "flatPath": "v2beta1/folders/{foldersId}",
	//   "httpMethod": "PATCH",
	//   "id": "cloudresourcemanager.folders.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Output only. The resource name of the Folder.\nIts format is `folders/{folder_id}`, for example: \"folders/1234\".",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "Fields to be updated.\nOnly the `display_name` can be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}",
	//   "request": {
	//     "$ref": "Folder"
	//   },
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudresourcemanager.folders.search":

type FoldersSearchCall struct {
	s                    *Service
	searchfoldersrequest *SearchFoldersRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Search: Search for folders that match specific filter
// criteria.
// Search provides an eventually consistent view of the folders a user
// has
// access to which meet the specified filter criteria.
//
// This will only return folders on which the caller has the
// permission `resourcemanager.folders.get`.
func (r *FoldersService) Search(searchfoldersrequest *SearchFoldersRequest) *FoldersSearchCall {
	c := &FoldersSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchfoldersrequest = searchfoldersrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersSearchCall) Fields(s ...googleapi.Field) *FoldersSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersSearchCall) Context(ctx context.Context) *FoldersSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchfoldersrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/folders:search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.search" call.
// Exactly one of *SearchFoldersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SearchFoldersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *FoldersSearchCall) Do(opts ...googleapi.CallOption) (*SearchFoldersResponse, error) {
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
	ret := &SearchFoldersResponse{
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
	//   "description": "Search for folders that match specific filter criteria.\nSearch provides an eventually consistent view of the folders a user has\naccess to which meet the specified filter criteria.\n\nThis will only return folders on which the caller has the\npermission `resourcemanager.folders.get`.",
	//   "flatPath": "v2beta1/folders:search",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/folders:search",
	//   "request": {
	//     "$ref": "SearchFoldersRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchFoldersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *FoldersSearchCall) Pages(ctx context.Context, f func(*SearchFoldersResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchfoldersrequest.PageToken = pt }(c.searchfoldersrequest.PageToken) // reset paging to original point
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
		c.searchfoldersrequest.PageToken = x.NextPageToken
	}
}

// method id "cloudresourcemanager.folders.setIamPolicy":

type FoldersSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on a Folder, replacing
// any existing policy.
// The `resource` field should be the Folder's resource name,
// e.g.
// "folders/1234".
// The caller must have `resourcemanager.folders.setIamPolicy`
// permission
// on the identified folder.
func (r *FoldersService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *FoldersSetIamPolicyCall {
	c := &FoldersSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersSetIamPolicyCall) Fields(s ...googleapi.Field) *FoldersSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersSetIamPolicyCall) Context(ctx context.Context) *FoldersSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the access control policy on a Folder, replacing any existing policy.\nThe `resource` field should be the Folder's resource name, e.g.\n\"folders/1234\".\nThe caller must have `resourcemanager.folders.setIamPolicy` permission\non the identified folder.",
	//   "flatPath": "v2beta1/folders/{foldersId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+resource}:setIamPolicy",
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

// method id "cloudresourcemanager.folders.testIamPermissions":

type FoldersTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified Folder.
// The `resource` field should be the Folder's resource name,
// e.g. "folders/1234".
//
// There are no permissions required for making this API call.
func (r *FoldersService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *FoldersTestIamPermissionsCall {
	c := &FoldersTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersTestIamPermissionsCall) Fields(s ...googleapi.Field) *FoldersTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersTestIamPermissionsCall) Context(ctx context.Context) *FoldersTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *FoldersTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	//   "description": "Returns permissions that a caller has on the specified Folder.\nThe `resource` field should be the Folder's resource name,\ne.g. \"folders/1234\".\n\nThere are no permissions required for making this API call.",
	//   "flatPath": "v2beta1/folders/{foldersId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+resource}:testIamPermissions",
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

// method id "cloudresourcemanager.folders.undelete":

type FoldersUndeleteCall struct {
	s                     *Service
	name                  string
	undeletefolderrequest *UndeleteFolderRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Undelete: Cancels the deletion request for a Folder. This method may
// only be
// called on a Folder in the [DELETE_REQUESTED] state.
// In order to succeed, the Folder's parent must be in the [ACTIVE]
// state.
// In addition, reintroducing the folder into the tree must not
// violate
// folder naming, height and fanout constraints described in
// the
// [CreateFolder] documentation.
// The caller must have `resourcemanager.folders.undelete` permission on
// the
// identified folder.
func (r *FoldersService) Undelete(name string, undeletefolderrequest *UndeleteFolderRequest) *FoldersUndeleteCall {
	c := &FoldersUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.undeletefolderrequest = undeletefolderrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FoldersUndeleteCall) Fields(s ...googleapi.Field) *FoldersUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FoldersUndeleteCall) Context(ctx context.Context) *FoldersUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FoldersUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FoldersUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.undeletefolderrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+name}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudresourcemanager.folders.undelete" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *FoldersUndeleteCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Cancels the deletion request for a Folder. This method may only be\ncalled on a Folder in the [DELETE_REQUESTED] state.\nIn order to succeed, the Folder's parent must be in the [ACTIVE] state.\nIn addition, reintroducing the folder into the tree must not violate\nfolder naming, height and fanout constraints described in the\n[CreateFolder] documentation.\nThe caller must have `resourcemanager.folders.undelete` permission on the\nidentified folder.",
	//   "flatPath": "v2beta1/folders/{foldersId}:undelete",
	//   "httpMethod": "POST",
	//   "id": "cloudresourcemanager.folders.undelete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The resource name of the Folder to undelete.\nMust be of the form `folders/{folder_id}`.",
	//       "location": "path",
	//       "pattern": "^folders/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+name}:undelete",
	//   "request": {
	//     "$ref": "UndeleteFolderRequest"
	//   },
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
