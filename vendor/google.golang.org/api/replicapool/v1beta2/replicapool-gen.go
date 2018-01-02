// Package replicapool provides access to the Google Compute Engine Instance Group Manager API.
//
// See https://developers.google.com/compute/docs/instance-groups/manager/v1beta2
//
// Usage example:
//
//   import "google.golang.org/api/replicapool/v1beta2"
//   ...
//   replicapoolService, err := replicapool.New(oauthHttpClient)
package replicapool // import "google.golang.org/api/replicapool/v1beta2"

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

const apiId = "replicapool:v1beta2"
const apiName = "replicapool"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/replicapool/v1beta2/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// View your Google Compute Engine resources
	ComputeReadonlyScope = "https://www.googleapis.com/auth/compute.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.InstanceGroupManagers = NewInstanceGroupManagersService(s)
	s.ZoneOperations = NewZoneOperationsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	InstanceGroupManagers *InstanceGroupManagersService

	ZoneOperations *ZoneOperationsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewInstanceGroupManagersService(s *Service) *InstanceGroupManagersService {
	rs := &InstanceGroupManagersService{s: s}
	return rs
}

type InstanceGroupManagersService struct {
	s *Service
}

func NewZoneOperationsService(s *Service) *ZoneOperationsService {
	rs := &ZoneOperationsService{s: s}
	return rs
}

type ZoneOperationsService struct {
	s *Service
}

// InstanceGroupManager: An Instance Group Manager resource.
type InstanceGroupManager struct {
	// AutoHealingPolicies: The autohealing policy for this managed instance
	// group. You can specify only one value.
	AutoHealingPolicies []*ReplicaPoolAutoHealingPolicy `json:"autoHealingPolicies,omitempty"`

	// BaseInstanceName: The base instance name to use for instances in this
	// group. The value must be a valid RFC1035 name. Supported characters
	// are lowercase letters, numbers, and hyphens (-). Instances are named
	// by appending a hyphen and a random four-character string to the base
	// instance name.
	BaseInstanceName string `json:"baseInstanceName,omitempty"`

	// CreationTimestamp: [Output only] The time the instance group manager
	// was created, in RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// CurrentSize: [Output only] The number of instances that currently
	// exist and are a part of this group. This includes instances that are
	// starting but are not yet RUNNING, and instances that are in the
	// process of being deleted or abandoned.
	CurrentSize int64 `json:"currentSize,omitempty"`

	// Description: An optional textual description of the instance group
	// manager.
	Description string `json:"description,omitempty"`

	// Fingerprint: [Output only] Fingerprint of the instance group manager.
	// This field is used for optimistic locking. An up-to-date fingerprint
	// must be provided in order to modify the Instance Group Manager
	// resource.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Group: [Output only] The full URL of the instance group created by
	// the manager. This group contains all of the instances being managed,
	// and cannot contain non-managed instances.
	Group string `json:"group,omitempty"`

	// Id: [Output only] A server-assigned unique identifier for the
	// resource.
	Id uint64 `json:"id,omitempty,string"`

	// InstanceTemplate: The full URL to an instance template from which all
	// new instances will be created.
	InstanceTemplate string `json:"instanceTemplate,omitempty"`

	// Kind: [Output only] The resource type. Always
	// replicapool#instanceGroupManager.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the instance group manager. Must be 1-63 characters
	// long and comply with RFC1035. Supported characters include lowercase
	// letters, numbers, and hyphens.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output only] The fully qualified URL for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// TargetPools: The full URL of all target pools to which new instances
	// in the group are added. Updating the target pool values does not
	// affect existing instances.
	TargetPools []string `json:"targetPools,omitempty"`

	// TargetSize: [Output only] The number of instances that the manager is
	// attempting to maintain. Deleting or abandoning instances affects this
	// number, as does resizing the group.
	TargetSize int64 `json:"targetSize,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AutoHealingPolicies")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AutoHealingPolicies") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManager) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManager
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagerList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: A list of instance resources.
	Items []*InstanceGroupManager `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
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

func (s *InstanceGroupManagerList) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagerList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagersAbandonInstancesRequest struct {
	// Instances: The names of one or more instances to abandon. For
	// example:
	// { 'instances': [ 'instance-c3po', 'instance-r2d2' ] }
	Instances []string `json:"instances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Instances") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Instances") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManagersAbandonInstancesRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagersAbandonInstancesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagersDeleteInstancesRequest struct {
	// Instances: Names of instances to delete.
	//
	// Example: 'instance-foo', 'instance-bar'
	Instances []string `json:"instances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Instances") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Instances") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManagersDeleteInstancesRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagersDeleteInstancesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagersRecreateInstancesRequest struct {
	// Instances: The names of one or more instances to recreate. For
	// example:
	// { 'instances': [ 'instance-c3po', 'instance-r2d2' ] }
	Instances []string `json:"instances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Instances") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Instances") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManagersRecreateInstancesRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagersRecreateInstancesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagersSetInstanceTemplateRequest struct {
	// InstanceTemplate: The full URL to an Instance Template from which all
	// new instances will be created.
	InstanceTemplate string `json:"instanceTemplate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InstanceTemplate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InstanceTemplate") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManagersSetInstanceTemplateRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagersSetInstanceTemplateRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type InstanceGroupManagersSetTargetPoolsRequest struct {
	// Fingerprint: The current fingerprint of the Instance Group Manager
	// resource. If this does not match the server-side fingerprint of the
	// resource, then the request will be rejected.
	Fingerprint string `json:"fingerprint,omitempty"`

	// TargetPools: A list of fully-qualified URLs to existing Target Pool
	// resources. New instances in the Instance Group Manager will be added
	// to the specified target pools; existing instances are not affected.
	TargetPools []string `json:"targetPools,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fingerprint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fingerprint") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupManagersSetTargetPoolsRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupManagersSetTargetPoolsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: An operation resource, used to manage asynchronous API
// requests.
type Operation struct {
	// ClientOperationId: [Output only] An optional identifier specified by
	// the client when the mutation was initiated. Must be unique for all
	// operation resources in the project.
	ClientOperationId string `json:"clientOperationId,omitempty"`

	// CreationTimestamp: [Output Only] The time that this operation was
	// requested, in RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// EndTime: [Output Only] The time that this operation was completed, in
	// RFC3339 text format.
	EndTime string `json:"endTime,omitempty"`

	// Error: [Output Only] If errors occurred during processing of this
	// operation, this field will be populated.
	Error *OperationError `json:"error,omitempty"`

	// HttpErrorMessage: [Output only] If operation fails, the HTTP error
	// message returned.
	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	// HttpErrorStatusCode: [Output only] If operation fails, the HTTP error
	// status code returned.
	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	// Id: [Output Only] Unique identifier for the resource, generated by
	// the server.
	Id uint64 `json:"id,omitempty,string"`

	// InsertTime: [Output Only] The time that this operation was requested,
	// in RFC3339 text format.
	InsertTime string `json:"insertTime,omitempty"`

	// Kind: [Output only] Type of the resource.
	Kind string `json:"kind,omitempty"`

	// Name: [Output Only] Name of the resource.
	Name string `json:"name,omitempty"`

	// OperationType: [Output only] Type of the operation. Operations
	// include insert, update, and delete.
	OperationType string `json:"operationType,omitempty"`

	// Progress: [Output only] An optional progress indicator that ranges
	// from 0 to 100. There is no requirement that this be linear or support
	// any granularity of operations. This should not be used to guess at
	// when the operation will be complete. This number should be
	// monotonically increasing as the operation progresses.
	Progress int64 `json:"progress,omitempty"`

	// Region: [Output Only] URL of the region where the operation resides.
	// Only available when performing regional operations.
	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server-defined fully-qualified URL for this
	// resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: [Output Only] The time that this operation was started by
	// the server, in RFC3339 text format.
	StartTime string `json:"startTime,omitempty"`

	// Status: [Output Only] Status of the operation.
	//
	// Possible values:
	//   "DONE"
	//   "PENDING"
	//   "RUNNING"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output Only] An optional textual description of the
	// current status of the operation.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetId: [Output Only] Unique target ID which identifies a
	// particular incarnation of the target.
	TargetId uint64 `json:"targetId,omitempty,string"`

	// TargetLink: [Output only] URL of the resource the operation is
	// mutating.
	TargetLink string `json:"targetLink,omitempty"`

	// User: [Output Only] User who requested the operation, for example:
	// user@example.com.
	User string `json:"user,omitempty"`

	// Warnings: [Output Only] If there are issues with this operation, a
	// warning is returned.
	Warnings []*OperationWarnings `json:"warnings,omitempty"`

	// Zone: [Output Only] URL of the zone where the operation resides. Only
	// available when performing per-zone operations.
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

// OperationError: [Output Only] If errors occurred during processing of
// this operation, this field will be populated.
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

	// Location: [Output Only] Indicates the field in the request which
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
	// Code: [Output only] The warning type identifier for this warning.
	//
	// Possible values:
	//   "DEPRECATED_RESOURCE_USED"
	//   "DISK_SIZE_LARGER_THAN_IMAGE_SIZE"
	//   "INJECTED_KERNELS_DEPRECATED"
	//   "NEXT_HOP_ADDRESS_NOT_ASSIGNED"
	//   "NEXT_HOP_CANNOT_IP_FORWARD"
	//   "NEXT_HOP_INSTANCE_NOT_FOUND"
	//   "NEXT_HOP_INSTANCE_NOT_ON_NETWORK"
	//   "NEXT_HOP_NOT_RUNNING"
	//   "NO_RESULTS_ON_PAGE"
	//   "REQUIRED_TOS_AGREEMENT"
	//   "RESOURCE_NOT_DELETED"
	//   "SINGLE_INSTANCE_PROPERTY_TEMPLATE"
	//   "UNREACHABLE"
	Code string `json:"code,omitempty"`

	// Data: [Output only] Metadata for this warning in key:value format.
	Data []*OperationWarningsData `json:"data,omitempty"`

	// Message: [Output only] Optional human-readable details for this
	// warning.
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
	// Key: [Output Only] Metadata key for this warning.
	Key string `json:"key,omitempty"`

	// Value: [Output Only] Metadata value for this warning.
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

type OperationList struct {
	// Id: Unique identifier for the resource; defined by the server (output
	// only).
	Id string `json:"id,omitempty"`

	// Items: The operation resources.
	Items []*Operation `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token used to continue a truncated list request
	// (output only).
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Server defined URL for this resource (output only).
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

type ReplicaPoolAutoHealingPolicy struct {
	// ActionType: The action to perform when an instance becomes unhealthy.
	// Possible values are RECREATE or REBOOT. RECREATE replaces an
	// unhealthy instance with a new instance that is based on the instance
	// template for this managed instance group. REBOOT performs a soft
	// reboot on an instance. If the instance cannot reboot, the instance
	// performs a hard restart.
	//
	// Possible values:
	//   "REBOOT"
	//   "RECREATE"
	ActionType string `json:"actionType,omitempty"`

	// HealthCheck: The URL for the HealthCheck that signals autohealing.
	HealthCheck string `json:"healthCheck,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ActionType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ActionType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReplicaPoolAutoHealingPolicy) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaPoolAutoHealingPolicy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "replicapool.instanceGroupManagers.abandonInstances":

type InstanceGroupManagersAbandonInstancesCall struct {
	s                                            *Service
	project                                      string
	zone                                         string
	instanceGroupManager                         string
	instancegroupmanagersabandoninstancesrequest *InstanceGroupManagersAbandonInstancesRequest
	urlParams_                                   gensupport.URLParams
	ctx_                                         context.Context
	header_                                      http.Header
}

// AbandonInstances: Removes the specified instances from the managed
// instance group, and from any target pools of which they were members,
// without deleting the instances.
func (r *InstanceGroupManagersService) AbandonInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersabandoninstancesrequest *InstanceGroupManagersAbandonInstancesRequest) *InstanceGroupManagersAbandonInstancesCall {
	c := &InstanceGroupManagersAbandonInstancesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersabandoninstancesrequest = instancegroupmanagersabandoninstancesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersAbandonInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersAbandonInstancesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersAbandonInstancesCall) Context(ctx context.Context) *InstanceGroupManagersAbandonInstancesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersAbandonInstancesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersAbandonInstancesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersabandoninstancesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/abandonInstances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.abandonInstances" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersAbandonInstancesCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Removes the specified instances from the managed instance group, and from any target pools of which they were members, without deleting the instances.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.abandonInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/abandonInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersAbandonInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.delete":

type InstanceGroupManagersDeleteCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Delete: Deletes the instance group manager and all instances
// contained within. If you'd like to delete the manager without
// deleting the instances, you must first abandon the instances to
// remove them from the group.
func (r *InstanceGroupManagersService) Delete(project string, zone string, instanceGroupManager string) *InstanceGroupManagersDeleteCall {
	c := &InstanceGroupManagersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersDeleteCall) Fields(s ...googleapi.Field) *InstanceGroupManagersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersDeleteCall) Context(ctx context.Context) *InstanceGroupManagersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Deletes the instance group manager and all instances contained within. If you'd like to delete the manager without deleting the instances, you must first abandon the instances to remove them from the group.",
	//   "httpMethod": "DELETE",
	//   "id": "replicapool.instanceGroupManagers.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "Name of the Instance Group Manager resource to delete.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.deleteInstances":

type InstanceGroupManagersDeleteInstancesCall struct {
	s                                           *Service
	project                                     string
	zone                                        string
	instanceGroupManager                        string
	instancegroupmanagersdeleteinstancesrequest *InstanceGroupManagersDeleteInstancesRequest
	urlParams_                                  gensupport.URLParams
	ctx_                                        context.Context
	header_                                     http.Header
}

// DeleteInstances: Deletes the specified instances. The instances are
// deleted, then removed from the instance group and any target pools of
// which they were a member. The targetSize of the instance group
// manager is reduced by the number of instances deleted.
func (r *InstanceGroupManagersService) DeleteInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersdeleteinstancesrequest *InstanceGroupManagersDeleteInstancesRequest) *InstanceGroupManagersDeleteInstancesCall {
	c := &InstanceGroupManagersDeleteInstancesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersdeleteinstancesrequest = instancegroupmanagersdeleteinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersDeleteInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersDeleteInstancesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersDeleteInstancesCall) Context(ctx context.Context) *InstanceGroupManagersDeleteInstancesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersDeleteInstancesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersDeleteInstancesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersdeleteinstancesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deleteInstances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.deleteInstances" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersDeleteInstancesCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Deletes the specified instances. The instances are deleted, then removed from the instance group and any target pools of which they were a member. The targetSize of the instance group manager is reduced by the number of instances deleted.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.deleteInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/deleteInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersDeleteInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.get":

type InstanceGroupManagersGetCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	urlParams_           gensupport.URLParams
	ifNoneMatch_         string
	ctx_                 context.Context
	header_              http.Header
}

// Get: Returns the specified Instance Group Manager resource.
func (r *InstanceGroupManagersService) Get(project string, zone string, instanceGroupManager string) *InstanceGroupManagersGetCall {
	c := &InstanceGroupManagersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersGetCall) Fields(s ...googleapi.Field) *InstanceGroupManagersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstanceGroupManagersGetCall) IfNoneMatch(entityTag string) *InstanceGroupManagersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersGetCall) Context(ctx context.Context) *InstanceGroupManagersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.get" call.
// Exactly one of *InstanceGroupManager or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InstanceGroupManager.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstanceGroupManagersGetCall) Do(opts ...googleapi.CallOption) (*InstanceGroupManager, error) {
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
	ret := &InstanceGroupManager{
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
	//   "description": "Returns the specified Instance Group Manager resource.",
	//   "httpMethod": "GET",
	//   "id": "replicapool.instanceGroupManagers.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "Name of the instance resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}",
	//   "response": {
	//     "$ref": "InstanceGroupManager"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.insert":

type InstanceGroupManagersInsertCall struct {
	s                    *Service
	project              string
	zone                 string
	instancegroupmanager *InstanceGroupManager
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Insert: Creates an instance group manager, as well as the instance
// group and the specified number of instances.
func (r *InstanceGroupManagersService) Insert(project string, zone string, size int64, instancegroupmanager *InstanceGroupManager) *InstanceGroupManagersInsertCall {
	c := &InstanceGroupManagersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.urlParams_.Set("size", fmt.Sprint(size))
	c.instancegroupmanager = instancegroupmanager
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersInsertCall) Fields(s ...googleapi.Field) *InstanceGroupManagersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersInsertCall) Context(ctx context.Context) *InstanceGroupManagersInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanager)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates an instance group manager, as well as the instance group and the specified number of instances.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "size"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "size": {
	//       "description": "Number of instances that should exist.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers",
	//   "request": {
	//     "$ref": "InstanceGroupManager"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.list":

type InstanceGroupManagersListCall struct {
	s            *Service
	project      string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the list of Instance Group Manager resources
// contained within the specified zone.
func (r *InstanceGroupManagersService) List(project string, zone string) *InstanceGroupManagersListCall {
	c := &InstanceGroupManagersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *InstanceGroupManagersListCall) Filter(filter string) *InstanceGroupManagersListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 500.
func (c *InstanceGroupManagersListCall) MaxResults(maxResults int64) *InstanceGroupManagersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *InstanceGroupManagersListCall) PageToken(pageToken string) *InstanceGroupManagersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersListCall) Fields(s ...googleapi.Field) *InstanceGroupManagersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstanceGroupManagersListCall) IfNoneMatch(entityTag string) *InstanceGroupManagersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersListCall) Context(ctx context.Context) *InstanceGroupManagersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.list" call.
// Exactly one of *InstanceGroupManagerList or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *InstanceGroupManagerList.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstanceGroupManagersListCall) Do(opts ...googleapi.CallOption) (*InstanceGroupManagerList, error) {
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
	ret := &InstanceGroupManagerList{
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
	//   "description": "Retrieves the list of Instance Group Manager resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "replicapool.instanceGroupManagers.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 500.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers",
	//   "response": {
	//     "$ref": "InstanceGroupManagerList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *InstanceGroupManagersListCall) Pages(ctx context.Context, f func(*InstanceGroupManagerList) error) error {
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

// method id "replicapool.instanceGroupManagers.recreateInstances":

type InstanceGroupManagersRecreateInstancesCall struct {
	s                                             *Service
	project                                       string
	zone                                          string
	instanceGroupManager                          string
	instancegroupmanagersrecreateinstancesrequest *InstanceGroupManagersRecreateInstancesRequest
	urlParams_                                    gensupport.URLParams
	ctx_                                          context.Context
	header_                                       http.Header
}

// RecreateInstances: Recreates the specified instances. The instances
// are deleted, then recreated using the instance group manager's
// current instance template.
func (r *InstanceGroupManagersService) RecreateInstances(project string, zone string, instanceGroupManager string, instancegroupmanagersrecreateinstancesrequest *InstanceGroupManagersRecreateInstancesRequest) *InstanceGroupManagersRecreateInstancesCall {
	c := &InstanceGroupManagersRecreateInstancesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagersrecreateinstancesrequest = instancegroupmanagersrecreateinstancesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersRecreateInstancesCall) Fields(s ...googleapi.Field) *InstanceGroupManagersRecreateInstancesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersRecreateInstancesCall) Context(ctx context.Context) *InstanceGroupManagersRecreateInstancesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersRecreateInstancesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersRecreateInstancesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagersrecreateinstancesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/recreateInstances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.recreateInstances" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersRecreateInstancesCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Recreates the specified instances. The instances are deleted, then recreated using the instance group manager's current instance template.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.recreateInstances",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/recreateInstances",
	//   "request": {
	//     "$ref": "InstanceGroupManagersRecreateInstancesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.resize":

type InstanceGroupManagersResizeCall struct {
	s                    *Service
	project              string
	zone                 string
	instanceGroupManager string
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Resize: Resizes the managed instance group up or down. If resized up,
// new instances are created using the current instance template. If
// resized down, instances are removed in the order outlined in Resizing
// a managed instance group.
func (r *InstanceGroupManagersService) Resize(project string, zone string, instanceGroupManager string, size int64) *InstanceGroupManagersResizeCall {
	c := &InstanceGroupManagersResizeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.urlParams_.Set("size", fmt.Sprint(size))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersResizeCall) Fields(s ...googleapi.Field) *InstanceGroupManagersResizeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersResizeCall) Context(ctx context.Context) *InstanceGroupManagersResizeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersResizeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersResizeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resize")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.resize" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersResizeCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Resizes the managed instance group up or down. If resized up, new instances are created using the current instance template. If resized down, instances are removed in the order outlined in Resizing a managed instance group.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.resize",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager",
	//     "size"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "size": {
	//       "description": "Number of instances that should exist in this Instance Group Manager.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resize",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.setInstanceTemplate":

type InstanceGroupManagersSetInstanceTemplateCall struct {
	s                                               *Service
	project                                         string
	zone                                            string
	instanceGroupManager                            string
	instancegroupmanagerssetinstancetemplaterequest *InstanceGroupManagersSetInstanceTemplateRequest
	urlParams_                                      gensupport.URLParams
	ctx_                                            context.Context
	header_                                         http.Header
}

// SetInstanceTemplate: Sets the instance template to use when creating
// new instances in this group. Existing instances are not affected.
func (r *InstanceGroupManagersService) SetInstanceTemplate(project string, zone string, instanceGroupManager string, instancegroupmanagerssetinstancetemplaterequest *InstanceGroupManagersSetInstanceTemplateRequest) *InstanceGroupManagersSetInstanceTemplateCall {
	c := &InstanceGroupManagersSetInstanceTemplateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagerssetinstancetemplaterequest = instancegroupmanagerssetinstancetemplaterequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersSetInstanceTemplateCall) Fields(s ...googleapi.Field) *InstanceGroupManagersSetInstanceTemplateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersSetInstanceTemplateCall) Context(ctx context.Context) *InstanceGroupManagersSetInstanceTemplateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersSetInstanceTemplateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersSetInstanceTemplateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagerssetinstancetemplaterequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.setInstanceTemplate" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersSetInstanceTemplateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Sets the instance template to use when creating new instances in this group. Existing instances are not affected.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.setInstanceTemplate",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setInstanceTemplate",
	//   "request": {
	//     "$ref": "InstanceGroupManagersSetInstanceTemplateRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.instanceGroupManagers.setTargetPools":

type InstanceGroupManagersSetTargetPoolsCall struct {
	s                                          *Service
	project                                    string
	zone                                       string
	instanceGroupManager                       string
	instancegroupmanagerssettargetpoolsrequest *InstanceGroupManagersSetTargetPoolsRequest
	urlParams_                                 gensupport.URLParams
	ctx_                                       context.Context
	header_                                    http.Header
}

// SetTargetPools: Modifies the target pools to which all new instances
// in this group are assigned. Existing instances in the group are not
// affected.
func (r *InstanceGroupManagersService) SetTargetPools(project string, zone string, instanceGroupManager string, instancegroupmanagerssettargetpoolsrequest *InstanceGroupManagersSetTargetPoolsRequest) *InstanceGroupManagersSetTargetPoolsCall {
	c := &InstanceGroupManagersSetTargetPoolsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.instanceGroupManager = instanceGroupManager
	c.instancegroupmanagerssettargetpoolsrequest = instancegroupmanagerssettargetpoolsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstanceGroupManagersSetTargetPoolsCall) Fields(s ...googleapi.Field) *InstanceGroupManagersSetTargetPoolsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstanceGroupManagersSetTargetPoolsCall) Context(ctx context.Context) *InstanceGroupManagersSetTargetPoolsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstanceGroupManagersSetTargetPoolsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstanceGroupManagersSetTargetPoolsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancegroupmanagerssettargetpoolsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setTargetPools")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":              c.project,
		"zone":                 c.zone,
		"instanceGroupManager": c.instanceGroupManager,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.instanceGroupManagers.setTargetPools" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstanceGroupManagersSetTargetPoolsCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Modifies the target pools to which all new instances in this group are assigned. Existing instances in the group are not affected.",
	//   "httpMethod": "POST",
	//   "id": "replicapool.instanceGroupManagers.setTargetPools",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "instanceGroupManager"
	//   ],
	//   "parameters": {
	//     "instanceGroupManager": {
	//       "description": "The name of the instance group manager.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The Google Developers Console project name.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the zone in which the instance group manager resides.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/setTargetPools",
	//   "request": {
	//     "$ref": "InstanceGroupManagersSetTargetPoolsRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.zoneOperations.get":

type ZoneOperationsGetCall struct {
	s            *Service
	project      string
	zone         string
	operation    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the specified zone-specific operation resource.
func (r *ZoneOperationsService) Get(project string, zone string, operation string) *ZoneOperationsGetCall {
	c := &ZoneOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsGetCall) Fields(s ...googleapi.Field) *ZoneOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ZoneOperationsGetCall) IfNoneMatch(entityTag string) *ZoneOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneOperationsGetCall) Context(ctx context.Context) *ZoneOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"zone":      c.zone,
		"operation": c.operation,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.zoneOperations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ZoneOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Retrieves the specified zone-specific operation resource.",
	//   "httpMethod": "GET",
	//   "id": "replicapool.zoneOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "Name of the operation resource to return.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "replicapool.zoneOperations.list":

type ZoneOperationsListCall struct {
	s            *Service
	project      string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the list of operation resources contained within the
// specified zone.
func (r *ZoneOperationsService) List(project string, zone string) *ZoneOperationsListCall {
	c := &ZoneOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter": Filter expression for
// filtering listed resources.
func (c *ZoneOperationsListCall) Filter(filter string) *ZoneOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Maximum value is 500 and default value is
// 500.
func (c *ZoneOperationsListCall) MaxResults(maxResults int64) *ZoneOperationsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Tag returned by a
// previous list request truncated by maxResults. Used to continue a
// previous list request.
func (c *ZoneOperationsListCall) PageToken(pageToken string) *ZoneOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsListCall) Fields(s ...googleapi.Field) *ZoneOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ZoneOperationsListCall) IfNoneMatch(entityTag string) *ZoneOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneOperationsListCall) Context(ctx context.Context) *ZoneOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ZoneOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ZoneOperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "replicapool.zoneOperations.list" call.
// Exactly one of *OperationList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *OperationList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ZoneOperationsListCall) Do(opts ...googleapi.CallOption) (*OperationList, error) {
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
	//   "description": "Retrieves the list of operation resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "replicapool.zoneOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional. Filter expression for filtering listed resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "description": "Optional. Maximum count of results to be returned. Maximum value is 500 and default value is 500.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. Tag returned by a previous list request truncated by maxResults. Used to continue a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Name of the project scoping this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Name of the zone scoping this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations",
	//   "response": {
	//     "$ref": "OperationList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ZoneOperationsListCall) Pages(ctx context.Context, f func(*OperationList) error) error {
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
