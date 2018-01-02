// Package servicecontrol provides access to the Google Service Control API.
//
// See https://cloud.google.com/service-control/
//
// Usage example:
//
//   import "google.golang.org/api/servicecontrol/v1"
//   ...
//   servicecontrolService, err := servicecontrol.New(oauthHttpClient)
package servicecontrol // import "google.golang.org/api/servicecontrol/v1"

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

const apiId = "servicecontrol:v1"
const apiName = "servicecontrol"
const apiVersion = "v1"
const basePath = "https://servicecontrol.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Manage your Google Service Control data
	ServicecontrolScope = "https://www.googleapis.com/auth/servicecontrol"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Services = NewServicesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Services *ServicesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewServicesService(s *Service) *ServicesService {
	rs := &ServicesService{s: s}
	return rs
}

type ServicesService struct {
	s *Service
}

// AllocateQuotaRequest: Request message for the AllocateQuota method.
type AllocateQuotaRequest struct {
	// AllocateOperation: Operation that describes the quota allocation.
	AllocateOperation *QuotaOperation `json:"allocateOperation,omitempty"`

	// AllocationMode: Allocation mode for this operation.
	// Deprecated: use QuotaMode inside the QuotaOperation.
	//
	// Possible values:
	//   "UNSPECIFIED"
	//   "NORMAL" - Allocates quota for the amount specified in the service
	// configuration or
	// specified using the quota_metrics. If the amount is higher than
	// the
	// available quota, allocation error will be returned and no quota will
	// be
	// allocated.
	//   "BEST_EFFORT" - Allocates quota for the amount specified in the
	// service configuration or
	// specified using the quota_metrics. If the amount is higher than
	// the
	// available quota, request does not fail but all available quota will
	// be
	// allocated.
	//   "CHECK_ONLY" - Only checks if there is enough quota available and
	// does not change the
	// available quota. No lock is placed on the available quota either.
	AllocationMode string `json:"allocationMode,omitempty"`

	// ServiceConfigId: Specifies which version of service configuration
	// should be used to process
	// the request. If unspecified or no matching version can be found, the
	// latest
	// one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllocateOperation")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllocateOperation") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AllocateQuotaRequest) MarshalJSON() ([]byte, error) {
	type noMethod AllocateQuotaRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AllocateQuotaResponse: Response message for the AllocateQuota method.
type AllocateQuotaResponse struct {
	// AllocateErrors: Indicates the decision of the allocate.
	AllocateErrors []*QuotaError `json:"allocateErrors,omitempty"`

	// OperationId: The same operation_id value used in the
	// AllocateQuotaRequest. Used for
	// logging and diagnostics purposes.
	OperationId string `json:"operationId,omitempty"`

	// QuotaMetrics: Quota metrics to indicate the result of allocation.
	// Depending on the
	// request, one or more of the following metrics will be included:
	//
	// 1. For rate quota, per quota group or per quota metric incremental
	// usage
	// will be specified using the following delta metric:
	//   "serviceruntime.googleapis.com/api/consumer/quota_used_count"
	//
	// 2. For allocation quota, per quota metric total usage will be
	// specified
	// using the following gauge metric:
	//
	// "serviceruntime.googleapis.com/allocation/consumer/quota_used_count"
	//
	//
	// 3. For both rate quota and allocation quota, the quota limit
	// reached
	// condition will be specified using the following boolean metric:
	//   "serviceruntime.googleapis.com/quota/exceeded"
	//
	// 4. For allocation quota, value for each quota limit associated
	// with
	// the metrics will be specified using the following gauge metric:
	//   "serviceruntime.googleapis.com/quota/limit"
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// ServiceConfigId: ID of the actual config used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AllocateErrors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllocateErrors") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AllocateQuotaResponse) MarshalJSON() ([]byte, error) {
	type noMethod AllocateQuotaResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuditLog: Common audit log format for Google Cloud Platform API
// operations.
//
//
type AuditLog struct {
	// AuthenticationInfo: Authentication information.
	AuthenticationInfo *AuthenticationInfo `json:"authenticationInfo,omitempty"`

	// AuthorizationInfo: Authorization information. If there are
	// multiple
	// resources or permissions involved, then there is
	// one AuthorizationInfo element for each {resource, permission} tuple.
	AuthorizationInfo []*AuthorizationInfo `json:"authorizationInfo,omitempty"`

	// MethodName: The name of the service method or operation.
	// For API calls, this should be the name of the API method.
	// For example,
	//
	//     "google.datastore.v1.Datastore.RunQuery"
	//     "google.logging.v1.LoggingService.DeleteLog"
	MethodName string `json:"methodName,omitempty"`

	// NumResponseItems: The number of items returned from a List or Query
	// API method,
	// if applicable.
	NumResponseItems int64 `json:"numResponseItems,omitempty,string"`

	// Request: The operation request. This may not include all request
	// parameters,
	// such as those that are too large, privacy-sensitive, or
	// duplicated
	// elsewhere in the log record.
	// It should never include user-generated data, such as file
	// contents.
	// When the JSON object represented here has a proto equivalent, the
	// proto
	// name will be indicated in the `@type` property.
	Request googleapi.RawMessage `json:"request,omitempty"`

	// RequestMetadata: Metadata about the operation.
	RequestMetadata *RequestMetadata `json:"requestMetadata,omitempty"`

	// ResourceName: The resource or collection that is the target of the
	// operation.
	// The name is a scheme-less URI, not including the API service
	// name.
	// For example:
	//
	//     "shelves/SHELF_ID/books"
	//     "shelves/SHELF_ID/books/BOOK_ID"
	ResourceName string `json:"resourceName,omitempty"`

	// Response: The operation response. This may not include all response
	// elements,
	// such as those that are too large, privacy-sensitive, or
	// duplicated
	// elsewhere in the log record.
	// It should never include user-generated data, such as file
	// contents.
	// When the JSON object represented here has a proto equivalent, the
	// proto
	// name will be indicated in the `@type` property.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServiceData: Other service-specific data about the request, response,
	// and other
	// activities.
	ServiceData googleapi.RawMessage `json:"serviceData,omitempty"`

	// ServiceName: The name of the API service performing the operation.
	// For example,
	// "datastore.googleapis.com".
	ServiceName string `json:"serviceName,omitempty"`

	// Status: The status of the overall operation.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuthenticationInfo")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuthenticationInfo") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuditLog) MarshalJSON() ([]byte, error) {
	type noMethod AuditLog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthenticationInfo: Authentication information for the operation.
type AuthenticationInfo struct {
	// AuthoritySelector: The authority selector specified by the requestor,
	// if any.
	// It is not guaranteed that the principal was allowed to use this
	// authority.
	AuthoritySelector string `json:"authoritySelector,omitempty"`

	// PrincipalEmail: The email address of the authenticated user (or
	// service account on behalf
	// of third party principal) making the request. For privacy reasons,
	// the
	// principal email address is redacted for all read-only operations that
	// fail
	// with a "permission denied" error.
	PrincipalEmail string `json:"principalEmail,omitempty"`

	// ThirdPartyPrincipal: The third party identification (if any) of the
	// authenticated user making
	// the request.
	// When the JSON object represented here has a proto equivalent, the
	// proto
	// name will be indicated in the `@type` property.
	ThirdPartyPrincipal googleapi.RawMessage `json:"thirdPartyPrincipal,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuthoritySelector")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuthoritySelector") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AuthenticationInfo) MarshalJSON() ([]byte, error) {
	type noMethod AuthenticationInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AuthorizationInfo: Authorization information for the operation.
type AuthorizationInfo struct {
	// Granted: Whether or not authorization for `resource` and
	// `permission`
	// was granted.
	Granted bool `json:"granted,omitempty"`

	// Permission: The required IAM permission.
	Permission string `json:"permission,omitempty"`

	// Resource: The resource being accessed, as a REST-style string. For
	// example:
	//
	//     bigquery.googleapis.com/projects/PROJECTID/datasets/DATASETID
	Resource string `json:"resource,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Granted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Granted") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AuthorizationInfo) MarshalJSON() ([]byte, error) {
	type noMethod AuthorizationInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CheckError: Defines the errors to be returned
// in
// google.api.servicecontrol.v1.CheckResponse.check_errors.
type CheckError struct {
	// Code: The error code.
	//
	// Possible values:
	//   "ERROR_CODE_UNSPECIFIED" - This is never used in `CheckResponse`.
	//   "NOT_FOUND" - The consumer's project id was not found.
	// Same as google.rpc.Code.NOT_FOUND.
	//   "PERMISSION_DENIED" - The consumer doesn't have access to the
	// specified resource.
	// Same as google.rpc.Code.PERMISSION_DENIED.
	//   "RESOURCE_EXHAUSTED" - Quota check failed. Same as
	// google.rpc.Code.RESOURCE_EXHAUSTED.
	//   "BUDGET_EXCEEDED" - Budget check failed.
	//   "DENIAL_OF_SERVICE_DETECTED" - The consumer's request has been
	// flagged as a DoS attack.
	//   "LOAD_SHEDDING" - The consumer's request should be rejected in
	// order to protect the service
	// from being overloaded.
	//   "ABUSER_DETECTED" - The consumer has been flagged as an abuser.
	//   "SERVICE_NOT_ACTIVATED" - The consumer hasn't activated the
	// service.
	//   "VISIBILITY_DENIED" - The consumer cannot access the service due to
	// visibility configuration.
	//   "BILLING_DISABLED" - The consumer cannot access the service because
	// billing is disabled.
	//   "PROJECT_DELETED" - The consumer's project has been marked as
	// deleted (soft deletion).
	//   "PROJECT_INVALID" - The consumer's project number or id does not
	// represent a valid project.
	//   "IP_ADDRESS_BLOCKED" - The IP address of the consumer is invalid
	// for the specific consumer
	// project.
	//   "REFERER_BLOCKED" - The referer address of the consumer request is
	// invalid for the specific
	// consumer project.
	//   "CLIENT_APP_BLOCKED" - The client application of the consumer
	// request is invalid for the
	// specific consumer project.
	//   "API_TARGET_BLOCKED" - The API targeted by this request is invalid
	// for the specified consumer
	// project.
	//   "API_KEY_INVALID" - The consumer's API key is invalid.
	//   "API_KEY_EXPIRED" - The consumer's API Key has expired.
	//   "API_KEY_NOT_FOUND" - The consumer's API Key was not found in
	// config record.
	//   "SPATULA_HEADER_INVALID" - The consumer's spatula header is
	// invalid.
	//   "LOAS_ROLE_INVALID" - The consumer's LOAS role is invalid.
	//   "NO_LOAS_PROJECT" - The consumer's LOAS role has no associated
	// project.
	//   "LOAS_PROJECT_DISABLED" - The consumer's LOAS project is not
	// `ACTIVE` in LoquatV2.
	//   "SECURITY_POLICY_VIOLATED" - Request is not allowed as per security
	// policies defined in Org Policy.
	//   "NAMESPACE_LOOKUP_UNAVAILABLE" - The backend server for looking up
	// project id/number is unavailable.
	//   "SERVICE_STATUS_UNAVAILABLE" - The backend server for checking
	// service status is unavailable.
	//   "BILLING_STATUS_UNAVAILABLE" - The backend server for checking
	// billing status is unavailable.
	//   "QUOTA_CHECK_UNAVAILABLE" - The backend server for checking quota
	// limits is unavailable.
	//   "LOAS_PROJECT_LOOKUP_UNAVAILABLE" - The Spanner for looking up LOAS
	// project is unavailable.
	//   "CLOUD_RESOURCE_MANAGER_BACKEND_UNAVAILABLE" - Cloud Resource
	// Manager backend server is unavailable.
	//   "SECURITY_POLICY_BACKEND_UNAVAILABLE" - Backend server for
	// evaluating security policy is unavailable.
	Code string `json:"code,omitempty"`

	// Detail: Free-form text providing details on the error cause of the
	// error.
	Detail string `json:"detail,omitempty"`

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

func (s *CheckError) MarshalJSON() ([]byte, error) {
	type noMethod CheckError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CheckInfo struct {
	// ConsumerInfo: Consumer info of this check.
	ConsumerInfo *ConsumerInfo `json:"consumerInfo,omitempty"`

	// UnusedArguments: A list of fields and label keys that are ignored by
	// the server.
	// The client doesn't need to send them for following requests to
	// improve
	// performance and allow better aggregation.
	UnusedArguments []string `json:"unusedArguments,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConsumerInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerInfo") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CheckInfo) MarshalJSON() ([]byte, error) {
	type noMethod CheckInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CheckRequest: Request message for the Check method.
type CheckRequest struct {
	// Operation: The operation to be checked.
	Operation *Operation `json:"operation,omitempty"`

	// RequestProjectSettings: Requests the project settings to be returned
	// as part of the check response.
	RequestProjectSettings bool `json:"requestProjectSettings,omitempty"`

	// ServiceConfigId: Specifies which version of service configuration
	// should be used to process
	// the request.
	//
	// If unspecified or no matching version can be found, the
	// latest one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// SkipActivationCheck: Indicates if service activation check should be
	// skipped for this request.
	// Default behavior is to perform the check and apply relevant quota.
	SkipActivationCheck bool `json:"skipActivationCheck,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Operation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Operation") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CheckRequest) MarshalJSON() ([]byte, error) {
	type noMethod CheckRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CheckResponse: Response message for the Check method.
type CheckResponse struct {
	// CheckErrors: Indicate the decision of the check.
	//
	// If no check errors are present, the service should process the
	// operation.
	// Otherwise the service should use the list of errors to determine
	// the
	// appropriate action.
	CheckErrors []*CheckError `json:"checkErrors,omitempty"`

	// CheckInfo: Feedback data returned from the server during processing a
	// Check request.
	CheckInfo *CheckInfo `json:"checkInfo,omitempty"`

	// OperationId: The same operation_id value used in the
	// CheckRequest.
	// Used for logging and diagnostics purposes.
	OperationId string `json:"operationId,omitempty"`

	// QuotaInfo: Quota information for the check request associated with
	// this response.
	//
	QuotaInfo *QuotaInfo `json:"quotaInfo,omitempty"`

	// ServiceConfigId: The actual config id used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CheckErrors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CheckErrors") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CheckResponse) MarshalJSON() ([]byte, error) {
	type noMethod CheckResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConsumerInfo: `ConsumerInfo` provides information about the consumer
// project.
type ConsumerInfo struct {
	// ProjectNumber: The Google cloud project number, e.g. 1234567890. A
	// value of 0 indicates
	// no project number is found.
	ProjectNumber int64 `json:"projectNumber,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "ProjectNumber") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProjectNumber") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ConsumerInfo) MarshalJSON() ([]byte, error) {
	type noMethod ConsumerInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Distribution: Distribution represents a frequency distribution of
// double-valued sample
// points. It contains the size of the population of sample points
// plus
// additional optional information:
//
//   - the arithmetic mean of the samples
//   - the minimum and maximum of the samples
//   - the sum-squared-deviation of the samples, used to compute
// variance
//   - a histogram of the values of the sample points
type Distribution struct {
	// BucketCounts: The number of samples in each histogram bucket.
	// `bucket_counts` are
	// optional. If present, they must sum to the `count` value.
	//
	// The buckets are defined below in `bucket_option`. There are N
	// buckets.
	// `bucket_counts[0]` is the number of samples in the underflow
	// bucket.
	// `bucket_counts[1]` to `bucket_counts[N-1]` are the numbers of
	// samples
	// in each of the finite buckets. And `bucket_counts[N] is the number
	// of samples in the overflow bucket. See the comments of
	// `bucket_option`
	// below for more details.
	//
	// Any suffix of trailing zeros may be omitted.
	BucketCounts googleapi.Int64s `json:"bucketCounts,omitempty"`

	// Count: The total number of samples in the distribution. Must be >= 0.
	Count int64 `json:"count,omitempty,string"`

	// ExplicitBuckets: Buckets with arbitrary user-provided width.
	ExplicitBuckets *ExplicitBuckets `json:"explicitBuckets,omitempty"`

	// ExponentialBuckets: Buckets with exponentially growing width.
	ExponentialBuckets *ExponentialBuckets `json:"exponentialBuckets,omitempty"`

	// LinearBuckets: Buckets with constant width.
	LinearBuckets *LinearBuckets `json:"linearBuckets,omitempty"`

	// Maximum: The maximum of the population of values. Ignored if `count`
	// is zero.
	Maximum float64 `json:"maximum,omitempty"`

	// Mean: The arithmetic mean of the samples in the distribution. If
	// `count` is
	// zero then this field must be zero.
	Mean float64 `json:"mean,omitempty"`

	// Minimum: The minimum of the population of values. Ignored if `count`
	// is zero.
	Minimum float64 `json:"minimum,omitempty"`

	// SumOfSquaredDeviation: The sum of squared deviations from the mean:
	//   Sum[i=1..count]((x_i - mean)^2)
	// where each x_i is a sample values. If `count` is zero then this
	// field
	// must be zero, otherwise validation of the request fails.
	SumOfSquaredDeviation float64 `json:"sumOfSquaredDeviation,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BucketCounts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketCounts") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Distribution) MarshalJSON() ([]byte, error) {
	type noMethod Distribution
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Distribution) UnmarshalJSON(data []byte) error {
	type noMethod Distribution
	var s1 struct {
		Maximum               gensupport.JSONFloat64 `json:"maximum"`
		Mean                  gensupport.JSONFloat64 `json:"mean"`
		Minimum               gensupport.JSONFloat64 `json:"minimum"`
		SumOfSquaredDeviation gensupport.JSONFloat64 `json:"sumOfSquaredDeviation"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Maximum = float64(s1.Maximum)
	s.Mean = float64(s1.Mean)
	s.Minimum = float64(s1.Minimum)
	s.SumOfSquaredDeviation = float64(s1.SumOfSquaredDeviation)
	return nil
}

type EndReconciliationRequest struct {
	// ReconciliationOperation: Operation that describes the quota
	// reconciliation.
	ReconciliationOperation *QuotaOperation `json:"reconciliationOperation,omitempty"`

	// ServiceConfigId: Specifies which version of service configuration
	// should be used to process
	// the request. If unspecified or no matching version can be found, the
	// latest
	// one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ReconciliationOperation") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReconciliationOperation")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *EndReconciliationRequest) MarshalJSON() ([]byte, error) {
	type noMethod EndReconciliationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type EndReconciliationResponse struct {
	// OperationId: The same operation_id value used in the
	// EndReconciliationRequest. Used for
	// logging and diagnostics purposes.
	OperationId string `json:"operationId,omitempty"`

	// QuotaMetrics: Metric values as tracked by One Platform before the
	// adjustment was made.
	// The following metrics will be included:
	//
	// 1. Per quota metric total usage will be specified using the following
	// gauge
	// metric:
	//
	// "serviceruntime.googleapis.com/allocation/consumer/quota_used_count"
	//
	//
	// 2. Value for each quota limit associated with the metrics will be
	// specified
	// using the following gauge metric:
	//   "serviceruntime.googleapis.com/quota/limit"
	//
	// 3. Delta value of the usage after the reconciliation for limits
	// associated
	// with the metrics will be specified using the following metric:
	//   "serviceruntime.googleapis.com/allocation/reconciliation_delta"
	// The delta value is defined as:
	//   new_usage_from_client - existing_value_in_spanner.
	// This metric is not defined in serviceruntime.yaml or in Cloud
	// Monarch.
	// This metric is meant for callers' use only. Since this metric is
	// not
	// defined in the monitoring backend, reporting on this metric will
	// result in
	// an error.
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// ReconciliationErrors: Indicates the decision of the reconciliation
	// end.
	ReconciliationErrors []*QuotaError `json:"reconciliationErrors,omitempty"`

	// ServiceConfigId: ID of the actual config used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EndReconciliationResponse) MarshalJSON() ([]byte, error) {
	type noMethod EndReconciliationResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExplicitBuckets: Describing buckets with arbitrary user-provided
// width.
type ExplicitBuckets struct {
	// Bounds: 'bound' is a list of strictly increasing boundaries
	// between
	// buckets. Note that a list of length N-1 defines N buckets because
	// of fenceposting. See comments on `bucket_options` for details.
	//
	// The i'th finite bucket covers the interval
	//   [bound[i-1], bound[i])
	// where i ranges from 1 to bound_size() - 1. Note that there are
	// no
	// finite buckets at all if 'bound' only contains a single element;
	// in
	// that special case the single bound defines the boundary between
	// the
	// underflow and overflow buckets.
	//
	// bucket number                   lower bound    upper bound
	//  i == 0 (underflow)              -inf           bound[i]
	//  0 < i < bound_size()            bound[i-1]     bound[i]
	//  i == bound_size() (overflow)    bound[i-1]     +inf
	Bounds []float64 `json:"bounds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bounds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bounds") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExplicitBuckets) MarshalJSON() ([]byte, error) {
	type noMethod ExplicitBuckets
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExponentialBuckets: Describing buckets with exponentially growing
// width.
type ExponentialBuckets struct {
	// GrowthFactor: The i'th exponential bucket covers the interval
	//   [scale * growth_factor^(i-1), scale * growth_factor^i)
	// where i ranges from 1 to num_finite_buckets inclusive.
	// Must be larger than 1.0.
	GrowthFactor float64 `json:"growthFactor,omitempty"`

	// NumFiniteBuckets: The number of finite buckets. With the underflow
	// and overflow buckets,
	// the total number of buckets is `num_finite_buckets` + 2.
	// See comments on `bucket_options` for details.
	NumFiniteBuckets int64 `json:"numFiniteBuckets,omitempty"`

	// Scale: The i'th exponential bucket covers the interval
	//   [scale * growth_factor^(i-1), scale * growth_factor^i)
	// where i ranges from 1 to num_finite_buckets inclusive.
	// Must be > 0.
	Scale float64 `json:"scale,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GrowthFactor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GrowthFactor") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExponentialBuckets) MarshalJSON() ([]byte, error) {
	type noMethod ExponentialBuckets
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ExponentialBuckets) UnmarshalJSON(data []byte) error {
	type noMethod ExponentialBuckets
	var s1 struct {
		GrowthFactor gensupport.JSONFloat64 `json:"growthFactor"`
		Scale        gensupport.JSONFloat64 `json:"scale"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.GrowthFactor = float64(s1.GrowthFactor)
	s.Scale = float64(s1.Scale)
	return nil
}

// LinearBuckets: Describing buckets with constant width.
type LinearBuckets struct {
	// NumFiniteBuckets: The number of finite buckets. With the underflow
	// and overflow buckets,
	// the total number of buckets is `num_finite_buckets` + 2.
	// See comments on `bucket_options` for details.
	NumFiniteBuckets int64 `json:"numFiniteBuckets,omitempty"`

	// Offset: The i'th linear bucket covers the interval
	//   [offset + (i-1) * width, offset + i * width)
	// where i ranges from 1 to num_finite_buckets, inclusive.
	Offset float64 `json:"offset,omitempty"`

	// Width: The i'th linear bucket covers the interval
	//   [offset + (i-1) * width, offset + i * width)
	// where i ranges from 1 to num_finite_buckets, inclusive.
	// Must be strictly positive.
	Width float64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NumFiniteBuckets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NumFiniteBuckets") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *LinearBuckets) MarshalJSON() ([]byte, error) {
	type noMethod LinearBuckets
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *LinearBuckets) UnmarshalJSON(data []byte) error {
	type noMethod LinearBuckets
	var s1 struct {
		Offset gensupport.JSONFloat64 `json:"offset"`
		Width  gensupport.JSONFloat64 `json:"width"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Offset = float64(s1.Offset)
	s.Width = float64(s1.Width)
	return nil
}

// LogEntry: An individual log entry.
type LogEntry struct {
	// InsertId: A unique ID for the log entry used for deduplication. If
	// omitted,
	// the implementation will generate one based on operation_id.
	InsertId string `json:"insertId,omitempty"`

	// Labels: A set of user-defined (key, value) data that provides
	// additional
	// information about the log entry.
	Labels map[string]string `json:"labels,omitempty"`

	// Name: Required. The log to which this log entry belongs. Examples:
	// "syslog",
	// "book_log".
	Name string `json:"name,omitempty"`

	// ProtoPayload: The log entry payload, represented as a protocol buffer
	// that is
	// expressed as a JSON object. The only accepted type currently
	// is
	// AuditLog.
	ProtoPayload googleapi.RawMessage `json:"protoPayload,omitempty"`

	// Severity: The severity of the log entry. The default value
	// is
	// `LogSeverity.DEFAULT`.
	//
	// Possible values:
	//   "DEFAULT" - (0) The log entry has no assigned severity level.
	//   "DEBUG" - (100) Debug or trace information.
	//   "INFO" - (200) Routine information, such as ongoing status or
	// performance.
	//   "NOTICE" - (300) Normal but significant events, such as start up,
	// shut down, or
	// a configuration change.
	//   "WARNING" - (400) Warning events might cause problems.
	//   "ERROR" - (500) Error events are likely to cause problems.
	//   "CRITICAL" - (600) Critical events cause more severe problems or
	// outages.
	//   "ALERT" - (700) A person must take an action immediately.
	//   "EMERGENCY" - (800) One or more systems are unusable.
	Severity string `json:"severity,omitempty"`

	// StructPayload: The log entry payload, represented as a structure
	// that
	// is expressed as a JSON object.
	StructPayload googleapi.RawMessage `json:"structPayload,omitempty"`

	// TextPayload: The log entry payload, represented as a Unicode string
	// (UTF-8).
	TextPayload string `json:"textPayload,omitempty"`

	// Timestamp: The time the event described by the log entry occurred.
	// If
	// omitted, defaults to operation start time.
	Timestamp string `json:"timestamp,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InsertId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InsertId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogEntry) MarshalJSON() ([]byte, error) {
	type noMethod LogEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricValue: Represents a single metric value.
type MetricValue struct {
	// BoolValue: A boolean value.
	BoolValue *bool `json:"boolValue,omitempty"`

	// DistributionValue: A distribution value.
	DistributionValue *Distribution `json:"distributionValue,omitempty"`

	// DoubleValue: A double precision floating point value.
	DoubleValue *float64 `json:"doubleValue,omitempty"`

	// EndTime: The end of the time period over which this metric value's
	// measurement
	// applies.
	EndTime string `json:"endTime,omitempty"`

	// Int64Value: A signed 64-bit integer value.
	Int64Value *int64 `json:"int64Value,omitempty,string"`

	// Labels: The labels describing the metric value.
	// See comments on google.api.servicecontrol.v1.Operation.labels for
	// the overriding relationship.
	Labels map[string]string `json:"labels,omitempty"`

	// MoneyValue: A money value.
	MoneyValue *Money `json:"moneyValue,omitempty"`

	// StartTime: The start of the time period over which this metric
	// value's measurement
	// applies. The time period has different semantics for different
	// metric
	// types (cumulative, delta, and gauge). See the metric
	// definition
	// documentation in the service configuration for details.
	StartTime string `json:"startTime,omitempty"`

	// StringValue: A text string value.
	StringValue *string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoolValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoolValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricValue) MarshalJSON() ([]byte, error) {
	type noMethod MetricValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *MetricValue) UnmarshalJSON(data []byte) error {
	type noMethod MetricValue
	var s1 struct {
		DoubleValue *gensupport.JSONFloat64 `json:"doubleValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	if s1.DoubleValue != nil {
		s.DoubleValue = (*float64)(s1.DoubleValue)
	}
	return nil
}

// MetricValueSet: Represents a set of metric values in the same
// metric.
// Each metric value in the set should have a unique combination of
// start time,
// end time, and label values.
type MetricValueSet struct {
	// MetricName: The metric name defined in the service configuration.
	MetricName string `json:"metricName,omitempty"`

	// MetricValues: The values in this metric.
	MetricValues []*MetricValue `json:"metricValues,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MetricName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MetricName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MetricValueSet) MarshalJSON() ([]byte, error) {
	type noMethod MetricValueSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Money: Represents an amount of money with its currency type.
type Money struct {
	// CurrencyCode: The 3-letter currency code defined in ISO 4217.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// Nanos: Number of nano (10^-9) units of the amount.
	// The value must be between -999,999,999 and +999,999,999 inclusive.
	// If `units` is positive, `nanos` must be positive or zero.
	// If `units` is zero, `nanos` can be positive, zero, or negative.
	// If `units` is negative, `nanos` must be negative or zero.
	// For example $-1.75 is represented as `units`=-1 and
	// `nanos`=-750,000,000.
	Nanos int64 `json:"nanos,omitempty"`

	// Units: The whole units of the amount.
	// For example if `currencyCode` is "USD", then 1 unit is one US
	// dollar.
	Units int64 `json:"units,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CurrencyCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CurrencyCode") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Money) MarshalJSON() ([]byte, error) {
	type noMethod Money
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: Represents information regarding an operation.
type Operation struct {
	// ConsumerId: Identity of the consumer who is using the service.
	// This field should be filled in for the operations initiated by
	// a
	// consumer, but not for service-initiated operations that are
	// not related to a specific consumer.
	//
	// This can be in one of the following formats:
	//   project:<project_id>,
	//   project_number:<project_number>,
	//   api_key:<api_key>.
	ConsumerId string `json:"consumerId,omitempty"`

	// EndTime: End time of the operation.
	// Required when the operation is used in ServiceController.Report,
	// but optional when the operation is used in ServiceController.Check.
	EndTime string `json:"endTime,omitempty"`

	// Importance: DO NOT USE. This is an experimental field.
	//
	// Possible values:
	//   "LOW" - The API implementation may cache and aggregate the
	// data.
	// The data may be lost when rare and unexpected system failures occur.
	//   "HIGH" - The API implementation doesn't cache and aggregate the
	// data.
	// If the method returns successfully, it's guaranteed that the data
	// has
	// been persisted in durable storage.
	//   "DEBUG" - In addition to the behavior described in HIGH, DEBUG
	// enables
	// additional validation logic that is only useful during the
	// onboarding
	// process. This is only available to Google internal services and
	// the service must be whitelisted by chemist-dev@google.com in order
	// to use this level.
	Importance string `json:"importance,omitempty"`

	// Labels: Labels describing the operation. Only the following labels
	// are allowed:
	//
	// - Labels describing monitored resources as defined in
	//   the service configuration.
	// - Default labels of metric values. When specified, labels defined in
	// the
	//   metric value override these default.
	// - The following labels defined by Google Cloud Platform:
	//     - `cloud.googleapis.com/location` describing the location where
	// the
	//        operation happened,
	//     - `servicecontrol.googleapis.com/user_agent` describing the user
	// agent
	//        of the API request,
	//     - `servicecontrol.googleapis.com/service_agent` describing the
	// service
	//        used to handle the API request (e.g. ESP),
	//     - `servicecontrol.googleapis.com/platform` describing the
	// platform
	//        where the API is served (e.g. GAE, GCE, GKE).
	Labels map[string]string `json:"labels,omitempty"`

	// LogEntries: Represents information to be logged.
	LogEntries []*LogEntry `json:"logEntries,omitempty"`

	// MetricValueSets: Represents information about this operation. Each
	// MetricValueSet
	// corresponds to a metric defined in the service configuration.
	// The data type used in the MetricValueSet must agree with
	// the data type specified in the metric definition.
	//
	// Within a single operation, it is not allowed to have more than
	// one
	// MetricValue instances that have the same metric names and
	// identical
	// label value combinations. If a request has such duplicated
	// MetricValue
	// instances, the entire request is rejected with
	// an invalid argument error.
	MetricValueSets []*MetricValueSet `json:"metricValueSets,omitempty"`

	// OperationId: Identity of the operation. This must be unique within
	// the scope of the
	// service that generated the operation. If the service calls
	// Check() and Report() on the same operation, the two calls should
	// carry
	// the same id.
	//
	// UUID version 4 is recommended, though not required.
	// In scenarios where an operation is computed from existing
	// information
	// and an idempotent id is desirable for deduplication purpose, UUID
	// version 5
	// is recommended. See RFC 4122 for details.
	OperationId string `json:"operationId,omitempty"`

	// OperationName: Fully qualified name of the operation. Reserved for
	// future use.
	OperationName string `json:"operationName,omitempty"`

	// QuotaProperties: Represents the properties needed for quota check.
	// Applicable only if this
	// operation is for a quota check request.
	QuotaProperties *QuotaProperties `json:"quotaProperties,omitempty"`

	// ResourceContainer: The resource name of the parent of a resource in
	// the resource hierarchy.
	//
	// This can be in one of the following formats:
	//     - “projects/<project-id or project-number>”
	//     - “folders/<folder-id>”
	//     - “organizations/<organization-id>”
	ResourceContainer string `json:"resourceContainer,omitempty"`

	// ResourceContainers: DO NOT USE.
	// This field is not ready for use yet.
	ResourceContainers []string `json:"resourceContainers,omitempty"`

	// StartTime: Required. Start time of the operation.
	StartTime string `json:"startTime,omitempty"`

	// UserLabels: User defined labels for the resource that this operation
	// is associated
	// with.
	UserLabels map[string]string `json:"userLabels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConsumerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerId") to include in
	// API requests with the JSON null value. By default, fields with empty
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

type QuotaError struct {
	// Code: Error code.
	//
	// Possible values:
	//   "UNSPECIFIED" - This is never used.
	//   "RESOURCE_EXHAUSTED" - Quota allocation failed.
	// Same as google.rpc.Code.RESOURCE_EXHAUSTED.
	//   "OUT_OF_RANGE" - Quota release failed.  This error is ONLY returned
	// on a NORMAL release.
	// More formally:  if a user requests a release of 10 tokens, but only
	// 5 tokens were previously allocated, in a BEST_EFFORT release, this
	// will
	// be considered a success, 5 tokens will be released, and the result
	// will
	// be "Ok".  If this is done in NORMAL mode, no tokens will be
	// released,
	// and an OUT_OF_RANGE error will be returned.
	// Same as google.rpc.Code.OUT_OF_RANGE.
	//   "BILLING_NOT_ACTIVE" - Consumer cannot access the service because
	// the service requires active
	// billing.
	//   "PROJECT_DELETED" - Consumer's project has been marked as deleted
	// (soft deletion).
	//   "API_KEY_INVALID" - Specified API key is invalid.
	//   "API_KEY_EXPIRED" - Specified API Key has expired.
	//   "SPATULA_HEADER_INVALID" - Consumer's spatula header is invalid.
	//   "LOAS_ROLE_INVALID" - The consumer's LOAS role is invalid.
	//   "NO_LOAS_PROJECT" - The consumer's LOAS role has no associated
	// project.
	//   "PROJECT_STATUS_UNAVAILABLE" - The backend server for looking up
	// project id/number is unavailable.
	//   "SERVICE_STATUS_UNAVAILABLE" - The backend server for checking
	// service status is unavailable.
	//   "BILLING_STATUS_UNAVAILABLE" - The backend server for checking
	// billing status is unavailable.
	//   "QUOTA_SYSTEM_UNAVAILABLE" - The backend server for checking quota
	// limits is unavailable.
	Code string `json:"code,omitempty"`

	// Description: Free-form text that provides details on the cause of the
	// error.
	Description string `json:"description,omitempty"`

	// Subject: Subject to whom this error applies. See the specific enum
	// for more details
	// on this field. For example, "clientip:<ip address of client>"
	// or
	// "project:<Google developer project id>".
	Subject string `json:"subject,omitempty"`

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

func (s *QuotaError) MarshalJSON() ([]byte, error) {
	type noMethod QuotaError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QuotaInfo: Contains the quota information for a quota check response.
type QuotaInfo struct {
	// LimitExceeded: Quota Metrics that have exceeded quota limits.
	// For QuotaGroup-based quota, this is QuotaGroup.name
	// For QuotaLimit-based quota, this is QuotaLimit.name
	// See: google.api.Quota
	// Deprecated: Use quota_metrics to get per quota group limit exceeded
	// status.
	LimitExceeded []string `json:"limitExceeded,omitempty"`

	// QuotaConsumed: Map of quota group name to the actual number of tokens
	// consumed. If the
	// quota check was not successful, then this will not be populated due
	// to no
	// quota consumption.
	//
	// We are not merging this field with 'quota_metrics' field because of
	// the
	// complexity of scaling in Chemist client code base. For simplicity, we
	// will
	// keep this field for Castor (that scales quota usage) and
	// 'quota_metrics'
	// for SuperQuota (that doesn't scale quota usage).
	//
	QuotaConsumed map[string]int64 `json:"quotaConsumed,omitempty"`

	// QuotaMetrics: Quota metrics to indicate the usage. Depending on the
	// check request, one or
	// more of the following metrics will be included:
	//
	// 1. For rate quota, per quota group or per quota metric incremental
	// usage
	// will be specified using the following delta metric:
	//   "serviceruntime.googleapis.com/api/consumer/quota_used_count"
	//
	// 2. For allocation quota, per quota metric total usage will be
	// specified
	// using the following gauge metric:
	//
	// "serviceruntime.googleapis.com/allocation/consumer/quota_used_count"
	//
	//
	// 3. For both rate quota and allocation quota, the quota limit
	// reached
	// condition will be specified using the following boolean metric:
	//   "serviceruntime.googleapis.com/quota/exceeded"
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LimitExceeded") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LimitExceeded") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QuotaInfo) MarshalJSON() ([]byte, error) {
	type noMethod QuotaInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QuotaOperation: Represents information regarding a quota operation.
type QuotaOperation struct {
	// ConsumerId: Identity of the consumer for whom this quota operation is
	// being performed.
	//
	// This can be in one of the following formats:
	//   project:<project_id>,
	//   project_number:<project_number>,
	//   api_key:<api_key>.
	ConsumerId string `json:"consumerId,omitempty"`

	// Labels: Labels describing the operation.
	Labels map[string]string `json:"labels,omitempty"`

	// MethodName: Fully qualified name of the API method for which this
	// quota operation is
	// requested. This name is used for matching quota rules or metric rules
	// and
	// billing status rules defined in service configuration. This field is
	// not
	// required if the quota operation is performed on non-API
	// resources.
	//
	// Example of an RPC method name:
	//     google.example.library.v1.LibraryService.CreateShelf
	MethodName string `json:"methodName,omitempty"`

	// OperationId: Identity of the operation. This is expected to be unique
	// within the scope
	// of the service that generated the operation, and guarantees
	// idempotency in
	// case of retries.
	//
	// UUID version 4 is recommended, though not required. In scenarios
	// where an
	// operation is computed from existing information and an idempotent id
	// is
	// desirable for deduplication purpose, UUID version 5 is recommended.
	// See
	// RFC 4122 for details.
	OperationId string `json:"operationId,omitempty"`

	// QuotaMetrics: Represents information about this operation. Each
	// MetricValueSet
	// corresponds to a metric defined in the service configuration.
	// The data type used in the MetricValueSet must agree with
	// the data type specified in the metric definition.
	//
	// Within a single operation, it is not allowed to have more than
	// one
	// MetricValue instances that have the same metric names and
	// identical
	// label value combinations. If a request has such duplicated
	// MetricValue
	// instances, the entire request is rejected with
	// an invalid argument error.
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// QuotaMode: Quota mode for this operation.
	//
	// Possible values:
	//   "UNSPECIFIED"
	//   "NORMAL" - For AllocateQuota request, allocates quota for the
	// amount specified in
	// the service configuration or specified using the quota metrics. If
	// the
	// amount is higher than the available quota, allocation error will
	// be
	// returned and no quota will be allocated.
	// For ReleaseQuota request, this mode is supported only for precise
	// quota
	// limits. In this case, this operation releases quota for the
	// amount
	// specified in the service configuration or specified using the
	// quota
	// metrics. If the release can make used quota negative, release
	// error
	// will be returned and no quota will be released.
	//   "BEST_EFFORT" - For AllocateQuota request, this mode is supported
	// only for imprecise
	// quota limits. In this case, the operation allocates quota for the
	// amount
	// specified in the service configuration or specified using the
	// quota
	// metrics. If the amount is higher than the available quota, request
	// does
	// not fail but all available quota will be allocated.
	// For ReleaseQuota request, this mode is supported for both precise
	// quota
	// limits and imprecise quota limits. In this case, this operation
	// releases
	// quota for the amount specified in the service configuration or
	// specified
	// using the quota metrics. If the release can make used quota
	// negative, request does not fail but only the used quota will
	// be
	// released. After the ReleaseQuota request completes, the used
	// quota
	// will be 0, and never goes to negative.
	//   "CHECK_ONLY" - For AllocateQuota request, only checks if there is
	// enough quota
	// available and does not change the available quota. No lock is placed
	// on
	// the available quota either. Not supported for ReleaseQuota request.
	QuotaMode string `json:"quotaMode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConsumerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConsumerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QuotaOperation) MarshalJSON() ([]byte, error) {
	type noMethod QuotaOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QuotaProperties: Represents the properties needed for quota
// operations.
type QuotaProperties struct {
	// LimitByIds: LimitType IDs that should be used for checking quota. Key
	// in this map
	// should be a valid LimitType string, and the value is the ID to be
	// used. For
	// example, an entry <USER, 123> will cause all user quota limits to use
	// 123
	// as the user ID. See google/api/quota.proto for the definition of
	// LimitType.
	// CLIENT_PROJECT: Not supported.
	// USER: Value of this entry will be used for enforcing user-level
	// quota
	//       limits. If none specified, caller IP passed in the
	//       servicecontrol.googleapis.com/caller_ip label will be used
	// instead.
	//       If the server cannot resolve a value for this LimitType, an
	// error
	//       will be thrown. No validation will be performed on this
	// ID.
	// Deprecated: use servicecontrol.googleapis.com/user label to send user
	// ID.
	LimitByIds map[string]string `json:"limitByIds,omitempty"`

	// QuotaMode: Quota mode for this operation.
	//
	// Possible values:
	//   "ACQUIRE" - Decreases available quota by the cost specified for the
	// operation.
	// If cost is higher than available quota, operation fails and
	// returns
	// error.
	//   "ACQUIRE_BEST_EFFORT" - Decreases available quota by the cost
	// specified for the operation.
	// If cost is higher than available quota, operation does not fail
	// and
	// available quota goes down to zero but it returns error.
	//   "CHECK" - Does not change any available quota. Only checks if there
	// is enough
	// quota.
	// No lock is placed on the checked tokens neither.
	//   "RELEASE" - Increases available quota by the operation cost
	// specified for the
	// operation.
	QuotaMode string `json:"quotaMode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LimitByIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LimitByIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QuotaProperties) MarshalJSON() ([]byte, error) {
	type noMethod QuotaProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReleaseQuotaRequest: Request message for the ReleaseQuota method.
type ReleaseQuotaRequest struct {
	// ReleaseOperation: Operation that describes the quota release.
	ReleaseOperation *QuotaOperation `json:"releaseOperation,omitempty"`

	// ServiceConfigId: Specifies which version of service configuration
	// should be used to process
	// the request. If unspecified or no matching version can be found, the
	// latest
	// one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReleaseOperation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReleaseOperation") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReleaseQuotaRequest) MarshalJSON() ([]byte, error) {
	type noMethod ReleaseQuotaRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReleaseQuotaResponse: Response message for the ReleaseQuota method.
type ReleaseQuotaResponse struct {
	// OperationId: The same operation_id value used in the
	// ReleaseQuotaRequest. Used for
	// logging and diagnostics purposes.
	OperationId string `json:"operationId,omitempty"`

	// QuotaMetrics: Quota metrics to indicate the result of release.
	// Depending on the
	// request, one or more of the following metrics will be included:
	//
	// 1. For rate quota, per quota group or per quota metric released
	// amount
	// will be specified using the following delta metric:
	//   "serviceruntime.googleapis.com/api/consumer/quota_refund_count"
	//
	// 2. For allocation quota, per quota metric total usage will be
	// specified
	// using the following gauge metric:
	//
	// "serviceruntime.googleapis.com/allocation/consumer/quota_used_count"
	//
	//
	// 3. For allocation quota, value for each quota limit associated
	// with
	// the metrics will be specified using the following gauge metric:
	//   "serviceruntime.googleapis.com/quota/limit"
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// ReleaseErrors: Indicates the decision of the release.
	ReleaseErrors []*QuotaError `json:"releaseErrors,omitempty"`

	// ServiceConfigId: ID of the actual config used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReleaseQuotaResponse) MarshalJSON() ([]byte, error) {
	type noMethod ReleaseQuotaResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportError: Represents the processing error of one Operation in the
// request.
type ReportError struct {
	// OperationId: The Operation.operation_id value from the request.
	OperationId string `json:"operationId,omitempty"`

	// Status: Details of the error when processing the Operation.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportError) MarshalJSON() ([]byte, error) {
	type noMethod ReportError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ReportInfo struct {
	// OperationId: The Operation.operation_id value from the request.
	OperationId string `json:"operationId,omitempty"`

	// QuotaInfo: Quota usage info when processing the `Operation`.
	QuotaInfo *QuotaInfo `json:"quotaInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportInfo) MarshalJSON() ([]byte, error) {
	type noMethod ReportInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportRequest: Request message for the Report method.
type ReportRequest struct {
	// Operations: Operations to be reported.
	//
	// Typically the service should report one operation per
	// request.
	// Putting multiple operations into a single request is allowed, but
	// should
	// be used only when multiple operations are natually available at the
	// time
	// of the report.
	//
	// If multiple operations are in a single request, the total request
	// size
	// should be no larger than 1MB. See ReportResponse.report_errors
	// for
	// partial failure behavior.
	Operations []*Operation `json:"operations,omitempty"`

	// ServiceConfigId: Specifies which version of service config should be
	// used to process the
	// request.
	//
	// If unspecified or no matching version can be found, the
	// latest one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Operations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Operations") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportRequest) MarshalJSON() ([]byte, error) {
	type noMethod ReportRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReportResponse: Response message for the Report method.
type ReportResponse struct {
	// ReportErrors: Partial failures, one for each `Operation` in the
	// request that failed
	// processing. There are three possible combinations of the RPC
	// status:
	//
	// 1. The combination of a successful RPC status and an empty
	// `report_errors`
	//    list indicates a complete success where all `Operations` in the
	//    request are processed successfully.
	// 2. The combination of a successful RPC status and a non-empty
	//    `report_errors` list indicates a partial success where some
	//    `Operations` in the request succeeded. Each
	//    `Operation` that failed processing has a corresponding item
	//    in this list.
	// 3. A failed RPC status indicates a general non-deterministic
	// failure.
	//    When this happens, it's impossible to know which of the
	//    'Operations' in the request succeeded or failed.
	ReportErrors []*ReportError `json:"reportErrors,omitempty"`

	// ReportInfos: Quota usage for each quota release `Operation`
	// request.
	//
	// Fully or partially failed quota release request may or may not be
	// present
	// in `report_quota_info`. For example, a failed quota release request
	// will
	// have the current quota usage info when precise quota library returns
	// the
	// info. A deadline exceeded quota request will not have quota usage
	// info.
	//
	// If there is no quota release request, report_quota_info will be
	// empty.
	//
	ReportInfos []*ReportInfo `json:"reportInfos,omitempty"`

	// ServiceConfigId: The actual config id used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ReportErrors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReportErrors") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReportResponse) MarshalJSON() ([]byte, error) {
	type noMethod ReportResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RequestMetadata: Metadata about the request.
type RequestMetadata struct {
	// CallerIp: The IP address of the caller.
	// For caller from internet, this will be public IPv4 or IPv6
	// address.
	// For caller from GCE VM with external IP address, this will be the
	// VM's
	// external IP address. For caller from GCE VM without external IP
	// address, if
	// the VM is in the same GCP organization (or project) as the
	// accessed
	// resource, `caller_ip` will be the GCE VM's internal IPv4 address,
	// otherwise
	// it will be redacted to "gce-internal-ip".
	// See https://cloud.google.com/compute/docs/vpc/ for more information.
	CallerIp string `json:"callerIp,omitempty"`

	// CallerSuppliedUserAgent: The user agent of the caller.
	// This information is not authenticated and should be treated
	// accordingly.
	// For example:
	//
	// +   `google-api-python-client/1.4.0`:
	//     The request was made by the Google API client for Python.
	// +   `Cloud SDK Command Line Tool apitools-client/1.0 gcloud/0.9.62`:
	//     The request was made by the Google Cloud SDK CLI (gcloud).
	// +   `AppEngine-Google; (+http://code.google.com/appengine; appid:
	// s~my-project`:
	//     The request was made from the `my-project` App Engine app.
	// NOLINT
	CallerSuppliedUserAgent string `json:"callerSuppliedUserAgent,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CallerIp") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallerIp") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RequestMetadata) MarshalJSON() ([]byte, error) {
	type noMethod RequestMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type StartReconciliationRequest struct {
	// ReconciliationOperation: Operation that describes the quota
	// reconciliation.
	ReconciliationOperation *QuotaOperation `json:"reconciliationOperation,omitempty"`

	// ServiceConfigId: Specifies which version of service configuration
	// should be used to process
	// the request. If unspecified or no matching version can be found, the
	// latest
	// one will be used.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ReconciliationOperation") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReconciliationOperation")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *StartReconciliationRequest) MarshalJSON() ([]byte, error) {
	type noMethod StartReconciliationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type StartReconciliationResponse struct {
	// OperationId: The same operation_id value used in the
	// StartReconciliationRequest. Used
	// for logging and diagnostics purposes.
	OperationId string `json:"operationId,omitempty"`

	// QuotaMetrics: Metric values as tracked by One Platform before the
	// start of
	// reconciliation. The following metrics will be included:
	//
	// 1. Per quota metric total usage will be specified using the following
	// gauge
	// metric:
	//
	// "serviceruntime.googleapis.com/allocation/consumer/quota_used_count"
	//
	//
	// 2. Value for each quota limit associated with the metrics will be
	// specified
	// using the following gauge metric:
	//   "serviceruntime.googleapis.com/quota/limit"
	QuotaMetrics []*MetricValueSet `json:"quotaMetrics,omitempty"`

	// ReconciliationErrors: Indicates the decision of the reconciliation
	// start.
	ReconciliationErrors []*QuotaError `json:"reconciliationErrors,omitempty"`

	// ServiceConfigId: ID of the actual config used to process the request.
	ServiceConfigId string `json:"serviceConfigId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "OperationId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OperationId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *StartReconciliationResponse) MarshalJSON() ([]byte, error) {
	type noMethod StartReconciliationResponse
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

// method id "servicecontrol.services.allocateQuota":

type ServicesAllocateQuotaCall struct {
	s                    *Service
	serviceName          string
	allocatequotarequest *AllocateQuotaRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// AllocateQuota: Attempts to allocate quota for the specified consumer.
// It should be called
// before the operation is executed.
//
// This method requires the
// `servicemanagement.services.quota`
// permission on the specified service. For more information,
// see
// [Google Cloud IAM](https://cloud.google.com/iam).
//
// **NOTE:** the client code **must** fail-open if the server returns
// one
// of the following quota errors:
// -   `PROJECT_STATUS_UNAVAILABLE`
// -   `SERVICE_STATUS_UNAVAILABLE`
// -   `BILLING_STATUS_UNAVAILABLE`
// -   `QUOTA_SYSTEM_UNAVAILABLE`
//
// The server may inject above errors to prohibit any hard dependency
// on the quota system.
func (r *ServicesService) AllocateQuota(serviceName string, allocatequotarequest *AllocateQuotaRequest) *ServicesAllocateQuotaCall {
	c := &ServicesAllocateQuotaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.allocatequotarequest = allocatequotarequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesAllocateQuotaCall) Fields(s ...googleapi.Field) *ServicesAllocateQuotaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesAllocateQuotaCall) Context(ctx context.Context) *ServicesAllocateQuotaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesAllocateQuotaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesAllocateQuotaCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.allocatequotarequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:allocateQuota")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.allocateQuota" call.
// Exactly one of *AllocateQuotaResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AllocateQuotaResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesAllocateQuotaCall) Do(opts ...googleapi.CallOption) (*AllocateQuotaResponse, error) {
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
	ret := &AllocateQuotaResponse{
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
	//   "description": "Attempts to allocate quota for the specified consumer. It should be called\nbefore the operation is executed.\n\nThis method requires the `servicemanagement.services.quota`\npermission on the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).\n\n**NOTE:** the client code **must** fail-open if the server returns one\nof the following quota errors:\n-   `PROJECT_STATUS_UNAVAILABLE`\n-   `SERVICE_STATUS_UNAVAILABLE`\n-   `BILLING_STATUS_UNAVAILABLE`\n-   `QUOTA_SYSTEM_UNAVAILABLE`\n\nThe server may inject above errors to prohibit any hard dependency\non the quota system.",
	//   "flatPath": "v1/services/{serviceName}:allocateQuota",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.allocateQuota",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service as specified in the service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee google.api.Service for the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:allocateQuota",
	//   "request": {
	//     "$ref": "AllocateQuotaRequest"
	//   },
	//   "response": {
	//     "$ref": "AllocateQuotaResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}

// method id "servicecontrol.services.check":

type ServicesCheckCall struct {
	s            *Service
	serviceName  string
	checkrequest *CheckRequest
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Check: Checks an operation with Google Service Control to decide
// whether
// the given operation should proceed. It should be called before
// the
// operation is executed.
//
// If feasible, the client should cache the check results and reuse them
// for
// 60 seconds. In case of server errors, the client can rely on the
// cached
// results for longer time.
//
// NOTE: the CheckRequest has the size limit of 64KB.
//
// This method requires the `servicemanagement.services.check`
// permission
// on the specified service. For more information, see
// [Google Cloud IAM](https://cloud.google.com/iam).
func (r *ServicesService) Check(serviceName string, checkrequest *CheckRequest) *ServicesCheckCall {
	c := &ServicesCheckCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.checkrequest = checkrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesCheckCall) Fields(s ...googleapi.Field) *ServicesCheckCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesCheckCall) Context(ctx context.Context) *ServicesCheckCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesCheckCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesCheckCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.checkrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:check")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.check" call.
// Exactly one of *CheckResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CheckResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesCheckCall) Do(opts ...googleapi.CallOption) (*CheckResponse, error) {
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
	ret := &CheckResponse{
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
	//   "description": "Checks an operation with Google Service Control to decide whether\nthe given operation should proceed. It should be called before the\noperation is executed.\n\nIf feasible, the client should cache the check results and reuse them for\n60 seconds. In case of server errors, the client can rely on the cached\nresults for longer time.\n\nNOTE: the CheckRequest has the size limit of 64KB.\n\nThis method requires the `servicemanagement.services.check` permission\non the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).",
	//   "flatPath": "v1/services/{serviceName}:check",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.check",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The service name as specified in its service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee\n[google.api.Service](https://cloud.google.com/service-management/reference/rpc/google.api#google.api.Service)\nfor the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:check",
	//   "request": {
	//     "$ref": "CheckRequest"
	//   },
	//   "response": {
	//     "$ref": "CheckResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}

// method id "servicecontrol.services.endReconciliation":

type ServicesEndReconciliationCall struct {
	s                        *Service
	serviceName              string
	endreconciliationrequest *EndReconciliationRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// EndReconciliation: Signals the quota controller that service ends the
// ongoing usage
// reconciliation.
//
// This method requires the
// `servicemanagement.services.quota`
// permission on the specified service. For more information,
// see
// [Google Cloud IAM](https://cloud.google.com/iam).
func (r *ServicesService) EndReconciliation(serviceName string, endreconciliationrequest *EndReconciliationRequest) *ServicesEndReconciliationCall {
	c := &ServicesEndReconciliationCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.endreconciliationrequest = endreconciliationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesEndReconciliationCall) Fields(s ...googleapi.Field) *ServicesEndReconciliationCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesEndReconciliationCall) Context(ctx context.Context) *ServicesEndReconciliationCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesEndReconciliationCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesEndReconciliationCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.endreconciliationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:endReconciliation")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.endReconciliation" call.
// Exactly one of *EndReconciliationResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *EndReconciliationResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesEndReconciliationCall) Do(opts ...googleapi.CallOption) (*EndReconciliationResponse, error) {
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
	ret := &EndReconciliationResponse{
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
	//   "description": "Signals the quota controller that service ends the ongoing usage\nreconciliation.\n\nThis method requires the `servicemanagement.services.quota`\npermission on the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).",
	//   "flatPath": "v1/services/{serviceName}:endReconciliation",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.endReconciliation",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service as specified in the service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee google.api.Service for the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:endReconciliation",
	//   "request": {
	//     "$ref": "EndReconciliationRequest"
	//   },
	//   "response": {
	//     "$ref": "EndReconciliationResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}

// method id "servicecontrol.services.releaseQuota":

type ServicesReleaseQuotaCall struct {
	s                   *Service
	serviceName         string
	releasequotarequest *ReleaseQuotaRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// ReleaseQuota: Releases previously allocated quota done through
// AllocateQuota method.
//
// This method requires the
// `servicemanagement.services.quota`
// permission on the specified service. For more information,
// see
// [Google Cloud IAM](https://cloud.google.com/iam).
//
// **NOTE:** the client code **must** fail-open if the server returns
// one
// of the following quota errors:
// -   `PROJECT_STATUS_UNAVAILABLE`
// -   `SERVICE_STATUS_UNAVAILABLE`
// -   `BILLING_STATUS_UNAVAILABLE`
// -   `QUOTA_SYSTEM_UNAVAILABLE`
//
// The server may inject above errors to prohibit any hard dependency
// on the quota system.
func (r *ServicesService) ReleaseQuota(serviceName string, releasequotarequest *ReleaseQuotaRequest) *ServicesReleaseQuotaCall {
	c := &ServicesReleaseQuotaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.releasequotarequest = releasequotarequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesReleaseQuotaCall) Fields(s ...googleapi.Field) *ServicesReleaseQuotaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesReleaseQuotaCall) Context(ctx context.Context) *ServicesReleaseQuotaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesReleaseQuotaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesReleaseQuotaCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.releasequotarequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:releaseQuota")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.releaseQuota" call.
// Exactly one of *ReleaseQuotaResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ReleaseQuotaResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesReleaseQuotaCall) Do(opts ...googleapi.CallOption) (*ReleaseQuotaResponse, error) {
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
	ret := &ReleaseQuotaResponse{
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
	//   "description": "Releases previously allocated quota done through AllocateQuota method.\n\nThis method requires the `servicemanagement.services.quota`\npermission on the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).\n\n**NOTE:** the client code **must** fail-open if the server returns one\nof the following quota errors:\n-   `PROJECT_STATUS_UNAVAILABLE`\n-   `SERVICE_STATUS_UNAVAILABLE`\n-   `BILLING_STATUS_UNAVAILABLE`\n-   `QUOTA_SYSTEM_UNAVAILABLE`\n\nThe server may inject above errors to prohibit any hard dependency\non the quota system.",
	//   "flatPath": "v1/services/{serviceName}:releaseQuota",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.releaseQuota",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service as specified in the service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee google.api.Service for the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:releaseQuota",
	//   "request": {
	//     "$ref": "ReleaseQuotaRequest"
	//   },
	//   "response": {
	//     "$ref": "ReleaseQuotaResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}

// method id "servicecontrol.services.report":

type ServicesReportCall struct {
	s             *Service
	serviceName   string
	reportrequest *ReportRequest
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Report: Reports operation results to Google Service Control, such as
// logs and
// metrics. It should be called after an operation is completed.
//
// If feasible, the client should aggregate reporting data for up to
// 5
// seconds to reduce API traffic. Limiting aggregation to 5 seconds is
// to
// reduce data loss during client crashes. Clients should carefully
// choose
// the aggregation time window to avoid data loss risk more than
// 0.01%
// for business and compliance reasons.
//
// NOTE: the ReportRequest has the size limit of 1MB.
//
// This method requires the `servicemanagement.services.report`
// permission
// on the specified service. For more information, see
// [Google Cloud IAM](https://cloud.google.com/iam).
func (r *ServicesService) Report(serviceName string, reportrequest *ReportRequest) *ServicesReportCall {
	c := &ServicesReportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.reportrequest = reportrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesReportCall) Fields(s ...googleapi.Field) *ServicesReportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesReportCall) Context(ctx context.Context) *ServicesReportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesReportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesReportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reportrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:report")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.report" call.
// Exactly one of *ReportResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReportResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesReportCall) Do(opts ...googleapi.CallOption) (*ReportResponse, error) {
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
	ret := &ReportResponse{
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
	//   "description": "Reports operation results to Google Service Control, such as logs and\nmetrics. It should be called after an operation is completed.\n\nIf feasible, the client should aggregate reporting data for up to 5\nseconds to reduce API traffic. Limiting aggregation to 5 seconds is to\nreduce data loss during client crashes. Clients should carefully choose\nthe aggregation time window to avoid data loss risk more than 0.01%\nfor business and compliance reasons.\n\nNOTE: the ReportRequest has the size limit of 1MB.\n\nThis method requires the `servicemanagement.services.report` permission\non the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).",
	//   "flatPath": "v1/services/{serviceName}:report",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.report",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "The service name as specified in its service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee\n[google.api.Service](https://cloud.google.com/service-management/reference/rpc/google.api#google.api.Service)\nfor the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:report",
	//   "request": {
	//     "$ref": "ReportRequest"
	//   },
	//   "response": {
	//     "$ref": "ReportResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}

// method id "servicecontrol.services.startReconciliation":

type ServicesStartReconciliationCall struct {
	s                          *Service
	serviceName                string
	startreconciliationrequest *StartReconciliationRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// StartReconciliation: Unlike rate quota, allocation quota does not get
// refilled periodically.
// So, it is possible that the quota usage as seen by the service
// differs from
// what the One Platform considers the usage is. This is expected to
// happen
// only rarely, but over time this can accumulate. Services can
// invoke
// StartReconciliation and EndReconciliation to correct this usage
// drift, as
// described below:
// 1. Service sends StartReconciliation with a timestamp in future for
// each
//    metric that needs to be reconciled. The timestamp being in future
// allows
//    to account for in-flight AllocateQuota and ReleaseQuota requests
// for the
//    same metric.
// 2. One Platform records this timestamp and starts tracking
// subsequent
//    AllocateQuota and ReleaseQuota requests until EndReconciliation
// is
//    called.
// 3. At or after the time specified in the StartReconciliation,
// service
//    sends EndReconciliation with the usage that needs to be reconciled
// to.
// 4. One Platform adjusts its own record of usage for that metric to
// the
//    value specified in EndReconciliation by taking in to account any
//    allocation or release between StartReconciliation and
// EndReconciliation.
//
// Signals the quota controller that the service wants to perform a
// usage
// reconciliation as specified in the request.
//
// This method requires the
// `servicemanagement.services.quota`
// permission on the specified service. For more information,
// see
// [Google Cloud IAM](https://cloud.google.com/iam).
func (r *ServicesService) StartReconciliation(serviceName string, startreconciliationrequest *StartReconciliationRequest) *ServicesStartReconciliationCall {
	c := &ServicesStartReconciliationCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.serviceName = serviceName
	c.startreconciliationrequest = startreconciliationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ServicesStartReconciliationCall) Fields(s ...googleapi.Field) *ServicesStartReconciliationCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ServicesStartReconciliationCall) Context(ctx context.Context) *ServicesStartReconciliationCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ServicesStartReconciliationCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ServicesStartReconciliationCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.startreconciliationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/services/{serviceName}:startReconciliation")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"serviceName": c.serviceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "servicecontrol.services.startReconciliation" call.
// Exactly one of *StartReconciliationResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *StartReconciliationResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ServicesStartReconciliationCall) Do(opts ...googleapi.CallOption) (*StartReconciliationResponse, error) {
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
	ret := &StartReconciliationResponse{
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
	//   "description": "Unlike rate quota, allocation quota does not get refilled periodically.\nSo, it is possible that the quota usage as seen by the service differs from\nwhat the One Platform considers the usage is. This is expected to happen\nonly rarely, but over time this can accumulate. Services can invoke\nStartReconciliation and EndReconciliation to correct this usage drift, as\ndescribed below:\n1. Service sends StartReconciliation with a timestamp in future for each\n   metric that needs to be reconciled. The timestamp being in future allows\n   to account for in-flight AllocateQuota and ReleaseQuota requests for the\n   same metric.\n2. One Platform records this timestamp and starts tracking subsequent\n   AllocateQuota and ReleaseQuota requests until EndReconciliation is\n   called.\n3. At or after the time specified in the StartReconciliation, service\n   sends EndReconciliation with the usage that needs to be reconciled to.\n4. One Platform adjusts its own record of usage for that metric to the\n   value specified in EndReconciliation by taking in to account any\n   allocation or release between StartReconciliation and EndReconciliation.\n\nSignals the quota controller that the service wants to perform a usage\nreconciliation as specified in the request.\n\nThis method requires the `servicemanagement.services.quota`\npermission on the specified service. For more information, see\n[Google Cloud IAM](https://cloud.google.com/iam).",
	//   "flatPath": "v1/services/{serviceName}:startReconciliation",
	//   "httpMethod": "POST",
	//   "id": "servicecontrol.services.startReconciliation",
	//   "parameterOrder": [
	//     "serviceName"
	//   ],
	//   "parameters": {
	//     "serviceName": {
	//       "description": "Name of the service as specified in the service configuration. For example,\n`\"pubsub.googleapis.com\"`.\n\nSee google.api.Service for the definition of a service name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/services/{serviceName}:startReconciliation",
	//   "request": {
	//     "$ref": "StartReconciliationRequest"
	//   },
	//   "response": {
	//     "$ref": "StartReconciliationResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/servicecontrol"
	//   ]
	// }

}
