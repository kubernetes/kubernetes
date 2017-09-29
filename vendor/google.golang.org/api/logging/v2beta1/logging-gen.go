// Package logging provides access to the Stackdriver Logging API.
//
// See https://cloud.google.com/logging/docs/
//
// Usage example:
//
//   import "google.golang.org/api/logging/v2beta1"
//   ...
//   loggingService, err := logging.New(oauthHttpClient)
package logging

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

const apiId = "logging:v2beta1"
const apiName = "logging"
const apiVersion = "v2beta1"
const basePath = "https://logging.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// Administrate log data for your projects
	LoggingAdminScope = "https://www.googleapis.com/auth/logging.admin"

	// View log data for your projects
	LoggingReadScope = "https://www.googleapis.com/auth/logging.read"

	// Submit log data for your projects
	LoggingWriteScope = "https://www.googleapis.com/auth/logging.write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.BillingAccounts = NewBillingAccountsService(s)
	s.Entries = NewEntriesService(s)
	s.MonitoredResourceDescriptors = NewMonitoredResourceDescriptorsService(s)
	s.Organizations = NewOrganizationsService(s)
	s.Projects = NewProjectsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	BillingAccounts *BillingAccountsService

	Entries *EntriesService

	MonitoredResourceDescriptors *MonitoredResourceDescriptorsService

	Organizations *OrganizationsService

	Projects *ProjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewBillingAccountsService(s *Service) *BillingAccountsService {
	rs := &BillingAccountsService{s: s}
	rs.Logs = NewBillingAccountsLogsService(s)
	return rs
}

type BillingAccountsService struct {
	s *Service

	Logs *BillingAccountsLogsService
}

func NewBillingAccountsLogsService(s *Service) *BillingAccountsLogsService {
	rs := &BillingAccountsLogsService{s: s}
	return rs
}

type BillingAccountsLogsService struct {
	s *Service
}

func NewEntriesService(s *Service) *EntriesService {
	rs := &EntriesService{s: s}
	return rs
}

type EntriesService struct {
	s *Service
}

func NewMonitoredResourceDescriptorsService(s *Service) *MonitoredResourceDescriptorsService {
	rs := &MonitoredResourceDescriptorsService{s: s}
	return rs
}

type MonitoredResourceDescriptorsService struct {
	s *Service
}

func NewOrganizationsService(s *Service) *OrganizationsService {
	rs := &OrganizationsService{s: s}
	rs.Logs = NewOrganizationsLogsService(s)
	return rs
}

type OrganizationsService struct {
	s *Service

	Logs *OrganizationsLogsService
}

func NewOrganizationsLogsService(s *Service) *OrganizationsLogsService {
	rs := &OrganizationsLogsService{s: s}
	return rs
}

type OrganizationsLogsService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Logs = NewProjectsLogsService(s)
	rs.Metrics = NewProjectsMetricsService(s)
	rs.Sinks = NewProjectsSinksService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Logs *ProjectsLogsService

	Metrics *ProjectsMetricsService

	Sinks *ProjectsSinksService
}

func NewProjectsLogsService(s *Service) *ProjectsLogsService {
	rs := &ProjectsLogsService{s: s}
	return rs
}

type ProjectsLogsService struct {
	s *Service
}

func NewProjectsMetricsService(s *Service) *ProjectsMetricsService {
	rs := &ProjectsMetricsService{s: s}
	return rs
}

type ProjectsMetricsService struct {
	s *Service
}

func NewProjectsSinksService(s *Service) *ProjectsSinksService {
	rs := &ProjectsSinksService{s: s}
	return rs
}

type ProjectsSinksService struct {
	s *Service
}

// BucketOptions: BucketOptions describes the bucket boundaries used to
// create a histogram for the distribution. The buckets can be in a
// linear sequence, an exponential sequence, or each bucket can be
// specified explicitly. BucketOptions does not include the number of
// values in each bucket.A bucket has an inclusive lower bound and
// exclusive upper bound for the values that are counted for that
// bucket. The upper bound of a bucket must be strictly greater than the
// lower bound. The sequence of N buckets for a distribution consists of
// an underflow bucket (number 0), zero or more finite buckets (number 1
// through N - 2) and an overflow bucket (number N - 1). The buckets are
// contiguous: the lower bound of bucket i (i > 0) is the same as the
// upper bound of bucket i - 1. The buckets span the whole range of
// finite values: lower bound of the underflow bucket is -infinity and
// the upper bound of the overflow bucket is +infinity. The finite
// buckets are so-called because both bounds are finite.
type BucketOptions struct {
	// ExplicitBuckets: The explicit buckets.
	ExplicitBuckets *Explicit `json:"explicitBuckets,omitempty"`

	// ExponentialBuckets: The exponential buckets.
	ExponentialBuckets *Exponential `json:"exponentialBuckets,omitempty"`

	// LinearBuckets: The linear bucket.
	LinearBuckets *Linear `json:"linearBuckets,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExplicitBuckets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExplicitBuckets") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BucketOptions) MarshalJSON() ([]byte, error) {
	type noMethod BucketOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance:
// service Foo {
//   rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
// }
// The JSON representation for Empty is empty JSON object {}.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// Explicit: Specifies a set of buckets with arbitrary widths.There are
// size(bounds) + 1 (= N) buckets. Bucket i has the following
// boundaries:Upper bound (0 <= i < N-1): boundsi  Lower bound (1 <= i <
// N); boundsi - 1The bounds field must contain at least one element. If
// bounds has only one element, then there are no finite buckets, and
// that single element is the common boundary of the overflow and
// underflow buckets.
type Explicit struct {
	// Bounds: The values must be monotonically increasing.
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

func (s *Explicit) MarshalJSON() ([]byte, error) {
	type noMethod Explicit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Exponential: Specifies an exponential sequence of buckets that have a
// width that is proportional to the value of the lower bound. Each
// bucket represents a constant relative uncertainty on a specific value
// in the bucket.There are num_finite_buckets + 2 (= N) buckets. Bucket
// i has the following boundaries:Upper bound (0 <= i < N-1): scale *
// (growth_factor ^ i).  Lower bound (1 <= i < N): scale *
// (growth_factor ^ (i - 1)).
type Exponential struct {
	// GrowthFactor: Must be greater than 1.
	GrowthFactor float64 `json:"growthFactor,omitempty"`

	// NumFiniteBuckets: Must be greater than 0.
	NumFiniteBuckets int64 `json:"numFiniteBuckets,omitempty"`

	// Scale: Must be greater than 0.
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

func (s *Exponential) MarshalJSON() ([]byte, error) {
	type noMethod Exponential
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Exponential) UnmarshalJSON(data []byte) error {
	type noMethod Exponential
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

// HttpRequest: A common proto for logging HTTP requests. Only contains
// semantics defined by the HTTP specification. Product-specific logging
// information MUST be defined in a separate message.
type HttpRequest struct {
	// CacheFillBytes: The number of HTTP response bytes inserted into
	// cache. Set only when a cache fill was attempted.
	CacheFillBytes int64 `json:"cacheFillBytes,omitempty,string"`

	// CacheHit: Whether or not an entity was served from cache (with or
	// without validation).
	CacheHit bool `json:"cacheHit,omitempty"`

	// CacheLookup: Whether or not a cache lookup was attempted.
	CacheLookup bool `json:"cacheLookup,omitempty"`

	// CacheValidatedWithOriginServer: Whether or not the response was
	// validated with the origin server before being served from cache. This
	// field is only meaningful if cache_hit is True.
	CacheValidatedWithOriginServer bool `json:"cacheValidatedWithOriginServer,omitempty"`

	// Latency: The request processing latency on the server, from the time
	// the request was received until the response was sent.
	Latency string `json:"latency,omitempty"`

	// Protocol: Protocol used for the request. Examples: "HTTP/1.1",
	// "HTTP/2", "websocket"
	Protocol string `json:"protocol,omitempty"`

	// Referer: The referer URL of the request, as defined in HTTP/1.1
	// Header Field Definitions
	// (http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html).
	Referer string `json:"referer,omitempty"`

	// RemoteIp: The IP address (IPv4 or IPv6) of the client that issued the
	// HTTP request. Examples: "192.168.1.1", "FE80::0202:B3FF:FE1E:8329".
	RemoteIp string `json:"remoteIp,omitempty"`

	// RequestMethod: The request method. Examples: "GET", "HEAD", "PUT",
	// "POST".
	RequestMethod string `json:"requestMethod,omitempty"`

	// RequestSize: The size of the HTTP request message in bytes, including
	// the request headers and the request body.
	RequestSize int64 `json:"requestSize,omitempty,string"`

	// RequestUrl: The scheme (http, https), the host name, the path and the
	// query portion of the URL that was requested. Example:
	// "http://example.com/some/info?color=red".
	RequestUrl string `json:"requestUrl,omitempty"`

	// ResponseSize: The size of the HTTP response message sent back to the
	// client, in bytes, including the response headers and the response
	// body.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// ServerIp: The IP address (IPv4 or IPv6) of the origin server that the
	// request was sent to.
	ServerIp string `json:"serverIp,omitempty"`

	// Status: The response code indicating the status of response.
	// Examples: 200, 404.
	Status int64 `json:"status,omitempty"`

	// UserAgent: The user agent sent by the client. Example: "Mozilla/4.0
	// (compatible; MSIE 6.0; Windows 98; Q312461; .NET CLR 1.0.3705)".
	UserAgent string `json:"userAgent,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CacheFillBytes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CacheFillBytes") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *HttpRequest) MarshalJSON() ([]byte, error) {
	type noMethod HttpRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LabelDescriptor: A description of a label.
type LabelDescriptor struct {
	// Description: A human-readable description for the label.
	Description string `json:"description,omitempty"`

	// Key: The label key.
	Key string `json:"key,omitempty"`

	// ValueType: The type of data that can be assigned to the label.
	//
	// Possible values:
	//   "STRING" - A variable-length string. This is the default.
	//   "BOOL" - Boolean; true or false.
	//   "INT64" - A 64-bit signed integer.
	ValueType string `json:"valueType,omitempty"`

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

func (s *LabelDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod LabelDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Linear: Specifies a linear sequence of buckets that all have the same
// width (except overflow and underflow). Each bucket represents a
// constant absolute uncertainty on the specific value in the
// bucket.There are num_finite_buckets + 2 (= N) buckets. Bucket i has
// the following boundaries:Upper bound (0 <= i < N-1): offset + (width
// * i).  Lower bound (1 <= i < N): offset + (width * (i - 1)).
type Linear struct {
	// NumFiniteBuckets: Must be greater than 0.
	NumFiniteBuckets int64 `json:"numFiniteBuckets,omitempty"`

	// Offset: Lower bound of the first bucket.
	Offset float64 `json:"offset,omitempty"`

	// Width: Must be greater than 0.
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

func (s *Linear) MarshalJSON() ([]byte, error) {
	type noMethod Linear
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Linear) UnmarshalJSON(data []byte) error {
	type noMethod Linear
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

// ListLogEntriesRequest: The parameters to ListLogEntries.
type ListLogEntriesRequest struct {
	// Filter: Optional. A filter that chooses which log entries to return.
	// See Advanced Logs Filters. Only log entries that match the filter are
	// returned. An empty filter matches all log entries in the resources
	// listed in resource_names. Referencing a parent resource that is not
	// listed in resource_names will cause the filter to return no results.
	// The maximum length of the filter is 20000 characters.
	Filter string `json:"filter,omitempty"`

	// OrderBy: Optional. How the results should be sorted. Presently, the
	// only permitted values are "timestamp asc" (default) and "timestamp
	// desc". The first option returns entries in order of increasing values
	// of LogEntry.timestamp (oldest first), and the second option returns
	// entries in order of decreasing timestamps (newest first). Entries
	// with equal timestamps are returned in order of their insert_id
	// values.
	OrderBy string `json:"orderBy,omitempty"`

	// PageSize: Optional. The maximum number of results to return from this
	// request. Non-positive values are ignored. The presence of
	// next_page_token in the response indicates that more results might be
	// available.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: Optional. If present, then retrieve the next batch of
	// results from the preceding call to this method. page_token must be
	// the value of next_page_token from the previous response. The values
	// of other method parameters should be identical to those in the
	// previous call.
	PageToken string `json:"pageToken,omitempty"`

	// ProjectIds: Deprecated. Use resource_names instead. One or more
	// project identifiers or project numbers from which to retrieve log
	// entries. Example: "my-project-1A". If present, these project
	// identifiers are converted to resource name format and added to the
	// list of resources in resource_names.
	ProjectIds []string `json:"projectIds,omitempty"`

	// ResourceNames: Required. Names of one or more parent resources from
	// which to retrieve log
	// entries:
	// "projects/[PROJECT_ID]"
	// "organizations/[ORGANIZATION_ID]"
	// "bi
	// llingAccounts/[BILLING_ACCOUNT_ID]"
	// "folders/[FOLDER_ID]"
	// Projects listed in the project_ids field are added to this list.
	ResourceNames []string `json:"resourceNames,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filter") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListLogEntriesRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListLogEntriesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListLogEntriesResponse: Result returned from ListLogEntries.
type ListLogEntriesResponse struct {
	// Entries: A list of log entries. If entries is empty, nextPageToken
	// may still be returned, indicating that more entries may exist. See
	// nextPageToken for more information.
	Entries []*LogEntry `json:"entries,omitempty"`

	// NextPageToken: If there might be more results than those appearing in
	// this response, then nextPageToken is included. To get the next set of
	// results, call this method again using the value of nextPageToken as
	// pageToken.If a value for next_page_token appears and the entries
	// field is empty, it means that the search found no log entries so far
	// but it did not have time to search all the possible log entries.
	// Retry the method with this value for page_token to continue the
	// search. Alternatively, consider speeding up the search by changing
	// your filter to specify a single log name or resource type, or to
	// narrow the time range of the search.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListLogEntriesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogEntriesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListLogMetricsResponse: Result returned from ListLogMetrics.
type ListLogMetricsResponse struct {
	// Metrics: A list of logs-based metrics.
	Metrics []*LogMetric `json:"metrics,omitempty"`

	// NextPageToken: If there might be more results than appear in this
	// response, then nextPageToken is included. To get the next set of
	// results, call this method again using the value of nextPageToken as
	// pageToken.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Metrics") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metrics") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListLogMetricsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogMetricsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListLogsResponse: Result returned from ListLogs.
type ListLogsResponse struct {
	// LogNames: A list of log names. For example,
	// "projects/my-project/syslog" or
	// "organizations/123/cloudresourcemanager.googleapis.com%2Factivity".
	LogNames []string `json:"logNames,omitempty"`

	// NextPageToken: If there might be more results than those appearing in
	// this response, then nextPageToken is included. To get the next set of
	// results, call this method again using the value of nextPageToken as
	// pageToken.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "LogNames") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LogNames") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListLogsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListMonitoredResourceDescriptorsResponse: Result returned from
// ListMonitoredResourceDescriptors.
type ListMonitoredResourceDescriptorsResponse struct {
	// NextPageToken: If there might be more results than those appearing in
	// this response, then nextPageToken is included. To get the next set of
	// results, call this method again using the value of nextPageToken as
	// pageToken.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResourceDescriptors: A list of resource descriptors.
	ResourceDescriptors []*MonitoredResourceDescriptor `json:"resourceDescriptors,omitempty"`

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

func (s *ListMonitoredResourceDescriptorsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListMonitoredResourceDescriptorsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListSinksResponse: Result returned from ListSinks.
type ListSinksResponse struct {
	// NextPageToken: If there might be more results than appear in this
	// response, then nextPageToken is included. To get the next set of
	// results, call the same method again using the value of nextPageToken
	// as pageToken.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Sinks: A list of sinks.
	Sinks []*LogSink `json:"sinks,omitempty"`

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

func (s *ListSinksResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListSinksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogEntry: An individual entry in a log.
type LogEntry struct {
	// HttpRequest: Optional. Information about the HTTP request associated
	// with this log entry, if applicable.
	HttpRequest *HttpRequest `json:"httpRequest,omitempty"`

	// InsertId: Optional. A unique identifier for the log entry. If you
	// provide a value, then Stackdriver Logging considers other log entries
	// in the same project, with the same timestamp, and with the same
	// insert_id to be duplicates which can be removed. If omitted in new
	// log entries, then Stackdriver Logging assigns its own unique
	// identifier. The insert_id is also used to order log entries that have
	// the same timestamp value.
	InsertId string `json:"insertId,omitempty"`

	// JsonPayload: The log entry payload, represented as a structure that
	// is expressed as a JSON object.
	JsonPayload googleapi.RawMessage `json:"jsonPayload,omitempty"`

	// Labels: Optional. A set of user-defined (key, value) data that
	// provides additional information about the log entry.
	Labels map[string]string `json:"labels,omitempty"`

	// LogName: Required. The resource name of the log to which this log
	// entry
	// belongs:
	// "projects/[PROJECT_ID]/logs/[LOG_ID]"
	// "organizations/[ORGANIZ
	// ATION_ID]/logs/[LOG_ID]"
	// "billingAccounts/[BILLING_ACCOUNT_ID]/logs/[L
	// OG_ID]"
	// "folders/[FOLDER_ID]/logs/[LOG_ID]"
	// [LOG_ID] must be URL-encoded within log_name. Example:
	// "organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Fa
	// ctivity". [LOG_ID] must be less than 512 characters long and can only
	// include the following characters: upper and lower case alphanumeric
	// characters, forward-slash, underscore, hyphen, and period.For
	// backward compatibility, if log_name begins with a forward-slash, such
	// as /projects/..., then the log entry is ingested as usual but the
	// forward-slash is removed. Listing the log entry will not show the
	// leading slash and filtering for a log name with a leading slash will
	// never return any results.
	LogName string `json:"logName,omitempty"`

	// Operation: Optional. Information about an operation associated with
	// the log entry, if applicable.
	Operation *LogEntryOperation `json:"operation,omitempty"`

	// ProtoPayload: The log entry payload, represented as a protocol
	// buffer. Some Google Cloud Platform services use this field for their
	// log entry payloads.
	ProtoPayload googleapi.RawMessage `json:"protoPayload,omitempty"`

	// ReceiveTimestamp: Output only. The time the log entry was received by
	// Stackdriver Logging.
	ReceiveTimestamp string `json:"receiveTimestamp,omitempty"`

	// Resource: Required. The monitored resource associated with this log
	// entry. Example: a log entry that reports a database error would be
	// associated with the monitored resource designating the particular
	// database that reported the error.
	Resource *MonitoredResource `json:"resource,omitempty"`

	// Severity: Optional. The severity of the log entry. The default value
	// is LogSeverity.DEFAULT.
	//
	// Possible values:
	//   "DEFAULT" - (0) The log entry has no assigned severity level.
	//   "DEBUG" - (100) Debug or trace information.
	//   "INFO" - (200) Routine information, such as ongoing status or
	// performance.
	//   "NOTICE" - (300) Normal but significant events, such as start up,
	// shut down, or a configuration change.
	//   "WARNING" - (400) Warning events might cause problems.
	//   "ERROR" - (500) Error events are likely to cause problems.
	//   "CRITICAL" - (600) Critical events cause more severe problems or
	// outages.
	//   "ALERT" - (700) A person must take an action immediately.
	//   "EMERGENCY" - (800) One or more systems are unusable.
	Severity string `json:"severity,omitempty"`

	// SourceLocation: Optional. Source code location information associated
	// with the log entry, if any.
	SourceLocation *LogEntrySourceLocation `json:"sourceLocation,omitempty"`

	// TextPayload: The log entry payload, represented as a Unicode string
	// (UTF-8).
	TextPayload string `json:"textPayload,omitempty"`

	// Timestamp: Optional. The time the event described by the log entry
	// occurred. This time is used to compute the log entry's age and to
	// enforce the logs retention period. If this field is omitted in a new
	// log entry, then Stackdriver Logging assigns it the current
	// time.Incoming log entries should have timestamps that are no more
	// than the logs retention period in the past, and no more than 24 hours
	// in the future. See the entries.write API method for more information.
	Timestamp string `json:"timestamp,omitempty"`

	// Trace: Optional. Resource name of the trace associated with the log
	// entry, if any. If it contains a relative resource name, the name is
	// assumed to be relative to //tracing.googleapis.com. Example:
	// projects/my-projectid/traces/06796866738c859f2f19b7cfb3214824
	Trace string `json:"trace,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HttpRequest") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HttpRequest") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogEntry) MarshalJSON() ([]byte, error) {
	type noMethod LogEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogEntryOperation: Additional information about a potentially
// long-running operation with which a log entry is associated.
type LogEntryOperation struct {
	// First: Optional. Set this to True if this is the first log entry in
	// the operation.
	First bool `json:"first,omitempty"`

	// Id: Optional. An arbitrary operation identifier. Log entries with the
	// same identifier are assumed to be part of the same operation.
	Id string `json:"id,omitempty"`

	// Last: Optional. Set this to True if this is the last log entry in the
	// operation.
	Last bool `json:"last,omitempty"`

	// Producer: Optional. An arbitrary producer identifier. The combination
	// of id and producer must be globally unique. Examples for producer:
	// "MyDivision.MyBigCompany.com", "github.com/MyProject/MyApplication".
	Producer string `json:"producer,omitempty"`

	// ForceSendFields is a list of field names (e.g. "First") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "First") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogEntryOperation) MarshalJSON() ([]byte, error) {
	type noMethod LogEntryOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogEntrySourceLocation: Additional information about the source code
// location that produced the log entry.
type LogEntrySourceLocation struct {
	// File: Optional. Source file name. Depending on the runtime
	// environment, this might be a simple name or a fully-qualified name.
	File string `json:"file,omitempty"`

	// Function: Optional. Human-readable name of the function or method
	// being invoked, with optional context such as the class or package
	// name. This information may be used in contexts such as the logs
	// viewer, where a file and line number are less meaningful. The format
	// can vary by language. For example: qual.if.ied.Class.method (Java),
	// dir/package.func (Go), function (Python).
	Function string `json:"function,omitempty"`

	// Line: Optional. Line within the source file. 1-based; 0 indicates no
	// line number available.
	Line int64 `json:"line,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "File") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "File") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogEntrySourceLocation) MarshalJSON() ([]byte, error) {
	type noMethod LogEntrySourceLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogLine: Application log line emitted while processing a request.
type LogLine struct {
	// LogMessage: App-provided log message.
	LogMessage string `json:"logMessage,omitempty"`

	// Severity: Severity of this log entry.
	//
	// Possible values:
	//   "DEFAULT" - (0) The log entry has no assigned severity level.
	//   "DEBUG" - (100) Debug or trace information.
	//   "INFO" - (200) Routine information, such as ongoing status or
	// performance.
	//   "NOTICE" - (300) Normal but significant events, such as start up,
	// shut down, or a configuration change.
	//   "WARNING" - (400) Warning events might cause problems.
	//   "ERROR" - (500) Error events are likely to cause problems.
	//   "CRITICAL" - (600) Critical events cause more severe problems or
	// outages.
	//   "ALERT" - (700) A person must take an action immediately.
	//   "EMERGENCY" - (800) One or more systems are unusable.
	Severity string `json:"severity,omitempty"`

	// SourceLocation: Where in the source code this log message was
	// written.
	SourceLocation *SourceLocation `json:"sourceLocation,omitempty"`

	// Time: Approximate time when this log entry was made.
	Time string `json:"time,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LogMessage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LogMessage") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogLine) MarshalJSON() ([]byte, error) {
	type noMethod LogLine
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogMetric: Describes a logs-based metric. The value of the metric is
// the number of log entries that match a logs filter in a given time
// interval.Logs-based metric can also be used to extract values from
// logs and create a a distribution of the values. The distribution
// records the statistics of the extracted values along with an optional
// histogram of the values as specified by the bucket options.
type LogMetric struct {
	// BucketOptions: Optional. The bucket_options are required when the
	// logs-based metric is using a DISTRIBUTION value type and it describes
	// the bucket boundaries used to create a histogram of the extracted
	// values.
	BucketOptions *BucketOptions `json:"bucketOptions,omitempty"`

	// Description: Optional. A description of this metric, which is used in
	// documentation.
	Description string `json:"description,omitempty"`

	// Filter: Required. An advanced logs filter which is used to match log
	// entries. Example:
	// "resource.type=gae_app AND severity>=ERROR"
	// The maximum length of the filter is 20000 characters.
	Filter string `json:"filter,omitempty"`

	// LabelExtractors: Optional. A map from a label key string to an
	// extractor expression which is used to extract data from a log entry
	// field and assign as the label value. Each label key specified in the
	// LabelDescriptor must have an associated extractor expression in this
	// map. The syntax of the extractor expression is the same as for the
	// value_extractor field.The extracted value is converted to the type
	// defined in the label descriptor. If the either the extraction or the
	// type conversion fails, the label will have a default value. The
	// default value for a string label is an empty string, for an integer
	// label its 0, and for a boolean label its false.Note that there are
	// upper bounds on the maximum number of labels and the number of active
	// time series that are allowed in a project.
	LabelExtractors map[string]string `json:"labelExtractors,omitempty"`

	// MetricDescriptor: Optional. The metric descriptor associated with the
	// logs-based metric. If unspecified, it uses a default metric
	// descriptor with a DELTA metric kind, INT64 value type, with no labels
	// and a unit of "1". Such a metric counts the number of log entries
	// matching the filter expression.The name, type, and description fields
	// in the metric_descriptor are output only, and is constructed using
	// the name and description field in the LogMetric.To create a
	// logs-based metric that records a distribution of log values, a DELTA
	// metric kind with a DISTRIBUTION value type must be used along with a
	// value_extractor expression in the LogMetric.Each label in the metric
	// descriptor must have a matching label name as the key and an
	// extractor expression as the value in the label_extractors map.The
	// metric_kind and value_type fields in the metric_descriptor cannot be
	// updated once initially configured. New labels can be added in the
	// metric_descriptor, but existing labels cannot be modified except for
	// their description.
	MetricDescriptor *MetricDescriptor `json:"metricDescriptor,omitempty"`

	// Name: Required. The client-assigned metric identifier. Examples:
	// "error_count", "nginx/requests".Metric identifiers are limited to 100
	// characters and can include only the following characters: A-Z, a-z,
	// 0-9, and the special characters _-.,+!*',()%/. The forward-slash
	// character (/) denotes a hierarchy of name pieces, and it cannot be
	// the first character of the name.The metric identifier in this field
	// must not be URL-encoded
	// (https://en.wikipedia.org/wiki/Percent-encoding). However, when the
	// metric identifier appears as the [METRIC_ID] part of a metric_name
	// API parameter, then the metric identifier must be URL-encoded.
	// Example: "projects/my-project/metrics/nginx%2Frequests".
	Name string `json:"name,omitempty"`

	// ValueExtractor: Optional. A value_extractor is required when using a
	// distribution logs-based metric to extract the values to record from a
	// log entry. Two functions are supported for value extraction:
	// EXTRACT(field) or REGEXP_EXTRACT(field, regex). The argument are:  1.
	// field: The name of the log entry field from which the value is to be
	// extracted.  2. regex: A regular expression using the Google RE2
	// syntax  (https://github.com/google/re2/wiki/Syntax) with a single
	// capture  group to extract data from the specified log entry field.
	// The value  of the field is converted to a string before applying the
	// regex.  It is an error to specify a regex that does not include
	// exactly one  capture group.The result of the extraction must be
	// convertible to a double type, as the distribution always records
	// double values. If either the extraction or the conversion to double
	// fails, then those values are not recorded in the
	// distribution.Example: REGEXP_EXTRACT(jsonPayload.request,
	// ".*quantity=(\d+).*")
	ValueExtractor string `json:"valueExtractor,omitempty"`

	// Version: Deprecated. The API version that created or updated this
	// metric. The v2 format is used by default and cannot be changed.
	//
	// Possible values:
	//   "V2" - Stackdriver Logging API v2.
	//   "V1" - Stackdriver Logging API v1.
	Version string `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BucketOptions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketOptions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogMetric) MarshalJSON() ([]byte, error) {
	type noMethod LogMetric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogSink: Describes a sink used to export log entries to one of the
// following destinations in any project: a Cloud Storage bucket, a
// BigQuery dataset, or a Cloud Pub/Sub topic. A logs filter controls
// which log entries are exported. The sink must be created within a
// project, organization, billing account, or folder.
type LogSink struct {
	// Destination: Required. The export
	// destination:
	// "storage.googleapis.com/[GCS_BUCKET]"
	// "bigquery.googleapi
	// s.com/projects/[PROJECT_ID]/datasets/[DATASET]"
	// "pubsub.googleapis.com
	// /projects/[PROJECT_ID]/topics/[TOPIC_ID]"
	// The sink's writer_identity, set when the sink is created, must have
	// permission to write to the destination or else the log entries are
	// not exported. For more information, see Exporting Logs With Sinks.
	Destination string `json:"destination,omitempty"`

	// EndTime: Deprecated. This field is ignored when creating or updating
	// sinks.
	EndTime string `json:"endTime,omitempty"`

	// Filter: Optional. An advanced logs filter. The only exported log
	// entries are those that are in the resource owning the sink and that
	// match the filter. For
	// example:
	// logName="projects/[PROJECT_ID]/logs/[LOG_ID]" AND severity>=ERROR
	//
	Filter string `json:"filter,omitempty"`

	// IncludeChildren: Optional. This field applies only to sinks owned by
	// organizations and folders. If the field is false, the default, only
	// the logs owned by the sink's parent resource are available for
	// export. If the field is true, then logs from all the projects,
	// folders, and billing accounts contained in the sink's parent resource
	// are also available for export. Whether a particular log entry from
	// the children is exported depends on the sink's filter expression. For
	// example, if this field is true, then the filter
	// resource.type=gce_instance would export all Compute Engine VM
	// instance log entries from all projects in the sink's parent. To only
	// export entries from certain child projects, filter on the project
	// part of the log name:
	// logName:("projects/test-project1/" OR "projects/test-project2/")
	// AND
	// resource.type=gce_instance
	//
	IncludeChildren bool `json:"includeChildren,omitempty"`

	// Name: Required. The client-assigned sink identifier, unique within
	// the project. Example: "my-syslog-errors-to-pubsub". Sink identifiers
	// are limited to 100 characters and can include only the following
	// characters: upper and lower-case alphanumeric characters,
	// underscores, hyphens, and periods.
	Name string `json:"name,omitempty"`

	// OutputVersionFormat: Deprecated. The log entry format to use for this
	// sink's exported log entries. The v2 format is used by default and
	// cannot be changed.
	//
	// Possible values:
	//   "VERSION_FORMAT_UNSPECIFIED" - An unspecified format version that
	// will default to V2.
	//   "V2" - LogEntry version 2 format.
	//   "V1" - LogEntry version 1 format.
	OutputVersionFormat string `json:"outputVersionFormat,omitempty"`

	// StartTime: Deprecated. This field is ignored when creating or
	// updating sinks.
	StartTime string `json:"startTime,omitempty"`

	// WriterIdentity: Output only. An IAM identity&mdash;a service account
	// or group&mdash;under which Stackdriver Logging writes the exported
	// log entries to the sink's destination. This field is set by
	// sinks.create and sinks.update, based on the setting of
	// unique_writer_identity in those methods.Until you grant this identity
	// write-access to the destination, log entry exports from this sink
	// will fail. For more information, see Granting access for a resource.
	// Consult the destination service's documentation to determine the
	// appropriate IAM roles to assign to the identity.
	WriterIdentity string `json:"writerIdentity,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Destination") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Destination") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LogSink) MarshalJSON() ([]byte, error) {
	type noMethod LogSink
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MetricDescriptor: Defines a metric type and its schema. Once a metric
// descriptor is created, deleting or altering it stops data collection
// and makes the metric type's existing data unusable.
type MetricDescriptor struct {
	// Description: A detailed description of the metric, which can be used
	// in documentation.
	Description string `json:"description,omitempty"`

	// DisplayName: A concise name for the metric, which can be displayed in
	// user interfaces. Use sentence case without an ending period, for
	// example "Request count".
	DisplayName string `json:"displayName,omitempty"`

	// Labels: The set of labels that can be used to describe a specific
	// instance of this metric type. For example, the
	// appengine.googleapis.com/http/server/response_latencies metric type
	// has a label for the HTTP response code, response_code, so you can
	// look at latencies for successful responses or just for responses that
	// failed.
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// MetricKind: Whether the metric records instantaneous values, changes
	// to a value, etc. Some combinations of metric_kind and value_type
	// might not be supported.
	//
	// Possible values:
	//   "METRIC_KIND_UNSPECIFIED" - Do not use this default value.
	//   "GAUGE" - An instantaneous measurement of a value.
	//   "DELTA" - The change in a value during a time interval.
	//   "CUMULATIVE" - A value accumulated over a time interval. Cumulative
	// measurements in a time series should have the same start time and
	// increasing end times, until an event resets the cumulative value to
	// zero and sets a new start time for the following points.
	MetricKind string `json:"metricKind,omitempty"`

	// Name: The resource name of the metric descriptor. Depending on the
	// implementation, the name typically includes: (1) the parent resource
	// name that defines the scope of the metric type or of its data; and
	// (2) the metric's URL-encoded type, which also appears in the type
	// field of this descriptor. For example, following is the resource name
	// of a custom metric within the GCP project
	// my-project-id:
	// "projects/my-project-id/metricDescriptors/custom.google
	// apis.com%2Finvoice%2Fpaid%2Famount"
	//
	Name string `json:"name,omitempty"`

	// Type: The metric type, including its DNS name prefix. The type is not
	// URL-encoded. All user-defined custom metric types have the DNS name
	// custom.googleapis.com. Metric types should use a natural hierarchical
	// grouping. For
	// example:
	// "custom.googleapis.com/invoice/paid/amount"
	// "appengine.google
	// apis.com/http/server/response_latencies"
	//
	Type string `json:"type,omitempty"`

	// Unit: The unit in which the metric value is reported. It is only
	// applicable if the value_type is INT64, DOUBLE, or DISTRIBUTION. The
	// supported units are a subset of The Unified Code for Units of Measure
	// (http://unitsofmeasure.org/ucum.html) standard:Basic units (UNIT)
	// bit bit
	// By byte
	// s second
	// min minute
	// h hour
	// d dayPrefixes (PREFIX)
	// k kilo (10**3)
	// M mega (10**6)
	// G giga (10**9)
	// T tera (10**12)
	// P peta (10**15)
	// E exa (10**18)
	// Z zetta (10**21)
	// Y yotta (10**24)
	// m milli (10**-3)
	// u micro (10**-6)
	// n nano (10**-9)
	// p pico (10**-12)
	// f femto (10**-15)
	// a atto (10**-18)
	// z zepto (10**-21)
	// y yocto (10**-24)
	// Ki kibi (2**10)
	// Mi mebi (2**20)
	// Gi gibi (2**30)
	// Ti tebi (2**40)GrammarThe grammar includes the dimensionless unit 1,
	// such as 1/s.The grammar also includes these connectors:
	// / division (as an infix operator, e.g. 1/s).
	// . multiplication (as an infix operator, e.g. GBy.d)The grammar for a
	// unit is as follows:
	// Expression = Component { "." Component } { "/" Component }
	// ;
	//
	// Component = [ PREFIX ] UNIT [ Annotation ]
	//           | Annotation
	//           | "1"
	//           ;
	//
	// Annotation = "{" NAME "}" ;
	// Notes:
	// Annotation is just a comment if it follows a UNIT and is  equivalent
	// to 1 if it is used alone. For examples,  {requests}/s == 1/s,
	// By{transmitted}/s == By/s.
	// NAME is a sequence of non-blank printable ASCII characters not
	// containing '{' or '}'.
	Unit string `json:"unit,omitempty"`

	// ValueType: Whether the measurement is an integer, a floating-point
	// number, etc. Some combinations of metric_kind and value_type might
	// not be supported.
	//
	// Possible values:
	//   "VALUE_TYPE_UNSPECIFIED" - Do not use this default value.
	//   "BOOL" - The value is a boolean. This value type can be used only
	// if the metric kind is GAUGE.
	//   "INT64" - The value is a signed 64-bit integer.
	//   "DOUBLE" - The value is a double precision floating point number.
	//   "STRING" - The value is a text string. This value type can be used
	// only if the metric kind is GAUGE.
	//   "DISTRIBUTION" - The value is a Distribution.
	//   "MONEY" - The value is money.
	ValueType string `json:"valueType,omitempty"`

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

func (s *MetricDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MetricDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MonitoredResource: An object representing a resource that can be used
// for monitoring, logging, billing, or other purposes. Examples include
// virtual machine instances, databases, and storage devices such as
// disks. The type field identifies a MonitoredResourceDescriptor object
// that describes the resource's schema. Information in the labels field
// identifies the actual resource and its attributes according to the
// schema. For example, a particular Compute Engine VM instance could be
// represented by the following object, because the
// MonitoredResourceDescriptor for "gce_instance" has labels
// "instance_id" and "zone":
// { "type": "gce_instance",
//   "labels": { "instance_id": "12345678901234",
//               "zone": "us-central1-a" }}
//
type MonitoredResource struct {
	// Labels: Required. Values for all of the labels listed in the
	// associated monitored resource descriptor. For example, Compute Engine
	// VM instances use the labels "project_id", "instance_id", and "zone".
	Labels map[string]string `json:"labels,omitempty"`

	// Type: Required. The monitored resource type. This field must match
	// the type field of a MonitoredResourceDescriptor object. For example,
	// the type of a Compute Engine VM instance is gce_instance.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Labels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Labels") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MonitoredResource) MarshalJSON() ([]byte, error) {
	type noMethod MonitoredResource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MonitoredResourceDescriptor: An object that describes the schema of a
// MonitoredResource object using a type name and a set of labels. For
// example, the monitored resource descriptor for Google Compute Engine
// VM instances has a type of "gce_instance" and specifies the use of
// the labels "instance_id" and "zone" to identify particular VM
// instances.Different APIs can support different monitored resource
// types. APIs generally provide a list method that returns the
// monitored resource descriptors used by the API.
type MonitoredResourceDescriptor struct {
	// Description: Optional. A detailed description of the monitored
	// resource type that might be used in documentation.
	Description string `json:"description,omitempty"`

	// DisplayName: Optional. A concise name for the monitored resource type
	// that might be displayed in user interfaces. It should be a Title
	// Cased Noun Phrase, without any article or other determiners. For
	// example, "Google Cloud SQL Database".
	DisplayName string `json:"displayName,omitempty"`

	// Labels: Required. A set of labels used to describe instances of this
	// monitored resource type. For example, an individual Google Cloud SQL
	// database is identified by values for the labels "database_id" and
	// "zone".
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// Name: Optional. The resource name of the monitored resource
	// descriptor:
	// "projects/{project_id}/monitoredResourceDescriptors/{type}" where
	// {type} is the value of the type field in this object and {project_id}
	// is a project ID that provides API-specific context for accessing the
	// type. APIs that do not use project information can use the resource
	// name format "monitoredResourceDescriptors/{type}".
	Name string `json:"name,omitempty"`

	// Type: Required. The monitored resource type. For example, the type
	// "cloudsql_database" represents databases in Google Cloud SQL. The
	// maximum length of this value is 256 characters.
	Type string `json:"type,omitempty"`

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

func (s *MonitoredResourceDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod MonitoredResourceDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RequestLog: Complete log information about a single HTTP request to
// an App Engine application.
type RequestLog struct {
	// AppEngineRelease: App Engine release version.
	AppEngineRelease string `json:"appEngineRelease,omitempty"`

	// AppId: Application that handled this request.
	AppId string `json:"appId,omitempty"`

	// Cost: An indication of the relative cost of serving this request.
	Cost float64 `json:"cost,omitempty"`

	// EndTime: Time when the request finished.
	EndTime string `json:"endTime,omitempty"`

	// Finished: Whether this request is finished or active.
	Finished bool `json:"finished,omitempty"`

	// First: Whether this is the first RequestLog entry for this request.
	// If an active request has several RequestLog entries written to
	// Stackdriver Logging, then this field will be set for one of them.
	First bool `json:"first,omitempty"`

	// Host: Internet host and port number of the resource being requested.
	Host string `json:"host,omitempty"`

	// HttpVersion: HTTP version of request. Example: "HTTP/1.1".
	HttpVersion string `json:"httpVersion,omitempty"`

	// InstanceId: An identifier for the instance that handled the request.
	InstanceId string `json:"instanceId,omitempty"`

	// InstanceIndex: If the instance processing this request belongs to a
	// manually scaled module, then this is the 0-based index of the
	// instance. Otherwise, this value is -1.
	InstanceIndex int64 `json:"instanceIndex,omitempty"`

	// Ip: Origin IP address.
	Ip string `json:"ip,omitempty"`

	// Latency: Latency of the request.
	Latency string `json:"latency,omitempty"`

	// Line: A list of log lines emitted by the application while serving
	// this request.
	Line []*LogLine `json:"line,omitempty"`

	// MegaCycles: Number of CPU megacycles used to process request.
	MegaCycles int64 `json:"megaCycles,omitempty,string"`

	// Method: Request method. Example: "GET", "HEAD", "PUT", "POST",
	// "DELETE".
	Method string `json:"method,omitempty"`

	// ModuleId: Module of the application that handled this request.
	ModuleId string `json:"moduleId,omitempty"`

	// Nickname: The logged-in user who made the request.Most likely, this
	// is the part of the user's email before the @ sign. The field value is
	// the same for different requests from the same user, but different
	// users can have similar names. This information is also available to
	// the application via the App Engine Users API.This field will be
	// populated starting with App Engine 1.9.21.
	Nickname string `json:"nickname,omitempty"`

	// PendingTime: Time this request spent in the pending request queue.
	PendingTime string `json:"pendingTime,omitempty"`

	// Referrer: Referrer URL of request.
	Referrer string `json:"referrer,omitempty"`

	// RequestId: Globally unique identifier for a request, which is based
	// on the request start time. Request IDs for requests which started
	// later will compare greater as strings than those for requests which
	// started earlier.
	RequestId string `json:"requestId,omitempty"`

	// Resource: Contains the path and query portion of the URL that was
	// requested. For example, if the URL was
	// "http://example.com/app?name=val", the resource would be
	// "/app?name=val". The fragment identifier, which is identified by the
	// # character, is not included.
	Resource string `json:"resource,omitempty"`

	// ResponseSize: Size in bytes sent back to client by request.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// SourceReference: Source code for the application that handled this
	// request. There can be more than one source reference per deployed
	// application if source code is distributed among multiple
	// repositories.
	SourceReference []*SourceReference `json:"sourceReference,omitempty"`

	// StartTime: Time when the request started.
	StartTime string `json:"startTime,omitempty"`

	// Status: HTTP response status code. Example: 200, 404.
	Status int64 `json:"status,omitempty"`

	// TaskName: Task name of the request, in the case of an offline
	// request.
	TaskName string `json:"taskName,omitempty"`

	// TaskQueueName: Queue name of the request, in the case of an offline
	// request.
	TaskQueueName string `json:"taskQueueName,omitempty"`

	// TraceId: Stackdriver Trace identifier for this request.
	TraceId string `json:"traceId,omitempty"`

	// UrlMapEntry: File or class that handled the request.
	UrlMapEntry string `json:"urlMapEntry,omitempty"`

	// UserAgent: User agent that made the request.
	UserAgent string `json:"userAgent,omitempty"`

	// VersionId: Version of the application that handled this request.
	VersionId string `json:"versionId,omitempty"`

	// WasLoadingRequest: Whether this was a loading request for the
	// instance.
	WasLoadingRequest bool `json:"wasLoadingRequest,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppEngineRelease") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineRelease") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RequestLog) MarshalJSON() ([]byte, error) {
	type noMethod RequestLog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *RequestLog) UnmarshalJSON(data []byte) error {
	type noMethod RequestLog
	var s1 struct {
		Cost gensupport.JSONFloat64 `json:"cost"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Cost = float64(s1.Cost)
	return nil
}

// SourceLocation: Specifies a location in a source code file.
type SourceLocation struct {
	// File: Source file name. Depending on the runtime environment, this
	// might be a simple name or a fully-qualified name.
	File string `json:"file,omitempty"`

	// FunctionName: Human-readable name of the function or method being
	// invoked, with optional context such as the class or package name.
	// This information is used in contexts such as the logs viewer, where a
	// file and line number are less meaningful. The format can vary by
	// language. For example: qual.if.ied.Class.method (Java),
	// dir/package.func (Go), function (Python).
	FunctionName string `json:"functionName,omitempty"`

	// Line: Line within the source file.
	Line int64 `json:"line,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "File") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "File") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourceLocation) MarshalJSON() ([]byte, error) {
	type noMethod SourceLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SourceReference: A reference to a particular snapshot of the source
// tree used to build and deploy an application.
type SourceReference struct {
	// Repository: Optional. A URI string identifying the repository.
	// Example: "https://github.com/GoogleCloudPlatform/kubernetes.git"
	Repository string `json:"repository,omitempty"`

	// RevisionId: The canonical and persistent identifier of the deployed
	// revision. Example (git): "0035781c50ec7aa23385dc841529ce8a4b70db1b"
	RevisionId string `json:"revisionId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Repository") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Repository") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourceReference) MarshalJSON() ([]byte, error) {
	type noMethod SourceReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WriteLogEntriesRequest: The parameters to WriteLogEntries.
type WriteLogEntriesRequest struct {
	// Entries: Required. The log entries to send to Stackdriver Logging.
	// The order of log entries in this list does not matter. Values
	// supplied in this method's log_name, resource, and labels fields are
	// copied into those log entries in this list that do not include values
	// for their corresponding fields. For more information, see the
	// LogEntry type.If the timestamp or insert_id fields are missing in log
	// entries, then this method supplies the current time or a unique
	// identifier, respectively. The supplied values are chosen so that,
	// among the log entries that did not supply their own values, the
	// entries earlier in the list will sort before the entries later in the
	// list. See the entries.list method.Log entries with timestamps that
	// are more than the logs retention period in the past or more than 24
	// hours in the future might be discarded. Discarding does not return an
	// error.To improve throughput and to avoid exceeding the quota limit
	// for calls to entries.write, you should try to include several log
	// entries in this list, rather than calling this method for each
	// individual log entry.
	Entries []*LogEntry `json:"entries,omitempty"`

	// Labels: Optional. Default labels that are added to the labels field
	// of all log entries in entries. If a log entry already has a label
	// with the same key as a label in this parameter, then the log entry's
	// label is not changed. See LogEntry.
	Labels map[string]string `json:"labels,omitempty"`

	// LogName: Optional. A default log resource name that is assigned to
	// all log entries in entries that do not specify a value for
	// log_name:
	// "projects/[PROJECT_ID]/logs/[LOG_ID]"
	// "organizations/[ORGANI
	// ZATION_ID]/logs/[LOG_ID]"
	// "billingAccounts/[BILLING_ACCOUNT_ID]/logs/[
	// LOG_ID]"
	// "folders/[FOLDER_ID]/logs/[LOG_ID]"
	// [LOG_ID] must be URL-encoded. For example,
	// "projects/my-project-id/logs/syslog" or
	// "organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Fa
	// ctivity". For more information about log names, see LogEntry.
	LogName string `json:"logName,omitempty"`

	// PartialSuccess: Optional. Whether valid entries should be written
	// even if some other entries fail due to INVALID_ARGUMENT or
	// PERMISSION_DENIED errors. If any entry is not written, then the
	// response status is the error associated with one of the failed
	// entries and the response includes error details keyed by the entries'
	// zero-based index in the entries.write method.
	PartialSuccess bool `json:"partialSuccess,omitempty"`

	// Resource: Optional. A default monitored resource object that is
	// assigned to all log entries in entries that do not specify a value
	// for resource. Example:
	// { "type": "gce_instance",
	//   "labels": {
	//     "zone": "us-central1-a", "instance_id": "00000000000000000000"
	// }}
	// See LogEntry.
	Resource *MonitoredResource `json:"resource,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WriteLogEntriesRequest) MarshalJSON() ([]byte, error) {
	type noMethod WriteLogEntriesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WriteLogEntriesResponse: Result returned from WriteLogEntries. empty
type WriteLogEntriesResponse struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// method id "logging.billingAccounts.logs.delete":

type BillingAccountsLogsDeleteCall struct {
	s          *Service
	logName    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes all the log entries in a log. The log reappears if it
// receives new entries. Log entries written shortly before the delete
// operation might not be deleted.
func (r *BillingAccountsLogsService) Delete(logName string) *BillingAccountsLogsDeleteCall {
	c := &BillingAccountsLogsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.logName = logName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BillingAccountsLogsDeleteCall) Fields(s ...googleapi.Field) *BillingAccountsLogsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BillingAccountsLogsDeleteCall) Context(ctx context.Context) *BillingAccountsLogsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BillingAccountsLogsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BillingAccountsLogsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+logName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"logName": c.logName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.billingAccounts.logs.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BillingAccountsLogsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes all the log entries in a log. The log reappears if it receives new entries. Log entries written shortly before the delete operation might not be deleted.",
	//   "flatPath": "v2beta1/billingAccounts/{billingAccountsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.billingAccounts.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete:\n\"projects/[PROJECT_ID]/logs/[LOG_ID]\"\n\"organizations/[ORGANIZATION_ID]/logs/[LOG_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]\"\n\"folders/[FOLDER_ID]/logs/[LOG_ID]\"\n[LOG_ID] must be URL-encoded. For example, \"projects/my-project-id/logs/syslog\", \"organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Factivity\". For more information about log names, see LogEntry.",
	//       "location": "path",
	//       "pattern": "^billingAccounts/[^/]+/logs/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+logName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.billingAccounts.logs.list":

type BillingAccountsLogsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the logs in projects, organizations, folders, or billing
// accounts. Only logs that have entries are listed.
func (r *BillingAccountsLogsService) List(parent string) *BillingAccountsLogsListCall {
	c := &BillingAccountsLogsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *BillingAccountsLogsListCall) PageSize(pageSize int64) *BillingAccountsLogsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *BillingAccountsLogsListCall) PageToken(pageToken string) *BillingAccountsLogsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BillingAccountsLogsListCall) Fields(s ...googleapi.Field) *BillingAccountsLogsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BillingAccountsLogsListCall) IfNoneMatch(entityTag string) *BillingAccountsLogsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BillingAccountsLogsListCall) Context(ctx context.Context) *BillingAccountsLogsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BillingAccountsLogsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BillingAccountsLogsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/logs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.billingAccounts.logs.list" call.
// Exactly one of *ListLogsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BillingAccountsLogsListCall) Do(opts ...googleapi.CallOption) (*ListLogsResponse, error) {
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
	ret := &ListLogsResponse{
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
	//   "description": "Lists the logs in projects, organizations, folders, or billing accounts. Only logs that have entries are listed.",
	//   "flatPath": "v2beta1/billingAccounts/{billingAccountsId}/logs",
	//   "httpMethod": "GET",
	//   "id": "logging.billingAccounts.logs.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The resource name that owns the logs:\n\"projects/[PROJECT_ID]\"\n\"organizations/[ORGANIZATION_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]\"\n\"folders/[FOLDER_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^billingAccounts/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/logs",
	//   "response": {
	//     "$ref": "ListLogsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *BillingAccountsLogsListCall) Pages(ctx context.Context, f func(*ListLogsResponse) error) error {
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

// method id "logging.entries.list":

type EntriesListCall struct {
	s                     *Service
	listlogentriesrequest *ListLogEntriesRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// List: Lists log entries. Use this method to retrieve log entries from
// Stackdriver Logging. For ways to export log entries, see Exporting
// Logs.
func (r *EntriesService) List(listlogentriesrequest *ListLogEntriesRequest) *EntriesListCall {
	c := &EntriesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.listlogentriesrequest = listlogentriesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntriesListCall) Fields(s ...googleapi.Field) *EntriesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntriesListCall) Context(ctx context.Context) *EntriesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntriesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntriesListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.listlogentriesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/entries:list")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.entries.list" call.
// Exactly one of *ListLogEntriesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogEntriesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EntriesListCall) Do(opts ...googleapi.CallOption) (*ListLogEntriesResponse, error) {
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
	ret := &ListLogEntriesResponse{
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
	//   "description": "Lists log entries. Use this method to retrieve log entries from Stackdriver Logging. For ways to export log entries, see Exporting Logs.",
	//   "flatPath": "v2beta1/entries:list",
	//   "httpMethod": "POST",
	//   "id": "logging.entries.list",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/entries:list",
	//   "request": {
	//     "$ref": "ListLogEntriesRequest"
	//   },
	//   "response": {
	//     "$ref": "ListLogEntriesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *EntriesListCall) Pages(ctx context.Context, f func(*ListLogEntriesResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.listlogentriesrequest.PageToken = pt }(c.listlogentriesrequest.PageToken) // reset paging to original point
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
		c.listlogentriesrequest.PageToken = x.NextPageToken
	}
}

// method id "logging.entries.write":

type EntriesWriteCall struct {
	s                      *Service
	writelogentriesrequest *WriteLogEntriesRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Write: Log entry resourcesWrites log entries to Stackdriver Logging.
// This API method is the only way to send log entries to Stackdriver
// Logging. This method is used, directly or indirectly, by the
// Stackdriver Logging agent (fluentd) and all logging libraries
// configured to use Stackdriver Logging.
func (r *EntriesService) Write(writelogentriesrequest *WriteLogEntriesRequest) *EntriesWriteCall {
	c := &EntriesWriteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.writelogentriesrequest = writelogentriesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EntriesWriteCall) Fields(s ...googleapi.Field) *EntriesWriteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EntriesWriteCall) Context(ctx context.Context) *EntriesWriteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EntriesWriteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EntriesWriteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.writelogentriesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/entries:write")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.entries.write" call.
// Exactly one of *WriteLogEntriesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WriteLogEntriesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EntriesWriteCall) Do(opts ...googleapi.CallOption) (*WriteLogEntriesResponse, error) {
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
	ret := &WriteLogEntriesResponse{
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
	//   "description": "Log entry resourcesWrites log entries to Stackdriver Logging. This API method is the only way to send log entries to Stackdriver Logging. This method is used, directly or indirectly, by the Stackdriver Logging agent (fluentd) and all logging libraries configured to use Stackdriver Logging.",
	//   "flatPath": "v2beta1/entries:write",
	//   "httpMethod": "POST",
	//   "id": "logging.entries.write",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v2beta1/entries:write",
	//   "request": {
	//     "$ref": "WriteLogEntriesRequest"
	//   },
	//   "response": {
	//     "$ref": "WriteLogEntriesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "logging.monitoredResourceDescriptors.list":

type MonitoredResourceDescriptorsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the descriptors for monitored resource types used by
// Stackdriver Logging.
func (r *MonitoredResourceDescriptorsService) List() *MonitoredResourceDescriptorsListCall {
	c := &MonitoredResourceDescriptorsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *MonitoredResourceDescriptorsListCall) PageSize(pageSize int64) *MonitoredResourceDescriptorsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *MonitoredResourceDescriptorsListCall) PageToken(pageToken string) *MonitoredResourceDescriptorsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MonitoredResourceDescriptorsListCall) Fields(s ...googleapi.Field) *MonitoredResourceDescriptorsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MonitoredResourceDescriptorsListCall) IfNoneMatch(entityTag string) *MonitoredResourceDescriptorsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *MonitoredResourceDescriptorsListCall) Context(ctx context.Context) *MonitoredResourceDescriptorsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *MonitoredResourceDescriptorsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *MonitoredResourceDescriptorsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/monitoredResourceDescriptors")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.monitoredResourceDescriptors.list" call.
// Exactly one of *ListMonitoredResourceDescriptorsResponse or error
// will be non-nil. Any non-2xx status code is an error. Response
// headers are in either
// *ListMonitoredResourceDescriptorsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *MonitoredResourceDescriptorsListCall) Do(opts ...googleapi.CallOption) (*ListMonitoredResourceDescriptorsResponse, error) {
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
	ret := &ListMonitoredResourceDescriptorsResponse{
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
	//   "description": "Lists the descriptors for monitored resource types used by Stackdriver Logging.",
	//   "flatPath": "v2beta1/monitoredResourceDescriptors",
	//   "httpMethod": "GET",
	//   "id": "logging.monitoredResourceDescriptors.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/monitoredResourceDescriptors",
	//   "response": {
	//     "$ref": "ListMonitoredResourceDescriptorsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *MonitoredResourceDescriptorsListCall) Pages(ctx context.Context, f func(*ListMonitoredResourceDescriptorsResponse) error) error {
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

// method id "logging.organizations.logs.delete":

type OrganizationsLogsDeleteCall struct {
	s          *Service
	logName    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes all the log entries in a log. The log reappears if it
// receives new entries. Log entries written shortly before the delete
// operation might not be deleted.
func (r *OrganizationsLogsService) Delete(logName string) *OrganizationsLogsDeleteCall {
	c := &OrganizationsLogsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.logName = logName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsLogsDeleteCall) Fields(s ...googleapi.Field) *OrganizationsLogsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsLogsDeleteCall) Context(ctx context.Context) *OrganizationsLogsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsLogsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsLogsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+logName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"logName": c.logName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.organizations.logs.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OrganizationsLogsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes all the log entries in a log. The log reappears if it receives new entries. Log entries written shortly before the delete operation might not be deleted.",
	//   "flatPath": "v2beta1/organizations/{organizationsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.organizations.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete:\n\"projects/[PROJECT_ID]/logs/[LOG_ID]\"\n\"organizations/[ORGANIZATION_ID]/logs/[LOG_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]\"\n\"folders/[FOLDER_ID]/logs/[LOG_ID]\"\n[LOG_ID] must be URL-encoded. For example, \"projects/my-project-id/logs/syslog\", \"organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Factivity\". For more information about log names, see LogEntry.",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+/logs/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+logName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.organizations.logs.list":

type OrganizationsLogsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the logs in projects, organizations, folders, or billing
// accounts. Only logs that have entries are listed.
func (r *OrganizationsLogsService) List(parent string) *OrganizationsLogsListCall {
	c := &OrganizationsLogsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *OrganizationsLogsListCall) PageSize(pageSize int64) *OrganizationsLogsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *OrganizationsLogsListCall) PageToken(pageToken string) *OrganizationsLogsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OrganizationsLogsListCall) Fields(s ...googleapi.Field) *OrganizationsLogsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OrganizationsLogsListCall) IfNoneMatch(entityTag string) *OrganizationsLogsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OrganizationsLogsListCall) Context(ctx context.Context) *OrganizationsLogsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OrganizationsLogsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OrganizationsLogsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/logs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.organizations.logs.list" call.
// Exactly one of *ListLogsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OrganizationsLogsListCall) Do(opts ...googleapi.CallOption) (*ListLogsResponse, error) {
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
	ret := &ListLogsResponse{
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
	//   "description": "Lists the logs in projects, organizations, folders, or billing accounts. Only logs that have entries are listed.",
	//   "flatPath": "v2beta1/organizations/{organizationsId}/logs",
	//   "httpMethod": "GET",
	//   "id": "logging.organizations.logs.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The resource name that owns the logs:\n\"projects/[PROJECT_ID]\"\n\"organizations/[ORGANIZATION_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]\"\n\"folders/[FOLDER_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^organizations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/logs",
	//   "response": {
	//     "$ref": "ListLogsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OrganizationsLogsListCall) Pages(ctx context.Context, f func(*ListLogsResponse) error) error {
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

// method id "logging.projects.logs.delete":

type ProjectsLogsDeleteCall struct {
	s          *Service
	logName    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes all the log entries in a log. The log reappears if it
// receives new entries. Log entries written shortly before the delete
// operation might not be deleted.
func (r *ProjectsLogsService) Delete(logName string) *ProjectsLogsDeleteCall {
	c := &ProjectsLogsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.logName = logName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsDeleteCall) Fields(s ...googleapi.Field) *ProjectsLogsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsDeleteCall) Context(ctx context.Context) *ProjectsLogsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLogsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLogsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+logName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"logName": c.logName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.logs.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes all the log entries in a log. The log reappears if it receives new entries. Log entries written shortly before the delete operation might not be deleted.",
	//   "flatPath": "v2beta1/projects/{projectsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete:\n\"projects/[PROJECT_ID]/logs/[LOG_ID]\"\n\"organizations/[ORGANIZATION_ID]/logs/[LOG_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]\"\n\"folders/[FOLDER_ID]/logs/[LOG_ID]\"\n[LOG_ID] must be URL-encoded. For example, \"projects/my-project-id/logs/syslog\", \"organizations/1234567890/logs/cloudresourcemanager.googleapis.com%2Factivity\". For more information about log names, see LogEntry.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/logs/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+logName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.projects.logs.list":

type ProjectsLogsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the logs in projects, organizations, folders, or billing
// accounts. Only logs that have entries are listed.
func (r *ProjectsLogsService) List(parent string) *ProjectsLogsListCall {
	c := &ProjectsLogsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *ProjectsLogsListCall) PageSize(pageSize int64) *ProjectsLogsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *ProjectsLogsListCall) PageToken(pageToken string) *ProjectsLogsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsListCall) Fields(s ...googleapi.Field) *ProjectsLogsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogsListCall) IfNoneMatch(entityTag string) *ProjectsLogsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsListCall) Context(ctx context.Context) *ProjectsLogsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLogsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLogsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/logs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.logs.list" call.
// Exactly one of *ListLogsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogsListCall) Do(opts ...googleapi.CallOption) (*ListLogsResponse, error) {
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
	ret := &ListLogsResponse{
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
	//   "description": "Lists the logs in projects, organizations, folders, or billing accounts. Only logs that have entries are listed.",
	//   "flatPath": "v2beta1/projects/{projectsId}/logs",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logs.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The resource name that owns the logs:\n\"projects/[PROJECT_ID]\"\n\"organizations/[ORGANIZATION_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]\"\n\"folders/[FOLDER_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/logs",
	//   "response": {
	//     "$ref": "ListLogsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsLogsListCall) Pages(ctx context.Context, f func(*ListLogsResponse) error) error {
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

// method id "logging.projects.metrics.create":

type ProjectsMetricsCreateCall struct {
	s          *Service
	parent     string
	logmetric  *LogMetric
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a logs-based metric.
func (r *ProjectsMetricsService) Create(parent string, logmetric *LogMetric) *ProjectsMetricsCreateCall {
	c := &ProjectsMetricsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.logmetric = logmetric
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMetricsCreateCall) Fields(s ...googleapi.Field) *ProjectsMetricsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsMetricsCreateCall) Context(ctx context.Context) *ProjectsMetricsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsMetricsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsMetricsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logmetric)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/metrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.metrics.create" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsCreateCall) Do(opts ...googleapi.CallOption) (*LogMetric, error) {
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
	ret := &LogMetric{
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
	//   "description": "Creates a logs-based metric.",
	//   "flatPath": "v2beta1/projects/{projectsId}/metrics",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.metrics.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The resource name of the project in which to create the metric:\n\"projects/[PROJECT_ID]\"\nThe new metric must be provided in the request.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/metrics",
	//   "request": {
	//     "$ref": "LogMetric"
	//   },
	//   "response": {
	//     "$ref": "LogMetric"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "logging.projects.metrics.delete":

type ProjectsMetricsDeleteCall struct {
	s          *Service
	metricName string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a logs-based metric.
func (r *ProjectsMetricsService) Delete(metricName string) *ProjectsMetricsDeleteCall {
	c := &ProjectsMetricsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.metricName = metricName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMetricsDeleteCall) Fields(s ...googleapi.Field) *ProjectsMetricsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsMetricsDeleteCall) Context(ctx context.Context) *ProjectsMetricsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsMetricsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsMetricsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+metricName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"metricName": c.metricName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.metrics.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsMetricsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a logs-based metric.",
	//   "flatPath": "v2beta1/projects/{projectsId}/metrics/{metricsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.metrics.delete",
	//   "parameterOrder": [
	//     "metricName"
	//   ],
	//   "parameters": {
	//     "metricName": {
	//       "description": "The resource name of the metric to delete:\n\"projects/[PROJECT_ID]/metrics/[METRIC_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/metrics/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+metricName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "logging.projects.metrics.get":

type ProjectsMetricsGetCall struct {
	s            *Service
	metricName   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a logs-based metric.
func (r *ProjectsMetricsService) Get(metricName string) *ProjectsMetricsGetCall {
	c := &ProjectsMetricsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.metricName = metricName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMetricsGetCall) Fields(s ...googleapi.Field) *ProjectsMetricsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsMetricsGetCall) IfNoneMatch(entityTag string) *ProjectsMetricsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsMetricsGetCall) Context(ctx context.Context) *ProjectsMetricsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsMetricsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsMetricsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+metricName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"metricName": c.metricName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.metrics.get" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsGetCall) Do(opts ...googleapi.CallOption) (*LogMetric, error) {
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
	ret := &LogMetric{
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
	//   "description": "Gets a logs-based metric.",
	//   "flatPath": "v2beta1/projects/{projectsId}/metrics/{metricsId}",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.metrics.get",
	//   "parameterOrder": [
	//     "metricName"
	//   ],
	//   "parameters": {
	//     "metricName": {
	//       "description": "The resource name of the desired metric:\n\"projects/[PROJECT_ID]/metrics/[METRIC_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/metrics/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+metricName}",
	//   "response": {
	//     "$ref": "LogMetric"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.metrics.list":

type ProjectsMetricsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists logs-based metrics.
func (r *ProjectsMetricsService) List(parent string) *ProjectsMetricsListCall {
	c := &ProjectsMetricsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *ProjectsMetricsListCall) PageSize(pageSize int64) *ProjectsMetricsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *ProjectsMetricsListCall) PageToken(pageToken string) *ProjectsMetricsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMetricsListCall) Fields(s ...googleapi.Field) *ProjectsMetricsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsMetricsListCall) IfNoneMatch(entityTag string) *ProjectsMetricsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsMetricsListCall) Context(ctx context.Context) *ProjectsMetricsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsMetricsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsMetricsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/metrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.metrics.list" call.
// Exactly one of *ListLogMetricsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogMetricsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsMetricsListCall) Do(opts ...googleapi.CallOption) (*ListLogMetricsResponse, error) {
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
	ret := &ListLogMetricsResponse{
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
	//   "description": "Lists logs-based metrics.",
	//   "flatPath": "v2beta1/projects/{projectsId}/metrics",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.metrics.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The name of the project containing the metrics:\n\"projects/[PROJECT_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/metrics",
	//   "response": {
	//     "$ref": "ListLogMetricsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsMetricsListCall) Pages(ctx context.Context, f func(*ListLogMetricsResponse) error) error {
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

// method id "logging.projects.metrics.update":

type ProjectsMetricsUpdateCall struct {
	s          *Service
	metricName string
	logmetric  *LogMetric
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Creates or updates a logs-based metric.
func (r *ProjectsMetricsService) Update(metricName string, logmetric *LogMetric) *ProjectsMetricsUpdateCall {
	c := &ProjectsMetricsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.metricName = metricName
	c.logmetric = logmetric
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsMetricsUpdateCall) Fields(s ...googleapi.Field) *ProjectsMetricsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsMetricsUpdateCall) Context(ctx context.Context) *ProjectsMetricsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsMetricsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsMetricsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logmetric)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+metricName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"metricName": c.metricName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.metrics.update" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsUpdateCall) Do(opts ...googleapi.CallOption) (*LogMetric, error) {
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
	ret := &LogMetric{
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
	//   "description": "Creates or updates a logs-based metric.",
	//   "flatPath": "v2beta1/projects/{projectsId}/metrics/{metricsId}",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.metrics.update",
	//   "parameterOrder": [
	//     "metricName"
	//   ],
	//   "parameters": {
	//     "metricName": {
	//       "description": "The resource name of the metric to update:\n\"projects/[PROJECT_ID]/metrics/[METRIC_ID]\"\nThe updated metric must be provided in the request and it's name field must be the same as [METRIC_ID] If the metric does not exist in [PROJECT_ID], then a new metric is created.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/metrics/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+metricName}",
	//   "request": {
	//     "$ref": "LogMetric"
	//   },
	//   "response": {
	//     "$ref": "LogMetric"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "logging.projects.sinks.create":

type ProjectsSinksCreateCall struct {
	s          *Service
	parent     string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a sink that exports specified log entries to a
// destination. The export of newly-ingested log entries begins
// immediately, unless the sink's writer_identity is not permitted to
// write to the destination. A sink can export log entries only from the
// resource owning the sink.
func (r *ProjectsSinksService) Create(parent string, logsink *LogSink) *ProjectsSinksCreateCall {
	c := &ProjectsSinksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.logsink = logsink
	return c
}

// UniqueWriterIdentity sets the optional parameter
// "uniqueWriterIdentity": Determines the kind of IAM identity returned
// as writer_identity in the new sink. If this value is omitted or set
// to false, and if the sink's parent is a project, then the value
// returned as writer_identity is the same group or service account used
// by Stackdriver Logging before the addition of writer identities to
// this API. The sink's destination must be in the same project as the
// sink itself.If this field is set to true, or if the sink is owned by
// a non-project resource such as an organization, then the value of
// writer_identity will be a unique service account used only for
// exports from the new sink. For more information, see writer_identity
// in LogSink.
func (c *ProjectsSinksCreateCall) UniqueWriterIdentity(uniqueWriterIdentity bool) *ProjectsSinksCreateCall {
	c.urlParams_.Set("uniqueWriterIdentity", fmt.Sprint(uniqueWriterIdentity))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSinksCreateCall) Fields(s ...googleapi.Field) *ProjectsSinksCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSinksCreateCall) Context(ctx context.Context) *ProjectsSinksCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsSinksCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsSinksCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.sinks.create" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksCreateCall) Do(opts ...googleapi.CallOption) (*LogSink, error) {
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
	ret := &LogSink{
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
	//   "description": "Creates a sink that exports specified log entries to a destination. The export of newly-ingested log entries begins immediately, unless the sink's writer_identity is not permitted to write to the destination. A sink can export log entries only from the resource owning the sink.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.sinks.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required. The resource in which to create the sink:\n\"projects/[PROJECT_ID]\"\n\"organizations/[ORGANIZATION_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]\"\n\"folders/[FOLDER_ID]\"\nExamples: \"projects/my-logging-project\", \"organizations/123456789\".",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "uniqueWriterIdentity": {
	//       "description": "Optional. Determines the kind of IAM identity returned as writer_identity in the new sink. If this value is omitted or set to false, and if the sink's parent is a project, then the value returned as writer_identity is the same group or service account used by Stackdriver Logging before the addition of writer identities to this API. The sink's destination must be in the same project as the sink itself.If this field is set to true, or if the sink is owned by a non-project resource such as an organization, then the value of writer_identity will be a unique service account used only for exports from the new sink. For more information, see writer_identity in LogSink.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/sinks",
	//   "request": {
	//     "$ref": "LogSink"
	//   },
	//   "response": {
	//     "$ref": "LogSink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.projects.sinks.delete":

type ProjectsSinksDeleteCall struct {
	s          *Service
	sinkNameid string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a sink. If the sink has a unique writer_identity,
// then that service account is also deleted.
func (r *ProjectsSinksService) Delete(sinkNameid string) *ProjectsSinksDeleteCall {
	c := &ProjectsSinksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.sinkNameid = sinkNameid
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSinksDeleteCall) Fields(s ...googleapi.Field) *ProjectsSinksDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSinksDeleteCall) Context(ctx context.Context) *ProjectsSinksDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsSinksDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsSinksDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+sinkName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"sinkName": c.sinkNameid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.sinks.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a sink. If the sink has a unique writer_identity, then that service account is also deleted.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks/{sinksId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.sinks.delete",
	//   "parameterOrder": [
	//     "sinkName"
	//   ],
	//   "parameters": {
	//     "sinkName": {
	//       "description": "Required. The full resource name of the sink to delete, including the parent resource and the sink identifier:\n\"projects/[PROJECT_ID]/sinks/[SINK_ID]\"\n\"organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]\"\n\"folders/[FOLDER_ID]/sinks/[SINK_ID]\"\nExample: \"projects/my-project-id/sinks/my-sink-id\".",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/sinks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+sinkName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.projects.sinks.get":

type ProjectsSinksGetCall struct {
	s            *Service
	sinkName     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a sink.
func (r *ProjectsSinksService) Get(sinkName string) *ProjectsSinksGetCall {
	c := &ProjectsSinksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.sinkName = sinkName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSinksGetCall) Fields(s ...googleapi.Field) *ProjectsSinksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsSinksGetCall) IfNoneMatch(entityTag string) *ProjectsSinksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSinksGetCall) Context(ctx context.Context) *ProjectsSinksGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsSinksGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsSinksGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+sinkName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"sinkName": c.sinkName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.sinks.get" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksGetCall) Do(opts ...googleapi.CallOption) (*LogSink, error) {
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
	ret := &LogSink{
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
	//   "description": "Gets a sink.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks/{sinksId}",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.sinks.get",
	//   "parameterOrder": [
	//     "sinkName"
	//   ],
	//   "parameters": {
	//     "sinkName": {
	//       "description": "Required. The resource name of the sink:\n\"projects/[PROJECT_ID]/sinks/[SINK_ID]\"\n\"organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]\"\n\"folders/[FOLDER_ID]/sinks/[SINK_ID]\"\nExample: \"projects/my-project-id/sinks/my-sink-id\".",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/sinks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+sinkName}",
	//   "response": {
	//     "$ref": "LogSink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.sinks.list":

type ProjectsSinksListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists sinks.
func (r *ProjectsSinksService) List(parent string) *ProjectsSinksListCall {
	c := &ProjectsSinksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request. Non-positive values are
// ignored. The presence of nextPageToken in the response indicates that
// more results might be available.
func (c *ProjectsSinksListCall) PageSize(pageSize int64) *ProjectsSinksListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the preceding call to this
// method. pageToken must be the value of nextPageToken from the
// previous response. The values of other method parameters should be
// identical to those in the previous call.
func (c *ProjectsSinksListCall) PageToken(pageToken string) *ProjectsSinksListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSinksListCall) Fields(s ...googleapi.Field) *ProjectsSinksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsSinksListCall) IfNoneMatch(entityTag string) *ProjectsSinksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSinksListCall) Context(ctx context.Context) *ProjectsSinksListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsSinksListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsSinksListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+parent}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.sinks.list" call.
// Exactly one of *ListSinksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListSinksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsSinksListCall) Do(opts ...googleapi.CallOption) (*ListSinksResponse, error) {
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
	ret := &ListSinksResponse{
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
	//   "description": "Lists sinks.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.sinks.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request. Non-positive values are ignored. The presence of nextPageToken in the response indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the preceding call to this method. pageToken must be the value of nextPageToken from the previous response. The values of other method parameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The parent resource whose sinks are to be listed:\n\"projects/[PROJECT_ID]\"\n\"organizations/[ORGANIZATION_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]\"\n\"folders/[FOLDER_ID]\"\n",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+parent}/sinks",
	//   "response": {
	//     "$ref": "ListSinksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsSinksListCall) Pages(ctx context.Context, f func(*ListSinksResponse) error) error {
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

// method id "logging.projects.sinks.update":

type ProjectsSinksUpdateCall struct {
	s          *Service
	sinkNameid string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a sink. This method replaces the following fields in
// the existing sink with values from the new sink: destination, and
// filter. The updated sink might also have a new writer_identity; see
// the unique_writer_identity field.
func (r *ProjectsSinksService) Update(sinkNameid string, logsink *LogSink) *ProjectsSinksUpdateCall {
	c := &ProjectsSinksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.sinkNameid = sinkNameid
	c.logsink = logsink
	return c
}

// UniqueWriterIdentity sets the optional parameter
// "uniqueWriterIdentity": See sinks.create for a description of this
// field. When updating a sink, the effect of this field on the value of
// writer_identity in the updated sink depends on both the old and new
// values of this field:
// If the old and new values of this field are both false or both true,
// then there is no change to the sink's writer_identity.
// If the old value is false and the new value is true, then
// writer_identity is changed to a unique service account.
// It is an error if the old value is true and the new value is set to
// false or defaulted to false.
func (c *ProjectsSinksUpdateCall) UniqueWriterIdentity(uniqueWriterIdentity bool) *ProjectsSinksUpdateCall {
	c.urlParams_.Set("uniqueWriterIdentity", fmt.Sprint(uniqueWriterIdentity))
	return c
}

// UpdateMask sets the optional parameter "updateMask": Field mask that
// specifies the fields in sink that need an update. A sink field will
// be overwritten if, and only if, it is in the update mask. name and
// output only fields cannot be updated.An empty updateMask is
// temporarily treated as using the following mask for backwards
// compatibility purposes:  destination,filter,includeChildren At some
// point in the future, behavior will be removed and specifying an empty
// updateMask will be an error.For a detailed FieldMask definition, see
// https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#fieldmaskExample:
// updateMask=filter.
func (c *ProjectsSinksUpdateCall) UpdateMask(updateMask string) *ProjectsSinksUpdateCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSinksUpdateCall) Fields(s ...googleapi.Field) *ProjectsSinksUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSinksUpdateCall) Context(ctx context.Context) *ProjectsSinksUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsSinksUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsSinksUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta1/{+sinkName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"sinkName": c.sinkNameid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "logging.projects.sinks.update" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksUpdateCall) Do(opts ...googleapi.CallOption) (*LogSink, error) {
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
	ret := &LogSink{
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
	//   "description": "Updates a sink. This method replaces the following fields in the existing sink with values from the new sink: destination, and filter. The updated sink might also have a new writer_identity; see the unique_writer_identity field.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks/{sinksId}",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.sinks.update",
	//   "parameterOrder": [
	//     "sinkName"
	//   ],
	//   "parameters": {
	//     "sinkName": {
	//       "description": "Required. The full resource name of the sink to update, including the parent resource and the sink identifier:\n\"projects/[PROJECT_ID]/sinks/[SINK_ID]\"\n\"organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]\"\n\"billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]\"\n\"folders/[FOLDER_ID]/sinks/[SINK_ID]\"\nExample: \"projects/my-project-id/sinks/my-sink-id\".",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/sinks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "uniqueWriterIdentity": {
	//       "description": "Optional. See sinks.create for a description of this field. When updating a sink, the effect of this field on the value of writer_identity in the updated sink depends on both the old and new values of this field:\nIf the old and new values of this field are both false or both true, then there is no change to the sink's writer_identity.\nIf the old value is false and the new value is true, then writer_identity is changed to a unique service account.\nIt is an error if the old value is true and the new value is set to false or defaulted to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "updateMask": {
	//       "description": "Optional. Field mask that specifies the fields in sink that need an update. A sink field will be overwritten if, and only if, it is in the update mask. name and output only fields cannot be updated.An empty updateMask is temporarily treated as using the following mask for backwards compatibility purposes:  destination,filter,includeChildren At some point in the future, behavior will be removed and specifying an empty updateMask will be an error.For a detailed FieldMask definition, see https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#fieldmaskExample: updateMask=filter.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta1/{+sinkName}",
	//   "request": {
	//     "$ref": "LogSink"
	//   },
	//   "response": {
	//     "$ref": "LogSink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}
