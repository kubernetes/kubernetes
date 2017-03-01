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

// HttpRequest: A common proto for logging HTTP requests. Only contains
// semantics
// defined by the HTTP specification. Product-specific
// logging
// information MUST be defined in a separate message.
type HttpRequest struct {
	// CacheFillBytes: The number of HTTP response bytes inserted into
	// cache. Set only when a
	// cache fill was attempted.
	CacheFillBytes int64 `json:"cacheFillBytes,omitempty,string"`

	// CacheHit: Whether or not an entity was served from cache
	// (with or without validation).
	CacheHit bool `json:"cacheHit,omitempty"`

	// CacheLookup: Whether or not a cache lookup was attempted.
	CacheLookup bool `json:"cacheLookup,omitempty"`

	// CacheValidatedWithOriginServer: Whether or not the response was
	// validated with the origin server before
	// being served from cache. This field is only meaningful if `cache_hit`
	// is
	// True.
	CacheValidatedWithOriginServer bool `json:"cacheValidatedWithOriginServer,omitempty"`

	// Latency: The request processing latency on the server, from the time
	// the request was
	// received until the response was sent.
	Latency string `json:"latency,omitempty"`

	// Referer: The referer URL of the request, as defined in
	// [HTTP/1.1 Header Field
	// Definitions](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html).
	Referer string `json:"referer,omitempty"`

	// RemoteIp: The IP address (IPv4 or IPv6) of the client that issued the
	// HTTP
	// request. Examples: "192.168.1.1", "FE80::0202:B3FF:FE1E:8329".
	RemoteIp string `json:"remoteIp,omitempty"`

	// RequestMethod: The request method. Examples: "GET", "HEAD",
	// "PUT", "POST".
	RequestMethod string `json:"requestMethod,omitempty"`

	// RequestSize: The size of the HTTP request message in bytes, including
	// the request
	// headers and the request body.
	RequestSize int64 `json:"requestSize,omitempty,string"`

	// RequestUrl: The scheme (http, https), the host name, the path and the
	// query
	// portion of the URL that was requested.
	// Example: "http://example.com/some/info?color=red".
	RequestUrl string `json:"requestUrl,omitempty"`

	// ResponseSize: The size of the HTTP response message sent back to the
	// client, in bytes,
	// including the response headers and the response body.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// ServerIp: The IP address (IPv4 or IPv6) of the origin server that the
	// request was
	// sent to.
	ServerIp string `json:"serverIp,omitempty"`

	// Status: The response code indicating the status of
	// response.
	// Examples: 200, 404.
	Status int64 `json:"status,omitempty"`

	// UserAgent: The user agent sent by the client. Example:
	// "Mozilla/4.0 (compatible; MSIE 6.0; Windows 98; Q312461; .NET CLR
	// 1.0.3705)".
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

// ListLogEntriesRequest: The parameters to `ListLogEntries`.
type ListLogEntriesRequest struct {
	// Filter: Optional. A filter that chooses which log entries to return.
	// See [Advanced
	// Logs Filters](/logging/docs/view/advanced_filters).  Only log entries
	// that
	// match the filter are returned.  An empty filter matches all log
	// entries.
	// The maximum length of the filter is 20000 characters.
	Filter string `json:"filter,omitempty"`

	// OrderBy: Optional. How the results should be sorted.  Presently, the
	// only permitted
	// values are "timestamp asc" (default) and "timestamp desc". The
	// first
	// option returns entries in order of increasing values
	// of
	// `LogEntry.timestamp` (oldest first), and the second option returns
	// entries
	// in order of decreasing timestamps (newest first).  Entries with
	// equal
	// timestamps are returned in order of `LogEntry.insertId`.
	OrderBy string `json:"orderBy,omitempty"`

	// PageSize: Optional. The maximum number of results to return from this
	// request.
	// Non-positive values are ignored.  The presence of `nextPageToken` in
	// the
	// response indicates that more results might be available.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: Optional. If present, then retrieve the next batch of
	// results from the
	// preceding call to this method.  `pageToken` must be the value
	// of
	// `nextPageToken` from the previous response.  The values of other
	// method
	// parameters should be identical to those in the previous call.
	PageToken string `json:"pageToken,omitempty"`

	// ProjectIds: Deprecated. One or more project identifiers or project
	// numbers from which
	// to retrieve log entries.  Examples: "my-project-1A",
	// "1234567890". If
	// present, these project identifiers are converted to resource format
	// and
	// added to the list of resources in `resourceNames`. Callers should
	// use
	// `resourceNames` rather than this parameter.
	ProjectIds []string `json:"projectIds,omitempty"`

	// ResourceNames: Required. One or more cloud resources from which to
	// retrieve log entries.
	// Example: "projects/my-project-1A", "projects/1234567890".
	// Projects
	// listed in `projectIds` are added to this list.
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

// ListLogEntriesResponse: Result returned from `ListLogEntries`.
type ListLogEntriesResponse struct {
	// Entries: A list of log entries.
	Entries []*LogEntry `json:"entries,omitempty"`

	// NextPageToken: If there might be more results than appear in this
	// response, then
	// `nextPageToken` is included.  To get the next set of results, call
	// this
	// method again using the value of `nextPageToken` as `pageToken`.
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
	// response, then
	// `nextPageToken` is included.  To get the next set of results, call
	// this
	// method again using the value of `nextPageToken` as `pageToken`.
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

// ListMonitoredResourceDescriptorsResponse: Result returned from
// ListMonitoredResourceDescriptors.
type ListMonitoredResourceDescriptorsResponse struct {
	// NextPageToken: If there might be more results than appear in this
	// response, then
	// `nextPageToken` is included.  To get the next set of results, call
	// this
	// method again using the value of `nextPageToken` as `pageToken`.
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

// ListSinksResponse: Result returned from `ListSinks`.
type ListSinksResponse struct {
	// NextPageToken: If there might be more results than appear in this
	// response, then
	// `nextPageToken` is included.  To get the next set of results, call
	// the same
	// method again using the value of `nextPageToken` as `pageToken`.
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
	// with this
	// log entry, if applicable.
	HttpRequest *HttpRequest `json:"httpRequest,omitempty"`

	// InsertId: Optional. A unique ID for the log entry. If you provide
	// this
	// field, the logging service considers other log entries in the
	// same project with the same ID as duplicates which can be removed.
	// If
	// omitted, Stackdriver Logging will generate a unique ID for this
	// log entry.
	InsertId string `json:"insertId,omitempty"`

	// JsonPayload: The log entry payload, represented as a structure
	// that
	// is expressed as a JSON object.
	JsonPayload googleapi.RawMessage `json:"jsonPayload,omitempty"`

	// Labels: Optional. A set of user-defined (key, value) data that
	// provides additional
	// information about the log entry.
	Labels map[string]string `json:"labels,omitempty"`

	// LogName: Required. The resource name of the log to which this log
	// entry
	// belongs. The format of the name
	// is
	// "projects/<project-id>/logs/<log-id>".
	// Examples:
	// "projects/my-projectid/logs/syslog",
	// "projects/my-project
	// id/logs/library.googleapis.com%2Fbook_log".
	//
	// The log ID part of resource name must be less than 512
	// characters
	// long and can only include the following characters: upper and
	// lower case alphanumeric characters: [A-Za-z0-9]; and
	// punctuation
	// characters: forward-slash, underscore, hyphen, and
	// period.
	// Forward-slash (`/`) characters in the log ID must be URL-encoded.
	LogName string `json:"logName,omitempty"`

	// Operation: Optional. Information about an operation associated with
	// the log entry, if
	// applicable.
	Operation *LogEntryOperation `json:"operation,omitempty"`

	// ProtoPayload: The log entry payload, represented as a protocol
	// buffer.  Some
	// Google Cloud Platform services use this field for their log
	// entry payloads.
	ProtoPayload googleapi.RawMessage `json:"protoPayload,omitempty"`

	// Resource: Required. The monitored resource associated with this log
	// entry.
	// Example: a log entry that reports a database error would
	// be
	// associated with the monitored resource designating the
	// particular
	// database that reported the error.
	Resource *MonitoredResource `json:"resource,omitempty"`

	// Severity: Optional. The severity of the log entry. The default value
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

	// TextPayload: The log entry payload, represented as a Unicode string
	// (UTF-8).
	TextPayload string `json:"textPayload,omitempty"`

	// Timestamp: Optional. The time the event described by the log entry
	// occurred.  If
	// omitted, Stackdriver Logging will use the time the log entry is
	// received.
	Timestamp string `json:"timestamp,omitempty"`

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
// long-running operation with which
// a log entry is associated.
type LogEntryOperation struct {
	// First: Optional. Set this to True if this is the first log entry in
	// the operation.
	First bool `json:"first,omitempty"`

	// Id: Optional. An arbitrary operation identifier. Log entries with
	// the
	// same identifier are assumed to be part of the same operation.
	Id string `json:"id,omitempty"`

	// Last: Optional. Set this to True if this is the last log entry in the
	// operation.
	Last bool `json:"last,omitempty"`

	// Producer: Optional. An arbitrary producer identifier. The combination
	// of
	// `id` and `producer` must be globally unique.  Examples for
	// `producer`:
	// "MyDivision.MyBigCompany.com",
	// "github.com/MyProject/MyApplication".
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
	// shut down, or
	// a configuration change.
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

// LogMetric: Describes a logs-based metric.  The value of the metric is
// the
// number of log entries that match a logs filter.
type LogMetric struct {
	// Description: Optional. A description of this metric, which is used in
	// documentation.
	Description string `json:"description,omitempty"`

	// Filter: Required. An [advanced logs
	// filter](/logging/docs/view/advanced_filters).
	// Example: "resource.type=gae_app AND severity>=ERROR".
	// The maximum length of the filter is 20000 characters.
	Filter string `json:"filter,omitempty"`

	// Name: Required. The client-assigned metric identifier.
	// Example:
	// "severe_errors".  Metric identifiers are limited to 100
	// characters and can include only the following characters:
	// `A-Z`,
	// `a-z`, `0-9`, and the special characters `_-.,+!*',()%/`.
	// The
	// forward-slash character (`/`) denotes a hierarchy of name pieces,
	// and it cannot be the first character of the name.  The '%'
	// character
	// is used to URL encode unsafe and reserved characters and must
	// be
	// followed by two hexadecimal digits according to RFC 1738.
	Name string `json:"name,omitempty"`

	// Version: Output only. The API version that created or updated this
	// metric.
	// The version also dictates the syntax of the filter expression. When a
	// value
	// for this field is missing, the default value of V2 should be assumed.
	//
	// Possible values:
	//   "V2" - Stackdriver Logging API v2.
	//   "V1" - Stackdriver Logging API v1.
	Version string `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *LogMetric) MarshalJSON() ([]byte, error) {
	type noMethod LogMetric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LogSink: Describes a sink used to export log entries outside of
// Stackdriver Logging.
// A logs filter controls which log entries are exported.  Sinks can
// have a
// start time and an end time; these can be used to place log entries
// from an
// exact time range into a particular destination.  If both `start_time`
// and
// `end_time` are present, then `start_time` must be less than
// `end_time`.
type LogSink struct {
	// Destination: Required. The export destination. See
	// [Exporting Logs With
	// Sinks](/logging/docs/api/tasks/exporting-logs).
	// Examples:
	//
	//     "storage.googleapis.com/my-gcs-bucket"
	//
	// "bigquery.googleapis.com/projects/my-project-id/datasets/my-dataset"
	//     "pubsub.googleapis.com/projects/my-project/topics/my-topic"
	Destination string `json:"destination,omitempty"`

	// EndTime: Optional. Time at which this sink will stop exporting log
	// entries.  If this
	// value is present, then log entries are exported only if
	// `entry.timestamp` <
	// `end_time`.
	EndTime string `json:"endTime,omitempty"`

	// Filter: Optional. An [advanced logs
	// filter](/logging/docs/view/advanced_filters).
	// Only log entries matching the filter are exported. The filter
	// must be consistent with the log entry format specified by
	// the
	// `outputVersionFormat` parameter, regardless of the format of the
	// log entry that was originally written to Stackdriver Logging.
	// Example filter (V2 format):
	//
	//     logName=projects/my-projectid/logs/syslog AND severity>=ERROR
	// The maximum length of the filter is 20000 characters.
	Filter string `json:"filter,omitempty"`

	// Name: Required. The client-assigned sink identifier, unique within
	// the
	// project. Example: "my-syslog-errors-to-pubsub".  Sink identifiers
	// are
	// limited to 1000 characters and can include only the following
	// characters:
	// `A-Z`, `a-z`, `0-9`, and the special characters `_-.`.  The maximum
	// length
	// of the name is 100 characters.
	Name string `json:"name,omitempty"`

	// OutputVersionFormat: Optional. The log entry version to use for this
	// sink's exported log
	// entries.  This version does not have to correspond to the version of
	// the
	// log entry that was written to Stackdriver Logging. If omitted, the
	// v2 format is used.
	//
	// Possible values:
	//   "VERSION_FORMAT_UNSPECIFIED" - An unspecified version format will
	// default to V2.
	//   "V2" - `LogEntry` version 2 format.
	//   "V1" - `LogEntry` version 1 format.
	OutputVersionFormat string `json:"outputVersionFormat,omitempty"`

	// StartTime: Optional. The time at which this sink will begin exporting
	// log entries.  If
	// this value is present, then log entries are exported only if
	// `start_time`
	// <=`entry.timestamp`.
	StartTime string `json:"startTime,omitempty"`

	// WriterIdentity: Output only. An IAM identity&mdash;a service account
	// or group&mdash;that
	// will write exported log entries to the destination on behalf of
	// Stackdriver
	// Logging. You must grant this identity write-access to the
	// destination.
	// Consult the destination service's documentation to determine the
	// exact role
	// that must be granted.
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

// MonitoredResource: An object representing a resource that can be used
// for monitoring, logging,
// billing, or other purposes. Examples include virtual machine
// instances,
// databases, and storage devices such as disks. The `type` field
// identifies a
// MonitoredResourceDescriptor object that describes the
// resource's
// schema. Information in the `labels` field identifies the actual
// resource and
// its attributes according to the schema. For example, a particular
// Compute
// Engine VM instance could be represented by the following object,
// because the
// MonitoredResourceDescriptor for "gce_instance" has
// labels
// "instance_id" and "zone":
//
//     { "type": "gce_instance",
//       "labels": { "instance_id": "12345678901234",
//                   "zone": "us-central1-a" }}
type MonitoredResource struct {
	// Labels: Required. Values for all of the labels listed in the
	// associated monitored
	// resource descriptor. For example, Cloud SQL databases use the
	// labels
	// "database_id" and "zone".
	Labels map[string]string `json:"labels,omitempty"`

	// Type: Required. The monitored resource type. This field must
	// match
	// the `type` field of a MonitoredResourceDescriptor object.
	// For
	// example, the type of a Cloud SQL database is "cloudsql_database".
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
// MonitoredResource object using a
// type name and a set of labels.  For example, the monitored
// resource
// descriptor for Google Compute Engine VM instances has a type
// of
// "gce_instance" and specifies the use of the labels "instance_id"
// and
// "zone" to identify particular VM instances.
//
// Different APIs can support different monitored resource types. APIs
// generally
// provide a `list` method that returns the monitored resource
// descriptors used
// by the API.
type MonitoredResourceDescriptor struct {
	// Description: Optional. A detailed description of the monitored
	// resource type that might
	// be used in documentation.
	Description string `json:"description,omitempty"`

	// DisplayName: Optional. A concise name for the monitored resource type
	// that might be
	// displayed in user interfaces. It should be a Title Cased Noun
	// Phrase,
	// without any article or other determiners. For example,
	// "Google Cloud SQL Database".
	DisplayName string `json:"displayName,omitempty"`

	// Labels: Required. A set of labels used to describe instances of this
	// monitored
	// resource type. For example, an individual Google Cloud SQL database
	// is
	// identified by values for the labels "database_id" and "zone".
	Labels []*LabelDescriptor `json:"labels,omitempty"`

	// Name: Optional. The resource name of the monitored resource
	// descriptor:
	// "projects/{project_id}/monitoredResourceDescriptors/{type
	// }" where
	// {type} is the value of the `type` field in this object
	// and
	// {project_id} is a project ID that provides API-specific context
	// for
	// accessing the type.  APIs that do not use project information can use
	// the
	// resource name format "monitoredResourceDescriptors/{type}".
	Name string `json:"name,omitempty"`

	// Type: Required. The monitored resource type. For example, the
	// type
	// "cloudsql_database" represents databases in Google Cloud SQL.
	// The maximum length of this value is 256 characters.
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
// an App Engine
// application.
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

	// First: Whether this is the first `RequestLog` entry for this request.
	//  If an
	// active request has several `RequestLog` entries written to
	// Stackdriver
	// Logging, then this field will be set for one of them.
	First bool `json:"first,omitempty"`

	// Host: Internet host and port number of the resource being requested.
	Host string `json:"host,omitempty"`

	// HttpVersion: HTTP version of request. Example: "HTTP/1.1".
	HttpVersion string `json:"httpVersion,omitempty"`

	// InstanceId: An identifier for the instance that handled the request.
	InstanceId string `json:"instanceId,omitempty"`

	// InstanceIndex: If the instance processing this request belongs to a
	// manually scaled
	// module, then this is the 0-based index of the instance. Otherwise,
	// this
	// value is -1.
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

	// Method: Request method. Example: "GET", "HEAD", "PUT",
	// "POST", "DELETE".
	Method string `json:"method,omitempty"`

	// ModuleId: Module of the application that handled this request.
	ModuleId string `json:"moduleId,omitempty"`

	// Nickname: The logged-in user who made the request.
	//
	// Most likely, this is the part of the user's email before the `@`
	// sign.  The
	// field value is the same for different requests from the same user,
	// but
	// different users can have similar names.  This information is
	// also
	// available to the application via the App Engine Users API.
	//
	// This field will be populated starting with App Engine 1.9.21.
	Nickname string `json:"nickname,omitempty"`

	// PendingTime: Time this request spent in the pending request queue.
	PendingTime string `json:"pendingTime,omitempty"`

	// Referrer: Referrer URL of request.
	Referrer string `json:"referrer,omitempty"`

	// RequestId: Globally unique identifier for a request, which is based
	// on the request
	// start time.  Request IDs for requests which started later will
	// compare
	// greater as strings than those for requests which started earlier.
	RequestId string `json:"requestId,omitempty"`

	// Resource: Contains the path and query portion of the URL that was
	// requested. For
	// example, if the URL was "http://example.com/app?name=val", the
	// resource
	// would be "/app?name=val".  The fragment identifier, which is
	// identified by
	// the `#` character, is not included.
	Resource string `json:"resource,omitempty"`

	// ResponseSize: Size in bytes sent back to client by request.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// SourceReference: Source code for the application that handled this
	// request. There can be
	// more than one source reference per deployed application if source
	// code is
	// distributed among multiple repositories.
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

// SourceLocation: Specifies a location in a source code file.
type SourceLocation struct {
	// File: Source file name. Depending on the runtime environment, this
	// might be a
	// simple name or a fully-qualified name.
	File string `json:"file,omitempty"`

	// FunctionName: Human-readable name of the function or method being
	// invoked, with optional
	// context such as the class or package name. This information is used
	// in
	// contexts such as the logs viewer, where a file and line number are
	// less
	// meaningful. The format can vary by language. For
	// example:
	// `qual.if.ied.Class.method` (Java), `dir/package.func` (Go),
	// `function`
	// (Python).
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
// tree used to build and
// deploy an application.
type SourceReference struct {
	// Repository: Optional. A URI string identifying the
	// repository.
	// Example: "https://github.com/GoogleCloudPlatform/kubernetes.git"
	Repository string `json:"repository,omitempty"`

	// RevisionId: The canonical and persistent identifier of the deployed
	// revision.
	// Example (git): "0035781c50ec7aa23385dc841529ce8a4b70db1b"
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
	// Entries: Required. The log entries to write. Values supplied for the
	// fields
	// `log_name`, `resource`, and `labels` in this `entries.write` request
	// are
	// added to those log entries that do not provide their own values for
	// the
	// fields.
	//
	// To improve throughput and to avoid exceeding the
	// [quota limit](/logging/quota-policy) for calls to
	// `entries.write`,
	// you should write multiple log entries at once rather than
	// calling this method for each individual log entry.
	Entries []*LogEntry `json:"entries,omitempty"`

	// Labels: Optional. Default labels that are added to the `labels` field
	// of all log
	// entries in `entries`. If a log entry already has a label with the
	// same key
	// as a label in this parameter, then the log entry's label is not
	// changed.
	// See LogEntry.
	Labels map[string]string `json:"labels,omitempty"`

	// LogName: Optional. A default log resource name that is assigned to
	// all log entries
	// in `entries` that do not specify a value for `log_name`.
	// Example:
	// "projects/my-project/logs/syslog".  See
	// LogEntry.
	LogName string `json:"logName,omitempty"`

	// PartialSuccess: Optional. Whether valid entries should be written
	// even if some other
	// entries fail due to INVALID_ARGUMENT or PERMISSION_DENIED errors. If
	// any
	// entry is not written, the response status will be the error
	// associated
	// with one of the failed entries and include error details in the form
	// of
	// WriteLogEntriesPartialErrors.
	PartialSuccess bool `json:"partialSuccess,omitempty"`

	// Resource: Optional. A default monitored resource object that is
	// assigned to all log
	// entries in `entries` that do not specify a value for `resource`.
	// Example:
	//
	//     { "type": "gce_instance",
	//       "labels": {
	//         "zone": "us-central1-a", "instance_id":
	// "00000000000000000000" }}
	//
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

// WriteLogEntriesResponse: Result returned from WriteLogEntries.
// empty
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

// Delete: Deletes a log and all its log entries.
// The log will reappear if it receives new entries.
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
	//   "description": "Deletes a log and all its log entries.\nThe log will reappear if it receives new entries.",
	//   "flatPath": "v2beta1/billingAccounts/{billingAccountsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.billingAccounts.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete.  Example:\n`\"projects/my-project/logs/syslog\"`.",
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

// method id "logging.entries.list":

type EntriesListCall struct {
	s                     *Service
	listlogentriesrequest *ListLogEntriesRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// List: Lists log entries.  Use this method to retrieve log entries
// from Cloud
// Logging.  For ways to export log entries, see
// [Exporting Logs](/logging/docs/export).
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
	//   "description": "Lists log entries.  Use this method to retrieve log entries from Cloud\nLogging.  For ways to export log entries, see\n[Exporting Logs](/logging/docs/export).",
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

// method id "logging.entries.write":

type EntriesWriteCall struct {
	s                      *Service
	writelogentriesrequest *WriteLogEntriesRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Write: Writes log entries to Stackdriver Logging.  All log entries
// are
// written by this method.
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
	//   "description": "Writes log entries to Stackdriver Logging.  All log entries are\nwritten by this method.",
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

// List: Lists the monitored resource descriptors used by Stackdriver
// Logging.
func (r *MonitoredResourceDescriptorsService) List() *MonitoredResourceDescriptorsListCall {
	c := &MonitoredResourceDescriptorsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return from this request.
// Non-positive values are ignored.  The presence of `nextPageToken` in
// the
// response indicates that more results might be available.
func (c *MonitoredResourceDescriptorsListCall) PageSize(pageSize int64) *MonitoredResourceDescriptorsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the
// preceding call to this method.  `pageToken` must be the value
// of
// `nextPageToken` from the previous response.  The values of other
// method
// parameters should be identical to those in the previous call.
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
	//   "description": "Lists the monitored resource descriptors used by Stackdriver Logging.",
	//   "flatPath": "v2beta1/monitoredResourceDescriptors",
	//   "httpMethod": "GET",
	//   "id": "logging.monitoredResourceDescriptors.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Optional. The maximum number of results to return from this request.\nNon-positive values are ignored.  The presence of `nextPageToken` in the\nresponse indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the\npreceding call to this method.  `pageToken` must be the value of\n`nextPageToken` from the previous response.  The values of other method\nparameters should be identical to those in the previous call.",
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

// Delete: Deletes a log and all its log entries.
// The log will reappear if it receives new entries.
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
	//   "description": "Deletes a log and all its log entries.\nThe log will reappear if it receives new entries.",
	//   "flatPath": "v2beta1/organizations/{organizationsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.organizations.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete.  Example:\n`\"projects/my-project/logs/syslog\"`.",
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

// method id "logging.projects.logs.delete":

type ProjectsLogsDeleteCall struct {
	s          *Service
	logName    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a log and all its log entries.
// The log will reappear if it receives new entries.
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
	//   "description": "Deletes a log and all its log entries.\nThe log will reappear if it receives new entries.",
	//   "flatPath": "v2beta1/projects/{projectsId}/logs/{logsId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.logs.delete",
	//   "parameterOrder": [
	//     "logName"
	//   ],
	//   "parameters": {
	//     "logName": {
	//       "description": "Required. The resource name of the log to delete.  Example:\n`\"projects/my-project/logs/syslog\"`.",
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
	//       "description": "The resource name of the project in which to create the metric.\nExample: `\"projects/my-project-id\"`.\n\nThe new metric must be provided in the request.",
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
	//       "description": "The resource name of the metric to delete.\nExample: `\"projects/my-project-id/metrics/my-metric-id\"`.",
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
	//       "description": "The resource name of the desired metric.\nExample: `\"projects/my-project-id/metrics/my-metric-id\"`.",
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
// of results to return from this request.
// Non-positive values are ignored.  The presence of `nextPageToken` in
// the
// response indicates that more results might be available.
func (c *ProjectsMetricsListCall) PageSize(pageSize int64) *ProjectsMetricsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the
// preceding call to this method.  `pageToken` must be the value
// of
// `nextPageToken` from the previous response.  The values of other
// method
// parameters should be identical to those in the previous call.
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
	//       "description": "Optional. The maximum number of results to return from this request.\nNon-positive values are ignored.  The presence of `nextPageToken` in the\nresponse indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the\npreceding call to this method.  `pageToken` must be the value of\n`nextPageToken` from the previous response.  The values of other method\nparameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The resource name containing the metrics.\nExample: `\"projects/my-project-id\"`.",
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
	s            *Service
	metricNameid string
	logmetric    *LogMetric
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Creates or updates a logs-based metric.
func (r *ProjectsMetricsService) Update(metricNameid string, logmetric *LogMetric) *ProjectsMetricsUpdateCall {
	c := &ProjectsMetricsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.metricNameid = metricNameid
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
		"metricName": c.metricNameid,
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
	//       "description": "The resource name of the metric to update.\nExample: `\"projects/my-project-id/metrics/my-metric-id\"`.\n\nThe updated metric must be provided in the request and have the\nsame identifier that is specified in `metricName`.\nIf the metric does not exist, it is created.",
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

// Create: Creates a sink.
func (r *ProjectsSinksService) Create(parent string, logsink *LogSink) *ProjectsSinksCreateCall {
	c := &ProjectsSinksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.logsink = logsink
	return c
}

// UniqueWriterIdentity sets the optional parameter
// "uniqueWriterIdentity": Whether the sink will have a dedicated
// service account returned
// in the sink's writer_identity. Set this field to be true to
// export
// logs from one project to a different project. This field is ignored
// for
// non-project sinks (e.g. organization sinks) because those sinks
// are
// required to have dedicated service accounts.
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
	//   "description": "Creates a sink.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.sinks.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required. The resource in which to create the sink.\nExample: `\"projects/my-project-id\"`.\nThe new sink must be provided in the request.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "uniqueWriterIdentity": {
	//       "description": "Optional. Whether the sink will have a dedicated service account returned\nin the sink's writer_identity. Set this field to be true to export\nlogs from one project to a different project. This field is ignored for\nnon-project sinks (e.g. organization sinks) because those sinks are\nrequired to have dedicated service accounts.",
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

// Delete: Deletes a sink.
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
	//   "description": "Deletes a sink.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks/{sinksId}",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.sinks.delete",
	//   "parameterOrder": [
	//     "sinkName"
	//   ],
	//   "parameters": {
	//     "sinkName": {
	//       "description": "Required. The resource name of the sink to delete, including the parent\nresource and the sink identifier.  Example:\n`\"projects/my-project-id/sinks/my-sink-id\"`.  It is an error if the sink\ndoes not exist.",
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
	//       "description": "Required. The resource name of the sink to return.\nExample: `\"projects/my-project-id/sinks/my-sink-id\"`.",
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
// of results to return from this request.
// Non-positive values are ignored.  The presence of `nextPageToken` in
// the
// response indicates that more results might be available.
func (c *ProjectsSinksListCall) PageSize(pageSize int64) *ProjectsSinksListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": If present, then
// retrieve the next batch of results from the
// preceding call to this method.  `pageToken` must be the value
// of
// `nextPageToken` from the previous response.  The values of other
// method
// parameters should be identical to those in the previous call.
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
	//       "description": "Optional. The maximum number of results to return from this request.\nNon-positive values are ignored.  The presence of `nextPageToken` in the\nresponse indicates that more results might be available.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Optional. If present, then retrieve the next batch of results from the\npreceding call to this method.  `pageToken` must be the value of\n`nextPageToken` from the previous response.  The values of other method\nparameters should be identical to those in the previous call.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required. The resource name where this sink was created.\nExample: `\"projects/my-logging-project\"`.",
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

// Update: Updates or creates a sink.
func (r *ProjectsSinksService) Update(sinkNameid string, logsink *LogSink) *ProjectsSinksUpdateCall {
	c := &ProjectsSinksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.sinkNameid = sinkNameid
	c.logsink = logsink
	return c
}

// UniqueWriterIdentity sets the optional parameter
// "uniqueWriterIdentity": Whether the sink will have a dedicated
// service account returned
// in the sink's writer_identity. Set this field to be true to
// export
// logs from one project to a different project. This field is ignored
// for
// non-project sinks (e.g. organization sinks) because those sinks
// are
// required to have dedicated service accounts.
func (c *ProjectsSinksUpdateCall) UniqueWriterIdentity(uniqueWriterIdentity bool) *ProjectsSinksUpdateCall {
	c.urlParams_.Set("uniqueWriterIdentity", fmt.Sprint(uniqueWriterIdentity))
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
	//   "description": "Updates or creates a sink.",
	//   "flatPath": "v2beta1/projects/{projectsId}/sinks/{sinksId}",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.sinks.update",
	//   "parameterOrder": [
	//     "sinkName"
	//   ],
	//   "parameters": {
	//     "sinkName": {
	//       "description": "Required. The resource name of the sink to update, including the parent\nresource and the sink identifier.  If the sink does not exist, this method\ncreates the sink.  Example: `\"projects/my-project-id/sinks/my-sink-id\"`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/sinks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "uniqueWriterIdentity": {
	//       "description": "Optional. Whether the sink will have a dedicated service account returned\nin the sink's writer_identity. Set this field to be true to export\nlogs from one project to a different project. This field is ignored for\nnon-project sinks (e.g. organization sinks) because those sinks are\nrequired to have dedicated service accounts.",
	//       "location": "query",
	//       "type": "boolean"
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
