// Package logging provides access to the Google Cloud Logging API.
//
// See https://cloud.google.com/logging/docs/
//
// Usage example:
//
//   import "google.golang.org/api/logging/v1beta3"
//   ...
//   loggingService, err := logging.New(oauthHttpClient)
package logging // import "google.golang.org/api/logging/v1beta3"

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

const apiId = "logging:v1beta3"
const apiName = "logging"
const apiVersion = "v1beta3"
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
	rs.LogServices = NewProjectsLogServicesService(s)
	rs.Logs = NewProjectsLogsService(s)
	rs.Metrics = NewProjectsMetricsService(s)
	rs.Sinks = NewProjectsSinksService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	LogServices *ProjectsLogServicesService

	Logs *ProjectsLogsService

	Metrics *ProjectsMetricsService

	Sinks *ProjectsSinksService
}

func NewProjectsLogServicesService(s *Service) *ProjectsLogServicesService {
	rs := &ProjectsLogServicesService{s: s}
	rs.Indexes = NewProjectsLogServicesIndexesService(s)
	rs.Sinks = NewProjectsLogServicesSinksService(s)
	return rs
}

type ProjectsLogServicesService struct {
	s *Service

	Indexes *ProjectsLogServicesIndexesService

	Sinks *ProjectsLogServicesSinksService
}

func NewProjectsLogServicesIndexesService(s *Service) *ProjectsLogServicesIndexesService {
	rs := &ProjectsLogServicesIndexesService{s: s}
	return rs
}

type ProjectsLogServicesIndexesService struct {
	s *Service
}

func NewProjectsLogServicesSinksService(s *Service) *ProjectsLogServicesSinksService {
	rs := &ProjectsLogServicesSinksService{s: s}
	return rs
}

type ProjectsLogServicesSinksService struct {
	s *Service
}

func NewProjectsLogsService(s *Service) *ProjectsLogsService {
	rs := &ProjectsLogsService{s: s}
	rs.Entries = NewProjectsLogsEntriesService(s)
	rs.Sinks = NewProjectsLogsSinksService(s)
	return rs
}

type ProjectsLogsService struct {
	s *Service

	Entries *ProjectsLogsEntriesService

	Sinks *ProjectsLogsSinksService
}

func NewProjectsLogsEntriesService(s *Service) *ProjectsLogsEntriesService {
	rs := &ProjectsLogsEntriesService{s: s}
	return rs
}

type ProjectsLogsEntriesService struct {
	s *Service
}

func NewProjectsLogsSinksService(s *Service) *ProjectsLogsSinksService {
	rs := &ProjectsLogsSinksService{s: s}
	return rs
}

type ProjectsLogsSinksService struct {
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
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance: service Foo { rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty); } The JSON representation for `Empty` is
// empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// HttpRequest: A common proto for logging HTTP requests.
type HttpRequest struct {
	// CacheHit: Whether or not an entity was served from cache (with or
	// without validation).
	CacheHit bool `json:"cacheHit,omitempty"`

	// Referer: Referer (a.k.a. referrer) URL of request, as defined in
	// http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html.
	Referer string `json:"referer,omitempty"`

	// RemoteIp: IP address of the client who issues the HTTP request. Could
	// be either IPv4 or IPv6.
	RemoteIp string `json:"remoteIp,omitempty"`

	// RequestMethod: Request method, such as `GET`, `HEAD`, `PUT` or
	// `POST`.
	RequestMethod string `json:"requestMethod,omitempty"`

	// RequestSize: Size of the HTTP request message in bytes, including
	// request headers and the request body.
	RequestSize int64 `json:"requestSize,omitempty,string"`

	// RequestUrl: Contains the scheme (http|https), the host name, the path
	// and the query portion of the URL that was requested.
	RequestUrl string `json:"requestUrl,omitempty"`

	// ResponseSize: Size of the HTTP response message in bytes sent back to
	// the client, including response headers and response body.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// Status: A response code indicates the status of response, e.g., 200.
	Status int64 `json:"status,omitempty"`

	// UserAgent: User agent sent by the client, e.g., "Mozilla/4.0
	// (compatible; MSIE 6.0; Windows 98; Q312461; .NET CLR 1.0.3705)".
	UserAgent string `json:"userAgent,omitempty"`

	// ValidatedWithOriginServer: Whether or not the response was validated
	// with the origin server before being served from cache. This field is
	// only meaningful if cache_hit is True.
	ValidatedWithOriginServer bool `json:"validatedWithOriginServer,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CacheHit") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HttpRequest) MarshalJSON() ([]byte, error) {
	type noMethod HttpRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogMetricsResponse: Result returned from ListLogMetrics.
type ListLogMetricsResponse struct {
	// Metrics: The list of metrics that was requested.
	Metrics []*LogMetric `json:"metrics,omitempty"`

	// NextPageToken: If there are more results, then `nextPageToken` is
	// returned in the response. To get the next batch of entries, use the
	// value of `nextPageToken` as `pageToken` in the next call of
	// `ListLogMetrics`. If `nextPageToken` is empty, then there are no more
	// results.
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
}

func (s *ListLogMetricsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogMetricsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogServiceIndexesResponse: Result returned from
// ListLogServiceIndexesRequest.
type ListLogServiceIndexesResponse struct {
	// NextPageToken: If there are more results, then `nextPageToken` is
	// returned in the response. To get the next batch of indexes, use the
	// value of `nextPageToken` as `pageToken` in the next call of
	// `ListLogServiceIndexes`. If `nextPageToken` is empty, then there are
	// no more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServiceIndexPrefixes: A list of log service index values. Each index
	// value has the form "/value1/value2/...", where `value1` is a value
	// in the primary index, `value2` is a value in the secondary index, and
	// so forth.
	ServiceIndexPrefixes []string `json:"serviceIndexPrefixes,omitempty"`

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
}

func (s *ListLogServiceIndexesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogServiceIndexesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogServiceSinksResponse: Result returned from
// `ListLogServiceSinks`.
type ListLogServiceSinksResponse struct {
	// Sinks: The requested log service sinks. If a returned `LogSink`
	// object has an empty `destination` field, the client can retrieve the
	// complete `LogSink` object by calling `logServices.sinks.get`.
	Sinks []*LogSink `json:"sinks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Sinks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListLogServiceSinksResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogServiceSinksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogServicesResponse: Result returned from
// `ListLogServicesRequest`.
type ListLogServicesResponse struct {
	// LogServices: A list of log services.
	LogServices []*LogService `json:"logServices,omitempty"`

	// NextPageToken: If there are more results, then `nextPageToken` is
	// returned in the response. To get the next batch of services, use the
	// value of `nextPageToken` as `pageToken` in the next call of
	// `ListLogServices`. If `nextPageToken` is empty, then there are no
	// more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "LogServices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListLogServicesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogServicesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogSinksResponse: Result returned from `ListLogSinks`.
type ListLogSinksResponse struct {
	// Sinks: The requested log sinks. If a returned `LogSink` object has an
	// empty `destination` field, the client can retrieve the complete
	// `LogSink` object by calling `log.sinks.get`.
	Sinks []*LogSink `json:"sinks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Sinks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListLogSinksResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogSinksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListLogsResponse: Result returned from ListLogs.
type ListLogsResponse struct {
	// Logs: A list of log descriptions matching the criteria.
	Logs []*Log `json:"logs,omitempty"`

	// NextPageToken: If there are more results, then `nextPageToken` is
	// returned in the response. To get the next batch of logs, use the
	// value of `nextPageToken` as `pageToken` in the next call of
	// `ListLogs`. If `nextPageToken` is empty, then there are no more
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Logs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListLogsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLogsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListSinksResponse: Result returned from `ListSinks`.
type ListSinksResponse struct {
	// Sinks: The requested sinks. If a returned `LogSink` object has an
	// empty `destination` field, the client can retrieve the complete
	// `LogSink` object by calling `projects.sinks.get`.
	Sinks []*LogSink `json:"sinks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Sinks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListSinksResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListSinksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Log: _Output only._ Describes a log, which is a named stream of log
// entries.
type Log struct {
	// DisplayName: _Optional._ The common name of the log. Example:
	// "request_log".
	DisplayName string `json:"displayName,omitempty"`

	// Name: The resource name of the log. Example:
	// "/projects/my-gcp-project-id/logs/LOG_NAME", where `LOG_NAME` is
	// the URL-encoded given name of the log. The log includes those log
	// entries whose `LogEntry.log` field contains this given name. To avoid
	// name collisions, it is a best practice to prefix the given log name
	// with the service name, but this is not required. Examples of log
	// given names: "appengine.googleapis.com/request_log",
	// "apache-access".
	Name string `json:"name,omitempty"`

	// PayloadType: _Optional_. A URI representing the expected payload type
	// for log entries.
	PayloadType string `json:"payloadType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Log) MarshalJSON() ([]byte, error) {
	type noMethod Log
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogEntry: An individual entry in a log.
type LogEntry struct {
	// HttpRequest: Information about the HTTP request associated with this
	// log entry, if applicable.
	HttpRequest *HttpRequest `json:"httpRequest,omitempty"`

	// InsertId: A unique ID for the log entry. If you provide this field,
	// the logging service considers other log entries in the same log with
	// the same ID as duplicates which can be removed.
	InsertId string `json:"insertId,omitempty"`

	// Log: The log to which this entry belongs. When a log entry is
	// ingested, the value of this field is set by the logging system.
	Log string `json:"log,omitempty"`

	// Metadata: Information about the log entry.
	Metadata *LogEntryMetadata `json:"metadata,omitempty"`

	// ProtoPayload: The log entry payload, represented as a protocol buffer
	// that is expressed as a JSON object. You can only pass `protoPayload`
	// values that belong to a set of approved types.
	ProtoPayload LogEntryProtoPayload `json:"protoPayload,omitempty"`

	// StructPayload: The log entry payload, represented as a structure that
	// is expressed as a JSON object.
	StructPayload LogEntryStructPayload `json:"structPayload,omitempty"`

	// TextPayload: The log entry payload, represented as a Unicode string
	// (UTF-8).
	TextPayload string `json:"textPayload,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HttpRequest") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogEntry) MarshalJSON() ([]byte, error) {
	type noMethod LogEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type LogEntryProtoPayload interface{}

type LogEntryStructPayload interface{}

// LogEntryMetadata: Additional data that is associated with a log
// entry, set by the service creating the log entry.
type LogEntryMetadata struct {
	// Labels: A set of (key, value) data that provides additional
	// information about the log entry. If the log entry is from one of the
	// Google Cloud Platform sources listed below, the indicated (key,
	// value) information must be provided: Google App Engine, service_name
	// `appengine.googleapis.com`: "appengine.googleapis.com/module_id",
	// "appengine.googleapis.com/version_id",  and one of:
	// "appengine.googleapis.com/replica_index",
	// "appengine.googleapis.com/clone_id",  or else provide the following
	// Compute Engine labels: Google Compute Engine, service_name
	// `compute.googleapis.com`: "compute.googleapis.com/resource_type",
	// "instance" "compute.googleapis.com/resource_id",
	Labels map[string]string `json:"labels,omitempty"`

	// ProjectId: The project ID of the Google Cloud Platform service that
	// created the log entry.
	ProjectId string `json:"projectId,omitempty"`

	// Region: The region name of the Google Cloud Platform service that
	// created the log entry. For example, "us-central1".
	Region string `json:"region,omitempty"`

	// ServiceName: The API name of the Google Cloud Platform service that
	// created the log entry. For example, "compute.googleapis.com".
	ServiceName string `json:"serviceName,omitempty"`

	// Severity: The severity of the log entry.
	//
	// Possible values:
	//   "DEFAULT"
	//   "DEBUG"
	//   "INFO"
	//   "NOTICE"
	//   "WARNING"
	//   "ERROR"
	//   "CRITICAL"
	//   "ALERT"
	//   "EMERGENCY"
	Severity string `json:"severity,omitempty"`

	// Timestamp: The time the event described by the log entry occurred.
	// Timestamps must be later than January 1, 1970.
	Timestamp string `json:"timestamp,omitempty"`

	// UserId: The fully-qualified email address of the authenticated user
	// that performed or requested the action represented by the log entry.
	// If the log entry does not apply to an action taken by an
	// authenticated user, then the field should be empty.
	UserId string `json:"userId,omitempty"`

	// Zone: The zone of the Google Cloud Platform service that created the
	// log entry. For example, "us-central1-a".
	Zone string `json:"zone,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Labels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogEntryMetadata) MarshalJSON() ([]byte, error) {
	type noMethod LogEntryMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogError: Describes a problem with a logging resource or operation.
type LogError struct {
	// Resource: A resource name associated with this error. For example,
	// the name of a Cloud Storage bucket that has insufficient permissions
	// to be a destination for log entries.
	Resource string `json:"resource,omitempty"`

	// Status: The error description, including a classification code, an
	// error message, and other details.
	Status *Status `json:"status,omitempty"`

	// TimeNanos: The time the error was observed, in nanoseconds since the
	// Unix epoch.
	TimeNanos int64 `json:"timeNanos,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Resource") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogError) MarshalJSON() ([]byte, error) {
	type noMethod LogError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogLine: Application log line emitted while processing a request.
type LogLine struct {
	// LogMessage: App provided log message.
	LogMessage string `json:"logMessage,omitempty"`

	// Severity: Severity of log.
	//
	// Possible values:
	//   "DEFAULT"
	//   "DEBUG"
	//   "INFO"
	//   "NOTICE"
	//   "WARNING"
	//   "ERROR"
	//   "CRITICAL"
	//   "ALERT"
	//   "EMERGENCY"
	Severity string `json:"severity,omitempty"`

	// SourceLocation: Line of code that generated this log message.
	SourceLocation *SourceLocation `json:"sourceLocation,omitempty"`

	// Time: Time when log entry was made. May be inaccurate.
	Time string `json:"time,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LogMessage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogLine) MarshalJSON() ([]byte, error) {
	type noMethod LogLine
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogMetric: Describes a logs-based metric. The value of the metric is
// the number of log entries in your project that match a logs filter.
type LogMetric struct {
	// Description: A description of this metric.
	Description string `json:"description,omitempty"`

	// Filter: An [advanced logs
	// filter](/logging/docs/view/advanced_filters). Example: "log:syslog
	// AND metadata.severity>=ERROR".
	Filter string `json:"filter,omitempty"`

	// Name: The client-assigned name for this metric, such as
	// "severe_errors". Metric names are limited to 1000 characters and
	// can include only the following characters: `A-Z`, `a-z`, `0-9`, and
	// the special characters `_-.,+!*',()%/\`. The slash character (`/`)
	// denotes a hierarchy of name pieces, and it cannot be the first
	// character of the name.
	Name string `json:"name,omitempty"`

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
}

func (s *LogMetric) MarshalJSON() ([]byte, error) {
	type noMethod LogMetric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogService: _Output only._ Describes a service that writes log
// entries.
type LogService struct {
	// IndexKeys: A list of the names of the keys used to index and label
	// individual log entries from this service. The first two keys are used
	// as the primary and secondary index, respectively. Additional keys may
	// be used to label the entries. For example, App Engine indexes its
	// entries by module and by version, so its `indexKeys` field is the
	// following: [ "appengine.googleapis.com/module_id",
	// "appengine.googleapis.com/version_id" ]
	IndexKeys []string `json:"indexKeys,omitempty"`

	// Name: The service's name. Example: "appengine.googleapis.com". Log
	// names beginning with this string are reserved for this service. This
	// value can appear in the `LogEntry.metadata.serviceName` field of log
	// entries associated with this log service.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IndexKeys") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogService) MarshalJSON() ([]byte, error) {
	type noMethod LogService
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogSink: Describes where log entries are written outside of Cloud
// Logging.
type LogSink struct {
	// Destination: The resource name of the destination. Cloud Logging
	// writes designated log entries to this destination. For example,
	// "storage.googleapis.com/my-output-bucket".
	Destination string `json:"destination,omitempty"`

	// Errors: _Output only._ If any errors occur when invoking a sink
	// method, then this field contains descriptions of the errors.
	Errors []*LogError `json:"errors,omitempty"`

	// Filter: An advanced logs filter. If present, only log entries
	// matching the filter are written. Only project sinks use this field;
	// log sinks and log service sinks must not include a filter.
	Filter string `json:"filter,omitempty"`

	// Name: The client-assigned name of this sink. For example,
	// "my-syslog-sink". The name must be unique among the sinks of a
	// similar kind in the project.
	Name string `json:"name,omitempty"`

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
}

func (s *LogSink) MarshalJSON() ([]byte, error) {
	type noMethod LogSink
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// RequestLog: Complete log information about a single request to an
// application.
type RequestLog struct {
	// AppEngineRelease: App Engine release version string.
	AppEngineRelease string `json:"appEngineRelease,omitempty"`

	// AppId: Identifies the application that handled this request.
	AppId string `json:"appId,omitempty"`

	// Cost: An indication of the relative cost of serving this request.
	Cost float64 `json:"cost,omitempty"`

	// EndTime: Time at which request was known to end processing.
	EndTime string `json:"endTime,omitempty"`

	// Finished: If true, represents a finished request. Otherwise, the
	// request is active.
	Finished bool `json:"finished,omitempty"`

	// Host: The Internet host and port number of the resource being
	// requested.
	Host string `json:"host,omitempty"`

	// HttpVersion: HTTP version of request.
	HttpVersion string `json:"httpVersion,omitempty"`

	// InstanceId: An opaque identifier for the instance that handled the
	// request.
	InstanceId string `json:"instanceId,omitempty"`

	// InstanceIndex: If the instance that processed this request was
	// individually addressable (i.e. belongs to a manually scaled module),
	// this is the index of the instance.
	InstanceIndex int64 `json:"instanceIndex,omitempty"`

	// Ip: Origin IP address.
	Ip string `json:"ip,omitempty"`

	// Latency: Latency of the request.
	Latency string `json:"latency,omitempty"`

	// Line: List of log lines emitted by the application while serving this
	// request, if requested.
	Line []*LogLine `json:"line,omitempty"`

	// MegaCycles: Number of CPU megacycles used to process request.
	MegaCycles int64 `json:"megaCycles,omitempty,string"`

	// Method: Request method, such as `GET`, `HEAD`, `PUT`, `POST`, or
	// `DELETE`.
	Method string `json:"method,omitempty"`

	// ModuleId: Identifies the module of the application that handled this
	// request.
	ModuleId string `json:"moduleId,omitempty"`

	// Nickname: A string that identifies a logged-in user who made this
	// request, or empty if the user is not logged in. Most likely, this is
	// the part of the user's email before the '@' sign. The field value is
	// the same for different requests from the same user, but different
	// users may have a similar name. This information is also available to
	// the application via Users API. This field will be populated starting
	// with App Engine 1.9.21.
	Nickname string `json:"nickname,omitempty"`

	// PendingTime: Time this request spent in the pending request queue, if
	// it was pending at all.
	PendingTime string `json:"pendingTime,omitempty"`

	// Referrer: Referrer URL of request.
	Referrer string `json:"referrer,omitempty"`

	// RequestId: Globally unique identifier for a request, based on request
	// start time. Request IDs for requests which started later will compare
	// greater as strings than those for requests which started earlier.
	RequestId string `json:"requestId,omitempty"`

	// Resource: Contains the path and query portion of the URL that was
	// requested. For example, if the URL was
	// "http://example.com/app?name=val", the resource would be
	// "/app?name=val". Any trailing fragment (separated by a '#' character)
	// will not be included.
	Resource string `json:"resource,omitempty"`

	// ResponseSize: Size in bytes sent back to client by request.
	ResponseSize int64 `json:"responseSize,omitempty,string"`

	// SourceReference: Source code for the application that handled this
	// request. There can be more than one source reference per deployed
	// application if source code is distributed among multiple
	// repositories.
	SourceReference []*SourceReference `json:"sourceReference,omitempty"`

	// StartTime: Time at which request was known to have begun processing.
	StartTime string `json:"startTime,omitempty"`

	// Status: Response status of request.
	Status int64 `json:"status,omitempty"`

	// TaskName: Task name of the request (for an offline request).
	TaskName string `json:"taskName,omitempty"`

	// TaskQueueName: Queue name of the request (for an offline request).
	TaskQueueName string `json:"taskQueueName,omitempty"`

	// TraceId: Cloud Trace identifier of the trace for this request.
	TraceId string `json:"traceId,omitempty"`

	// UrlMapEntry: File or class within URL mapping used for request.
	// Useful for tracking down the source code which was responsible for
	// managing request. Especially for multiply mapped handlers.
	UrlMapEntry string `json:"urlMapEntry,omitempty"`

	// UserAgent: User agent used for making request.
	UserAgent string `json:"userAgent,omitempty"`

	// VersionId: Version of the application that handled this request.
	VersionId string `json:"versionId,omitempty"`

	// WasLoadingRequest: Was this request a loading request for this
	// instance?
	WasLoadingRequest bool `json:"wasLoadingRequest,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppEngineRelease") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RequestLog) MarshalJSON() ([]byte, error) {
	type noMethod RequestLog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SourceLocation: Specifies a location in a source file.
type SourceLocation struct {
	// File: Source file name. May or may not be a fully qualified name,
	// depending on the runtime environment.
	File string `json:"file,omitempty"`

	// FunctionName: Human-readable name of the function or method being
	// invoked, with optional context such as the class or package name, for
	// use in contexts such as the logs viewer where file:line number is
	// less meaningful. This may vary by language, for example: in Java:
	// qual.if.ied.Class.method in Go: dir/package.func in Python: function
	// ...
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
}

func (s *SourceLocation) MarshalJSON() ([]byte, error) {
	type noMethod SourceLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SourceReference: A reference to a particular snapshot of the source
// tree used to build and deploy an application.
type SourceReference struct {
	// Repository: Optional. A URI string identifying the repository.
	// Example: "https://github.com/GoogleCloudPlatform/kubernetes.git"
	Repository string `json:"repository,omitempty"`

	// RevisionId: The canonical (and persistent) identifier of the deployed
	// revision. Example (git): "0035781c50ec7aa23385dc841529ce8a4b70db1b"
	RevisionId string `json:"revisionId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Repository") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SourceReference) MarshalJSON() ([]byte, error) {
	type noMethod SourceReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Status: The `Status` type defines a logical error model that is
// suitable for different programming environments, including REST APIs
// and RPC APIs. It is used by [gRPC](https://github.com/grpc). The
// error model is designed to be: - Simple to use and understand for
// most users - Flexible enough to meet unexpected needs # Overview The
// `Status` message contains three pieces of data: error code, error
// message, and error details. The error code should be an enum value of
// google.rpc.Code, but it may accept additional error codes if needed.
// The error message should be a developer-facing English message that
// helps developers *understand* and *resolve* the error. If a localized
// user-facing error message is needed, put the localized message in the
// error details or localize it in the client. The optional error
// details may contain arbitrary information about the error. There is a
// predefined set of error detail types in the package `google.rpc`
// which can be used for common error conditions. # Language mapping The
// `Status` message is the logical representation of the error model,
// but it is not necessarily the actual wire format. When the `Status`
// message is exposed in different client libraries and different wire
// protocols, it can be mapped differently. For example, it will likely
// be mapped to some exceptions in Java, but more likely mapped to some
// error codes in C. # Other uses The error model and the `Status`
// message can be used in a variety of environments, either with or
// without APIs, to provide a consistent developer experience across
// different environments. Example uses of this error model include: -
// Partial errors. If a service needs to return partial errors to the
// client, it may embed the `Status` in the normal response to indicate
// the partial errors. - Workflow errors. A typical workflow has
// multiple steps. Each step may have a `Status` message for error
// reporting purpose. - Batch operations. If a client uses batch request
// and batch response, the `Status` message should be used directly
// inside batch response, one for each error sub-response. -
// Asynchronous operations. If an API call embeds asynchronous operation
// results in its response, the status of those operations should be
// represented directly using the `Status` message. - Logging. If some
// API errors are stored in logs, the message `Status` could be used
// directly after any stripping needed for security/privacy reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details. There will
	// be a common set of message types for APIs to use.
	Details []StatusDetails `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any user-facing error message should be localized and sent
	// in the google.rpc.Status.details field, or localized by the client.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type StatusDetails interface{}

// WriteLogEntriesRequest: The parameters to WriteLogEntries.
type WriteLogEntriesRequest struct {
	// CommonLabels: Metadata labels that apply to all log entries in this
	// request, so that you don't have to repeat them in each log entry's
	// `metadata.labels` field. If any of the log entries contains a (key,
	// value) with the same key that is in `commonLabels`, then the entry's
	// (key, value) overrides the one in `commonLabels`.
	CommonLabels map[string]string `json:"commonLabels,omitempty"`

	// Entries: Log entries to insert.
	Entries []*LogEntry `json:"entries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommonLabels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *WriteLogEntriesRequest) MarshalJSON() ([]byte, error) {
	type noMethod WriteLogEntriesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// WriteLogEntriesResponse: Result returned from WriteLogEntries. empty
type WriteLogEntriesResponse struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// method id "logging.projects.logServices.list":

type ProjectsLogServicesListCall struct {
	s            *Service
	projectsId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the log services that have log entries in this project.
func (r *ProjectsLogServicesService) List(projectsId string) *ProjectsLogServicesListCall {
	c := &ProjectsLogServicesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of `LogService` objects to return in one operation.
func (c *ProjectsLogServicesListCall) PageSize(pageSize int64) *ProjectsLogServicesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": An opaque token,
// returned as `nextPageToken` by a prior `ListLogServices` operation.
// If `pageToken` is supplied, then the other fields of this request are
// ignored, and instead the previous `ListLogServices` operation is
// continued.
func (c *ProjectsLogServicesListCall) PageToken(pageToken string) *ProjectsLogServicesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesListCall) QuotaUser(quotaUser string) *ProjectsLogServicesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesListCall) Fields(s ...googleapi.Field) *ProjectsLogServicesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogServicesListCall) IfNoneMatch(entityTag string) *ProjectsLogServicesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesListCall) Context(ctx context.Context) *ProjectsLogServicesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.list" call.
// Exactly one of *ListLogServicesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogServicesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogServicesListCall) Do() (*ListLogServicesResponse, error) {
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
	ret := &ListLogServicesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the log services that have log entries in this project.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logServices.list",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of `LogService` objects to return in one operation.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "An opaque token, returned as `nextPageToken` by a prior `ListLogServices` operation. If `pageToken` is supplied, then the other fields of this request are ignored, and instead the previous `ListLogServices` operation is continued.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `projectName`. The resource name of the project whose services are to be listed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices",
	//   "response": {
	//     "$ref": "ListLogServicesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.logServices.indexes.list":

type ProjectsLogServicesIndexesListCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: Lists the current index values for a log service.
func (r *ProjectsLogServicesIndexesService) List(projectsId string, logServicesId string) *ProjectsLogServicesIndexesListCall {
	c := &ProjectsLogServicesIndexesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	return c
}

// Depth sets the optional parameter "depth": A non-negative integer
// that limits the number of levels of the index hierarchy that are
// returned. If `depth` is 1 (default), only the first index key value
// is returned. If `depth` is 2, both primary and secondary key values
// are returned. If `depth` is 0, the depth is the number of
// slash-separators in the `indexPrefix` field, not counting a slash
// appearing as the last character of the prefix. If the `indexPrefix`
// field is empty, the default depth is 1. It is an error for `depth` to
// be any positive value less than the number of components in
// `indexPrefix`.
func (c *ProjectsLogServicesIndexesListCall) Depth(depth int64) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("depth", fmt.Sprint(depth))
	return c
}

// IndexPrefix sets the optional parameter "indexPrefix": Restricts the
// index values returned to be those with a specified prefix for each
// index key. This field has the form "/prefix1/prefix2/...", in order
// corresponding to the `LogService indexKeys`. Non-empty prefixes must
// begin with `/`. For example, App Engine's two keys are the module ID
// and the version ID. Following is the effect of using various values
// for `indexPrefix`: + "/Mod/" retrieves `/Mod/10` and `/Mod/11` but
// not `/ModA/10`. + "/Mod` retrieves `/Mod/10`, `/Mod/11` and
// `/ModA/10` but not `/XXX/33`. + "/Mod/1" retrieves `/Mod/10` and
// `/Mod/11` but not `/ModA/10`. + "/Mod/10/" retrieves `/Mod/10`
// only. + An empty prefix or "/" retrieves all values.
func (c *ProjectsLogServicesIndexesListCall) IndexPrefix(indexPrefix string) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("indexPrefix", indexPrefix)
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of log service index resources to return in one operation.
func (c *ProjectsLogServicesIndexesListCall) PageSize(pageSize int64) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": An opaque token,
// returned as `nextPageToken` by a prior `ListLogServiceIndexes`
// operation. If `pageToken` is supplied, then the other fields of this
// request are ignored, and instead the previous `ListLogServiceIndexes`
// operation is continued.
func (c *ProjectsLogServicesIndexesListCall) PageToken(pageToken string) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesIndexesListCall) QuotaUser(quotaUser string) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesIndexesListCall) Fields(s ...googleapi.Field) *ProjectsLogServicesIndexesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogServicesIndexesListCall) IfNoneMatch(entityTag string) *ProjectsLogServicesIndexesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesIndexesListCall) Context(ctx context.Context) *ProjectsLogServicesIndexesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesIndexesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/indexes")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.indexes.list" call.
// Exactly one of *ListLogServiceIndexesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListLogServiceIndexesResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogServicesIndexesListCall) Do() (*ListLogServiceIndexesResponse, error) {
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
	ret := &ListLogServiceIndexesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the current index values for a log service.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logServices.indexes.list",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId"
	//   ],
	//   "parameters": {
	//     "depth": {
	//       "description": "A non-negative integer that limits the number of levels of the index hierarchy that are returned. If `depth` is 1 (default), only the first index key value is returned. If `depth` is 2, both primary and secondary key values are returned. If `depth` is 0, the depth is the number of slash-separators in the `indexPrefix` field, not counting a slash appearing as the last character of the prefix. If the `indexPrefix` field is empty, the default depth is 1. It is an error for `depth` to be any positive value less than the number of components in `indexPrefix`.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "indexPrefix": {
	//       "description": "Restricts the index values returned to be those with a specified prefix for each index key. This field has the form `\"/prefix1/prefix2/...\"`, in order corresponding to the `LogService indexKeys`. Non-empty prefixes must begin with `/`. For example, App Engine's two keys are the module ID and the version ID. Following is the effect of using various values for `indexPrefix`: + `\"/Mod/\"` retrieves `/Mod/10` and `/Mod/11` but not `/ModA/10`. + `\"/Mod` retrieves `/Mod/10`, `/Mod/11` and `/ModA/10` but not `/XXX/33`. + `\"/Mod/1\"` retrieves `/Mod/10` and `/Mod/11` but not `/ModA/10`. + `\"/Mod/10/\"` retrieves `/Mod/10` only. + An empty prefix or `\"/\"` retrieves all values.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "logServicesId": {
	//       "description": "Part of `serviceName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of log service index resources to return in one operation.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "An opaque token, returned as `nextPageToken` by a prior `ListLogServiceIndexes` operation. If `pageToken` is supplied, then the other fields of this request are ignored, and instead the previous `ListLogServiceIndexes` operation is continued.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `serviceName`. The resource name of a log service whose service indexes are requested. Example: `\"projects/my-project-id/logServices/appengine.googleapis.com\"`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/indexes",
	//   "response": {
	//     "$ref": "ListLogServiceIndexesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.logServices.sinks.create":

type ProjectsLogServicesSinksCreateCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	logsink       *LogSink
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Create: Creates a log service sink. All log entries from a specified
// log service are written to the destination.
func (r *ProjectsLogServicesSinksService) Create(projectsId string, logServicesId string, logsink *LogSink) *ProjectsLogServicesSinksCreateCall {
	c := &ProjectsLogServicesSinksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesSinksCreateCall) QuotaUser(quotaUser string) *ProjectsLogServicesSinksCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesSinksCreateCall) Fields(s ...googleapi.Field) *ProjectsLogServicesSinksCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesSinksCreateCall) Context(ctx context.Context) *ProjectsLogServicesSinksCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesSinksCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.sinks.create" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogServicesSinksCreateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a log service sink. All log entries from a specified log service are written to the destination.",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.logServices.sinks.create",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId"
	//   ],
	//   "parameters": {
	//     "logServicesId": {
	//       "description": "Part of `serviceName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `serviceName`. The resource name of the log service to which the sink is bound.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks",
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

// method id "logging.projects.logServices.sinks.delete":

type ProjectsLogServicesSinksDeleteCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	sinksId       string
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Delete: Deletes a log service sink. After deletion, no new log
// entries are written to the destination.
func (r *ProjectsLogServicesSinksService) Delete(projectsId string, logServicesId string, sinksId string) *ProjectsLogServicesSinksDeleteCall {
	c := &ProjectsLogServicesSinksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesSinksDeleteCall) QuotaUser(quotaUser string) *ProjectsLogServicesSinksDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesSinksDeleteCall) Fields(s ...googleapi.Field) *ProjectsLogServicesSinksDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesSinksDeleteCall) Context(ctx context.Context) *ProjectsLogServicesSinksDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesSinksDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
		"sinksId":       c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.sinks.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogServicesSinksDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a log service sink. After deletion, no new log entries are written to the destination.",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.logServices.sinks.delete",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logServicesId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the log service sink to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.projects.logServices.sinks.get":

type ProjectsLogServicesSinksGetCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	sinksId       string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Gets a log service sink.
func (r *ProjectsLogServicesSinksService) Get(projectsId string, logServicesId string, sinksId string) *ProjectsLogServicesSinksGetCall {
	c := &ProjectsLogServicesSinksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesSinksGetCall) QuotaUser(quotaUser string) *ProjectsLogServicesSinksGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesSinksGetCall) Fields(s ...googleapi.Field) *ProjectsLogServicesSinksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogServicesSinksGetCall) IfNoneMatch(entityTag string) *ProjectsLogServicesSinksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesSinksGetCall) Context(ctx context.Context) *ProjectsLogServicesSinksGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesSinksGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
		"sinksId":       c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.sinks.get" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogServicesSinksGetCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a log service sink.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logServices.sinks.get",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logServicesId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the log service sink to return.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}",
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

// method id "logging.projects.logServices.sinks.list":

type ProjectsLogServicesSinksListCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// List: Lists log service sinks associated with a log service.
func (r *ProjectsLogServicesSinksService) List(projectsId string, logServicesId string) *ProjectsLogServicesSinksListCall {
	c := &ProjectsLogServicesSinksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesSinksListCall) QuotaUser(quotaUser string) *ProjectsLogServicesSinksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesSinksListCall) Fields(s ...googleapi.Field) *ProjectsLogServicesSinksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogServicesSinksListCall) IfNoneMatch(entityTag string) *ProjectsLogServicesSinksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesSinksListCall) Context(ctx context.Context) *ProjectsLogServicesSinksListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesSinksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.sinks.list" call.
// Exactly one of *ListLogServiceSinksResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListLogServiceSinksResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogServicesSinksListCall) Do() (*ListLogServiceSinksResponse, error) {
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
	ret := &ListLogServiceSinksResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists log service sinks associated with a log service.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logServices.sinks.list",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId"
	//   ],
	//   "parameters": {
	//     "logServicesId": {
	//       "description": "Part of `serviceName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `serviceName`. The log service whose sinks are wanted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks",
	//   "response": {
	//     "$ref": "ListLogServiceSinksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.logServices.sinks.update":

type ProjectsLogServicesSinksUpdateCall struct {
	s             *Service
	projectsId    string
	logServicesId string
	sinksId       string
	logsink       *LogSink
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Updates a log service sink. If the sink does not exist, it is
// created.
func (r *ProjectsLogServicesSinksService) Update(projectsId string, logServicesId string, sinksId string, logsink *LogSink) *ProjectsLogServicesSinksUpdateCall {
	c := &ProjectsLogServicesSinksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logServicesId = logServicesId
	c.sinksId = sinksId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogServicesSinksUpdateCall) QuotaUser(quotaUser string) *ProjectsLogServicesSinksUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogServicesSinksUpdateCall) Fields(s ...googleapi.Field) *ProjectsLogServicesSinksUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogServicesSinksUpdateCall) Context(ctx context.Context) *ProjectsLogServicesSinksUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogServicesSinksUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId":    c.projectsId,
		"logServicesId": c.logServicesId,
		"sinksId":       c.sinksId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logServices.sinks.update" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogServicesSinksUpdateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a log service sink. If the sink does not exist, it is created.",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.logServices.sinks.update",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logServicesId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logServicesId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the log service sink to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logServices/{logServicesId}/sinks/{sinksId}",
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

// method id "logging.projects.logs.delete":

type ProjectsLogsDeleteCall struct {
	s          *Service
	projectsId string
	logsId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a log and all its log entries. The log will reappear
// if it receives new entries.
func (r *ProjectsLogsService) Delete(projectsId string, logsId string) *ProjectsLogsDeleteCall {
	c := &ProjectsLogsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsDeleteCall) QuotaUser(quotaUser string) *ProjectsLogsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsLogsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a log and all its log entries. The log will reappear if it receives new entries.",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.logs.delete",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `logName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `logName`. The resource name of the log to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}",
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
	projectsId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the logs in the project. Only logs that have entries are
// listed.
func (r *ProjectsLogsService) List(projectsId string) *ProjectsLogsListCall {
	c := &ProjectsLogsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return.
func (c *ProjectsLogsListCall) PageSize(pageSize int64) *ProjectsLogsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": An opaque token,
// returned as `nextPageToken` by a prior `ListLogs` operation. If
// `pageToken` is supplied, then the other fields of this request are
// ignored, and instead the previous `ListLogs` operation is continued.
func (c *ProjectsLogsListCall) PageToken(pageToken string) *ProjectsLogsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsListCall) QuotaUser(quotaUser string) *ProjectsLogsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// ServiceIndexPrefix sets the optional parameter "serviceIndexPrefix":
// The purpose of this field is to restrict the listed logs to those
// with entries of a certain kind. If `serviceName` is the name of a log
// service, then this field may contain values for the log service's
// indexes. Only logs that have entries whose indexes include the values
// are listed. The format for this field is "/val1/val2.../valN",
// where `val1` is a value for the first index, `val2` for the second
// index, etc. An empty value (a single slash) for an index matches all
// values, and you can omit values for later indexes entirely.
func (c *ProjectsLogsListCall) ServiceIndexPrefix(serviceIndexPrefix string) *ProjectsLogsListCall {
	c.urlParams_.Set("serviceIndexPrefix", serviceIndexPrefix)
	return c
}

// ServiceName sets the optional parameter "serviceName": If not empty,
// this field must be a log service name such as
// "compute.googleapis.com". Only logs associated with that that log
// service are listed.
func (c *ProjectsLogsListCall) ServiceName(serviceName string) *ProjectsLogsListCall {
	c.urlParams_.Set("serviceName", serviceName)
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

func (c *ProjectsLogsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.list" call.
// Exactly one of *ListLogsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogsListCall) Do() (*ListLogsResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the logs in the project. Only logs that have entries are listed.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logs.list",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of results to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "An opaque token, returned as `nextPageToken` by a prior `ListLogs` operation. If `pageToken` is supplied, then the other fields of this request are ignored, and instead the previous `ListLogs` operation is continued.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `projectName`. The resource name of the project whose logs are requested. If both `serviceName` and `serviceIndexPrefix` are empty, then all logs with entries in this project are listed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "serviceIndexPrefix": {
	//       "description": "The purpose of this field is to restrict the listed logs to those with entries of a certain kind. If `serviceName` is the name of a log service, then this field may contain values for the log service's indexes. Only logs that have entries whose indexes include the values are listed. The format for this field is `\"/val1/val2.../valN\"`, where `val1` is a value for the first index, `val2` for the second index, etc. An empty value (a single slash) for an index matches all values, and you can omit values for later indexes entirely.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "serviceName": {
	//       "description": "If not empty, this field must be a log service name such as `\"compute.googleapis.com\"`. Only logs associated with that that log service are listed.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs",
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

// method id "logging.projects.logs.entries.write":

type ProjectsLogsEntriesWriteCall struct {
	s                      *Service
	projectsId             string
	logsId                 string
	writelogentriesrequest *WriteLogEntriesRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
}

// Write: Writes log entries to Cloud Logging. Each entry consists of a
// `LogEntry` object. You must fill in all the fields of the object,
// including one of the payload fields. You may supply a map,
// `commonLabels`, that holds default (key, value) data for the
// `entries[].metadata.labels` map in each entry, saving you the trouble
// of creating identical copies for each entry.
func (r *ProjectsLogsEntriesService) Write(projectsId string, logsId string, writelogentriesrequest *WriteLogEntriesRequest) *ProjectsLogsEntriesWriteCall {
	c := &ProjectsLogsEntriesWriteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	c.writelogentriesrequest = writelogentriesrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsEntriesWriteCall) QuotaUser(quotaUser string) *ProjectsLogsEntriesWriteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsEntriesWriteCall) Fields(s ...googleapi.Field) *ProjectsLogsEntriesWriteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsEntriesWriteCall) Context(ctx context.Context) *ProjectsLogsEntriesWriteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsEntriesWriteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.writelogentriesrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/entries:write")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.entries.write" call.
// Exactly one of *WriteLogEntriesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WriteLogEntriesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogsEntriesWriteCall) Do() (*WriteLogEntriesResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Writes log entries to Cloud Logging. Each entry consists of a `LogEntry` object. You must fill in all the fields of the object, including one of the payload fields. You may supply a map, `commonLabels`, that holds default (key, value) data for the `entries[].metadata.labels` map in each entry, saving you the trouble of creating identical copies for each entry.",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.logs.entries.write",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `logName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `logName`. The resource name of the log that will receive the log entries.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/entries:write",
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

// method id "logging.projects.logs.sinks.create":

type ProjectsLogsSinksCreateCall struct {
	s          *Service
	projectsId string
	logsId     string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a log sink. All log entries for a specified log are
// written to the destination.
func (r *ProjectsLogsSinksService) Create(projectsId string, logsId string, logsink *LogSink) *ProjectsLogsSinksCreateCall {
	c := &ProjectsLogsSinksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsSinksCreateCall) QuotaUser(quotaUser string) *ProjectsLogsSinksCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsSinksCreateCall) Fields(s ...googleapi.Field) *ProjectsLogsSinksCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsSinksCreateCall) Context(ctx context.Context) *ProjectsLogsSinksCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsSinksCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.sinks.create" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsSinksCreateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a log sink. All log entries for a specified log are written to the destination.",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.logs.sinks.create",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `logName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `logName`. The resource name of the log to which to the sink is bound.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/sinks",
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

// method id "logging.projects.logs.sinks.delete":

type ProjectsLogsSinksDeleteCall struct {
	s          *Service
	projectsId string
	logsId     string
	sinksId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a log sink. After deletion, no new log entries are
// written to the destination.
func (r *ProjectsLogsSinksService) Delete(projectsId string, logsId string, sinksId string) *ProjectsLogsSinksDeleteCall {
	c := &ProjectsLogsSinksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsSinksDeleteCall) QuotaUser(quotaUser string) *ProjectsLogsSinksDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsSinksDeleteCall) Fields(s ...googleapi.Field) *ProjectsLogsSinksDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsSinksDeleteCall) Context(ctx context.Context) *ProjectsLogsSinksDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsSinksDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.sinks.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsSinksDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a log sink. After deletion, no new log entries are written to the destination.",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.logs.sinks.delete",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the log sink to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin"
	//   ]
	// }

}

// method id "logging.projects.logs.sinks.get":

type ProjectsLogsSinksGetCall struct {
	s            *Service
	projectsId   string
	logsId       string
	sinksId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a log sink.
func (r *ProjectsLogsSinksService) Get(projectsId string, logsId string, sinksId string) *ProjectsLogsSinksGetCall {
	c := &ProjectsLogsSinksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsSinksGetCall) QuotaUser(quotaUser string) *ProjectsLogsSinksGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsSinksGetCall) Fields(s ...googleapi.Field) *ProjectsLogsSinksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogsSinksGetCall) IfNoneMatch(entityTag string) *ProjectsLogsSinksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsSinksGetCall) Context(ctx context.Context) *ProjectsLogsSinksGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsSinksGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.sinks.get" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsSinksGetCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a log sink.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logs.sinks.get",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the log sink to return.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}",
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

// method id "logging.projects.logs.sinks.list":

type ProjectsLogsSinksListCall struct {
	s            *Service
	projectsId   string
	logsId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists log sinks associated with a log.
func (r *ProjectsLogsSinksService) List(projectsId string, logsId string) *ProjectsLogsSinksListCall {
	c := &ProjectsLogsSinksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsSinksListCall) QuotaUser(quotaUser string) *ProjectsLogsSinksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsSinksListCall) Fields(s ...googleapi.Field) *ProjectsLogsSinksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLogsSinksListCall) IfNoneMatch(entityTag string) *ProjectsLogsSinksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsSinksListCall) Context(ctx context.Context) *ProjectsLogsSinksListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsSinksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.sinks.list" call.
// Exactly one of *ListLogSinksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogSinksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLogsSinksListCall) Do() (*ListLogSinksResponse, error) {
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
	ret := &ListLogSinksResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists log sinks associated with a log.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.logs.sinks.list",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `logName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `logName`. The log whose sinks are wanted. For example, `\"compute.google.com/syslog\"`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/sinks",
	//   "response": {
	//     "$ref": "ListLogSinksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read"
	//   ]
	// }

}

// method id "logging.projects.logs.sinks.update":

type ProjectsLogsSinksUpdateCall struct {
	s          *Service
	projectsId string
	logsId     string
	sinksId    string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a log sink. If the sink does not exist, it is
// created.
func (r *ProjectsLogsSinksService) Update(projectsId string, logsId string, sinksId string, logsink *LogSink) *ProjectsLogsSinksUpdateCall {
	c := &ProjectsLogsSinksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsId = logsId
	c.sinksId = sinksId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsLogsSinksUpdateCall) QuotaUser(quotaUser string) *ProjectsLogsSinksUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLogsSinksUpdateCall) Fields(s ...googleapi.Field) *ProjectsLogsSinksUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLogsSinksUpdateCall) Context(ctx context.Context) *ProjectsLogsSinksUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsLogsSinksUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"logsId":     c.logsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.logs.sinks.update" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLogsSinksUpdateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a log sink. If the sink does not exist, it is created.",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.logs.sinks.update",
	//   "parameterOrder": [
	//     "projectsId",
	//     "logsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "logsId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the sink to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/logs/{logsId}/sinks/{sinksId}",
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

// method id "logging.projects.metrics.create":

type ProjectsMetricsCreateCall struct {
	s          *Service
	projectsId string
	logmetric  *LogMetric
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a logs-based metric.
func (r *ProjectsMetricsService) Create(projectsId string, logmetric *LogMetric) *ProjectsMetricsCreateCall {
	c := &ProjectsMetricsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logmetric = logmetric
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsMetricsCreateCall) QuotaUser(quotaUser string) *ProjectsMetricsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsMetricsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logmetric)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/metrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.metrics.create" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsCreateCall) Do() (*LogMetric, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a logs-based metric.",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.metrics.create",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `projectName`. The resource name of the project in which to create the metric.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/metrics",
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
	projectsId string
	metricsId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a logs-based metric.
func (r *ProjectsMetricsService) Delete(projectsId string, metricsId string) *ProjectsMetricsDeleteCall {
	c := &ProjectsMetricsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.metricsId = metricsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsMetricsDeleteCall) QuotaUser(quotaUser string) *ProjectsMetricsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsMetricsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/metrics/{metricsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"metricsId":  c.metricsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.metrics.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsMetricsDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a logs-based metric.",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.metrics.delete",
	//   "parameterOrder": [
	//     "projectsId",
	//     "metricsId"
	//   ],
	//   "parameters": {
	//     "metricsId": {
	//       "description": "Part of `metricName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `metricName`. The resource name of the metric to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/metrics/{metricsId}",
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
	projectsId   string
	metricsId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a logs-based metric.
func (r *ProjectsMetricsService) Get(projectsId string, metricsId string) *ProjectsMetricsGetCall {
	c := &ProjectsMetricsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.metricsId = metricsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsMetricsGetCall) QuotaUser(quotaUser string) *ProjectsMetricsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsMetricsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/metrics/{metricsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"metricsId":  c.metricsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.metrics.get" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsGetCall) Do() (*LogMetric, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a logs-based metric.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.metrics.get",
	//   "parameterOrder": [
	//     "projectsId",
	//     "metricsId"
	//   ],
	//   "parameters": {
	//     "metricsId": {
	//       "description": "Part of `metricName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `metricName`. The resource name of the desired metric.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/metrics/{metricsId}",
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
	projectsId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the logs-based metrics associated with a project.
func (r *ProjectsMetricsService) List(projectsId string) *ProjectsMetricsListCall {
	c := &ProjectsMetricsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of `LogMetric` objects to return in one operation.
func (c *ProjectsMetricsListCall) PageSize(pageSize int64) *ProjectsMetricsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": An opaque token,
// returned as `nextPageToken` by a prior `ListLogMetrics` operation. If
// `pageToken` is supplied, then the other fields of this request are
// ignored, and instead the previous `ListLogMetrics` operation is
// continued.
func (c *ProjectsMetricsListCall) PageToken(pageToken string) *ProjectsMetricsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsMetricsListCall) QuotaUser(quotaUser string) *ProjectsMetricsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsMetricsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/metrics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.metrics.list" call.
// Exactly one of *ListLogMetricsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLogMetricsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsMetricsListCall) Do() (*ListLogMetricsResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the logs-based metrics associated with a project.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.metrics.list",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of `LogMetric` objects to return in one operation.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "An opaque token, returned as `nextPageToken` by a prior `ListLogMetrics` operation. If `pageToken` is supplied, then the other fields of this request are ignored, and instead the previous `ListLogMetrics` operation is continued.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `projectName`. The resource name for the project whose metrics are wanted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/metrics",
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

// method id "logging.projects.metrics.update":

type ProjectsMetricsUpdateCall struct {
	s          *Service
	projectsId string
	metricsId  string
	logmetric  *LogMetric
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Creates or updates a logs-based metric.
func (r *ProjectsMetricsService) Update(projectsId string, metricsId string, logmetric *LogMetric) *ProjectsMetricsUpdateCall {
	c := &ProjectsMetricsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.metricsId = metricsId
	c.logmetric = logmetric
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsMetricsUpdateCall) QuotaUser(quotaUser string) *ProjectsMetricsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsMetricsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logmetric)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/metrics/{metricsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"metricsId":  c.metricsId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.metrics.update" call.
// Exactly one of *LogMetric or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *LogMetric.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsMetricsUpdateCall) Do() (*LogMetric, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates or updates a logs-based metric.",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.metrics.update",
	//   "parameterOrder": [
	//     "projectsId",
	//     "metricsId"
	//   ],
	//   "parameters": {
	//     "metricsId": {
	//       "description": "Part of `metricName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectsId": {
	//       "description": "Part of `metricName`. The resource name of the metric to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/metrics/{metricsId}",
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
	projectsId string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a project sink. A logs filter determines which log
// entries are written to the destination.
func (r *ProjectsSinksService) Create(projectsId string, logsink *LogSink) *ProjectsSinksCreateCall {
	c := &ProjectsSinksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSinksCreateCall) QuotaUser(quotaUser string) *ProjectsSinksCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsSinksCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.sinks.create" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksCreateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a project sink. A logs filter determines which log entries are written to the destination.",
	//   "httpMethod": "POST",
	//   "id": "logging.projects.sinks.create",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `projectName`. The resource name of the project to which the sink is bound.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/sinks",
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
	projectsId string
	sinksId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a project sink. After deletion, no new log entries
// are written to the destination.
func (r *ProjectsSinksService) Delete(projectsId string, sinksId string) *ProjectsSinksDeleteCall {
	c := &ProjectsSinksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSinksDeleteCall) QuotaUser(quotaUser string) *ProjectsSinksDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsSinksDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.sinks.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a project sink. After deletion, no new log entries are written to the destination.",
	//   "httpMethod": "DELETE",
	//   "id": "logging.projects.sinks.delete",
	//   "parameterOrder": [
	//     "projectsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the project sink to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/sinks/{sinksId}",
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
	projectsId   string
	sinksId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a project sink.
func (r *ProjectsSinksService) Get(projectsId string, sinksId string) *ProjectsSinksGetCall {
	c := &ProjectsSinksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.sinksId = sinksId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSinksGetCall) QuotaUser(quotaUser string) *ProjectsSinksGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsSinksGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.sinks.get" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksGetCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a project sink.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.sinks.get",
	//   "parameterOrder": [
	//     "projectsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the project sink to return.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/sinks/{sinksId}",
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
	projectsId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists project sinks associated with a project.
func (r *ProjectsSinksService) List(projectsId string) *ProjectsSinksListCall {
	c := &ProjectsSinksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSinksListCall) QuotaUser(quotaUser string) *ProjectsSinksListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsSinksListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/sinks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.sinks.list" call.
// Exactly one of *ListSinksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListSinksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsSinksListCall) Do() (*ListSinksResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists project sinks associated with a project.",
	//   "httpMethod": "GET",
	//   "id": "logging.projects.sinks.list",
	//   "parameterOrder": [
	//     "projectsId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `projectName`. The project whose sinks are wanted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/sinks",
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

// method id "logging.projects.sinks.update":

type ProjectsSinksUpdateCall struct {
	s          *Service
	projectsId string
	sinksId    string
	logsink    *LogSink
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a project sink. If the sink does not exist, it is
// created. The destination, filter, or both may be updated.
func (r *ProjectsSinksService) Update(projectsId string, sinksId string, logsink *LogSink) *ProjectsSinksUpdateCall {
	c := &ProjectsSinksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectsId = projectsId
	c.sinksId = sinksId
	c.logsink = logsink
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSinksUpdateCall) QuotaUser(quotaUser string) *ProjectsSinksUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *ProjectsSinksUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logsink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta3/projects/{projectsId}/sinks/{sinksId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectsId": c.projectsId,
		"sinksId":    c.sinksId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "logging.projects.sinks.update" call.
// Exactly one of *LogSink or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *LogSink.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSinksUpdateCall) Do() (*LogSink, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a project sink. If the sink does not exist, it is created. The destination, filter, or both may be updated.",
	//   "httpMethod": "PUT",
	//   "id": "logging.projects.sinks.update",
	//   "parameterOrder": [
	//     "projectsId",
	//     "sinksId"
	//   ],
	//   "parameters": {
	//     "projectsId": {
	//       "description": "Part of `sinkName`. The resource name of the project sink to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sinksId": {
	//       "description": "Part of `sinkName`. See documentation of `projectsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta3/projects/{projectsId}/sinks/{sinksId}",
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
