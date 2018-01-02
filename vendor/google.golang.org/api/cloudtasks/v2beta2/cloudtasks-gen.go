// Package cloudtasks provides access to the Cloud Tasks API.
//
// See https://cloud.google.com/cloud-tasks/
//
// Usage example:
//
//   import "google.golang.org/api/cloudtasks/v2beta2"
//   ...
//   cloudtasksService, err := cloudtasks.New(oauthHttpClient)
package cloudtasks // import "google.golang.org/api/cloudtasks/v2beta2"

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

const apiId = "cloudtasks:v2beta2"
const apiName = "cloudtasks"
const apiVersion = "v2beta2"
const basePath = "https://cloudtasks.googleapis.com/"

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
	rs.Locations = NewProjectsLocationsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Locations *ProjectsLocationsService
}

func NewProjectsLocationsService(s *Service) *ProjectsLocationsService {
	rs := &ProjectsLocationsService{s: s}
	rs.Queues = NewProjectsLocationsQueuesService(s)
	return rs
}

type ProjectsLocationsService struct {
	s *Service

	Queues *ProjectsLocationsQueuesService
}

func NewProjectsLocationsQueuesService(s *Service) *ProjectsLocationsQueuesService {
	rs := &ProjectsLocationsQueuesService{s: s}
	rs.Tasks = NewProjectsLocationsQueuesTasksService(s)
	return rs
}

type ProjectsLocationsQueuesService struct {
	s *Service

	Tasks *ProjectsLocationsQueuesTasksService
}

func NewProjectsLocationsQueuesTasksService(s *Service) *ProjectsLocationsQueuesTasksService {
	rs := &ProjectsLocationsQueuesTasksService{s: s}
	return rs
}

type ProjectsLocationsQueuesTasksService struct {
	s *Service
}

// AcknowledgeTaskRequest: Request message for acknowledging a task
// using
// CloudTasks.AcknowledgeTask.
type AcknowledgeTaskRequest struct {
	// ScheduleTime: Required.
	//
	// The task's current schedule time, available in the
	// Task.schedule_time
	// returned in PullTasksResponse.tasks or
	// CloudTasks.RenewLease. This restriction is to check that
	// the caller is acknowledging the correct task.
	ScheduleTime string `json:"scheduleTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ScheduleTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ScheduleTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AcknowledgeTaskRequest) MarshalJSON() ([]byte, error) {
	type noMethod AcknowledgeTaskRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppEngineHttpRequest: App Engine HTTP request.
//
// The message defines the HTTP request that is sent to an App Engine
// app when
// the task is dispatched.
//
// This proto can only be used for tasks in a queue which
// has
// Queue.app_engine_http_target set.
//
// Using this type of target
// requires
// [`appengine.applications.get`](/appengine/docs/admin-api/acce
// ss-control)
// Google IAM permission for the project
// and the following
// scope:
//
// `https://www.googleapis.com/auth/cloud-platform`
//
// The task will be delivered to the App Engine app which belongs to the
// same
// project as the queue. For more information, see
// [How Requests are
// Routed](/appengine/docs/standard/python/how-requests-are-routed)
// and how routing is affected by
// [dispatch files](/appengine/docs/python/config/dispatchref).
//
// The AppEngineRouting used to construct the URL that the task
// is
// delivered to can be set at the queue-level or task-level:
//
// *  If set, AppEngineHttpTarget.app_engine_routing_override is used
// for
//    all tasks in the queue, no matter what the setting is for the
//    task-level app_engine_routing.
//
//
// The `url` that the task will be sent to is:
//
// * `url =` AppEngineRouting.host `+`
// AppEngineHttpRequest.relative_url
//
// The task will be sent to a task handler by an HTTP
// request using the specified AppEngineHttpRequest.http_method (for
// example
// POST, HTTP GET, etc). The task attempt has succeeded if the task
// handler
// returns an HTTP response code in the range [200 - 299]. Error 503
// is
// considered an App Engine system error instead of an application
// error.
// Requests returning error 503 will be retried regardless of
// retry
// configuration and not counted against retry counts.
// Any other response code or a failure to receive a response before
// the
// deadline is a failed attempt.
type AppEngineHttpRequest struct {
	// AppEngineRouting: Task-level setting for App Engine routing.
	//
	// If set, AppEngineHttpTarget.app_engine_routing_override is used
	// for
	// all tasks in the queue, no matter what the setting is for
	// the
	// task-level app_engine_routing.
	AppEngineRouting *AppEngineRouting `json:"appEngineRouting,omitempty"`

	// Headers: HTTP request headers.
	//
	// This map contains the header field names and values.
	// Headers can be set when the
	// [task is
	// created](google.cloud.tasks.v2beta2.CloudTasks.CreateTask).
	// Repeated headers are not supported but a header value can contain
	// commas.
	//
	// Cloud Tasks sets some headers to default values:
	//
	// * `User-Agent`: By default, this header is
	//   "AppEngine-Google; (+http://code.google.com/appengine)".
	//   This header can be modified, but Cloud Tasks will append
	//   "AppEngine-Google; (+http://code.google.com/appengine)" to the
	//   modified `User-Agent`.
	//
	// If the task has an AppEngineHttpRequest.payload, Cloud Tasks sets
	// the
	// following headers:
	//
	// * `Content-Type`: By default, the `Content-Type` header is set to
	//   "application/octet-stream". The default can be overridden by
	// explictly
	//   setting `Content-Type` to a particular media type when the
	//   [task is
	// created](google.cloud.tasks.v2beta2.CloudTasks.CreateTask).
	//   For example, `Content-Type` can be set to "application/json".
	// * `Content-Length`: This is computed by Cloud Tasks. This value is
	//   output only. It cannot be changed.
	//
	// The headers below cannot be set or overridden:
	//
	// * `Host`
	// * `X-Google-*`
	// * `X-AppEngine-*`
	//
	// In addition, some App Engine headers, which contain
	// task-specific information, are also be sent to the task handler;
	// see
	// [request
	// headers](/appengine/docs/python/taskqueue/push/creating-handlers#readi
	// ng_request_headers).
	Headers map[string]string `json:"headers,omitempty"`

	// HttpMethod: The HTTP method to use for the request. The default is
	// POST.
	//
	// The app's request handler for the task's target URL must be able to
	// handle
	// HTTP requests with this http_method, otherwise the task attempt will
	// fail
	// with error code 405 "Method Not Allowed" because "the method
	// specified in
	// the Request-Line is not allowed for the resource identified by
	// the
	// Request-URI". See
	// [Writing a push task request
	// handler](/appengine/docs/java/taskqueue/push/creating-handlers#writing
	// _a_push_task_request_handler)
	// and the documentation for the request handlers in the language your
	// app is
	// written in e.g.
	// [python
	// RequestHandler](/appengine/docs/python/tools/webapp/requesthandlerclas
	// s).
	//
	// Possible values:
	//   "HTTP_METHOD_UNSPECIFIED" - HTTP method unspecified
	//   "POST" - HTTP Post
	//   "GET" - HTTP Get
	//   "HEAD" - HTTP Head
	//   "PUT" - HTTP Put
	//   "DELETE" - HTTP Delete
	HttpMethod string `json:"httpMethod,omitempty"`

	// Payload: Payload.
	//
	// The payload will be sent as the HTTP message body. A message
	// body, and thus a payload, is allowed only if the HTTP method is
	// POST or PUT. It is an error to set a data payload on a task with
	// an incompatible HttpMethod.
	Payload string `json:"payload,omitempty"`

	// RelativeUrl: The relative URL.
	//
	// The relative URL must begin with "/" and must be a valid HTTP
	// relative URL.
	// It can contain a path, query string arguments, and `#` fragments.
	// If the relative URL is empty, then the root path "/" will be used.
	// No spaces are allowed, and the maximum length allowed is 2083
	// characters.
	RelativeUrl string `json:"relativeUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppEngineRouting") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineRouting") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AppEngineHttpRequest) MarshalJSON() ([]byte, error) {
	type noMethod AppEngineHttpRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppEngineHttpTarget: App Engine HTTP target.
//
// The task will be delivered to the App Engine application
// hostname
// specified by its AppEngineHttpTarget and AppEngineHttpRequest.
// The documentation for AppEngineHttpRequest explains how the
// task's host URL is constructed.
//
// Using this type of queue configuration
// requires
// [`appengine.applications.get`](/appengine/docs/admin-api/acce
// ss-control)
// Google IAM permission for the project
// and the following
// scope:
//
// `https://www.googleapis.com/auth/cloud-platform`
type AppEngineHttpTarget struct {
	// AppEngineRoutingOverride: Overrides for the
	// task-level app_engine_routing.
	//
	// If set, AppEngineHttpTarget.app_engine_routing_override is used
	// for
	// all tasks in the queue, no matter what the setting is for
	// the
	// task-level app_engine_routing.
	AppEngineRoutingOverride *AppEngineRouting `json:"appEngineRoutingOverride,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AppEngineRoutingOverride") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineRoutingOverride")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AppEngineHttpTarget) MarshalJSON() ([]byte, error) {
	type noMethod AppEngineHttpTarget
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppEngineQueueConfig: Deprecated. Use AppEngineTarget.
type AppEngineQueueConfig struct {
	// AppEngineRoutingOverride: Deprecated. Use
	// AppEngineTarget.app_engine_routing_override.
	AppEngineRoutingOverride *AppEngineRouting `json:"appEngineRoutingOverride,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AppEngineRoutingOverride") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineRoutingOverride")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AppEngineQueueConfig) MarshalJSON() ([]byte, error) {
	type noMethod AppEngineQueueConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppEngineRouting: App Engine Routing.
//
// For more information about services, versions, and instances see
// [An Overview of App
// Engine](/appengine/docs/python/an-overview-of-app-engine),
// [Microservi
// ces Architecture on Google App
// Engine](/appengine/docs/python/microservices-on-app-engine),
// [App Engine Standard request
// routing](/appengine/docs/standard/python/how-requests-are-routed),
// and
//  [App Engine Flex request
// routing](/appengine/docs/flexible/python/how-requests-are-routed).
type AppEngineRouting struct {
	// Host: Output only.
	//
	// The host that the task is sent to. For more information, see
	// [How Requests are
	// Routed](/appengine/docs/standard/python/how-requests-are-routed).
	//
	// The
	//  host is constructed as:
	//
	//
	// * `host = [application_domain_name]`</br>
	//   `| [service] + '.' + [application_domain_name]`</br>
	//   `| [version] + '.' + [application_domain_name]`</br>
	//   `| [version_dot_service]+ '.' + [application_domain_name]`</br>
	//   `| [instance] + '.' + [application_domain_name]`</br>
	//   `| [instance_dot_service] + '.' + [application_domain_name]`</br>
	//   `| [instance_dot_version] + '.' + [application_domain_name]`</br>
	//   `| [instance_dot_version_dot_service] + '.' +
	// [application_domain_name]`
	//
	// * `application_domain_name` = The domain name of the app, for
	//   example <app-id>.appspot.com, which is associated with the
	//   queue's project ID. Some tasks which were created using the App
	// Engine
	//   SDK use a custom domain name.
	//
	// * `service =` AppEngineRouting.service
	//
	// * `version =` AppEngineRouting.version
	//
	// * `version_dot_service =`
	//   AppEngineRouting.version `+ '.' +` AppEngineRouting.service
	//
	// * `instance =` AppEngineRouting.instance
	//
	// * `instance_dot_service =`
	//   AppEngineRouting.instance `+ '.' +` AppEngineRouting.service
	//
	// * `instance_dot_version =`
	//   AppEngineRouting.instance `+ '.' +` AppEngineRouting.version
	//
	// * `instance_dot_version_dot_service =`
	//   AppEngineRouting.instance `+ '.' +`
	//   AppEngineRouting.version `+ '.' +` AppEngineRouting.service
	//
	// If AppEngineRouting.service is empty, then the task will be sent
	// to the service which is the default service when the task is
	// attempted.
	//
	// If AppEngineRouting.version is empty, then the task will be sent
	// to the version which is the default version when the task is
	// attempted.
	//
	// If AppEngineRouting.instance is empty, then the task will be sent
	// to an instance which is available when the task is attempted.
	//
	// When AppEngineRouting.service is "default",
	// AppEngineRouting.version is "default", and
	// AppEngineRouting.instance is empty, AppEngineRouting.host
	// is
	// shortened to just the `application_domain_name`.
	//
	// If AppEngineRouting.service, AppEngineRouting.version,
	// or
	// AppEngineRouting.instance is invalid, then the task will be sent
	// to the default version of the default service when the task is
	// attempted.
	Host string `json:"host,omitempty"`

	// Instance: App instance.
	//
	// By default, the task is sent to an instance which is available
	// when
	// the task is attempted.
	//
	// Requests can only be sent to a specific instance if
	// [manual scaling is used in App Engine
	// Standard](/appengine/docs/python/an-overview-of-app-engine?hl=en_US#sc
	// aling_types_and_instance_classes).
	// App Engine Flex does not support instances. For more information,
	// see
	// [App Engine Standard request
	// routing](/appengine/docs/standard/python/how-requests-are-routed)
	// and [App Engine Flex request
	// routing](/appengine/docs/flexible/python/how-requests-are-routed).
	Instance string `json:"instance,omitempty"`

	// Service: App service.
	//
	// By default, the task is sent to the service which is the
	// default
	// service when the task is attempted ("default").
	//
	// For some queues or tasks which were created using the App Engine Task
	// Queue
	// API, AppEngineRouting.host is not parsable
	// into
	// AppEngineRouting.service, AppEngineRouting.version,
	// and
	// AppEngineRouting.instance. For example, some tasks which were
	// created
	// using the App Engine SDK use a custom domain name; custom domains are
	// not
	// parsed by Cloud Tasks. If AppEngineRouting.host is not parsable,
	// then
	// AppEngineRouting.service, AppEngineRouting.version,
	// and
	// AppEngineRouting.instance are the empty string.
	Service string `json:"service,omitempty"`

	// Version: App version.
	//
	// By default, the task is sent to the version which is the
	// default
	// version when the task is attempted ("default").
	//
	// For some queues or tasks which were created using the App Engine Task
	// Queue
	// API, AppEngineRouting.host is not parsable
	// into
	// AppEngineRouting.service, AppEngineRouting.version,
	// and
	// AppEngineRouting.instance. For example, some tasks which were
	// created
	// using the App Engine SDK use a custom domain name; custom domains are
	// not
	// parsed by Cloud Tasks. If AppEngineRouting.host is not parsable,
	// then
	// AppEngineRouting.service, AppEngineRouting.version,
	// and
	// AppEngineRouting.instance are the empty string.
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Host") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Host") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppEngineRouting) MarshalJSON() ([]byte, error) {
	type noMethod AppEngineRouting
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppEngineTaskTarget: App Engine task target.
//
// An App Engine task is a task that has AppEngineTaskTarget set.
//
// This proto can only be used for tasks in a queue which
// has
// Queue.app_engine_queue_config set.
//
// Using this type of task target
// requires
// [`appengine.applications.get`](/appengine/docs/admin-api/acce
// ss-control)
// Google IAM permission for the project
// and the following
// scope:
//
// `https://www.googleapis.com/auth/cloud-platform`
//
// The task will be delivered to the URL specified by
// the
// AppEngineQueueConfig and AppEngineTaskTarget in the App Engine
// app
// which belongs to the same project as the queue. For more information,
// see
// [How Requests are
// Routed](/appengine/docs/standard/python/how-requests-are-routed)
// and how routing is affected by
// [dispatch files](/appengine/docs/python/config/dispatchref).
//
// The AppEngineRouting used to construct the URL can be set at
// the queue-level or task-level:
//
// *  If set, AppEngineQueueConfig.app_engine_routing_override is used
// for
//    all tasks in the queue, no matter what the setting is for the
//    task-level app_engine_routing.
//
//
// The `url` that the task will be sent to is:
//
// * `url =` AppEngineRouting.host `+`
// AppEngineTaskTarget.relative_url
//
// The task will be sent to a task handler by an HTTP
// request using the specified AppEngineTaskTarget.http_method (for
// example
// POST, HTTP GET, etc). The task attempt has succeeded if the task
// handler
// returns an HTTP response code in the range [200 - 299]. Error 503
// is
// considered an App Engine system error instead of an application
// error.
// Requests returning error 503 will be retried regardless of
// retry
// configuration and not counted against retry counts.
// Any other response code or a failure to receive a response before
// the
// deadline is a failed attempt.
type AppEngineTaskTarget struct {
	// AppEngineRouting: Task-level setting for App Engine routing.
	//
	// If set, AppEngineQueueConfig.app_engine_routing_override is used
	// for
	// all tasks in the queue, no matter what the setting is for
	// the
	// task-level app_engine_routing.
	AppEngineRouting *AppEngineRouting `json:"appEngineRouting,omitempty"`

	// Headers: HTTP request headers.
	//
	// This map contains the header field names and values.
	// Headers can be set when the
	// [task is
	// created](google.cloud.tasks.v2beta2.CloudTasks.CreateTask).
	// Repeated headers are not supported but a header value can contain
	// commas.
	//
	// Cloud Tasks sets some headers to default values:
	//
	// * `User-Agent`: By default, this header is
	//   "AppEngine-Google; (+http://code.google.com/appengine)".
	//   This header can be modified, but Cloud Tasks will append
	//   "AppEngine-Google; (+http://code.google.com/appengine)" to the
	//   modified `User-Agent`.
	//
	// If the task has an AppEngineTaskTarget.payload, Cloud Tasks sets
	// the
	// following headers:
	//
	// * `Content-Type`: By default, the `Content-Type` header is set to
	//   "application/octet-stream". The default can be overridden by
	// explictly
	//   setting `Content-Type` to a particular media type when the
	//   [task is
	// created](google.cloud.tasks.v2beta2.CloudTasks.CreateTask).
	//   For example, `Content-Type` can be set to "application/json".
	// * `Content-Length`: This is computed by Cloud Tasks. This value is
	//   output only. It cannot be changed.
	//
	// The headers below cannot be set or overridden:
	//
	// * `Host`
	// * `X-Google-*`
	// * `X-AppEngine-*`
	//
	// In addition, some App Engine headers, which contain
	// task-specific information, are also be sent to the task handler;
	// see
	// [request
	// headers](/appengine/docs/python/taskqueue/push/creating-handlers#readi
	// ng_request_headers).
	Headers map[string]string `json:"headers,omitempty"`

	// HttpMethod: The HTTP method to use for the request. The default is
	// POST.
	//
	// The app's request handler for the task's target URL must be able to
	// handle
	// HTTP requests with this http_method, otherwise the task attempt will
	// fail
	// with error code 405 "Method Not Allowed" because "the method
	// specified in
	// the Request-Line is not allowed for the resource identified by
	// the
	// Request-URI". See
	// [Writing a push task request
	// handler](/appengine/docs/java/taskqueue/push/creating-handlers#writing
	// _a_push_task_request_handler)
	// and the documentation for the request handlers in the language your
	// app is
	// written in e.g.
	// [python
	// RequestHandler](/appengine/docs/python/tools/webapp/requesthandlerclas
	// s).
	//
	// Possible values:
	//   "HTTP_METHOD_UNSPECIFIED" - HTTP method unspecified
	//   "POST" - HTTP Post
	//   "GET" - HTTP Get
	//   "HEAD" - HTTP Head
	//   "PUT" - HTTP Put
	//   "DELETE" - HTTP Delete
	HttpMethod string `json:"httpMethod,omitempty"`

	// Payload: Payload.
	//
	// The payload will be sent as the HTTP message body. A message
	// body, and thus a payload, is allowed only if the HTTP method is
	// POST or PUT. It is an error to set a data payload on a task with
	// an incompatible HttpMethod.
	Payload string `json:"payload,omitempty"`

	// RelativeUrl: The relative URL.
	//
	// The relative URL must begin with "/" and must be a valid HTTP
	// relative URL.
	// It can contain a path, query string arguments, and `#` fragments.
	// If the relative URL is empty, then the root path "/" will be used.
	// No spaces are allowed, and the maximum length allowed is 2083
	// characters.
	RelativeUrl string `json:"relativeUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AppEngineRouting") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineRouting") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AppEngineTaskTarget) MarshalJSON() ([]byte, error) {
	type noMethod AppEngineTaskTarget
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AttemptStatus: The status of a task attempt.
type AttemptStatus struct {
	// DispatchTime: Output only.
	//
	// The time that this attempt was dispatched.
	//
	// `dispatch_time` will be truncated to the nearest microsecond.
	DispatchTime string `json:"dispatchTime,omitempty"`

	// ResponseStatus: Output only.
	//
	// The response from the target for this attempt.
	//
	// If the task has not been attempted or the task is currently
	// running
	// then the response status is google.rpc.Code.UNKNOWN.
	ResponseStatus *Status `json:"responseStatus,omitempty"`

	// ResponseTime: Output only.
	//
	// The time that this attempt response was received.
	//
	// `response_time` will be truncated to the nearest microsecond.
	ResponseTime string `json:"responseTime,omitempty"`

	// ScheduleTime: Output only.
	//
	// The time that this attempt was scheduled.
	//
	// `schedule_time` will be truncated to the nearest microsecond.
	ScheduleTime string `json:"scheduleTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DispatchTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DispatchTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AttemptStatus) MarshalJSON() ([]byte, error) {
	type noMethod AttemptStatus
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

// CancelLeaseRequest: Request message for canceling a lease
// using
// CloudTasks.CancelLease.
type CancelLeaseRequest struct {
	// ResponseView: The response_view specifies which subset of the Task
	// will be
	// returned.
	//
	// By default response_view is Task.View.BASIC; not all
	// information is retrieved by default because some data, such
	// as
	// payloads, might be desirable to return only when needed because
	// of its large size or because of the sensitivity of data that
	// it
	// contains.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView`
	// [Google IAM](/iam/) permission on the
	// Task.name resource.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	ResponseView string `json:"responseView,omitempty"`

	// ScheduleTime: Required.
	//
	// The task's current schedule time, available in the
	// Task.schedule_time
	// returned in PullTasksResponse.tasks or
	// CloudTasks.RenewLease. This restriction is to check that
	// the caller is canceling the correct task.
	ScheduleTime string `json:"scheduleTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResponseView") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ResponseView") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CancelLeaseRequest) MarshalJSON() ([]byte, error) {
	type noMethod CancelLeaseRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateTaskRequest: Request message for CloudTasks.CreateTask.
type CreateTaskRequest struct {
	// ResponseView: The response_view specifies which subset of the Task
	// will be
	// returned.
	//
	// By default response_view is Task.View.BASIC; not all
	// information is retrieved by default because some data, such
	// as
	// payloads, might be desirable to return only when needed because
	// of its large size or because of the sensitivity of data that
	// it
	// contains.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView`
	// [Google IAM](/iam/) permission on the
	// Task.name resource.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	ResponseView string `json:"responseView,omitempty"`

	// Task: Required.
	//
	// The task to add.
	//
	// Task names have the following
	// format:
	// `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tas
	// ks/TASK_ID`.
	// The user can optionally specify a name for the task in
	// Task.name. If a name is not specified then the system will
	// generate a random unique task id, which will be returned in
	// the
	// response's Task.name.
	//
	// Explicitly specifying a Task.name enables task
	// de-duplication. If a task's name is identical to the name of
	// an
	// existing task or a task that was deleted or completed within the
	// last ~10 days then the call to CloudTasks.CreateTask will
	// fail with google.rpc.Code.ALREADY_EXISTS. Because there is an
	// extra lookup cost to identify duplicate task names,
	// these
	// CloudTasks.CreateTask calls have significantly increased
	// latency. Using hashed strings for the task id or for the prefix
	// of the task id is recommended. Choosing task ids that are
	// sequential or have sequential prefixes, for example using
	// a
	// timestamp, causes an increase in latency and error rates in all
	// task commands. The infrastructure relies on an approximately
	// uniform distribution of task ids to store and serve
	// tasks
	// efficiently.
	//
	// If Task.schedule_time is not set or is in the past then Cloud
	// Tasks will set it to the current time.
	Task *Task `json:"task,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResponseView") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ResponseView") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateTaskRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateTaskRequest
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

// GetIamPolicyRequest: Request message for `GetIamPolicy` method.
type GetIamPolicyRequest struct {
}

// ListQueuesResponse: Response message for CloudTasks.ListQueues.
type ListQueuesResponse struct {
	// NextPageToken: A token to retrieve next page of results.
	//
	// To return the next page of results, call
	// CloudTasks.ListQueues with this value as
	// the
	// ListQueuesRequest.page_token.
	//
	// If the next_page_token is empty, there are no more results.
	//
	// The page token is valid for only 2 hours.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Queues: The list of queues.
	Queues []*Queue `json:"queues,omitempty"`

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

func (s *ListQueuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListQueuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTasksResponse: Response message for listing tasks using
// CloudTasks.ListTasks.
type ListTasksResponse struct {
	// NextPageToken: A token to retrieve next page of results.
	//
	// To return the next page of results, call
	// CloudTasks.ListTasks with this value as
	// the
	// ListTasksRequest.page_token.
	//
	// If the next_page_token is empty, there are no more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Tasks: The list of tasks.
	Tasks []*Task `json:"tasks,omitempty"`

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

func (s *ListTasksResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTasksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PauseQueueRequest: Request message for CloudTasks.PauseQueue.
type PauseQueueRequest struct {
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

// PullMessage: The pull message contains data that can be used by the
// caller of
// CloudTasks.PullTasks to process the task.
//
// This proto can only be used for tasks in a queue which
// has
// Queue.pull_target set.
type PullMessage struct {
	// Payload: A data payload consumed by the task worker to execute the
	// task.
	Payload string `json:"payload,omitempty"`

	// Tag: A meta-data tag for this task.
	//
	// This value is used by CloudTasks.PullTasks calls
	// when
	// PullTasksRequest.filter is `tag=<tag>`.
	//
	// The tag must be less than 500 bytes.
	Tag string `json:"tag,omitempty"`

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

func (s *PullMessage) MarshalJSON() ([]byte, error) {
	type noMethod PullMessage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PullQueueConfig: Deprecated. Use PullTarget.
type PullQueueConfig struct {
}

// PullTarget: Pull target.
type PullTarget struct {
}

// PullTaskTarget: Pull task target.
//
// A pull task is a task that has PullTaskTarget set.
//
// This proto can only be used for tasks in a queue which
// has
// Queue.pull_queue_config set.
type PullTaskTarget struct {
	// Payload: A data payload consumed by the task worker to execute the
	// task.
	Payload string `json:"payload,omitempty"`

	// Tag: A meta-data tag for this task.
	//
	// This value is used by CloudTasks.PullTasks calls
	// when
	// PullTasksRequest.filter is `tag=<tag>`.
	//
	// The tag must be less than 500 bytes.
	Tag string `json:"tag,omitempty"`

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

func (s *PullTaskTarget) MarshalJSON() ([]byte, error) {
	type noMethod PullTaskTarget
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PullTasksRequest: Request message for pulling tasks using
// CloudTasks.PullTasks.
type PullTasksRequest struct {
	// Filter: `filter` can be used to specify a subset of tasks to
	// lease.
	//
	// When `filter` is set to `tag=<my-tag>` then the
	// PullTasksResponse will contain only tasks whose
	// PullTaskTarget.tag is equal to `<my-tag>`. `<my-tag>` can be
	// a bytes encoded as a string and must be less than 500 bytes.
	// If `<my-tag>` includes whitespace or special characters (characters
	// which
	// aren't letters, numbers, or underscores), then it must be
	// double-quoted.
	// Double quotes and backslashes in quoted strings must be escaped
	// by
	// preceding it with a backslash (`\`).
	//
	// When `filter` is set to `tag=oldest_tag()`, only tasks which have the
	// same
	// tag as the task with the oldest schedule_time will be
	// returned.
	//
	// Grammar Syntax:
	//
	// * `filter = "tag=" comparable`
	//
	// *  `comparable = tag | function`
	//
	// * `tag = string | bytes`
	//
	// * `function = "oldest_tag()"
	//
	//
	//
	// The `oldest_tag()` function returns tasks which have the same tag as
	// the
	// oldest task (ordered by schedule time).
	Filter string `json:"filter,omitempty"`

	// LeaseDuration: The duration of the lease.
	//
	// Each task returned in the PullTasksResponse will have
	// its
	// Task.schedule_time set to the current time plus the
	// `lease_duration`. A task that has been returned in
	// a
	// PullTasksResponse is leased -- that task will not be
	// returned in a different PullTasksResponse before
	// the
	// Task.schedule_time.
	//
	// After the lease holder has successfully finished the work
	// associated with the task, the lease holder must
	// call
	// CloudTasks.AcknowledgeTask. If the task is not acknowledged
	// via CloudTasks.AcknowledgeTask before the
	// Task.schedule_time then it will be returned in a
	// later
	// PullTasksResponse so that another lease holder can process
	// it.
	//
	// The maximum lease duration is 1 week.
	// `lease_duration` will be truncated to the nearest second.
	LeaseDuration string `json:"leaseDuration,omitempty"`

	// MaxTasks: The maximum number of tasks to lease. The maximum that can
	// be
	// requested is 1000.
	MaxTasks int64 `json:"maxTasks,omitempty"`

	// ResponseView: The response_view specifies which subset of the Task
	// will be
	// returned.
	//
	// By default response_view is Task.View.BASIC; not all
	// information is retrieved by default because some data, such
	// as
	// payloads, might be desirable to return only when needed because
	// of its large size or because of the sensitivity of data that
	// it
	// contains.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView`
	// [Google IAM](/iam/) permission on the
	// Task.name resource.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	ResponseView string `json:"responseView,omitempty"`

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

func (s *PullTasksRequest) MarshalJSON() ([]byte, error) {
	type noMethod PullTasksRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PullTasksResponse: Response message for pulling tasks using
// CloudTasks.PullTasks.
type PullTasksResponse struct {
	// Tasks: The leased tasks.
	Tasks []*Task `json:"tasks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Tasks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Tasks") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PullTasksResponse) MarshalJSON() ([]byte, error) {
	type noMethod PullTasksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PurgeQueueRequest: Request message for CloudTasks.PurgeQueue.
type PurgeQueueRequest struct {
}

// Queue: A queue is a container of related tasks. Queues are configured
// to manage
// how those tasks are dispatched. Configurable properties include rate
// limits,
// retry options, target types, and others.
type Queue struct {
	// AppEngineHttpTarget: App Engine HTTP target.
	//
	// An App Engine queue is a queue that has an AppEngineHttpTarget.
	AppEngineHttpTarget *AppEngineHttpTarget `json:"appEngineHttpTarget,omitempty"`

	// AppEngineQueueConfig: Deprecated. Use Queue.app_engine_http_target.
	AppEngineQueueConfig *AppEngineQueueConfig `json:"appEngineQueueConfig,omitempty"`

	// Name: The queue name.
	//
	// The queue name must have the following
	// format:
	// `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`
	//
	// *
	//  `PROJECT_ID` can contain uppercase and lowercase letters,
	//   numbers, hyphens, colons, and periods; that is, it must match
	//   the regular expression: `[a-zA-Z\\d-:\\.]+`.
	// * `QUEUE_ID` can contain uppercase and lowercase letters,
	//   numbers, and hyphens; that is, it must match the regular
	//   expression: `[a-zA-Z\\d-]+`. The maximum length is 100
	//   characters.
	//
	// Caller-specified and required in CreateQueueRequest, after which
	// it becomes output only.
	Name string `json:"name,omitempty"`

	// PullQueueConfig: Deprecated. Use Queue.pull_target.
	PullQueueConfig *PullQueueConfig `json:"pullQueueConfig,omitempty"`

	// PullTarget: Pull target.
	//
	// A pull queue is a queue that has a PullTarget.
	PullTarget *PullTarget `json:"pullTarget,omitempty"`

	// PurgeTime: Output only.
	//
	// The last time this queue was purged. All tasks that were
	// created before this time were purged.
	//
	// A queue can be purged using CloudTasks.PurgeQueue, the
	// [App Engine Task Queue SDK, or the Cloud
	// Console](/appengine/docs/standard/python/taskqueue/push/deleting-tasks
	// -and-queues#purging_all_tasks_from_a_queue).
	//
	// Purge time will be truncated to the nearest microsecond. Purge
	// time will be zero if the queue has never been purged.
	PurgeTime string `json:"purgeTime,omitempty"`

	// QueueState: Output only.
	//
	// The state of the queue.
	//
	// `queue_state` can only be changed by called
	// CloudTasks.PauseQueue, CloudTasks.ResumeQueue, or
	// uploading
	// [queue.yaml](/appengine/docs/python/config/queueref).
	// CloudT
	// asks.UpdateQueue cannot be used to change `queue_state`.
	//
	// Possible values:
	//   "QUEUE_STATE_UNSPECIFIED" - Unspecified state.
	//   "RUNNING" - The queue is running. Tasks can be dispatched.
	//   "PAUSED" - Tasks are paused by the user. If the queue is paused
	// then Cloud
	// Tasks will stop delivering tasks from it, but more tasks can
	// still be added to it by the user. When a pull queue is paused,
	// all CloudTasks.PullTasks calls will return a
	// `FAILED_PRECONDITION` error.
	//   "DISABLED" - The queue is disabled.
	//
	// A queue becomes `DISABLED`
	// when
	// [queue.yaml](/appengine/docs/python/config/queueref)
	// or
	// [queue.xml](appengine/docs/standard/java/config/queueref) is
	// uploaded
	// which does not contain the queue. You cannot directly disable a
	// queue.
	//
	// When a queue is disabled, tasks can still be added to a queue
	// but the tasks are not dispatched and CloudTasks.PullTasks
	// calls
	// return a `FAILED_PRECONDITION` error.
	//
	// To permanently delete this queue and all of its tasks,
	// call
	// CloudTasks.DeleteQueue.
	QueueState string `json:"queueState,omitempty"`

	// RetryConfig: Settings that determine the retry behavior.
	//
	// * For tasks created using Cloud Tasks: the queue-level retry
	// settings
	//   apply to all tasks in the queue that were created using Cloud
	// Tasks.
	//   Retry settings cannot be set on individual tasks.
	// * For tasks created using the App Engine SDK: the queue-level retry
	//   settings apply to all tasks in the queue which do not have retry
	// settings
	//   explicitly set on the task and were created by the App Engine SDK.
	// See
	//   [App Engine
	// documentation](/appengine/docs/standard/python/taskqueue/push/retrying
	// -tasks).
	RetryConfig *RetryConfig `json:"retryConfig,omitempty"`

	// ThrottleConfig: Config for throttling task dispatches.
	ThrottleConfig *ThrottleConfig `json:"throttleConfig,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AppEngineHttpTarget")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineHttpTarget") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Queue) MarshalJSON() ([]byte, error) {
	type noMethod Queue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RenewLeaseRequest: Request message for renewing a lease using
// CloudTasks.RenewLease.
type RenewLeaseRequest struct {
	// NewLeaseDuration: Required.
	//
	// The desired new lease duration, starting from now.
	//
	//
	// The maximum lease duration is 1 week.
	// `new_lease_duration` will be truncated to the nearest second.
	NewLeaseDuration string `json:"newLeaseDuration,omitempty"`

	// ResponseView: The response_view specifies which subset of the Task
	// will be
	// returned.
	//
	// By default response_view is Task.View.BASIC; not all
	// information is retrieved by default because some data, such
	// as
	// payloads, might be desirable to return only when needed because
	// of its large size or because of the sensitivity of data that
	// it
	// contains.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView`
	// [Google IAM](/iam/) permission on the
	// Task.name resource.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	ResponseView string `json:"responseView,omitempty"`

	// ScheduleTime: Required.
	//
	// The task's current schedule time, available in the
	// Task.schedule_time
	// returned in PullTasksResponse.tasks or
	// CloudTasks.RenewLease. This restriction is to check that
	// the caller is renewing the correct task.
	ScheduleTime string `json:"scheduleTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewLeaseDuration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewLeaseDuration") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RenewLeaseRequest) MarshalJSON() ([]byte, error) {
	type noMethod RenewLeaseRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ResumeQueueRequest: Request message for CloudTasks.ResumeQueue.
type ResumeQueueRequest struct {
}

// RetryConfig: Retry config.
//
// These settings determine retry behavior.
//
// If a task does not complete successfully, meaning that
// an
// acknowledgement is not received from the handler before
// the
// [deadline](/appengine/docs/python/taskqueue/push/#the_task_deadlin
// e),
// then it will be retried with exponential backoff according to
// the
// settings in RetryConfig.
type RetryConfig struct {
	// MaxAttempts: The maximum number of attempts for a task.
	//
	// Cloud Tasks will attempt the task `max_attempts` times (that
	// is, if the first attempt fails, then there will be
	// `max_attempts - 1` retries).  Must be > 0.
	MaxAttempts int64 `json:"maxAttempts,omitempty"`

	// MaxBackoff: The maximum amount of time to wait before retrying a task
	// after
	// it fails. The default is 1 hour.
	//
	// * For [App Engine
	// queues](google.cloud.tasks.v2beta2.AppEngineHttpTarget),
	//   this field is 1 hour by default.
	// * For [pull queues](google.cloud.tasks.v2beta2.PullTarget), this
	// field
	//   is output only and always 0.
	//
	// `max_backoff` will be truncated to the nearest second.
	MaxBackoff string `json:"maxBackoff,omitempty"`

	// MaxDoublings: The maximum number of times that the interval between
	// failed task
	// retries will be doubled before the increase becomes constant.
	// The
	// constant is: 2**(max_doublings - 1) *
	// RetryConfig.min_backoff.
	//
	// * For [App Engine
	// queues](google.cloud.tasks.v2beta2.AppEngineHttpTarget),
	//   this field is 16 by default.
	// * For [pull queues](google.cloud.tasks.v2beta2.PullTarget), this
	// field
	//   is output only and always 0.
	MaxDoublings int64 `json:"maxDoublings,omitempty"`

	// MinBackoff: The minimum amount of time to wait before retrying a task
	// after
	// it fails.
	//
	// * For [App Engine
	// queues](google.cloud.tasks.v2beta2.AppEngineHttpTarget),
	//   this field is 0.1 seconds by default.
	// * For [pull queues](google.cloud.tasks.v2beta2.PullTarget), this
	//   field is output only and always 0.
	//
	// `min_backoff` will be truncated to the nearest second.
	MinBackoff string `json:"minBackoff,omitempty"`

	// TaskAgeLimit: If positive, task_age_limit specifies the time limit
	// for retrying a failed
	// task, measured from when the task was first run. If specified
	// with
	// RetryConfig.max_attempts, the task will be retried until both
	// limits are reached.
	//
	// If zero, then the task age is unlimited. This field is zero by
	// default.
	//
	// `task_age_limit` will be truncated to the nearest second.
	TaskAgeLimit string `json:"taskAgeLimit,omitempty"`

	// UnlimitedAttempts: If true, then the number of attempts is unlimited.
	UnlimitedAttempts bool `json:"unlimitedAttempts,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxAttempts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxAttempts") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RetryConfig) MarshalJSON() ([]byte, error) {
	type noMethod RetryConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RunTaskRequest: Request message for forcing a task to run now
// using
// CloudTasks.RunTask.
type RunTaskRequest struct {
	// ResponseView: The response_view specifies which subset of the Task
	// will be
	// returned.
	//
	// By default response_view is Task.View.BASIC; not all
	// information is retrieved by default because some data, such
	// as
	// payloads, might be desirable to return only when needed because
	// of its large size or because of the sensitivity of data that
	// it
	// contains.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView`
	// [Google IAM](/iam/) permission on the
	// Task.name resource.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	ResponseView string `json:"responseView,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResponseView") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ResponseView") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RunTaskRequest) MarshalJSON() ([]byte, error) {
	type noMethod RunTaskRequest
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

// Task: A unit of scheduled work.
type Task struct {
	// AppEngineHttpRequest: App Engine HTTP request that is sent to the
	// task's target. Can be set
	// only if Queue.app_engine_http_target is set.
	//
	// An App Engine task is a task that has AppEngineHttpRequest set.
	AppEngineHttpRequest *AppEngineHttpRequest `json:"appEngineHttpRequest,omitempty"`

	// AppEngineTaskTarget: Deprecated. Use Task.app_engine_http_request.
	AppEngineTaskTarget *AppEngineTaskTarget `json:"appEngineTaskTarget,omitempty"`

	// CreateTime: Output only.
	//
	// The time that the task was created.
	//
	// `create_time` will be truncated to the nearest second.
	CreateTime string `json:"createTime,omitempty"`

	// Name: The task name.
	//
	// The task name must have the following
	// format:
	// `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tas
	// ks/TASK_ID`
	//
	// * `PROJECT_ID` can contain uppercase and lowercase letters,
	//   numbers, hyphens, colons, and periods; that is, it must match
	//   the regular expression: `[a-zA-Z\\d-:\\.]+`.
	// * `QUEUE_ID` can contain uppercase and lowercase letters,
	//   numbers, and hyphens; that is, it must match the regular
	//   expression: `[a-zA-Z\\d-]+`. The maximum length is 100
	//   characters.
	// * `TASK_ID` contain uppercase and lowercase letters, numbers,
	//   underscores, and hyphens; that is, it must match the regular
	//   expression: `[a-zA-Z\\d_-]+`. The maximum length is 500
	//   characters.
	//
	// Optionally caller-specified in CreateTaskRequest.
	Name string `json:"name,omitempty"`

	// PullMessage: Pull message contains data that should be used by the
	// caller of
	// CloudTasks.PullTasks to process the task. Can be set only
	// if
	// Queue.pull_target is set.
	//
	// A pull task is a task that has PullMessage set.
	PullMessage *PullMessage `json:"pullMessage,omitempty"`

	// PullTaskTarget: Deprecated. Use Task.pull_message.
	PullTaskTarget *PullTaskTarget `json:"pullTaskTarget,omitempty"`

	// ScheduleTime: The time when the task is scheduled to be
	// attempted.
	//
	// For pull queues, this is the time when the task is available to
	// be leased; if a task is currently leased, this is the time when
	// the current lease expires, that is, the time that the task was
	// leased plus the PullTasksRequest.lease_duration.
	//
	// For App Engine queues, this is when the task will be attempted or
	// retried.
	//
	// `schedule_time` will be truncated to the nearest microsecond.
	ScheduleTime string `json:"scheduleTime,omitempty"`

	// TaskStatus: Output only.
	//
	// Task status.
	TaskStatus *TaskStatus `json:"taskStatus,omitempty"`

	// View: Output only.
	//
	// The view specifies which subset of the Task has been
	// returned.
	//
	// Possible values:
	//   "VIEW_UNSPECIFIED" - Unspecified. Defaults to BASIC.
	//   "BASIC" - The basic view omits fields which can be large or can
	// contain
	// sensitive data.
	//
	// This view does not include the payload.
	//   "FULL" - All information is returned.
	//
	// Payloads might be desirable to return only when needed, because
	// they can be large and because of the sensitivity of the data
	// that you choose to store in it.
	//
	// Authorization for Task.View.FULL requires
	// `cloudtasks.tasks.fullView` [Google
	// IAM](https://cloud.google.com/iam/)
	// permission on the Queue.name resource.
	View string `json:"view,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "AppEngineHttpRequest") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppEngineHttpRequest") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Task) MarshalJSON() ([]byte, error) {
	type noMethod Task
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TaskStatus: Status of the task.
type TaskStatus struct {
	// AttemptDispatchCount: Output only.
	//
	// The number of attempts dispatched. This count includes tasks which
	// have
	// been dispatched but haven't received a response.
	AttemptDispatchCount int64 `json:"attemptDispatchCount,omitempty,string"`

	// AttemptResponseCount: Output only.
	//
	// The number of attempts which have received a response.
	//
	// This field is not calculated for
	// [pull tasks](google.cloud.tasks.v2beta2.PullTaskTarget).
	AttemptResponseCount int64 `json:"attemptResponseCount,omitempty,string"`

	// FirstAttemptStatus: Output only.
	//
	// The status of the task's first attempt.
	//
	// Only AttemptStatus.dispatch_time will be set.
	// The other AttemptStatus information is not retained by Cloud
	// Tasks.
	//
	// This field is not calculated for
	// [pull tasks](google.cloud.tasks.v2beta2.PullTaskTarget).
	FirstAttemptStatus *AttemptStatus `json:"firstAttemptStatus,omitempty"`

	// LastAttemptStatus: Output only.
	//
	// The status of the task's last attempt.
	//
	// This field is not calculated for
	// [pull tasks](google.cloud.tasks.v2beta2.PullTaskTarget).
	LastAttemptStatus *AttemptStatus `json:"lastAttemptStatus,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AttemptDispatchCount") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AttemptDispatchCount") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TaskStatus) MarshalJSON() ([]byte, error) {
	type noMethod TaskStatus
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

// ThrottleConfig: Throttle config.
//
// These settings determine the throttling behavior.
type ThrottleConfig struct {
	// MaxBurstSize: Output only.
	//
	// The max burst size limits how fast the queue is processed when
	// many tasks are in the queue and the rate is high. This field
	// allows the queue to have a high rate so processing starts
	// shortly
	// after a task is enqueued, but still limits resource usage when
	// many tasks are enqueued in a short period of time.
	//
	// * For App Engine queues, if
	//   ThrottleConfig.max_tasks_dispatched_per_second is 1, this
	//   field is 10; otherwise this field is
	//   ThrottleConfig.max_tasks_dispatched_per_second / 5.
	// * For pull queues, this field is output only and always
	// 10,000.
	//
	// Note: For App Engine queues that were created
	// through
	// `queue.yaml/xml`, `max_burst_size` might not have the same
	// settings as specified above; CloudTasks.UpdateQueue can be
	// used to set `max_burst_size` only to the values specified
	// above.
	//
	// This field has the same meaning as
	// [bucket_size in
	// queue.yaml](/appengine/docs/standard/python/config/queueref#bucket_siz
	// e).
	MaxBurstSize float64 `json:"maxBurstSize,omitempty"`

	// MaxOutstandingTasks: The maximum number of outstanding tasks that
	// Cloud Tasks allows
	// to be dispatched for this queue. After this threshold has
	// been
	// reached, Cloud Tasks stops dispatching tasks until the number
	// of
	// outstanding requests decreases.
	//
	// The maximum allowed value is 5,000.
	//
	// * For App Engine queues, this field is 10 by default.
	// * For pull queues, this field is output only and always -1, which
	//   indicates no limit.
	//
	// This field has the same meaning as
	// [max_concurrent_requests in
	// queue.yaml](/appengine/docs/standard/python/config/queueref#max_concur
	// rent_requests).
	MaxOutstandingTasks int64 `json:"maxOutstandingTasks,omitempty"`

	// MaxTasksDispatchedPerSecond: The maximum rate at which tasks are
	// dispatched from this
	// queue.
	//
	// The maximum allowed value is 500.
	//
	// * For App Engine queues, this field is 1 by default.
	// * For pull queues, this field is output only and always 10,000.
	//
	// This field has the same meaning as
	// [rate in
	// queue.yaml](/appengine/docs/standard/python/config/queueref#rate).
	MaxTasksDispatchedPerSecond float64 `json:"maxTasksDispatchedPerSecond,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxBurstSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxBurstSize") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ThrottleConfig) MarshalJSON() ([]byte, error) {
	type noMethod ThrottleConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ThrottleConfig) UnmarshalJSON(data []byte) error {
	type noMethod ThrottleConfig
	var s1 struct {
		MaxBurstSize                gensupport.JSONFloat64 `json:"maxBurstSize"`
		MaxTasksDispatchedPerSecond gensupport.JSONFloat64 `json:"maxTasksDispatchedPerSecond"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.MaxBurstSize = float64(s1.MaxBurstSize)
	s.MaxTasksDispatchedPerSecond = float64(s1.MaxTasksDispatchedPerSecond)
	return nil
}

// method id "cloudtasks.projects.locations.queues.create":

type ProjectsLocationsQueuesCreateCall struct {
	s          *Service
	parent     string
	queue      *Queue
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a queue.
//
// WARNING: This method is only available to whitelisted
// users. Using this method carries some risk. Read
// [Overview of Queue Management and
// queue.yaml](/cloud-tasks/docs/queue-yaml)
// carefully and then sign up for
// [whitelist access to this method](https://goo.gl/Fe5mUy).
func (r *ProjectsLocationsQueuesService) Create(parent string, queue *Queue) *ProjectsLocationsQueuesCreateCall {
	c := &ProjectsLocationsQueuesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.queue = queue
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesCreateCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesCreateCall) Context(ctx context.Context) *ProjectsLocationsQueuesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.queue)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+parent}/queues")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.create" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesCreateCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Creates a queue.\n\nWARNING: This method is only available to whitelisted\nusers. Using this method carries some risk. Read\n[Overview of Queue Management and queue.yaml](/cloud-tasks/docs/queue-yaml)\ncarefully and then sign up for\n[whitelist access to this method](https://goo.gl/Fe5mUy).",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required.\n\nThe location name in which the queue will be created.\nFor example: `projects/PROJECT_ID/locations/LOCATION_ID`\n\nThe list of allowed locations can be obtained by calling Cloud\nTasks' implementation of\ngoogle.cloud.location.Locations.ListLocations.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+parent}/queues",
	//   "request": {
	//     "$ref": "Queue"
	//   },
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.delete":

type ProjectsLocationsQueuesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a queue.
//
// This command will delete the queue even if it has tasks in it.
//
// Note: If you delete a queue, a queue with the same name can't be
// created
// for 7 days.
//
// WARNING: This method is only available to whitelisted
// users. Using this method carries some risk. Read
// [Overview of Queue Management and
// queue.yaml](/cloud-tasks/docs/queue-yaml)
// carefully and then sign up for
// [whitelist access to this method](https://goo.gl/Fe5mUy).
func (r *ProjectsLocationsQueuesService) Delete(name string) *ProjectsLocationsQueuesDeleteCall {
	c := &ProjectsLocationsQueuesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesDeleteCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesDeleteCall) Context(ctx context.Context) *ProjectsLocationsQueuesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a queue.\n\nThis command will delete the queue even if it has tasks in it.\n\nNote: If you delete a queue, a queue with the same name can't be created\nfor 7 days.\n\nWARNING: This method is only available to whitelisted\nusers. Using this method carries some risk. Read\n[Overview of Queue Management and queue.yaml](/cloud-tasks/docs/queue-yaml)\ncarefully and then sign up for\n[whitelist access to this method](https://goo.gl/Fe5mUy).",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}",
	//   "httpMethod": "DELETE",
	//   "id": "cloudtasks.projects.locations.queues.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.get":

type ProjectsLocationsQueuesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a queue.
func (r *ProjectsLocationsQueuesService) Get(name string) *ProjectsLocationsQueuesGetCall {
	c := &ProjectsLocationsQueuesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesGetCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLocationsQueuesGetCall) IfNoneMatch(entityTag string) *ProjectsLocationsQueuesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesGetCall) Context(ctx context.Context) *ProjectsLocationsQueuesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.get" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesGetCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Gets a queue.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}",
	//   "httpMethod": "GET",
	//   "id": "cloudtasks.projects.locations.queues.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe resource name of the queue. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}",
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.getIamPolicy":

type ProjectsLocationsQueuesGetIamPolicyCall struct {
	s                   *Service
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for a Queue.
// Returns an empty policy if the resource exists and does not have a
// policy
// set.
//
// Authorization requires the following [Google IAM](/iam) permission on
// the
// specified resource parent:
//
// * `cloudtasks.queues.getIamPolicy`
func (r *ProjectsLocationsQueuesService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *ProjectsLocationsQueuesGetIamPolicyCall {
	c := &ProjectsLocationsQueuesGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesGetIamPolicyCall) Context(ctx context.Context) *ProjectsLocationsQueuesGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets the access control policy for a Queue.\nReturns an empty policy if the resource exists and does not have a policy\nset.\n\nAuthorization requires the following [Google IAM](/iam) permission on the\nspecified resource parent:\n\n* `cloudtasks.queues.getIamPolicy`",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+resource}:getIamPolicy",
	//   "request": {
	//     "$ref": "GetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.list":

type ProjectsLocationsQueuesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists queues.
//
// Queues are returned in lexicographical order.
func (r *ProjectsLocationsQueuesService) List(parent string) *ProjectsLocationsQueuesListCall {
	c := &ProjectsLocationsQueuesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// Filter sets the optional parameter "filter": `filter` can be used to
// specify a subset of queues. Any Queue
// field can be used as a filter and several operators as supported.
// For example: `<=, <, >=, >, !=, =, :`. The filter syntax is the same
// as
// described in
// [Stackdriver's Advanced Logs
// Filters](/logging/docs/view/advanced_filters).
//
// Sample filter "app_engine_http_target: *".
//
// Note that using filters might cause fewer queues than
// the
// requested_page size to be returned.
func (c *ProjectsLocationsQueuesListCall) Filter(filter string) *ProjectsLocationsQueuesListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page
// size.
//
// The maximum page size is 9800. If unspecified, the page size will
// be the maximum. Fewer queues than requested might be returned,
// even if more queues exist; use
// ListQueuesResponse.next_page_token to determine if more
// queues exist.
func (c *ProjectsLocationsQueuesListCall) PageSize(pageSize int64) *ProjectsLocationsQueuesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying the page of results to return.
//
// To request the first page results, page_token must be empty.
// To
// request the next page of results, page_token must be the value
// of
// ListQueuesResponse.next_page_token returned from the previous
// call to CloudTasks.ListQueues method. It is an error to
// switch the value of ListQueuesRequest.filter while iterating
// through pages.
func (c *ProjectsLocationsQueuesListCall) PageToken(pageToken string) *ProjectsLocationsQueuesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesListCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLocationsQueuesListCall) IfNoneMatch(entityTag string) *ProjectsLocationsQueuesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesListCall) Context(ctx context.Context) *ProjectsLocationsQueuesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+parent}/queues")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.list" call.
// Exactly one of *ListQueuesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListQueuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLocationsQueuesListCall) Do(opts ...googleapi.CallOption) (*ListQueuesResponse, error) {
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
	ret := &ListQueuesResponse{
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
	//   "description": "Lists queues.\n\nQueues are returned in lexicographical order.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues",
	//   "httpMethod": "GET",
	//   "id": "cloudtasks.projects.locations.queues.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "`filter` can be used to specify a subset of queues. Any Queue\nfield can be used as a filter and several operators as supported.\nFor example: `\u003c=, \u003c, \u003e=, \u003e, !=, =, :`. The filter syntax is the same as\ndescribed in\n[Stackdriver's Advanced Logs Filters](/logging/docs/view/advanced_filters).\n\nSample filter \"app_engine_http_target: *\".\n\nNote that using filters might cause fewer queues than the\nrequested_page size to be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size.\n\nThe maximum page size is 9800. If unspecified, the page size will\nbe the maximum. Fewer queues than requested might be returned,\neven if more queues exist; use\nListQueuesResponse.next_page_token to determine if more\nqueues exist.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying the page of results to return.\n\nTo request the first page results, page_token must be empty. To\nrequest the next page of results, page_token must be the value of\nListQueuesResponse.next_page_token returned from the previous\ncall to CloudTasks.ListQueues method. It is an error to\nswitch the value of ListQueuesRequest.filter while iterating\nthrough pages.",
	//       "format": "byte",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required.\n\nThe location name.\nFor example: `projects/PROJECT_ID/locations/LOCATION_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+parent}/queues",
	//   "response": {
	//     "$ref": "ListQueuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsLocationsQueuesListCall) Pages(ctx context.Context, f func(*ListQueuesResponse) error) error {
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

// method id "cloudtasks.projects.locations.queues.patch":

type ProjectsLocationsQueuesPatchCall struct {
	s          *Service
	name       string
	queue      *Queue
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a queue.
//
// This method creates the queue if it does not exist and updates
// the queue if it does exist.
//
// WARNING: This method is only available to whitelisted
// users. Using this method carries some risk. Read
// [Overview of Queue Management and
// queue.yaml](/cloud-tasks/docs/queue-yaml)
// carefully and then sign up for
// [whitelist access to this method](https://goo.gl/Fe5mUy).
func (r *ProjectsLocationsQueuesService) Patch(name string, queue *Queue) *ProjectsLocationsQueuesPatchCall {
	c := &ProjectsLocationsQueuesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.queue = queue
	return c
}

// UpdateMask sets the optional parameter "updateMask": A mask used to
// specify which fields of the queue are being updated.
//
// If empty, then all fields will be updated.
func (c *ProjectsLocationsQueuesPatchCall) UpdateMask(updateMask string) *ProjectsLocationsQueuesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesPatchCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesPatchCall) Context(ctx context.Context) *ProjectsLocationsQueuesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.queue)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.patch" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesPatchCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Updates a queue.\n\nThis method creates the queue if it does not exist and updates\nthe queue if it does exist.\n\nWARNING: This method is only available to whitelisted\nusers. Using this method carries some risk. Read\n[Overview of Queue Management and queue.yaml](/cloud-tasks/docs/queue-yaml)\ncarefully and then sign up for\n[whitelist access to this method](https://goo.gl/Fe5mUy).",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}",
	//   "httpMethod": "PATCH",
	//   "id": "cloudtasks.projects.locations.queues.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The queue name.\n\nThe queue name must have the following format:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`\n\n* `PROJECT_ID` can contain uppercase and lowercase letters,\n  numbers, hyphens, colons, and periods; that is, it must match\n  the regular expression: `[a-zA-Z\\\\d-:\\\\.]+`.\n* `QUEUE_ID` can contain uppercase and lowercase letters,\n  numbers, and hyphens; that is, it must match the regular\n  expression: `[a-zA-Z\\\\d-]+`. The maximum length is 100\n  characters.\n\nCaller-specified and required in CreateQueueRequest, after which\nit becomes output only.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "A mask used to specify which fields of the queue are being updated.\n\nIf empty, then all fields will be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}",
	//   "request": {
	//     "$ref": "Queue"
	//   },
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.pause":

type ProjectsLocationsQueuesPauseCall struct {
	s                 *Service
	name              string
	pausequeuerequest *PauseQueueRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Pause: Pauses the queue.
//
// If a queue is paused then the system will stop executing the
// tasks in the queue until it is resumed via
// CloudTasks.ResumeQueue. Tasks can still be added when the
// queue is paused. The state of the queue is stored
// in
// Queue.queue_state; if paused it will be set
// to
// Queue.QueueState.PAUSED.
//
// WARNING: This method is only available to whitelisted
// users. Using this method carries some risk. Read
// [Overview of Queue Management and
// queue.yaml](/cloud-tasks/docs/queue-yaml)
// carefully and then sign up for
// [whitelist access to this method](https://goo.gl/Fe5mUy).
func (r *ProjectsLocationsQueuesService) Pause(name string, pausequeuerequest *PauseQueueRequest) *ProjectsLocationsQueuesPauseCall {
	c := &ProjectsLocationsQueuesPauseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.pausequeuerequest = pausequeuerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesPauseCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesPauseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesPauseCall) Context(ctx context.Context) *ProjectsLocationsQueuesPauseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesPauseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesPauseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pausequeuerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:pause")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.pause" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesPauseCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Pauses the queue.\n\nIf a queue is paused then the system will stop executing the\ntasks in the queue until it is resumed via\nCloudTasks.ResumeQueue. Tasks can still be added when the\nqueue is paused. The state of the queue is stored in\nQueue.queue_state; if paused it will be set to\nQueue.QueueState.PAUSED.\n\nWARNING: This method is only available to whitelisted\nusers. Using this method carries some risk. Read\n[Overview of Queue Management and queue.yaml](/cloud-tasks/docs/queue-yaml)\ncarefully and then sign up for\n[whitelist access to this method](https://goo.gl/Fe5mUy).",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:pause",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.pause",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:pause",
	//   "request": {
	//     "$ref": "PauseQueueRequest"
	//   },
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.purge":

type ProjectsLocationsQueuesPurgeCall struct {
	s                 *Service
	name              string
	purgequeuerequest *PurgeQueueRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Purge: Purges a queue by deleting all of its tasks.
//
// All tasks created before this method is called are permanently
// deleted.
//
// Purge operations can take up to one minute to take effect.
// Tasks
// might be dispatched before the purge takes effect. A purge is
// irreversible.
func (r *ProjectsLocationsQueuesService) Purge(name string, purgequeuerequest *PurgeQueueRequest) *ProjectsLocationsQueuesPurgeCall {
	c := &ProjectsLocationsQueuesPurgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.purgequeuerequest = purgequeuerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesPurgeCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesPurgeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesPurgeCall) Context(ctx context.Context) *ProjectsLocationsQueuesPurgeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesPurgeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesPurgeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.purgequeuerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:purge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.purge" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesPurgeCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Purges a queue by deleting all of its tasks.\n\nAll tasks created before this method is called are permanently deleted.\n\nPurge operations can take up to one minute to take effect. Tasks\nmight be dispatched before the purge takes effect. A purge is irreversible.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:purge",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.purge",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:purge",
	//   "request": {
	//     "$ref": "PurgeQueueRequest"
	//   },
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.resume":

type ProjectsLocationsQueuesResumeCall struct {
	s                  *Service
	name               string
	resumequeuerequest *ResumeQueueRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Resume: Resume a queue.
//
// This method resumes a queue after it has been
// Queue.QueueState.PAUSED or Queue.QueueState.DISABLED. The state of
// a queue is stored in Queue.queue_state; after calling this method
// it
// will be set to Queue.QueueState.RUNNING.
//
// WARNING: This method is only available to whitelisted
// users. Using this method carries some risk. Read
// [Overview of Queue Management and
// queue.yaml](/cloud-tasks/docs/queue-yaml)
// carefully and then sign up for
// [whitelist access to this method](https://goo.gl/Fe5mUy).
//
// WARNING: Resuming many high-QPS queues at the same time can
// lead to target overloading. If you are resuming high-QPS
// queues, follow the 500/50/5 pattern described in
// [Managing Cloud Tasks Scaling
// Risks](/cloud-tasks/pdfs/managing-cloud-tasks-scaling-risks-2017-06-05
// .pdf).
func (r *ProjectsLocationsQueuesService) Resume(name string, resumequeuerequest *ResumeQueueRequest) *ProjectsLocationsQueuesResumeCall {
	c := &ProjectsLocationsQueuesResumeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.resumequeuerequest = resumequeuerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesResumeCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesResumeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesResumeCall) Context(ctx context.Context) *ProjectsLocationsQueuesResumeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesResumeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesResumeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.resumequeuerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:resume")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.resume" call.
// Exactly one of *Queue or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Queue.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesResumeCall) Do(opts ...googleapi.CallOption) (*Queue, error) {
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
	ret := &Queue{
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
	//   "description": "Resume a queue.\n\nThis method resumes a queue after it has been\nQueue.QueueState.PAUSED or Queue.QueueState.DISABLED. The state of\na queue is stored in Queue.queue_state; after calling this method it\nwill be set to Queue.QueueState.RUNNING.\n\nWARNING: This method is only available to whitelisted\nusers. Using this method carries some risk. Read\n[Overview of Queue Management and queue.yaml](/cloud-tasks/docs/queue-yaml)\ncarefully and then sign up for\n[whitelist access to this method](https://goo.gl/Fe5mUy).\n\nWARNING: Resuming many high-QPS queues at the same time can\nlead to target overloading. If you are resuming high-QPS\nqueues, follow the 500/50/5 pattern described in\n[Managing Cloud Tasks Scaling Risks](/cloud-tasks/pdfs/managing-cloud-tasks-scaling-risks-2017-06-05.pdf).",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:resume",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.resume",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:resume",
	//   "request": {
	//     "$ref": "ResumeQueueRequest"
	//   },
	//   "response": {
	//     "$ref": "Queue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.setIamPolicy":

type ProjectsLocationsQueuesSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy for a Queue. Replaces
// any existing
// policy.
//
// Authorization requires the following [Google IAM](/iam) permission on
// the
// specified resource parent:
//
// * `cloudtasks.queues.setIamPolicy`
func (r *ProjectsLocationsQueuesService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsLocationsQueuesSetIamPolicyCall {
	c := &ProjectsLocationsQueuesSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesSetIamPolicyCall) Context(ctx context.Context) *ProjectsLocationsQueuesSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Sets the access control policy for a Queue. Replaces any existing\npolicy.\n\nAuthorization requires the following [Google IAM](/iam) permission on the\nspecified resource parent:\n\n* `cloudtasks.queues.setIamPolicy`",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy is being specified.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+resource}:setIamPolicy",
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

// method id "cloudtasks.projects.locations.queues.testIamPermissions":

type ProjectsLocationsQueuesTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on a
// Queue.
// If the resource does not exist, this will return an empty set
// of
// permissions, not a google.rpc.Code.NOT_FOUND error.
//
// Note: This operation is designed to be used for building
// permission-aware
// UIs and command-line tools, not for authorization checking. This
// operation
// may "fail open" without warning.
func (r *ProjectsLocationsQueuesService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsLocationsQueuesTestIamPermissionsCall {
	c := &ProjectsLocationsQueuesTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTestIamPermissionsCall) Context(ctx context.Context) *ProjectsLocationsQueuesTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLocationsQueuesTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
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
	//   "description": "Returns permissions that a caller has on a Queue.\nIf the resource does not exist, this will return an empty set of\npermissions, not a google.rpc.Code.NOT_FOUND error.\n\nNote: This operation is designed to be used for building permission-aware\nUIs and command-line tools, not for authorization checking. This operation\nmay \"fail open\" without warning.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which the policy detail is being requested.\nSee the operation documentation for the appropriate value for this field.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+resource}:testIamPermissions",
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

// method id "cloudtasks.projects.locations.queues.tasks.acknowledge":

type ProjectsLocationsQueuesTasksAcknowledgeCall struct {
	s                      *Service
	name                   string
	acknowledgetaskrequest *AcknowledgeTaskRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Acknowledge: Acknowledges a pull task.
//
// The lease holder, that is, the entity that received this task in
// a PullTasksResponse, must call this method to indicate that
// the work associated with the task has finished.
//
// The lease holder must acknowledge a task within
// the
// PullTasksRequest.lease_duration or the lease will expire and
// the task will become ready to be returned in a
// different
// PullTasksResponse. After the task is acknowledged, it will
// not be returned by a later CloudTasks.PullTasks,
// CloudTasks.GetTask, or CloudTasks.ListTasks.
func (r *ProjectsLocationsQueuesTasksService) Acknowledge(name string, acknowledgetaskrequest *AcknowledgeTaskRequest) *ProjectsLocationsQueuesTasksAcknowledgeCall {
	c := &ProjectsLocationsQueuesTasksAcknowledgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.acknowledgetaskrequest = acknowledgetaskrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksAcknowledgeCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksAcknowledgeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksAcknowledgeCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksAcknowledgeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksAcknowledgeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksAcknowledgeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.acknowledgetaskrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:acknowledge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.acknowledge" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesTasksAcknowledgeCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Acknowledges a pull task.\n\nThe lease holder, that is, the entity that received this task in\na PullTasksResponse, must call this method to indicate that\nthe work associated with the task has finished.\n\nThe lease holder must acknowledge a task within the\nPullTasksRequest.lease_duration or the lease will expire and\nthe task will become ready to be returned in a different\nPullTasksResponse. After the task is acknowledged, it will\nnot be returned by a later CloudTasks.PullTasks,\nCloudTasks.GetTask, or CloudTasks.ListTasks.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}:acknowledge",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.acknowledge",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:acknowledge",
	//   "request": {
	//     "$ref": "AcknowledgeTaskRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.cancelLease":

type ProjectsLocationsQueuesTasksCancelLeaseCall struct {
	s                  *Service
	name               string
	cancelleaserequest *CancelLeaseRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// CancelLease: Cancel a pull task's lease.
//
// The lease holder can use this method to cancel a task's lease
// by setting Task.schedule_time to now. This will make the
// task
// available to be leased to the next caller of CloudTasks.PullTasks.
func (r *ProjectsLocationsQueuesTasksService) CancelLease(name string, cancelleaserequest *CancelLeaseRequest) *ProjectsLocationsQueuesTasksCancelLeaseCall {
	c := &ProjectsLocationsQueuesTasksCancelLeaseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.cancelleaserequest = cancelleaserequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksCancelLeaseCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksCancelLeaseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksCancelLeaseCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksCancelLeaseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksCancelLeaseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksCancelLeaseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.cancelleaserequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:cancelLease")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.cancelLease" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsLocationsQueuesTasksCancelLeaseCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Cancel a pull task's lease.\n\nThe lease holder can use this method to cancel a task's lease\nby setting Task.schedule_time to now. This will make the task\navailable to be leased to the next caller of CloudTasks.PullTasks.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}:cancelLease",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.cancelLease",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:cancelLease",
	//   "request": {
	//     "$ref": "CancelLeaseRequest"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.create":

type ProjectsLocationsQueuesTasksCreateCall struct {
	s                 *Service
	parent            string
	createtaskrequest *CreateTaskRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// Create: Creates a task and adds it to a queue.
//
// To add multiple tasks at the same time, use
// [HTTP batching](/storage/docs/json_api/v1/how-tos/batch)
// or the batching documentation for your client library, for
// example
// https://developers.google.com/api-client-library/python/guide/
// batch.
//
// Tasks cannot be updated after creation; there is no UpdateTask
// command.
func (r *ProjectsLocationsQueuesTasksService) Create(parent string, createtaskrequest *CreateTaskRequest) *ProjectsLocationsQueuesTasksCreateCall {
	c := &ProjectsLocationsQueuesTasksCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createtaskrequest = createtaskrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksCreateCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksCreateCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createtaskrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+parent}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.create" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsLocationsQueuesTasksCreateCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Creates a task and adds it to a queue.\n\nTo add multiple tasks at the same time, use\n[HTTP batching](/storage/docs/json_api/v1/how-tos/batch)\nor the batching documentation for your client library, for example\nhttps://developers.google.com/api-client-library/python/guide/batch.\n\nTasks cannot be updated after creation; there is no UpdateTask command.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`\n\nThe queue must already exist.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+parent}/tasks",
	//   "request": {
	//     "$ref": "CreateTaskRequest"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.delete":

type ProjectsLocationsQueuesTasksDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a task.
//
// A task can be deleted if it is scheduled or dispatched. A task
// cannot be deleted if it has completed successfully or
// permanently
// failed.
func (r *ProjectsLocationsQueuesTasksService) Delete(name string) *ProjectsLocationsQueuesTasksDeleteCall {
	c := &ProjectsLocationsQueuesTasksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksDeleteCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksDeleteCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsLocationsQueuesTasksDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a task.\n\nA task can be deleted if it is scheduled or dispatched. A task\ncannot be deleted if it has completed successfully or permanently\nfailed.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}",
	//   "httpMethod": "DELETE",
	//   "id": "cloudtasks.projects.locations.queues.tasks.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.get":

type ProjectsLocationsQueuesTasksGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a task.
func (r *ProjectsLocationsQueuesTasksService) Get(name string) *ProjectsLocationsQueuesTasksGetCall {
	c := &ProjectsLocationsQueuesTasksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// ResponseView sets the optional parameter "responseView": The
// response_view specifies which subset of the Task will
// be
// returned.
//
// By default response_view is Task.View.BASIC; not all
// information is retrieved by default because some data, such
// as
// payloads, might be desirable to return only when needed because
// of its large size or because of the sensitivity of data that
// it
// contains.
//
// Authorization for Task.View.FULL requires
// `cloudtasks.tasks.fullView`
// [Google IAM](/iam/) permission on the
// Task.name resource.
//
// Possible values:
//   "VIEW_UNSPECIFIED"
//   "BASIC"
//   "FULL"
func (c *ProjectsLocationsQueuesTasksGetCall) ResponseView(responseView string) *ProjectsLocationsQueuesTasksGetCall {
	c.urlParams_.Set("responseView", responseView)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksGetCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLocationsQueuesTasksGetCall) IfNoneMatch(entityTag string) *ProjectsLocationsQueuesTasksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksGetCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.get" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsLocationsQueuesTasksGetCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Gets a task.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}",
	//   "httpMethod": "GET",
	//   "id": "cloudtasks.projects.locations.queues.tasks.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "responseView": {
	//       "description": "The response_view specifies which subset of the Task will be\nreturned.\n\nBy default response_view is Task.View.BASIC; not all\ninformation is retrieved by default because some data, such as\npayloads, might be desirable to return only when needed because\nof its large size or because of the sensitivity of data that it\ncontains.\n\nAuthorization for Task.View.FULL requires `cloudtasks.tasks.fullView`\n[Google IAM](/iam/) permission on the\nTask.name resource.",
	//       "enum": [
	//         "VIEW_UNSPECIFIED",
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}",
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.list":

type ProjectsLocationsQueuesTasksListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the tasks in a queue.
//
// By default response_view is Task.View.BASIC; not all
// information is retrieved by default due to
// performance
// considerations; ListTasksRequest.response_view controls the
// subset of information which is returned.
func (r *ProjectsLocationsQueuesTasksService) List(parent string) *ProjectsLocationsQueuesTasksListCall {
	c := &ProjectsLocationsQueuesTasksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort order used for
// the query. The fields supported for sorting
// are Task.schedule_time and PullTaskTarget.tag. All results will
// be
// returned in ascending order. The default ordering is
// by
// Task.schedule_time.
func (c *ProjectsLocationsQueuesTasksListCall) OrderBy(orderBy string) *ProjectsLocationsQueuesTasksListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Fewer tasks than requested might be returned.
//
// The maximum page size is 1000. If unspecified, the page size will
// be the maximum. Fewer tasks than requested might be returned,
// even if more tasks exist; use
// ListTasksResponse.next_page_token to determine if more tasks
// exist.
func (c *ProjectsLocationsQueuesTasksListCall) PageSize(pageSize int64) *ProjectsLocationsQueuesTasksListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying the page of results to return.
//
// To request the first page results, page_token must be empty.
// To
// request the next page of results, page_token must be the value
// of
// ListTasksResponse.next_page_token returned from the previous
// call to CloudTasks.ListTasks method.
//
// The page token is valid for only 2 hours.
func (c *ProjectsLocationsQueuesTasksListCall) PageToken(pageToken string) *ProjectsLocationsQueuesTasksListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ResponseView sets the optional parameter "responseView": The
// response_view specifies which subset of the Task will
// be
// returned.
//
// By default response_view is Task.View.BASIC; not all
// information is retrieved by default because some data, such
// as
// payloads, might be desirable to return only when needed because
// of its large size or because of the sensitivity of data that
// it
// contains.
//
// Authorization for Task.View.FULL requires
// `cloudtasks.tasks.fullView`
// [Google IAM](/iam/) permission on the
// Task.name resource.
//
// Possible values:
//   "VIEW_UNSPECIFIED"
//   "BASIC"
//   "FULL"
func (c *ProjectsLocationsQueuesTasksListCall) ResponseView(responseView string) *ProjectsLocationsQueuesTasksListCall {
	c.urlParams_.Set("responseView", responseView)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksListCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsLocationsQueuesTasksListCall) IfNoneMatch(entityTag string) *ProjectsLocationsQueuesTasksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksListCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+parent}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.list" call.
// Exactly one of *ListTasksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTasksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLocationsQueuesTasksListCall) Do(opts ...googleapi.CallOption) (*ListTasksResponse, error) {
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
	ret := &ListTasksResponse{
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
	//   "description": "Lists the tasks in a queue.\n\nBy default response_view is Task.View.BASIC; not all\ninformation is retrieved by default due to performance\nconsiderations; ListTasksRequest.response_view controls the\nsubset of information which is returned.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks",
	//   "httpMethod": "GET",
	//   "id": "cloudtasks.projects.locations.queues.tasks.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "orderBy": {
	//       "description": "\nSort order used for the query. The fields supported for sorting\nare Task.schedule_time and PullTaskTarget.tag. All results will be\nreturned in ascending order. The default ordering is by\nTask.schedule_time.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Fewer tasks than requested might be returned.\n\nThe maximum page size is 1000. If unspecified, the page size will\nbe the maximum. Fewer tasks than requested might be returned,\neven if more tasks exist; use\nListTasksResponse.next_page_token to determine if more tasks\nexist.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying the page of results to return.\n\nTo request the first page results, page_token must be empty. To\nrequest the next page of results, page_token must be the value of\nListTasksResponse.next_page_token returned from the previous\ncall to CloudTasks.ListTasks method.\n\nThe page token is valid for only 2 hours.",
	//       "format": "byte",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "responseView": {
	//       "description": "The response_view specifies which subset of the Task will be\nreturned.\n\nBy default response_view is Task.View.BASIC; not all\ninformation is retrieved by default because some data, such as\npayloads, might be desirable to return only when needed because\nof its large size or because of the sensitivity of data that it\ncontains.\n\nAuthorization for Task.View.FULL requires `cloudtasks.tasks.fullView`\n[Google IAM](/iam/) permission on the\nTask.name resource.",
	//       "enum": [
	//         "VIEW_UNSPECIFIED",
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+parent}/tasks",
	//   "response": {
	//     "$ref": "ListTasksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsLocationsQueuesTasksListCall) Pages(ctx context.Context, f func(*ListTasksResponse) error) error {
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

// method id "cloudtasks.projects.locations.queues.tasks.pull":

type ProjectsLocationsQueuesTasksPullCall struct {
	s                *Service
	name             string
	pulltasksrequest *PullTasksRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Pull: Pulls tasks from a pull queue and acquires a lease on them for
// a
// specified PullTasksRequest.lease_duration.
//
// This method is invoked by the lease holder to obtain the
// lease. The lease holder must acknowledge the task
// via
// CloudTasks.AcknowledgeTask after they have performed the
// work
// associated with the task.
//
// The payload is intended to store data that the lease holder needs
// to perform the work associated with the task. To return the
// payloads in the PullTasksResponse, set
// PullTasksRequest.response_view to Task.View.FULL.
func (r *ProjectsLocationsQueuesTasksService) Pull(name string, pulltasksrequest *PullTasksRequest) *ProjectsLocationsQueuesTasksPullCall {
	c := &ProjectsLocationsQueuesTasksPullCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.pulltasksrequest = pulltasksrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksPullCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksPullCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksPullCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksPullCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksPullCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksPullCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pulltasksrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}/tasks:pull")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.pull" call.
// Exactly one of *PullTasksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PullTasksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsLocationsQueuesTasksPullCall) Do(opts ...googleapi.CallOption) (*PullTasksResponse, error) {
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
	ret := &PullTasksResponse{
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
	//   "description": "Pulls tasks from a pull queue and acquires a lease on them for a\nspecified PullTasksRequest.lease_duration.\n\nThis method is invoked by the lease holder to obtain the\nlease. The lease holder must acknowledge the task via\nCloudTasks.AcknowledgeTask after they have performed the work\nassociated with the task.\n\nThe payload is intended to store data that the lease holder needs\nto perform the work associated with the task. To return the\npayloads in the PullTasksResponse, set\nPullTasksRequest.response_view to Task.View.FULL.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks:pull",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.pull",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe queue name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}/tasks:pull",
	//   "request": {
	//     "$ref": "PullTasksRequest"
	//   },
	//   "response": {
	//     "$ref": "PullTasksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.renewLease":

type ProjectsLocationsQueuesTasksRenewLeaseCall struct {
	s                 *Service
	name              string
	renewleaserequest *RenewLeaseRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
	header_           http.Header
}

// RenewLease: Renew the current lease of a pull task.
//
// The lease holder can use this method to extend the lease by a
// new
// duration, starting from now. The new task lease will be
// returned in Task.schedule_time.
func (r *ProjectsLocationsQueuesTasksService) RenewLease(name string, renewleaserequest *RenewLeaseRequest) *ProjectsLocationsQueuesTasksRenewLeaseCall {
	c := &ProjectsLocationsQueuesTasksRenewLeaseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.renewleaserequest = renewleaserequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksRenewLeaseCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksRenewLeaseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksRenewLeaseCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksRenewLeaseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksRenewLeaseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksRenewLeaseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.renewleaserequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:renewLease")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.renewLease" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsLocationsQueuesTasksRenewLeaseCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Renew the current lease of a pull task.\n\nThe lease holder can use this method to extend the lease by a new\nduration, starting from now. The new task lease will be\nreturned in Task.schedule_time.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}:renewLease",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.renewLease",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:renewLease",
	//   "request": {
	//     "$ref": "RenewLeaseRequest"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "cloudtasks.projects.locations.queues.tasks.run":

type ProjectsLocationsQueuesTasksRunCall struct {
	s              *Service
	name           string
	runtaskrequest *RunTaskRequest
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Run: Forces a task to run now.
//
// This command is meant to be used for manual debugging. For
// example, CloudTasks.RunTask can be used to retry a failed
// task after a fix has been made or to manually force a task to
// be
// dispatched now.
//
// When this method is called, Cloud Tasks will dispatch the task to
// its
// target, even if the queue is Queue.QueueState.PAUSED.
//
// The dispatched task is returned. That is, the task that is
// returned
// contains the Task.task_status after the task is dispatched but
// before the task is received by its target.
//
// If Cloud Tasks receives a successful response from the
// task's
// handler, then the task will be deleted; otherwise the
// task's
// Task.schedule_time will be reset to the time that
// CloudTasks.RunTask was called plus the retry delay specified
// in the queue and task's RetryConfig.
//
// CloudTasks.RunTask returns google.rpc.Code.NOT_FOUND when
// it is called on a task that has already succeeded or
// permanently
// failed. google.rpc.Code.FAILED_PRECONDITION is returned
// when
// CloudTasks.RunTask is called on task that is dispatched or
// already running.
func (r *ProjectsLocationsQueuesTasksService) Run(name string, runtaskrequest *RunTaskRequest) *ProjectsLocationsQueuesTasksRunCall {
	c := &ProjectsLocationsQueuesTasksRunCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.runtaskrequest = runtaskrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsLocationsQueuesTasksRunCall) Fields(s ...googleapi.Field) *ProjectsLocationsQueuesTasksRunCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsLocationsQueuesTasksRunCall) Context(ctx context.Context) *ProjectsLocationsQueuesTasksRunCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsLocationsQueuesTasksRunCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsLocationsQueuesTasksRunCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.runtaskrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2beta2/{+name}:run")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "cloudtasks.projects.locations.queues.tasks.run" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsLocationsQueuesTasksRunCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Forces a task to run now.\n\nThis command is meant to be used for manual debugging. For\nexample, CloudTasks.RunTask can be used to retry a failed\ntask after a fix has been made or to manually force a task to be\ndispatched now.\n\nWhen this method is called, Cloud Tasks will dispatch the task to its\ntarget, even if the queue is Queue.QueueState.PAUSED.\n\nThe dispatched task is returned. That is, the task that is returned\ncontains the Task.task_status after the task is dispatched but\nbefore the task is received by its target.\n\nIf Cloud Tasks receives a successful response from the task's\nhandler, then the task will be deleted; otherwise the task's\nTask.schedule_time will be reset to the time that\nCloudTasks.RunTask was called plus the retry delay specified\nin the queue and task's RetryConfig.\n\nCloudTasks.RunTask returns google.rpc.Code.NOT_FOUND when\nit is called on a task that has already succeeded or permanently\nfailed. google.rpc.Code.FAILED_PRECONDITION is returned when\nCloudTasks.RunTask is called on task that is dispatched or\nalready running.",
	//   "flatPath": "v2beta2/projects/{projectsId}/locations/{locationsId}/queues/{queuesId}/tasks/{tasksId}:run",
	//   "httpMethod": "POST",
	//   "id": "cloudtasks.projects.locations.queues.tasks.run",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Required.\n\nThe task name. For example:\n`projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID`",
	//       "location": "path",
	//       "pattern": "^projects/[^/]+/locations/[^/]+/queues/[^/]+/tasks/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2beta2/{+name}:run",
	//   "request": {
	//     "$ref": "RunTaskRequest"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
