// Package appengine provides access to the Google App Engine Admin API.
//
// See https://cloud.google.com/appengine/docs/admin-api/
//
// Usage example:
//
//   import "google.golang.org/api/appengine/v1beta4"
//   ...
//   appengineService, err := appengine.New(oauthHttpClient)
package appengine // import "google.golang.org/api/appengine/v1beta4"

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

const apiId = "appengine:v1beta4"
const apiName = "appengine"
const apiVersion = "v1beta4"
const basePath = "https://appengine.googleapis.com/"

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
	s.Apps = NewAppsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Apps *AppsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAppsService(s *Service) *AppsService {
	rs := &AppsService{s: s}
	rs.Modules = NewAppsModulesService(s)
	rs.Operations = NewAppsOperationsService(s)
	return rs
}

type AppsService struct {
	s *Service

	Modules *AppsModulesService

	Operations *AppsOperationsService
}

func NewAppsModulesService(s *Service) *AppsModulesService {
	rs := &AppsModulesService{s: s}
	rs.Versions = NewAppsModulesVersionsService(s)
	return rs
}

type AppsModulesService struct {
	s *Service

	Versions *AppsModulesVersionsService
}

func NewAppsModulesVersionsService(s *Service) *AppsModulesVersionsService {
	rs := &AppsModulesVersionsService{s: s}
	return rs
}

type AppsModulesVersionsService struct {
	s *Service
}

func NewAppsOperationsService(s *Service) *AppsOperationsService {
	rs := &AppsOperationsService{s: s}
	return rs
}

type AppsOperationsService struct {
	s *Service
}

// ApiConfigHandler: API Serving configuration for Cloud Endpoints.
type ApiConfigHandler struct {
	// AuthFailAction: For users not logged in, how to handle access to
	// resources with required login. Defaults to "redirect".
	//
	// Possible values:
	//   "AUTH_FAIL_ACTION_UNSPECIFIED"
	//   "AUTH_FAIL_ACTION_REDIRECT"
	//   "AUTH_FAIL_ACTION_UNAUTHORIZED"
	AuthFailAction string `json:"authFailAction,omitempty"`

	// Login: What level of login is required to access this resource.
	// Default is "optional".
	//
	// Possible values:
	//   "LOGIN_UNSPECIFIED"
	//   "LOGIN_OPTIONAL"
	//   "LOGIN_ADMIN"
	//   "LOGIN_REQUIRED"
	Login string `json:"login,omitempty"`

	// Script: Specifies the path to the script from the application root
	// directory.
	Script string `json:"script,omitempty"`

	// SecurityLevel: Configures whether security (HTTPS) should be enforced
	// for this URL.
	//
	// Possible values:
	//   "SECURE_UNSPECIFIED"
	//   "SECURE_DEFAULT"
	//   "SECURE_NEVER"
	//   "SECURE_OPTIONAL"
	//   "SECURE_ALWAYS"
	SecurityLevel string `json:"securityLevel,omitempty"`

	// Url: URL to serve the endpoint at.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuthFailAction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApiConfigHandler) MarshalJSON() ([]byte, error) {
	type noMethod ApiConfigHandler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ApiEndpointHandler: Use Google Cloud Endpoints to handle requests.
type ApiEndpointHandler struct {
	// ScriptPath: Specifies the path to the script from the application
	// root directory.
	ScriptPath string `json:"scriptPath,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ScriptPath") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ApiEndpointHandler) MarshalJSON() ([]byte, error) {
	type noMethod ApiEndpointHandler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Application: An Application contains the top-level configuration of
// an App Engine application.
type Application struct {
	// CodeBucket: A Google Cloud Storage bucket which can be used for
	// storing files associated with an application. This bucket is
	// associated with the application and can be used by the gcloud
	// deployment commands. @OutputOnly
	CodeBucket string `json:"codeBucket,omitempty"`

	// DispatchRules: HTTP path dispatch rules for requests to the app that
	// do not explicitly target a module or version. The rules are
	// order-dependent.
	DispatchRules []*UrlDispatchRule `json:"dispatchRules,omitempty"`

	// Id: The relative name/path of the application. Example: "myapp".
	// @OutputOnly
	Id string `json:"id,omitempty"`

	// Location: The location from which the application will be run.
	// Choices are "us-central" for United States and "europe-west" for
	// European Union. Application instances will run out of data centers in
	// the chosen location and all of the application's End User Content
	// will be stored at rest in the chosen location. The default is
	// "us-central".
	Location string `json:"location,omitempty"`

	// Name: The full path to the application in the API. Example:
	// "apps/myapp". @OutputOnly
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CodeBucket") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Application) MarshalJSON() ([]byte, error) {
	type noMethod Application
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AutomaticScaling: Automatic scaling is the scaling policy that App
// Engine has used since its inception. It is based on request rate,
// response latencies, and other application metrics.
type AutomaticScaling struct {
	// CoolDownPeriod: The amount of time that the
	// [Autoscaler](https://cloud.google.com/compute/docs/autoscaler/)
	// should wait between changes to the number of virtual machines.
	// Applies only to the VM runtime.
	CoolDownPeriod string `json:"coolDownPeriod,omitempty"`

	// CpuUtilization: Target scaling by CPU usage.
	CpuUtilization *CpuUtilization `json:"cpuUtilization,omitempty"`

	// DiskUtilization: Target scaling by disk usage.
	DiskUtilization *DiskUtilization `json:"diskUtilization,omitempty"`

	// MaxConcurrentRequests: The number of concurrent requests an automatic
	// scaling instance can accept before the scheduler spawns a new
	// instance. Default value is chosen based on the runtime.
	MaxConcurrentRequests int64 `json:"maxConcurrentRequests,omitempty"`

	// MaxIdleInstances: The maximum number of idle instances that App
	// Engine should maintain for this version.
	MaxIdleInstances int64 `json:"maxIdleInstances,omitempty"`

	// MaxPendingLatency: The maximum amount of time that App Engine should
	// allow a request to wait in the pending queue before starting a new
	// instance to handle it.
	MaxPendingLatency string `json:"maxPendingLatency,omitempty"`

	// MaxTotalInstances: Max number of instances that App Engine should
	// start to handle requests.
	MaxTotalInstances int64 `json:"maxTotalInstances,omitempty"`

	// MinIdleInstances: The minimum number of idle instances that App
	// Engine should maintain for this version. Only applies to the default
	// version of a module, since other versions are not expected to receive
	// significant traffic.
	MinIdleInstances int64 `json:"minIdleInstances,omitempty"`

	// MinPendingLatency: The minimum amount of time that App Engine should
	// allow a request to wait in the pending queue before starting a new
	// instance to handle it.
	MinPendingLatency string `json:"minPendingLatency,omitempty"`

	// MinTotalInstances: Minimum number of instances that App Engine should
	// maintain.
	MinTotalInstances int64 `json:"minTotalInstances,omitempty"`

	// NetworkUtilization: Target scaling by network usage.
	NetworkUtilization *NetworkUtilization `json:"networkUtilization,omitempty"`

	// RequestUtilization: Target scaling by request utilization.
	RequestUtilization *RequestUtilization `json:"requestUtilization,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CoolDownPeriod") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutomaticScaling) MarshalJSON() ([]byte, error) {
	type noMethod AutomaticScaling
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BasicScaling: A module with basic scaling will create an instance
// when the application receives a request. The instance will be turned
// down when the app becomes idle. Basic scaling is ideal for work that
// is intermittent or driven by user activity.
type BasicScaling struct {
	// IdleTimeout: The instance will be shut down this amount of time after
	// receiving its last request.
	IdleTimeout string `json:"idleTimeout,omitempty"`

	// MaxInstances: The maximum number of instances for App Engine to
	// create for this version.
	MaxInstances int64 `json:"maxInstances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IdleTimeout") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BasicScaling) MarshalJSON() ([]byte, error) {
	type noMethod BasicScaling
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ContainerInfo: A Docker (container) image which should be used to
// start the application.
type ContainerInfo struct {
	// Image: Reference to a hosted container image. Must be a URI to a
	// resource in a Docker repository. Must be fully qualified, including
	// tag or digest. e.g. gcr.io/my-project/image:tag or
	// gcr.io/my-project/image@digest
	Image string `json:"image,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Image") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ContainerInfo) MarshalJSON() ([]byte, error) {
	type noMethod ContainerInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CpuUtilization: Target scaling by CPU usage.
type CpuUtilization struct {
	// AggregationWindowLength: The period of time over which CPU
	// utilization is calculated.
	AggregationWindowLength string `json:"aggregationWindowLength,omitempty"`

	// TargetUtilization: Target (0-1) CPU utilization ratio to maintain
	// when scaling.
	TargetUtilization float64 `json:"targetUtilization,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AggregationWindowLength") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CpuUtilization) MarshalJSON() ([]byte, error) {
	type noMethod CpuUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Deployment: Code and application artifacts used to deploy a version
// to App Engine.
type Deployment struct {
	// Container: If supplied, a docker (container) image which should be
	// used to start the application. Only applicable to the 'vm' runtime.
	Container *ContainerInfo `json:"container,omitempty"`

	// Files: A manifest of files stored in Google Cloud Storage which
	// should be included as part of this application. All files must be
	// readable using the credentials supplied with this call.
	Files map[string]FileInfo `json:"files,omitempty"`

	// SourceReferences: The origin of the source code for this deployment.
	// There can be more than one source reference per Version if source
	// code is distributed among multiple repositories.
	SourceReferences []*SourceReference `json:"sourceReferences,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Container") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Deployment) MarshalJSON() ([]byte, error) {
	type noMethod Deployment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DiskUtilization: Target scaling by disk usage (for VM runtimes only).
type DiskUtilization struct {
	// TargetReadBytesPerSec: Target bytes per second read.
	TargetReadBytesPerSec int64 `json:"targetReadBytesPerSec,omitempty"`

	// TargetReadOpsPerSec: Target ops per second read.
	TargetReadOpsPerSec int64 `json:"targetReadOpsPerSec,omitempty"`

	// TargetWriteBytesPerSec: Target bytes per second written.
	TargetWriteBytesPerSec int64 `json:"targetWriteBytesPerSec,omitempty"`

	// TargetWriteOpsPerSec: Target ops per second written.
	TargetWriteOpsPerSec int64 `json:"targetWriteOpsPerSec,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "TargetReadBytesPerSec") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DiskUtilization) MarshalJSON() ([]byte, error) {
	type noMethod DiskUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ErrorHandler: A custom static error page to be served when an error
// occurs.
type ErrorHandler struct {
	// ErrorCode: The error condition this handler applies to.
	//
	// Possible values:
	//   "ERROR_CODE_UNSPECIFIED"
	//   "ERROR_CODE_DEFAULT"
	//   "ERROR_CODE_OVER_QUOTA"
	//   "ERROR_CODE_DOS_API_DENIAL"
	//   "ERROR_CODE_TIMEOUT"
	ErrorCode string `json:"errorCode,omitempty"`

	// MimeType: MIME type of file. If unspecified, "text/html" is assumed.
	MimeType string `json:"mimeType,omitempty"`

	// StaticFile: Static file content to be served for this error.
	StaticFile string `json:"staticFile,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ErrorCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ErrorHandler) MarshalJSON() ([]byte, error) {
	type noMethod ErrorHandler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FileInfo: A single source file which is part of the application to be
// deployed.
type FileInfo struct {
	// MimeType: The MIME type of the file; if unspecified, the value from
	// Google Cloud Storage will be used.
	MimeType string `json:"mimeType,omitempty"`

	// Sha1Sum: The SHA1 (160 bits) hash of the file in hex.
	Sha1Sum string `json:"sha1Sum,omitempty"`

	// SourceUrl: The URL source to use to fetch this file. Must be a URL to
	// a resource in Google Cloud Storage in the form
	// 'http(s)://storage.googleapis.com/\/\'.
	SourceUrl string `json:"sourceUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MimeType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FileInfo) MarshalJSON() ([]byte, error) {
	type noMethod FileInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// HealthCheck: Configure health checking for the VM instances.
// Unhealthy VM instances will be killed and replaced with new
// instances.
type HealthCheck struct {
	// CheckInterval: The interval between health checks.
	CheckInterval string `json:"checkInterval,omitempty"`

	// DisableHealthCheck: Whether to explicitly disable health checks for
	// this instance.
	DisableHealthCheck bool `json:"disableHealthCheck,omitempty"`

	// HealthyThreshold: The number of consecutive successful health checks
	// before receiving traffic.
	HealthyThreshold int64 `json:"healthyThreshold,omitempty"`

	// Host: The host header to send when performing an HTTP health check
	// (e.g. myapp.appspot.com)
	Host string `json:"host,omitempty"`

	// RestartThreshold: The number of consecutive failed health checks
	// before an instance is restarted.
	RestartThreshold int64 `json:"restartThreshold,omitempty"`

	// Timeout: The amount of time before the health check is considered
	// failed.
	Timeout string `json:"timeout,omitempty"`

	// UnhealthyThreshold: The number of consecutive failed health checks
	// before removing traffic.
	UnhealthyThreshold int64 `json:"unhealthyThreshold,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CheckInterval") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HealthCheck) MarshalJSON() ([]byte, error) {
	type noMethod HealthCheck
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Library: A Python runtime third-party library required by the
// application.
type Library struct {
	// Name: The name of the library, e.g. "PIL" or "django".
	Name string `json:"name,omitempty"`

	// Version: The version of the library to select, or "latest".
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Library) MarshalJSON() ([]byte, error) {
	type noMethod Library
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListModulesResponse: Response message for `Modules.ListModules`.
type ListModulesResponse struct {
	// Modules: The modules belonging to the requested application.
	Modules []*Module `json:"modules,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Modules") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListModulesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListModulesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListOperationsResponse: The response message for
// Operations.ListOperations.
type ListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that matches the specified filter in
	// the request.
	Operations []*Operation `json:"operations,omitempty"`

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

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListVersionsResponse: Response message for `Versions.ListVersions`.
type ListVersionsResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Versions: The versions belonging to the requested application module.
	Versions []*Version `json:"versions,omitempty"`

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

func (s *ListVersionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListVersionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ManualScaling: A module with manual scaling runs continuously,
// allowing you to perform complex initialization and rely on the state
// of its memory over time.
type ManualScaling struct {
	// Instances: The number of instances to assign to the module at the
	// start. This number can later be altered by using the [Modules
	// API](https://cloud.google.com/appengine/docs/python/modules/functions)
	//  `set_num_instances()` function.
	Instances int64 `json:"instances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Instances") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ManualScaling) MarshalJSON() ([]byte, error) {
	type noMethod ManualScaling
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Module: A module is a component of an application that provides a
// single service or configuration. A module has a collection of
// versions that define a specific set of code used to implement the
// functionality of that module.
type Module struct {
	// Id: The relative name/path of the module within the application.
	// Example: "default" @OutputOnly
	Id string `json:"id,omitempty"`

	// Name: The full path to the Module resource in the API. Example:
	// "apps/myapp/modules/default" @OutputOnly
	Name string `json:"name,omitempty"`

	// Split: A mapping that defines fractional HTTP traffic diversion to
	// different versions within the module.
	Split *TrafficSplit `json:"split,omitempty"`

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
}

func (s *Module) MarshalJSON() ([]byte, error) {
	type noMethod Module
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Network: Used to specify extra network settings (for VM runtimes
// only).
type Network struct {
	// ForwardedPorts: A list of ports (or port pairs) to forward from the
	// VM into the app container.
	ForwardedPorts []string `json:"forwardedPorts,omitempty"`

	// InstanceTag: A tag to apply to the VM instance during creation.
	InstanceTag string `json:"instanceTag,omitempty"`

	// Name: The Google Compute Engine network where the VMs will be
	// created. If not specified, or empty, the network named "default" will
	// be used. (The short name should be specified, not the resource path.)
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ForwardedPorts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Network) MarshalJSON() ([]byte, error) {
	type noMethod Network
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// NetworkUtilization: Target scaling by network usage (for VM runtimes
// only).
type NetworkUtilization struct {
	// TargetReceivedBytesPerSec: Target bytes per second received.
	TargetReceivedBytesPerSec int64 `json:"targetReceivedBytesPerSec,omitempty"`

	// TargetReceivedPacketsPerSec: Target packets per second received.
	TargetReceivedPacketsPerSec int64 `json:"targetReceivedPacketsPerSec,omitempty"`

	// TargetSentBytesPerSec: Target bytes per second sent.
	TargetSentBytesPerSec int64 `json:"targetSentBytesPerSec,omitempty"`

	// TargetSentPacketsPerSec: Target packets per second sent.
	TargetSentPacketsPerSec int64 `json:"targetSentPacketsPerSec,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "TargetReceivedBytesPerSec") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NetworkUtilization) MarshalJSON() ([]byte, error) {
	type noMethod NetworkUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a network API call.
type Operation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress. If true, the operation is completed, and either `error` or
	// `response` is available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation. It
	// typically contains progress information and common metadata such as
	// create time. Some services might not provide such metadata. Any
	// method that returns a long-running operation should document the
	// metadata type, if any.
	Metadata OperationMetadata `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that originally returns it. If you use the default HTTP
	// mapping above, the `name` should have the format of
	// `operations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success. If
	// the original method returns no data on success, such as `Delete`, the
	// response is `google.protobuf.Empty`. If the original method is
	// standard `Get`/`Create`/`Update`, the response should be the
	// resource. For other methods, the response should have the type
	// `XxxResponse`, where `Xxx` is the original method name. For example,
	// if the original method name is `TakeSnapshot()`, the inferred
	// response type is `TakeSnapshotResponse`.
	Response OperationResponse `json:"response,omitempty"`

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
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationMetadata interface{}

type OperationResponse interface{}

// OperationMetadata1: Metadata for the given
// google.longrunning.Operation.
type OperationMetadata1 struct {
	// EndTime: Timestamp that this operation was completed. (Not present if
	// the operation is still in progress.) @OutputOnly
	EndTime string `json:"endTime,omitempty"`

	// InsertTime: Timestamp that this operation was received. @OutputOnly
	InsertTime string `json:"insertTime,omitempty"`

	// Method: API method name that initiated the operation. Example:
	// "google.appengine.v1beta4.Version.CreateVersion". @OutputOnly
	Method string `json:"method,omitempty"`

	// OperationType: The type of the operation (deprecated, use method
	// field instead). Example: "create_version". @OutputOnly
	OperationType string `json:"operationType,omitempty"`

	// Target: Resource that this operation is acting on. Example:
	// "apps/myapp/modules/default". @OutputOnly
	Target string `json:"target,omitempty"`

	// User: The user who requested this operation. @OutputOnly
	User string `json:"user,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationMetadata1) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata1
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// RequestUtilization: Target scaling by request utilization (for VM
// runtimes only).
type RequestUtilization struct {
	// TargetConcurrentRequests: Target number of concurrent requests.
	TargetConcurrentRequests int64 `json:"targetConcurrentRequests,omitempty"`

	// TargetRequestCountPerSec: Target requests per second.
	TargetRequestCountPerSec int64 `json:"targetRequestCountPerSec,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "TargetConcurrentRequests") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RequestUtilization) MarshalJSON() ([]byte, error) {
	type noMethod RequestUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Resources: Used to specify how many machine resources an app version
// needs.
type Resources struct {
	// Cpu: How many CPU cores an app version needs.
	Cpu float64 `json:"cpu,omitempty"`

	// DiskGb: How much disk size, in GB, an app version needs.
	DiskGb float64 `json:"diskGb,omitempty"`

	// MemoryGb: How much memory, in GB, an app version needs.
	MemoryGb float64 `json:"memoryGb,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cpu") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Resources) MarshalJSON() ([]byte, error) {
	type noMethod Resources
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ScriptHandler: Executes a script to handle the request that matches
// the URL pattern.
type ScriptHandler struct {
	// ScriptPath: Specifies the path to the script from the application
	// root directory.
	ScriptPath string `json:"scriptPath,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ScriptPath") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ScriptHandler) MarshalJSON() ([]byte, error) {
	type noMethod ScriptHandler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SourceReference: A reference to a particular snapshot of the source
// tree used to build and deploy the application.
type SourceReference struct {
	// Repository: Optional. A URI string identifying the repository.
	// Example: "https://source.developers.google.com/p/app-123/r/default"
	Repository string `json:"repository,omitempty"`

	// RevisionId: The canonical (and persistent) identifier of the deployed
	// revision, i.e. any kind of aliases including tags or branch names are
	// not allowed. Example (git):
	// "2198322f89e0bb2e25021667c2ed489d1fd34e6b"
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

// StaticDirectoryHandler: Files served directly to the user for a given
// URL, such as images, CSS stylesheets, or JavaScript source files.
// Static directory handlers make it easy to serve the entire contents
// of a directory as static files.
type StaticDirectoryHandler struct {
	// ApplicationReadable: By default, files declared in static file
	// handlers are uploaded as static data and are only served to end
	// users, they cannot be read by an application. If this field is set to
	// true, the files are also uploaded as code data so your application
	// can read them. Both uploads are charged against your code and static
	// data storage resource quotas.
	ApplicationReadable bool `json:"applicationReadable,omitempty"`

	// Directory: The path to the directory containing the static files,
	// from the application root directory. Everything after the end of the
	// matched url pattern is appended to static_dir to form the full path
	// to the requested file.
	Directory string `json:"directory,omitempty"`

	// Expiration: The length of time a static file served by this handler
	// ought to be cached by web proxies and browsers.
	Expiration string `json:"expiration,omitempty"`

	// HttpHeaders: HTTP headers to use for all responses from these URLs.
	HttpHeaders map[string]string `json:"httpHeaders,omitempty"`

	// MimeType: If specified, all files served by this handler will be
	// served using the specified MIME type. If not specified, the MIME type
	// for a file will be derived from the file's filename extension.
	MimeType string `json:"mimeType,omitempty"`

	// RequireMatchingFile: If true, this UrlMap entry does not match the
	// request unless the file referenced by the handler also exists. If no
	// such file exists, processing will continue with the next UrlMap that
	// matches the requested URL.
	RequireMatchingFile bool `json:"requireMatchingFile,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApplicationReadable")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *StaticDirectoryHandler) MarshalJSON() ([]byte, error) {
	type noMethod StaticDirectoryHandler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// StaticFilesHandler: Files served directly to the user for a given
// URL, such as images, CSS stylesheets, or JavaScript source files.
// Static file handlers describe which files in the application
// directory are static files, and which URLs serve them.
type StaticFilesHandler struct {
	// ApplicationReadable: By default, files declared in static file
	// handlers are uploaded as static data and are only served to end
	// users, they cannot be read by an application. If this field is set to
	// true, the files are also uploaded as code data so your application
	// can read them. Both uploads are charged against your code and static
	// data storage resource quotas.
	ApplicationReadable bool `json:"applicationReadable,omitempty"`

	// Expiration: The length of time a static file served by this handler
	// ought to be cached by web proxies and browsers.
	Expiration string `json:"expiration,omitempty"`

	// HttpHeaders: HTTP headers to use for all responses from these URLs.
	HttpHeaders map[string]string `json:"httpHeaders,omitempty"`

	// MimeType: If specified, all files served by this handler will be
	// served using the specified MIME type. If not specified, the MIME type
	// for a file will be derived from the file's filename extension.
	MimeType string `json:"mimeType,omitempty"`

	// Path: The path to the static files matched by the URL pattern, from
	// the application root directory. The path can refer to text matched in
	// groupings in the URL pattern.
	Path string `json:"path,omitempty"`

	// RequireMatchingFile: If true, this UrlMap entry does not match the
	// request unless the file referenced by the handler also exists. If no
	// such file exists, processing will continue with the next UrlMap that
	// matches the requested URL.
	RequireMatchingFile bool `json:"requireMatchingFile,omitempty"`

	// UploadPathRegex: A regular expression that matches the file paths for
	// all files that will be referenced by this handler.
	UploadPathRegex string `json:"uploadPathRegex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApplicationReadable")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *StaticFilesHandler) MarshalJSON() ([]byte, error) {
	type noMethod StaticFilesHandler
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

// TrafficSplit: Configuration for traffic splitting for versions within
// a single module. Traffic splitting allows traffic directed to the
// module to be assigned to one of several versions in a fractional way,
// enabling experiments and canarying new builds, for example.
type TrafficSplit struct {
	// Allocations: Mapping from module version IDs within the module to
	// fractional (0.000, 1] allocations of traffic for that version. Each
	// version may only be specified once, but some versions in the module
	// may not have any traffic allocation. Modules that have traffic
	// allocated in this field may not be deleted until the module is
	// deleted, or their traffic allocation is removed. Allocations must sum
	// to 1. Supports precision up to two decimal places for IP-based splits
	// and up to three decimal places for cookie-based splits.
	Allocations *TrafficSplitAllocations `json:"allocations,omitempty"`

	// ShardBy: Which mechanism should be used as a selector when choosing a
	// version to send a request to. The traffic selection algorithm will be
	// stable for either type until allocations are changed.
	//
	// Possible values:
	//   "UNSPECIFIED"
	//   "COOKIE"
	//   "IP"
	ShardBy string `json:"shardBy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Allocations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TrafficSplit) MarshalJSON() ([]byte, error) {
	type noMethod TrafficSplit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TrafficSplitAllocations: Mapping from module version IDs within the
// module to fractional (0.000, 1] allocations of traffic for that
// version. Each version may only be specified once, but some versions
// in the module may not have any traffic allocation. Modules that have
// traffic allocated in this field may not be deleted until the module
// is deleted, or their traffic allocation is removed. Allocations must
// sum to 1. Supports precision up to two decimal places for IP-based
// splits and up to three decimal places for cookie-based splits.
type TrafficSplitAllocations struct {
}

// UrlDispatchRule: Rules to match an HTTP request and dispatch that
// request to a module.
type UrlDispatchRule struct {
	// Domain: The domain name to match on. Supports '*' (glob) wildcarding
	// on the left-hand side of a '.'. If empty, all domains will be matched
	// (the same as '*').
	Domain string `json:"domain,omitempty"`

	// Module: The resource id of a Module in this application that should
	// service the matched request. The Module must already exist. Example:
	// "default".
	Module string `json:"module,omitempty"`

	// Path: The pathname within the host. This must start with a '/'. A
	// single '*' (glob) can be included at the end of the path. The sum of
	// the lengths of the domain and path may not exceed 100 characters.
	Path string `json:"path,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Domain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UrlDispatchRule) MarshalJSON() ([]byte, error) {
	type noMethod UrlDispatchRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UrlMap: A URL pattern and description of how it should be handled.
// App Engine can handle URLs by executing application code, or by
// serving static files uploaded with the code, such as images, CSS or
// JavaScript.
type UrlMap struct {
	// ApiEndpoint: Use API Endpoints to handle requests.
	ApiEndpoint *ApiEndpointHandler `json:"apiEndpoint,omitempty"`

	// AuthFailAction: For users not logged in, how to handle access to
	// resources with required login. Defaults to "redirect".
	//
	// Possible values:
	//   "AUTH_FAIL_ACTION_UNSPECIFIED"
	//   "AUTH_FAIL_ACTION_REDIRECT"
	//   "AUTH_FAIL_ACTION_UNAUTHORIZED"
	AuthFailAction string `json:"authFailAction,omitempty"`

	// Login: What level of login is required to access this resource.
	//
	// Possible values:
	//   "LOGIN_UNSPECIFIED"
	//   "LOGIN_OPTIONAL"
	//   "LOGIN_ADMIN"
	//   "LOGIN_REQUIRED"
	Login string `json:"login,omitempty"`

	// RedirectHttpResponseCode: `30x` code to use when performing redirects
	// for the `secure` field. A `302` is used by default.
	//
	// Possible values:
	//   "REDIRECT_HTTP_RESPONSE_CODE_UNSPECIFIED"
	//   "REDIRECT_HTTP_RESPONSE_CODE_301"
	//   "REDIRECT_HTTP_RESPONSE_CODE_302"
	//   "REDIRECT_HTTP_RESPONSE_CODE_303"
	//   "REDIRECT_HTTP_RESPONSE_CODE_307"
	RedirectHttpResponseCode string `json:"redirectHttpResponseCode,omitempty"`

	// Script: Executes a script to handle the request that matches the URL
	// pattern.
	Script *ScriptHandler `json:"script,omitempty"`

	// SecurityLevel: Configures whether security (HTTPS) should be enforced
	// for this URL.
	//
	// Possible values:
	//   "SECURE_UNSPECIFIED"
	//   "SECURE_DEFAULT"
	//   "SECURE_NEVER"
	//   "SECURE_OPTIONAL"
	//   "SECURE_ALWAYS"
	SecurityLevel string `json:"securityLevel,omitempty"`

	// StaticDirectory: Serves the entire contents of a directory as static
	// files. This attribute is deprecated. You can mimic the behavior of
	// static directories using static files.
	StaticDirectory *StaticDirectoryHandler `json:"staticDirectory,omitempty"`

	// StaticFiles: Returns the contents of a file, such as an image, as the
	// response.
	StaticFiles *StaticFilesHandler `json:"staticFiles,omitempty"`

	// UrlRegex: A URL prefix. This value uses regular expression syntax
	// (and so regexp special characters must be escaped), but it should not
	// contain groupings. All URLs that begin with this prefix are handled
	// by this handler, using the portion of the URL after the prefix as
	// part of the file path. This is always required.
	UrlRegex string `json:"urlRegex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApiEndpoint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UrlMap) MarshalJSON() ([]byte, error) {
	type noMethod UrlMap
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Version: A Version is a specific set of source code and configuration
// files deployed to a module.
type Version struct {
	// ApiConfig: Serving configuration for Google Cloud Endpoints. Only
	// returned in `GET` requests if `view=FULL` is set. May only be set on
	// create requests; once created, is immutable.
	ApiConfig *ApiConfigHandler `json:"apiConfig,omitempty"`

	// AutomaticScaling: Automatic scaling is the scaling policy that App
	// Engine has used since its inception. It is based on request rate,
	// response latencies, and other application metrics.
	AutomaticScaling *AutomaticScaling `json:"automaticScaling,omitempty"`

	// BasicScaling: A module with basic scaling will create an instance
	// when the application receives a request. The instance will be turned
	// down when the app becomes idle. Basic scaling is ideal for work that
	// is intermittent or driven by user activity.
	BasicScaling *BasicScaling `json:"basicScaling,omitempty"`

	// BetaSettings: Beta settings supplied to the application via metadata.
	BetaSettings map[string]string `json:"betaSettings,omitempty"`

	// CreationTime: Creation time of this version. This will be between the
	// start and end times of the operation that creates this version.
	// @OutputOnly
	CreationTime string `json:"creationTime,omitempty"`

	// DefaultExpiration: The length of time a static file served by a
	// static file handler ought to be cached by web proxies and browsers,
	// if the handler does not specify its own expiration. Only returned in
	// `GET` requests if `view=FULL` is set. May only be set on create
	// requests; once created, is immutable.
	DefaultExpiration string `json:"defaultExpiration,omitempty"`

	// Deployer: The email address of the user who created this version.
	// @OutputOnly
	Deployer string `json:"deployer,omitempty"`

	// Deployment: Code and application artifacts that make up this version.
	// Only returned in `GET` requests if `view=FULL` is set. May only be
	// set on create requests; once created, is immutable.
	Deployment *Deployment `json:"deployment,omitempty"`

	// Env: The App Engine execution environment to use for this version.
	// Default: "1"
	Env string `json:"env,omitempty"`

	// EnvVariables: Environment variables made available to the
	// application. Only returned in `GET` requests if `view=FULL` is set.
	// May only be set on create requests; once created, is immutable.
	EnvVariables map[string]string `json:"envVariables,omitempty"`

	// ErrorHandlers: Custom static error pages instead of these generic
	// error pages, (limit 10 KB/page) Only returned in `GET` requests if
	// `view=FULL` is set. May only be set on create requests; once created,
	// is immutable.
	ErrorHandlers []*ErrorHandler `json:"errorHandlers,omitempty"`

	// Handlers: An ordered list of URL Matching patterns that should be
	// applied to incoming requests. The first matching URL consumes the
	// request, and subsequent handlers are not attempted. Only returned in
	// `GET` requests if `view=FULL` is set. May only be set on create
	// requests; once created, is immutable.
	Handlers []*UrlMap `json:"handlers,omitempty"`

	// HealthCheck: Configure health checking for the VM instances.
	// Unhealthy VM instances will be stopped and replaced with new
	// instances. Only returned in `GET` requests if `view=FULL` is set. May
	// only be set on create requests; once created, is immutable.
	HealthCheck *HealthCheck `json:"healthCheck,omitempty"`

	// Id: The relative name/path of the Version within the module. Example:
	// "v1". Version specifiers can contain lowercase letters, digits, and
	// hyphens. It cannot begin with the prefix `ah-` and the names
	// `default` and `latest` are reserved and cannot be used.
	Id string `json:"id,omitempty"`

	// InboundServices: Before an application can receive email or XMPP
	// messages, the application must be configured to enable the service.
	//
	// Possible values:
	//   "INBOUND_SERVICE_UNSPECIFIED" - Not specified.
	//   "INBOUND_SERVICE_MAIL" - Allows an application to receive mail.
	//   "INBOUND_SERVICE_MAIL_BOUNCE" - Allows an application receive email
	// bound notifications.
	//   "INBOUND_SERVICE_XMPP_ERROR" - Allows an application to receive
	// error stanzas.
	//   "INBOUND_SERVICE_XMPP_MESSAGE" - Allows an application to receive
	// instant messages.
	//   "INBOUND_SERVICE_XMPP_SUBSCRIBE" - Allows an application to receive
	// user subscription POSTs.
	//   "INBOUND_SERVICE_XMPP_PRESENCE" - Allows an application to receive
	// a user's chat presence.
	//   "INBOUND_SERVICE_CHANNEL_PRESENCE" - Registers an application for
	// notifications when a client connects or disconnects from a channel.
	//   "INBOUND_SERVICE_WARMUP" - Enables warmup requests.
	InboundServices []string `json:"inboundServices,omitempty"`

	// InstanceClass: The frontend instance class to use to run this app.
	// Valid values are `[F1, F2, F4, F4_1G]`. Default: "F1"
	InstanceClass string `json:"instanceClass,omitempty"`

	// Libraries: Configuration for Python runtime third-party libraries
	// required by the application. Only returned in `GET` requests if
	// `view=FULL` is set. May only be set on create requests; once created,
	// is immutable.
	Libraries []*Library `json:"libraries,omitempty"`

	// ManualScaling: A module with manual scaling runs continuously,
	// allowing you to perform complex initialization and rely on the state
	// of its memory over time.
	ManualScaling *ManualScaling `json:"manualScaling,omitempty"`

	// Name: The full path to the Version resource in the API. Example:
	// "apps/myapp/modules/default/versions/v1". @OutputOnly
	Name string `json:"name,omitempty"`

	// Network: Used to specify extra network settings (for VM runtimes
	// only).
	Network *Network `json:"network,omitempty"`

	// NobuildFilesRegex: Go only. Files that match this pattern will not be
	// built into the app. May only be set on create requests.
	NobuildFilesRegex string `json:"nobuildFilesRegex,omitempty"`

	// Resources: Used to specify how many machine resources an app version
	// needs (for VM runtimes only).
	Resources *Resources `json:"resources,omitempty"`

	// Runtime: The desired runtime. Values can include python27, java7, go,
	// etc.
	Runtime string `json:"runtime,omitempty"`

	// ServingStatus: The current serving status of this version. Only
	// `SERVING` versions will have instances created or billed for. If this
	// field is unset when a version is created, `SERVING` status will be
	// assumed. It is an error to explicitly set this field to
	// `SERVING_STATUS_UNSPECIFIED`.
	//
	// Possible values:
	//   "SERVING_STATUS_UNSPECIFIED"
	//   "SERVING"
	//   "STOPPED"
	ServingStatus string `json:"servingStatus,omitempty"`

	// Threadsafe: If true, multiple requests can be dispatched to the app
	// at once.
	Threadsafe bool `json:"threadsafe,omitempty"`

	// Vm: Whether to deploy this app in a VM container.
	Vm bool `json:"vm,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ApiConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Version) MarshalJSON() ([]byte, error) {
	type noMethod Version
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "appengine.apps.get":

type AppsGetCall struct {
	s            *Service
	appsId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets information about an application.
func (r *AppsService) Get(appsId string) *AppsGetCall {
	c := &AppsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	return c
}

// EnsureResourcesExist sets the optional parameter
// "ensureResourcesExist": Certain resources associated with an
// application are created on-demand. Controls whether these resources
// should be created when performing the `GET` operation. If specified
// and any resources cloud not be created, the request will fail with an
// error code.
func (c *AppsGetCall) EnsureResourcesExist(ensureResourcesExist bool) *AppsGetCall {
	c.urlParams_.Set("ensureResourcesExist", fmt.Sprint(ensureResourcesExist))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsGetCall) QuotaUser(quotaUser string) *AppsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsGetCall) Fields(s ...googleapi.Field) *AppsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsGetCall) IfNoneMatch(entityTag string) *AppsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsGetCall) Context(ctx context.Context) *AppsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId": c.appsId,
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

// Do executes the "appengine.apps.get" call.
// Exactly one of *Application or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Application.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsGetCall) Do() (*Application, error) {
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
	ret := &Application{
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
	//   "description": "Gets information about an application.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.get",
	//   "parameterOrder": [
	//     "appsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the application to get. For example: \"apps/myapp\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "ensureResourcesExist": {
	//       "description": "Certain resources associated with an application are created on-demand. Controls whether these resources should be created when performing the `GET` operation. If specified and any resources cloud not be created, the request will fail with an error code.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}",
	//   "response": {
	//     "$ref": "Application"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.delete":

type AppsModulesDeleteCall struct {
	s          *Service
	appsId     string
	modulesId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a module and all enclosed versions.
func (r *AppsModulesService) Delete(appsId string, modulesId string) *AppsModulesDeleteCall {
	c := &AppsModulesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesDeleteCall) QuotaUser(quotaUser string) *AppsModulesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesDeleteCall) Fields(s ...googleapi.Field) *AppsModulesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesDeleteCall) Context(ctx context.Context) *AppsModulesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":    c.appsId,
		"modulesId": c.modulesId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "appengine.apps.modules.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsModulesDeleteCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a module and all enclosed versions.",
	//   "httpMethod": "DELETE",
	//   "id": "appengine.apps.modules.delete",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp/modules/default\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.get":

type AppsModulesGetCall struct {
	s            *Service
	appsId       string
	modulesId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the current configuration of the module.
func (r *AppsModulesService) Get(appsId string, modulesId string) *AppsModulesGetCall {
	c := &AppsModulesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesGetCall) QuotaUser(quotaUser string) *AppsModulesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesGetCall) Fields(s ...googleapi.Field) *AppsModulesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsModulesGetCall) IfNoneMatch(entityTag string) *AppsModulesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesGetCall) Context(ctx context.Context) *AppsModulesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":    c.appsId,
		"modulesId": c.modulesId,
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

// Do executes the "appengine.apps.modules.get" call.
// Exactly one of *Module or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Module.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AppsModulesGetCall) Do() (*Module, error) {
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
	ret := &Module{
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
	//   "description": "Gets the current configuration of the module.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.modules.get",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp/modules/default\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}",
	//   "response": {
	//     "$ref": "Module"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.list":

type AppsModulesListCall struct {
	s            *Service
	appsId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists all the modules in the application.
func (r *AppsModulesService) List(appsId string) *AppsModulesListCall {
	c := &AppsModulesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum results to
// return per page.
func (c *AppsModulesListCall) PageSize(pageSize int64) *AppsModulesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AppsModulesListCall) PageToken(pageToken string) *AppsModulesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesListCall) QuotaUser(quotaUser string) *AppsModulesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesListCall) Fields(s ...googleapi.Field) *AppsModulesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsModulesListCall) IfNoneMatch(entityTag string) *AppsModulesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesListCall) Context(ctx context.Context) *AppsModulesListCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId": c.appsId,
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

// Do executes the "appengine.apps.modules.list" call.
// Exactly one of *ListModulesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListModulesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AppsModulesListCall) Do() (*ListModulesResponse, error) {
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
	ret := &ListModulesResponse{
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
	//   "description": "Lists all the modules in the application.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.modules.list",
	//   "parameterOrder": [
	//     "appsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum results to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules",
	//   "response": {
	//     "$ref": "ListModulesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.patch":

type AppsModulesPatchCall struct {
	s          *Service
	appsId     string
	modulesId  string
	module     *Module
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates the configuration of the specified module.
func (r *AppsModulesService) Patch(appsId string, modulesId string, module *Module) *AppsModulesPatchCall {
	c := &AppsModulesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	c.module = module
	return c
}

// Mask sets the optional parameter "mask": Standard field mask for the
// set of fields to be updated.
func (c *AppsModulesPatchCall) Mask(mask string) *AppsModulesPatchCall {
	c.urlParams_.Set("mask", mask)
	return c
}

// MigrateTraffic sets the optional parameter "migrateTraffic": Whether
// to use Traffic Migration to shift traffic gradually. Traffic can only
// be migrated from a single version to another single version.
func (c *AppsModulesPatchCall) MigrateTraffic(migrateTraffic bool) *AppsModulesPatchCall {
	c.urlParams_.Set("migrateTraffic", fmt.Sprint(migrateTraffic))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesPatchCall) QuotaUser(quotaUser string) *AppsModulesPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesPatchCall) Fields(s ...googleapi.Field) *AppsModulesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesPatchCall) Context(ctx context.Context) *AppsModulesPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.module)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":    c.appsId,
		"modulesId": c.modulesId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "appengine.apps.modules.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsModulesPatchCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the configuration of the specified module.",
	//   "httpMethod": "PATCH",
	//   "id": "appengine.apps.modules.patch",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource to update. For example: \"apps/myapp/modules/default\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "mask": {
	//       "description": "Standard field mask for the set of fields to be updated.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "migrateTraffic": {
	//       "description": "Whether to use Traffic Migration to shift traffic gradually. Traffic can only be migrated from a single version to another single version.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}",
	//   "request": {
	//     "$ref": "Module"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.versions.create":

type AppsModulesVersionsCreateCall struct {
	s          *Service
	appsId     string
	modulesId  string
	version    *Version
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Deploys new code and resource files to a version.
func (r *AppsModulesVersionsService) Create(appsId string, modulesId string, version *Version) *AppsModulesVersionsCreateCall {
	c := &AppsModulesVersionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	c.version = version
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesVersionsCreateCall) QuotaUser(quotaUser string) *AppsModulesVersionsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesVersionsCreateCall) Fields(s ...googleapi.Field) *AppsModulesVersionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesVersionsCreateCall) Context(ctx context.Context) *AppsModulesVersionsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesVersionsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.version)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}/versions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":    c.appsId,
		"modulesId": c.modulesId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "appengine.apps.modules.versions.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsModulesVersionsCreateCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deploys new code and resource files to a version.",
	//   "httpMethod": "POST",
	//   "id": "appengine.apps.modules.versions.create",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource to update. For example: \"apps/myapp/modules/default\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}/versions",
	//   "request": {
	//     "$ref": "Version"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.versions.delete":

type AppsModulesVersionsDeleteCall struct {
	s          *Service
	appsId     string
	modulesId  string
	versionsId string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes an existing version.
func (r *AppsModulesVersionsService) Delete(appsId string, modulesId string, versionsId string) *AppsModulesVersionsDeleteCall {
	c := &AppsModulesVersionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	c.versionsId = versionsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesVersionsDeleteCall) QuotaUser(quotaUser string) *AppsModulesVersionsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesVersionsDeleteCall) Fields(s ...googleapi.Field) *AppsModulesVersionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesVersionsDeleteCall) Context(ctx context.Context) *AppsModulesVersionsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesVersionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}/versions/{versionsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":     c.appsId,
		"modulesId":  c.modulesId,
		"versionsId": c.versionsId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "appengine.apps.modules.versions.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsModulesVersionsDeleteCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes an existing version.",
	//   "httpMethod": "DELETE",
	//   "id": "appengine.apps.modules.versions.delete",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId",
	//     "versionsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp/modules/default/versions/v1\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "versionsId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}/versions/{versionsId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.versions.get":

type AppsModulesVersionsGetCall struct {
	s            *Service
	appsId       string
	modulesId    string
	versionsId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets application deployment information.
func (r *AppsModulesVersionsService) Get(appsId string, modulesId string, versionsId string) *AppsModulesVersionsGetCall {
	c := &AppsModulesVersionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	c.versionsId = versionsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesVersionsGetCall) QuotaUser(quotaUser string) *AppsModulesVersionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// View sets the optional parameter "view": Controls the set of fields
// returned in the `Get` response.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *AppsModulesVersionsGetCall) View(view string) *AppsModulesVersionsGetCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesVersionsGetCall) Fields(s ...googleapi.Field) *AppsModulesVersionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsModulesVersionsGetCall) IfNoneMatch(entityTag string) *AppsModulesVersionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesVersionsGetCall) Context(ctx context.Context) *AppsModulesVersionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesVersionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}/versions/{versionsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":     c.appsId,
		"modulesId":  c.modulesId,
		"versionsId": c.versionsId,
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

// Do executes the "appengine.apps.modules.versions.get" call.
// Exactly one of *Version or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Version.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AppsModulesVersionsGetCall) Do() (*Version, error) {
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
	ret := &Version{
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
	//   "description": "Gets application deployment information.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.modules.versions.get",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId",
	//     "versionsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp/modules/default/versions/v1\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "versionsId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Controls the set of fields returned in the `Get` response.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}/versions/{versionsId}",
	//   "response": {
	//     "$ref": "Version"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.modules.versions.list":

type AppsModulesVersionsListCall struct {
	s            *Service
	appsId       string
	modulesId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the versions of a module.
func (r *AppsModulesVersionsService) List(appsId string, modulesId string) *AppsModulesVersionsListCall {
	c := &AppsModulesVersionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.modulesId = modulesId
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum results to
// return per page.
func (c *AppsModulesVersionsListCall) PageSize(pageSize int64) *AppsModulesVersionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AppsModulesVersionsListCall) PageToken(pageToken string) *AppsModulesVersionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsModulesVersionsListCall) QuotaUser(quotaUser string) *AppsModulesVersionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// View sets the optional parameter "view": Controls the set of fields
// returned in the `List` response.
//
// Possible values:
//   "BASIC"
//   "FULL"
func (c *AppsModulesVersionsListCall) View(view string) *AppsModulesVersionsListCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsModulesVersionsListCall) Fields(s ...googleapi.Field) *AppsModulesVersionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsModulesVersionsListCall) IfNoneMatch(entityTag string) *AppsModulesVersionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsModulesVersionsListCall) Context(ctx context.Context) *AppsModulesVersionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsModulesVersionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/modules/{modulesId}/versions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":    c.appsId,
		"modulesId": c.modulesId,
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

// Do executes the "appengine.apps.modules.versions.list" call.
// Exactly one of *ListVersionsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListVersionsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AppsModulesVersionsListCall) Do() (*ListVersionsResponse, error) {
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
	ret := &ListVersionsResponse{
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
	//   "description": "Lists the versions of a module.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.modules.versions.list",
	//   "parameterOrder": [
	//     "appsId",
	//     "modulesId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. Name of the resource requested. For example: \"apps/myapp/modules/default\".",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "modulesId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum results to return per page.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Controls the set of fields returned in the `List` response.",
	//       "enum": [
	//         "BASIC",
	//         "FULL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/modules/{modulesId}/versions",
	//   "response": {
	//     "$ref": "ListVersionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.operations.get":

type AppsOperationsGetCall struct {
	s            *Service
	appsId       string
	operationsId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the latest state of a long-running operation. Clients can
// use this method to poll the operation result at intervals as
// recommended by the API service.
func (r *AppsOperationsService) Get(appsId string, operationsId string) *AppsOperationsGetCall {
	c := &AppsOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	c.operationsId = operationsId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsOperationsGetCall) QuotaUser(quotaUser string) *AppsOperationsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsOperationsGetCall) Fields(s ...googleapi.Field) *AppsOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsOperationsGetCall) IfNoneMatch(entityTag string) *AppsOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsOperationsGetCall) Context(ctx context.Context) *AppsOperationsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsOperationsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/operations/{operationsId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId":       c.appsId,
		"operationsId": c.operationsId,
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

// Do executes the "appengine.apps.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AppsOperationsGetCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.operations.get",
	//   "parameterOrder": [
	//     "appsId",
	//     "operationsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. The name of the operation resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "operationsId": {
	//       "description": "Part of `name`. See documentation of `appsId`.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/operations/{operationsId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "appengine.apps.operations.list":

type AppsOperationsListCall struct {
	s            *Service
	appsId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists operations that match the specified filter in the
// request. If the server doesn't support this method, it returns
// `UNIMPLEMENTED`. NOTE: the `name` binding below allows API services
// to override the binding to use different resource name schemes, such
// as `users/*/operations`.
func (r *AppsOperationsService) List(appsId string) *AppsOperationsListCall {
	c := &AppsOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.appsId = appsId
	return c
}

// Filter sets the optional parameter "filter": The standard list
// filter.
func (c *AppsOperationsListCall) Filter(filter string) *AppsOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard list
// page size.
func (c *AppsOperationsListCall) PageSize(pageSize int64) *AppsOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *AppsOperationsListCall) PageToken(pageToken string) *AppsOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *AppsOperationsListCall) QuotaUser(quotaUser string) *AppsOperationsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AppsOperationsListCall) Fields(s ...googleapi.Field) *AppsOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AppsOperationsListCall) IfNoneMatch(entityTag string) *AppsOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AppsOperationsListCall) Context(ctx context.Context) *AppsOperationsListCall {
	c.ctx_ = ctx
	return c
}

func (c *AppsOperationsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta4/apps/{appsId}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"appsId": c.appsId,
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

// Do executes the "appengine.apps.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AppsOperationsListCall) Do() (*ListOperationsResponse, error) {
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
	ret := &ListOperationsResponse{
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
	//   "description": "Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`. NOTE: the `name` binding below allows API services to override the binding to use different resource name schemes, such as `users/*/operations`.",
	//   "httpMethod": "GET",
	//   "id": "appengine.apps.operations.list",
	//   "parameterOrder": [
	//     "appsId"
	//   ],
	//   "parameters": {
	//     "appsId": {
	//       "description": "Part of `name`. The name of the operation collection.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filter": {
	//       "description": "The standard list filter.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The standard list page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta4/apps/{appsId}/operations",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
