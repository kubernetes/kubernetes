// Package deploymentmanager provides access to the Google Cloud Deployment Manager API.
//
// See https://developers.google.com/deployment-manager/
//
// Usage example:
//
//   import "google.golang.org/api/deploymentmanager/v2beta1"
//   ...
//   deploymentmanagerService, err := deploymentmanager.New(oauthHttpClient)
package deploymentmanager // import "google.golang.org/api/deploymentmanager/v2beta1"

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

const apiId = "deploymentmanager:v2beta1"
const apiName = "deploymentmanager"
const apiVersion = "v2beta1"
const basePath = "https://www.googleapis.com/deploymentmanager/v2beta1/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// View and manage your Google Cloud Platform management resources and
	// deployment status information
	NdevCloudmanScope = "https://www.googleapis.com/auth/ndev.cloudman"

	// View your Google Cloud Platform management resources and deployment
	// status information
	NdevCloudmanReadonlyScope = "https://www.googleapis.com/auth/ndev.cloudman.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Deployments = NewDeploymentsService(s)
	s.Manifests = NewManifestsService(s)
	s.Operations = NewOperationsService(s)
	s.Resources = NewResourcesService(s)
	s.Types = NewTypesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Deployments *DeploymentsService

	Manifests *ManifestsService

	Operations *OperationsService

	Resources *ResourcesService

	Types *TypesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewDeploymentsService(s *Service) *DeploymentsService {
	rs := &DeploymentsService{s: s}
	return rs
}

type DeploymentsService struct {
	s *Service
}

func NewManifestsService(s *Service) *ManifestsService {
	rs := &ManifestsService{s: s}
	return rs
}

type ManifestsService struct {
	s *Service
}

func NewOperationsService(s *Service) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *Service
}

func NewResourcesService(s *Service) *ResourcesService {
	rs := &ResourcesService{s: s}
	return rs
}

type ResourcesService struct {
	s *Service
}

func NewTypesService(s *Service) *TypesService {
	rs := &TypesService{s: s}
	return rs
}

type TypesService struct {
	s *Service
}

// Deployment: Next available tag: 8
type Deployment struct {
	// Description: ! An optional user-provided description of the
	// deployment.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Manifest: ! [Output Only] URL of the manifest representing the full
	// configuration ! of this deployment.
	Manifest string `json:"manifest,omitempty"`

	// Name: ! The name of the deployment, which must be unique within the
	// project.
	Name string `json:"name,omitempty"`

	// TargetConfig: ! [Input Only] The YAML configuration to use in
	// processing this deployment. ! ! When you create a deployment, the
	// server creates a new manifest with the ! given YAML configuration and
	// sets the `manifest` property to the URL of ! the manifest resource.
	TargetConfig string `json:"targetConfig,omitempty"`

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

func (s *Deployment) MarshalJSON() ([]byte, error) {
	type noMethod Deployment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DeploymentsListResponse: ! A response containing a partial list of
// deployments and a page token used ! to build the next request if the
// request has been truncated. Next available tag: 4
type DeploymentsListResponse struct {
	// Deployments: ! The deployments contained in this response.
	Deployments []*Deployment `json:"deployments,omitempty"`

	// NextPageToken: ! A token used to continue a truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Deployments") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeploymentsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeploymentsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Manifest: Next available tag: 10
type Manifest struct {
	// Config: v2beta1: YAML with config - described above v2beta2: YAML +
	// templates. ! The YAML configuration for this manifest.
	Config string `json:"config,omitempty"`

	// EvaluatedConfig: ! [Output Only] The fully-expanded configuration
	// file, including any ! templates and references.
	EvaluatedConfig string `json:"evaluatedConfig,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Name: ! [Output Only] The name of the manifest.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] Self link for the manifest.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Config") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Manifest) MarshalJSON() ([]byte, error) {
	type noMethod Manifest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ManifestsListResponse: ! A response containing a partial list of
// manifests and a page token used ! to build the next request if the
// request has been truncated. Next available tag: 4
type ManifestsListResponse struct {
	// Manifests: ! Manifests contained in this list response.
	Manifests []*Manifest `json:"manifests,omitempty"`

	// NextPageToken: ! A token used to continue a truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Manifests") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ManifestsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ManifestsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Operation: ! An operation resource, used to manage asynchronous API
// requests. Next available tag: 24
type Operation struct {
	// CreationTimestamp: ! [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// EndTime: ! [Output Only] The time that this operation was completed.
	// This is in ! RFC3339 format.
	EndTime string `json:"endTime,omitempty"`

	// Error: ! [Output Only] If errors occurred during processing of this
	// operation, ! this field will be populated.
	Error *OperationError `json:"error,omitempty"`

	// HttpErrorMessage: ! [Output Only] If operation fails, the HTTP error
	// message returned, ! e.g. NOT FOUND.
	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	// HttpErrorStatusCode: ! [Output Only] If operation fails, the HTTP
	// error status code returned, ! e.g. 404.
	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	// Id: ! [Output Only] Unique identifier for the resource; defined by
	// the server.
	Id uint64 `json:"id,omitempty,string"`

	// InsertTime: ! [Output Only] The time that this operation was
	// requested. ! This is in RFC 3339 format.
	InsertTime string `json:"insertTime,omitempty"`

	// Name: ! [Output Only] Name of the operation.
	Name string `json:"name,omitempty"`

	// OperationType: ! [Output Only] Type of the operation. Examples
	// include "insert", or ! "delete"
	OperationType string `json:"operationType,omitempty"`

	// Progress: ! [Output Only] An optional progress indicator that ranges
	// from 0 to 100. ! There is no requirement that this be linear or
	// support any granularity ! of operations. This should not be used to
	// guess at when the operation will ! be complete. This number should be
	// monotonically increasing as the ! operation progresses.
	Progress int64 `json:"progress,omitempty"`

	// SelfLink: [Output Only] Self link for the manifest.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: ! [Output Only] The time that this operation was started
	// by the server. ! This is in RFC 3339 format.
	StartTime string `json:"startTime,omitempty"`

	// Status: ! [Output Only] Status of the operation. Can be one of the
	// following: ! "PENDING", "RUNNING", or "DONE".
	Status string `json:"status,omitempty"`

	// StatusMessage: ! [Output Only] An optional textual description of the
	// current status of ! the operation.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetId: ! [Output Only] Unique target id which identifies a
	// particular ! incarnation of the target.
	TargetId uint64 `json:"targetId,omitempty,string"`

	// TargetLink: ! [Output Only] URL of the resource the operation is
	// mutating.
	TargetLink string `json:"targetLink,omitempty"`

	// User: ! [Output Only] User who requested the operation, for example !
	// "user@example.com"
	User string `json:"user,omitempty"`

	// Warnings: ! [Output Only] If warning messages generated during
	// processing of this ! operation, this field will be populated.
	Warnings []*OperationWarnings `json:"warnings,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationTimestamp")
	// to unconditionally include in API requests. By default, fields with
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

// OperationError: ! [Output Only] If errors occurred during processing
// of this operation, ! this field will be populated.
type OperationError struct {
	// Errors: ! The array of errors encountered while processing this
	// operation.
	Errors []*OperationErrorErrors `json:"errors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Errors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationError) MarshalJSON() ([]byte, error) {
	type noMethod OperationError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationErrorErrors struct {
	// Code: ! The error type identifier for this error.
	Code string `json:"code,omitempty"`

	// Location: ! Indicates the field in the request which caused the
	// error. ! This property is optional.
	Location string `json:"location,omitempty"`

	// Message: ! An optional, human-readable error message.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationErrorErrors) MarshalJSON() ([]byte, error) {
	type noMethod OperationErrorErrors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationWarnings struct {
	// Code: ! The warning type identifier for this warning.
	Code interface{} `json:"code,omitempty"`

	// Data: ! Metadata for this warning in 'key: value' format.
	Data []*OperationWarningsData `json:"data,omitempty"`

	// Message: ! Optional human-readable details for this warning.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationWarnings) MarshalJSON() ([]byte, error) {
	type noMethod OperationWarnings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationWarningsData struct {
	// Key: ! A key for the warning data.
	Key string `json:"key,omitempty"`

	// Value: ! A warning data value corresponding to the key.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationWarningsData) MarshalJSON() ([]byte, error) {
	type noMethod OperationWarningsData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// OperationsListResponse: ! A response containing a partial list of
// operations and a page token used ! to build the next request if the
// request has been truncated. Next available tag: 4
type OperationsListResponse struct {
	// NextPageToken: ! A token used to continue a truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: ! Operations contained in this list response.
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

func (s *OperationsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod OperationsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Resource: Next available tag: 12
type Resource struct {
	// Errors: ! [Output Only] A list of any errors that occurred during
	// deployment.
	Errors []string `json:"errors,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Intent: ! [Output Only] The intended state of the resource.
	Intent string `json:"intent,omitempty"`

	// Manifest: ! [Output Only] URL of the manifest representing the
	// current configuration ! of this resource.
	Manifest string `json:"manifest,omitempty"`

	// Name: ! [Output Only] The name of the resource as it appears in the
	// YAML config.
	Name string `json:"name,omitempty"`

	// State: ! [Output Only] The state of the resource.
	State string `json:"state,omitempty"`

	// Type: ! [Output Only] The type of the resource, for example !
	// ?compute.v1.instance?, or ?replicaPools.v1beta2.instanceGroupManager?
	Type string `json:"type,omitempty"`

	// Url: ! [Output Only] The URL of the actual resource.
	Url string `json:"url,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Errors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Resource) MarshalJSON() ([]byte, error) {
	type noMethod Resource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ResourcesListResponse: ! A response containing a partial list of
// resources and a page token used ! to build the next request if the
// request has been truncated. Next available tag: 4
type ResourcesListResponse struct {
	// NextPageToken: ! A token used to continue a truncated list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Resources: ! Resources contained in this list response.
	Resources []*Resource `json:"resources,omitempty"`

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

func (s *ResourcesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod ResourcesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Type: ! A type supported by Deployment Manager. Next available tag: 4
type Type struct {
	// Name: ! Name of the type.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Type) MarshalJSON() ([]byte, error) {
	type noMethod Type
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TypesListResponse: ! A response that returns all Types supported by
// Deployment Manager Next available tag: 3
type TypesListResponse struct {
	// Types: ! Types supported by Deployment Manager
	Types []*Type `json:"types,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Types") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TypesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod TypesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "deploymentmanager.deployments.delete":

type DeploymentsDeleteCall struct {
	s          *Service
	project    string
	deployment string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: ! Deletes a deployment and all of the resources in the
// deployment.
func (r *DeploymentsService) Delete(project string, deployment string) *DeploymentsDeleteCall {
	c := &DeploymentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DeploymentsDeleteCall) QuotaUser(quotaUser string) *DeploymentsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DeploymentsDeleteCall) UserIP(userIP string) *DeploymentsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DeploymentsDeleteCall) Fields(s ...googleapi.Field) *DeploymentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DeploymentsDeleteCall) Context(ctx context.Context) *DeploymentsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *DeploymentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "deploymentmanager.deployments.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DeploymentsDeleteCall) Do() (*Operation, error) {
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
	//   "description": "! Deletes a deployment and all of the resources in the deployment.",
	//   "httpMethod": "DELETE",
	//   "id": "deploymentmanager.deployments.delete",
	//   "parameterOrder": [
	//     "project",
	//     "deployment"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "deploymentmanager.deployments.get":

type DeploymentsGetCall struct {
	s            *Service
	project      string
	deployment   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: ! Gets information about a specific deployment.
func (r *DeploymentsService) Get(project string, deployment string) *DeploymentsGetCall {
	c := &DeploymentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DeploymentsGetCall) QuotaUser(quotaUser string) *DeploymentsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DeploymentsGetCall) UserIP(userIP string) *DeploymentsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DeploymentsGetCall) Fields(s ...googleapi.Field) *DeploymentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DeploymentsGetCall) IfNoneMatch(entityTag string) *DeploymentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DeploymentsGetCall) Context(ctx context.Context) *DeploymentsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *DeploymentsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
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

// Do executes the "deploymentmanager.deployments.get" call.
// Exactly one of *Deployment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Deployment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DeploymentsGetCall) Do() (*Deployment, error) {
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
	ret := &Deployment{
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
	//   "description": "! Gets information about a specific deployment.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.deployments.get",
	//   "parameterOrder": [
	//     "project",
	//     "deployment"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}",
	//   "response": {
	//     "$ref": "Deployment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.deployments.insert":

type DeploymentsInsertCall struct {
	s          *Service
	project    string
	deployment *Deployment
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: ! Creates a deployment and all of the resources described by
// the ! deployment manifest.
func (r *DeploymentsService) Insert(project string, deployment *Deployment) *DeploymentsInsertCall {
	c := &DeploymentsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DeploymentsInsertCall) QuotaUser(quotaUser string) *DeploymentsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DeploymentsInsertCall) UserIP(userIP string) *DeploymentsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DeploymentsInsertCall) Fields(s ...googleapi.Field) *DeploymentsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DeploymentsInsertCall) Context(ctx context.Context) *DeploymentsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *DeploymentsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.deployment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "deploymentmanager.deployments.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DeploymentsInsertCall) Do() (*Operation, error) {
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
	//   "description": "! Creates a deployment and all of the resources described by the ! deployment manifest.",
	//   "httpMethod": "POST",
	//   "id": "deploymentmanager.deployments.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments",
	//   "request": {
	//     "$ref": "Deployment"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "deploymentmanager.deployments.list":

type DeploymentsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: ! Lists all deployments for a given project.
func (r *DeploymentsService) List(project string) *DeploymentsListCall {
	c := &DeploymentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": ! Maximum count
// of results to be returned. ! Acceptable values are 0 to 100,
// inclusive. (Default: 50)
func (c *DeploymentsListCall) MaxResults(maxResults int64) *DeploymentsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": ! Specifies a
// nextPageToken returned by a previous list request. This ! token can
// be used to request the next page of results from a previous ! list
// request.
func (c *DeploymentsListCall) PageToken(pageToken string) *DeploymentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *DeploymentsListCall) QuotaUser(quotaUser string) *DeploymentsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *DeploymentsListCall) UserIP(userIP string) *DeploymentsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DeploymentsListCall) Fields(s ...googleapi.Field) *DeploymentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DeploymentsListCall) IfNoneMatch(entityTag string) *DeploymentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DeploymentsListCall) Context(ctx context.Context) *DeploymentsListCall {
	c.ctx_ = ctx
	return c
}

func (c *DeploymentsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
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

// Do executes the "deploymentmanager.deployments.list" call.
// Exactly one of *DeploymentsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DeploymentsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DeploymentsListCall) Do() (*DeploymentsListResponse, error) {
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
	ret := &DeploymentsListResponse{
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
	//   "description": "! Lists all deployments for a given project.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.deployments.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "50",
	//       "description": "! Maximum count of results to be returned. ! Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "! Specifies a nextPageToken returned by a previous list request. This ! token can be used to request the next page of results from a previous ! list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments",
	//   "response": {
	//     "$ref": "DeploymentsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.manifests.get":

type ManifestsGetCall struct {
	s            *Service
	project      string
	deployment   string
	manifest     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: ! Gets information about a specific manifest.
func (r *ManifestsService) Get(project string, deployment string, manifest string) *ManifestsGetCall {
	c := &ManifestsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	c.manifest = manifest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ManifestsGetCall) QuotaUser(quotaUser string) *ManifestsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ManifestsGetCall) UserIP(userIP string) *ManifestsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManifestsGetCall) Fields(s ...googleapi.Field) *ManifestsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManifestsGetCall) IfNoneMatch(entityTag string) *ManifestsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManifestsGetCall) Context(ctx context.Context) *ManifestsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ManifestsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}/manifests/{manifest}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
		"manifest":   c.manifest,
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

// Do executes the "deploymentmanager.manifests.get" call.
// Exactly one of *Manifest or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Manifest.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ManifestsGetCall) Do() (*Manifest, error) {
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
	ret := &Manifest{
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
	//   "description": "! Gets information about a specific manifest.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.manifests.get",
	//   "parameterOrder": [
	//     "project",
	//     "deployment",
	//     "manifest"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "manifest": {
	//       "description": "! The name of the manifest for this request.",
	//       "location": "path",
	//       "pattern": "[-a-z0-9]{1,61}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}/manifests/{manifest}",
	//   "response": {
	//     "$ref": "Manifest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.manifests.list":

type ManifestsListCall struct {
	s            *Service
	project      string
	deployment   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: ! Lists all manifests for a given deployment.
func (r *ManifestsService) List(project string, deployment string) *ManifestsListCall {
	c := &ManifestsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	return c
}

// MaxResults sets the optional parameter "maxResults": ! Maximum count
// of results to be returned. ! Acceptable values are 0 to 100,
// inclusive. (Default: 50)
func (c *ManifestsListCall) MaxResults(maxResults int64) *ManifestsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": ! Specifies a
// nextPageToken returned by a previous list request. This ! token can
// be used to request the next page of results from a previous ! list
// request.
func (c *ManifestsListCall) PageToken(pageToken string) *ManifestsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ManifestsListCall) QuotaUser(quotaUser string) *ManifestsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ManifestsListCall) UserIP(userIP string) *ManifestsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ManifestsListCall) Fields(s ...googleapi.Field) *ManifestsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ManifestsListCall) IfNoneMatch(entityTag string) *ManifestsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ManifestsListCall) Context(ctx context.Context) *ManifestsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ManifestsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}/manifests")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
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

// Do executes the "deploymentmanager.manifests.list" call.
// Exactly one of *ManifestsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ManifestsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ManifestsListCall) Do() (*ManifestsListResponse, error) {
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
	ret := &ManifestsListResponse{
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
	//   "description": "! Lists all manifests for a given deployment.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.manifests.list",
	//   "parameterOrder": [
	//     "project",
	//     "deployment"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "50",
	//       "description": "! Maximum count of results to be returned. ! Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "! Specifies a nextPageToken returned by a previous list request. This ! token can be used to request the next page of results from a previous ! list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}/manifests",
	//   "response": {
	//     "$ref": "ManifestsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.operations.get":

type OperationsGetCall struct {
	s            *Service
	project      string
	operation    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: ! Gets information about a specific Operation.
func (r *OperationsService) Get(project string, operation string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.operation = operation
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OperationsGetCall) QuotaUser(quotaUser string) *OperationsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OperationsGetCall) UserIP(userIP string) *OperationsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsGetCall) Fields(s ...googleapi.Field) *OperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsGetCall) IfNoneMatch(entityTag string) *OperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsGetCall) Context(ctx context.Context) *OperationsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *OperationsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
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

// Do executes the "deploymentmanager.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *OperationsGetCall) Do() (*Operation, error) {
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
	//   "description": "! Gets information about a specific Operation.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.operations.get",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "description": "! The name of the operation for this request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.operations.list":

type OperationsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: ! Lists all Operations for a project.
func (r *OperationsService) List(project string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": ! Maximum count
// of results to be returned. ! Acceptable values are 0 to 100,
// inclusive. (Default: 50)
func (c *OperationsListCall) MaxResults(maxResults int64) *OperationsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": ! Specifies a
// nextPageToken returned by a previous list request. This ! token can
// be used to request the next page of results from a previous ! list
// request.
func (c *OperationsListCall) PageToken(pageToken string) *OperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *OperationsListCall) QuotaUser(quotaUser string) *OperationsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *OperationsListCall) UserIP(userIP string) *OperationsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsListCall) Fields(s ...googleapi.Field) *OperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsListCall) IfNoneMatch(entityTag string) *OperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsListCall) Context(ctx context.Context) *OperationsListCall {
	c.ctx_ = ctx
	return c
}

func (c *OperationsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
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

// Do executes the "deploymentmanager.operations.list" call.
// Exactly one of *OperationsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OperationsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do() (*OperationsListResponse, error) {
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
	ret := &OperationsListResponse{
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
	//   "description": "! Lists all Operations for a project.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.operations.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "50",
	//       "description": "! Maximum count of results to be returned. ! Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "! Specifies a nextPageToken returned by a previous list request. This ! token can be used to request the next page of results from a previous ! list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/operations",
	//   "response": {
	//     "$ref": "OperationsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.resources.get":

type ResourcesGetCall struct {
	s            *Service
	project      string
	deployment   string
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: ! Gets information about a single resource.
func (r *ResourcesService) Get(project string, deployment string, resource string) *ResourcesGetCall {
	c := &ResourcesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	c.resource = resource
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ResourcesGetCall) QuotaUser(quotaUser string) *ResourcesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ResourcesGetCall) UserIP(userIP string) *ResourcesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ResourcesGetCall) Fields(s ...googleapi.Field) *ResourcesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ResourcesGetCall) IfNoneMatch(entityTag string) *ResourcesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ResourcesGetCall) Context(ctx context.Context) *ResourcesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ResourcesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}/resources/{resource}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
		"resource":   c.resource,
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

// Do executes the "deploymentmanager.resources.get" call.
// Exactly one of *Resource or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Resource.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ResourcesGetCall) Do() (*Resource, error) {
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
	ret := &Resource{
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
	//   "description": "! Gets information about a single resource.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.resources.get",
	//   "parameterOrder": [
	//     "project",
	//     "deployment",
	//     "resource"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "resource": {
	//       "description": "! The name of the resource for this request.",
	//       "location": "path",
	//       "pattern": "[-a-z0-9]{1,61}",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}/resources/{resource}",
	//   "response": {
	//     "$ref": "Resource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.resources.list":

type ResourcesListCall struct {
	s            *Service
	project      string
	deployment   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: ! Lists all resources in a given deployment.
func (r *ResourcesService) List(project string, deployment string) *ResourcesListCall {
	c := &ResourcesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.deployment = deployment
	return c
}

// MaxResults sets the optional parameter "maxResults": ! Maximum count
// of results to be returned. ! Acceptable values are 0 to 100,
// inclusive. (Default: 50)
func (c *ResourcesListCall) MaxResults(maxResults int64) *ResourcesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": ! Specifies a
// nextPageToken returned by a previous list request. This ! token can
// be used to request the next page of results from a previous ! list
// request.
func (c *ResourcesListCall) PageToken(pageToken string) *ResourcesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ResourcesListCall) QuotaUser(quotaUser string) *ResourcesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ResourcesListCall) UserIP(userIP string) *ResourcesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ResourcesListCall) Fields(s ...googleapi.Field) *ResourcesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ResourcesListCall) IfNoneMatch(entityTag string) *ResourcesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ResourcesListCall) Context(ctx context.Context) *ResourcesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ResourcesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/deployments/{deployment}/resources")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"deployment": c.deployment,
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

// Do executes the "deploymentmanager.resources.list" call.
// Exactly one of *ResourcesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ResourcesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ResourcesListCall) Do() (*ResourcesListResponse, error) {
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
	ret := &ResourcesListResponse{
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
	//   "description": "! Lists all resources in a given deployment.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.resources.list",
	//   "parameterOrder": [
	//     "project",
	//     "deployment"
	//   ],
	//   "parameters": {
	//     "deployment": {
	//       "description": "! The name of the deployment for this request.",
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "50",
	//       "description": "! Maximum count of results to be returned. ! Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "! Specifies a nextPageToken returned by a previous list request. This ! token can be used to request the next page of results from a previous ! list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/deployments/{deployment}/resources",
	//   "response": {
	//     "$ref": "ResourcesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "deploymentmanager.types.list":

type TypesListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: ! Lists all Types for Deployment Manager.
func (r *TypesService) List(project string) *TypesListCall {
	c := &TypesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": ! Maximum count
// of results to be returned. ! Acceptable values are 0 to 100,
// inclusive. (Default: 50)
func (c *TypesListCall) MaxResults(maxResults int64) *TypesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": ! Specifies a
// nextPageToken returned by a previous list request. This ! token can
// be used to request the next page of results from a previous ! list
// request.
func (c *TypesListCall) PageToken(pageToken string) *TypesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TypesListCall) QuotaUser(quotaUser string) *TypesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TypesListCall) UserIP(userIP string) *TypesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TypesListCall) Fields(s ...googleapi.Field) *TypesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TypesListCall) IfNoneMatch(entityTag string) *TypesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TypesListCall) Context(ctx context.Context) *TypesListCall {
	c.ctx_ = ctx
	return c
}

func (c *TypesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/global/types")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
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

// Do executes the "deploymentmanager.types.list" call.
// Exactly one of *TypesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TypesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TypesListCall) Do() (*TypesListResponse, error) {
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
	ret := &TypesListResponse{
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
	//   "description": "! Lists all Types for Deployment Manager.",
	//   "httpMethod": "GET",
	//   "id": "deploymentmanager.types.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "50",
	//       "description": "! Maximum count of results to be returned. ! Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "! Specifies a nextPageToken returned by a previous list request. This ! token can be used to request the next page of results from a previous ! list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "! The project ID for this request.",
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/global/types",
	//   "response": {
	//     "$ref": "TypesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}
