// Package autoscaler provides access to the Google Compute Engine Autoscaler API.
//
// See http://developers.google.com/compute/docs/autoscaler
//
// Usage example:
//
//   import "google.golang.org/api/autoscaler/v1beta2"
//   ...
//   autoscalerService, err := autoscaler.New(oauthHttpClient)
package autoscaler // import "google.golang.org/api/autoscaler/v1beta2"

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

const apiId = "autoscaler:v1beta2"
const apiName = "autoscaler"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/autoscaler/v1beta2/"

// OAuth2 scopes used by this API.
const (
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
	s.Autoscalers = NewAutoscalersService(s)
	s.ZoneOperations = NewZoneOperationsService(s)
	s.Zones = NewZonesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Autoscalers *AutoscalersService

	ZoneOperations *ZoneOperationsService

	Zones *ZonesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAutoscalersService(s *Service) *AutoscalersService {
	rs := &AutoscalersService{s: s}
	return rs
}

type AutoscalersService struct {
	s *Service
}

func NewZoneOperationsService(s *Service) *ZoneOperationsService {
	rs := &ZoneOperationsService{s: s}
	return rs
}

type ZoneOperationsService struct {
	s *Service
}

func NewZonesService(s *Service) *ZonesService {
	rs := &ZonesService{s: s}
	return rs
}

type ZonesService struct {
	s *Service
}

// Autoscaler: Cloud Autoscaler resource.
type Autoscaler struct {
	// AutoscalingPolicy: Configuration parameters for autoscaling
	// algorithm.
	AutoscalingPolicy *AutoscalingPolicy `json:"autoscalingPolicy,omitempty"`

	// CreationTimestamp: [Output Only] Creation timestamp in RFC3339 text
	// format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional textual description of the resource provided
	// by the client.
	Description string `json:"description,omitempty"`

	// Id: [Output Only] Unique identifier for the resource; defined by the
	// server.
	Id uint64 `json:"id,omitempty,string"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the Autoscaler resource. Must be unique per project and
	// zone.
	Name string `json:"name,omitempty"`

	// SelfLink: [Output Only] A self-link to the Autoscaler configuration
	// resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Target: URL to the entity which will be autoscaled. Currently the
	// only supported value is ReplicaPool?s URL. Note: it is illegal to
	// specify multiple Autoscalers for the same target.
	Target string `json:"target,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AutoscalingPolicy")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Autoscaler) MarshalJSON() ([]byte, error) {
	type noMethod Autoscaler
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AutoscalerListResponse struct {
	// Items: Autoscaler resources.
	Items []*Autoscaler `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: [Output only] A token used to continue a truncated
	// list request.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalerListResponse) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalerListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AutoscalingPolicy: Cloud Autoscaler policy.
type AutoscalingPolicy struct {
	// CoolDownPeriodSec: The number of seconds that the Autoscaler should
	// wait between two succeeding changes to the number of virtual
	// machines. You should define an interval that is at least as long as
	// the initialization time of a virtual machine and the time it may take
	// for replica pool to create the virtual machine. The default is 60
	// seconds.
	CoolDownPeriodSec int64 `json:"coolDownPeriodSec,omitempty"`

	// CpuUtilization: Exactly one utilization policy should be provided.
	// Configuration parameters of CPU based autoscaling policy.
	CpuUtilization *AutoscalingPolicyCpuUtilization `json:"cpuUtilization,omitempty"`

	// CustomMetricUtilizations: Configuration parameters of autoscaling
	// based on custom metric.
	CustomMetricUtilizations []*AutoscalingPolicyCustomMetricUtilization `json:"customMetricUtilizations,omitempty"`

	// LoadBalancingUtilization: Configuration parameters of autoscaling
	// based on load balancer.
	LoadBalancingUtilization *AutoscalingPolicyLoadBalancingUtilization `json:"loadBalancingUtilization,omitempty"`

	// MaxNumReplicas: The maximum number of replicas that the Autoscaler
	// can scale up to.
	MaxNumReplicas int64 `json:"maxNumReplicas,omitempty"`

	// MinNumReplicas: The minimum number of replicas that the Autoscaler
	// can scale down to.
	MinNumReplicas int64 `json:"minNumReplicas,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CoolDownPeriodSec")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingPolicy) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingPolicy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AutoscalingPolicyCpuUtilization: CPU utilization policy.
type AutoscalingPolicyCpuUtilization struct {
	// UtilizationTarget: The target utilization that the Autoscaler should
	// maintain. It is represented as a fraction of used cores. For example:
	// 6 cores used in 8-core VM are represented here as 0.75. Must be a
	// float value between (0, 1]. If not defined, the default is 0.8.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`

	// ForceSendFields is a list of field names (e.g. "UtilizationTarget")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingPolicyCpuUtilization) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingPolicyCpuUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AutoscalingPolicyCustomMetricUtilization: Custom utilization metric
// policy.
type AutoscalingPolicyCustomMetricUtilization struct {
	// Metric: Identifier of the metric. It should be a Cloud Monitoring
	// metric. The metric can not have negative values. The metric should be
	// an utilization metric (increasing number of VMs handling requests x
	// times should reduce average value of the metric roughly x times). For
	// example you could use:
	// compute.googleapis.com/instance/network/received_bytes_count.
	Metric string `json:"metric,omitempty"`

	// UtilizationTarget: Target value of the metric which Autoscaler should
	// maintain. Must be a positive value.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`

	// UtilizationTargetType: Defines type in which utilization_target is
	// expressed.
	UtilizationTargetType string `json:"utilizationTargetType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metric") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingPolicyCustomMetricUtilization) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingPolicyCustomMetricUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AutoscalingPolicyLoadBalancingUtilization: Load balancing utilization
// policy.
type AutoscalingPolicyLoadBalancingUtilization struct {
	// UtilizationTarget: Fraction of backend capacity utilization (set in
	// HTTP load balancing configuration) that Autoscaler should maintain.
	// Must be a positive float value. If not defined, the default is 0.8.
	// For example if your maxRatePerInstance capacity (in HTTP Load
	// Balancing configuration) is set at 10 and you would like to keep
	// number of instances such that each instance receives 7 QPS on
	// average, set this to 0.7.
	UtilizationTarget float64 `json:"utilizationTarget,omitempty"`

	// ForceSendFields is a list of field names (e.g. "UtilizationTarget")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingPolicyLoadBalancingUtilization) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingPolicyLoadBalancingUtilization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DeprecationStatus struct {
	Deleted string `json:"deleted,omitempty"`

	Deprecated string `json:"deprecated,omitempty"`

	Obsolete string `json:"obsolete,omitempty"`

	Replacement string `json:"replacement,omitempty"`

	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeprecationStatus) MarshalJSON() ([]byte, error) {
	type noMethod DeprecationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Operation struct {
	ClientOperationId string `json:"clientOperationId,omitempty"`

	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	EndTime string `json:"endTime,omitempty"`

	Error *OperationError `json:"error,omitempty"`

	HttpErrorMessage string `json:"httpErrorMessage,omitempty"`

	HttpErrorStatusCode int64 `json:"httpErrorStatusCode,omitempty"`

	Id uint64 `json:"id,omitempty,string"`

	InsertTime string `json:"insertTime,omitempty"`

	// Kind: [Output Only] Type of the resource. Always compute#Operation
	// for Operation resources.
	Kind string `json:"kind,omitempty"`

	Name string `json:"name,omitempty"`

	OperationType string `json:"operationType,omitempty"`

	Progress int64 `json:"progress,omitempty"`

	Region string `json:"region,omitempty"`

	SelfLink string `json:"selfLink,omitempty"`

	StartTime string `json:"startTime,omitempty"`

	Status string `json:"status,omitempty"`

	StatusMessage string `json:"statusMessage,omitempty"`

	TargetId uint64 `json:"targetId,omitempty,string"`

	TargetLink string `json:"targetLink,omitempty"`

	User string `json:"user,omitempty"`

	Warnings []*OperationWarnings `json:"warnings,omitempty"`

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
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationError struct {
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
	Code string `json:"code,omitempty"`

	Location string `json:"location,omitempty"`

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
	Code string `json:"code,omitempty"`

	Data []*OperationWarningsData `json:"data,omitempty"`

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
	Key string `json:"key,omitempty"`

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

type OperationList struct {
	Id string `json:"id,omitempty"`

	Items []*Operation `json:"items,omitempty"`

	// Kind: [Output Only] Type of resource. Always compute#operations for
	// Operations resource.
	Kind string `json:"kind,omitempty"`

	NextPageToken string `json:"nextPageToken,omitempty"`

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
}

func (s *OperationList) MarshalJSON() ([]byte, error) {
	type noMethod OperationList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Zone struct {
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	Deprecated *DeprecationStatus `json:"deprecated,omitempty"`

	Description string `json:"description,omitempty"`

	Id uint64 `json:"id,omitempty,string"`

	// Kind: [Output Only] Type of the resource. Always kind#zone for zones.
	Kind string `json:"kind,omitempty"`

	MaintenanceWindows []*ZoneMaintenanceWindows `json:"maintenanceWindows,omitempty"`

	Name string `json:"name,omitempty"`

	Region string `json:"region,omitempty"`

	// SelfLink: [Output Only] Server defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CreationTimestamp")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Zone) MarshalJSON() ([]byte, error) {
	type noMethod Zone
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ZoneMaintenanceWindows struct {
	BeginTime string `json:"beginTime,omitempty"`

	Description string `json:"description,omitempty"`

	EndTime string `json:"endTime,omitempty"`

	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BeginTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ZoneMaintenanceWindows) MarshalJSON() ([]byte, error) {
	type noMethod ZoneMaintenanceWindows
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ZoneList struct {
	Id string `json:"id,omitempty"`

	Items []*Zone `json:"items,omitempty"`

	// Kind: Type of resource.
	Kind string `json:"kind,omitempty"`

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
}

func (s *ZoneList) MarshalJSON() ([]byte, error) {
	type noMethod ZoneList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "autoscaler.autoscalers.delete":

type AutoscalersDeleteCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes the specified Autoscaler resource.
func (r *AutoscalersService) Delete(project string, zone string, autoscaler string) *AutoscalersDeleteCall {
	c := &AutoscalersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersDeleteCall) QuotaUser(quotaUser string) *AutoscalersDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersDeleteCall) UserIP(userIP string) *AutoscalersDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersDeleteCall) Fields(s ...googleapi.Field) *AutoscalersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersDeleteCall) Context(ctx context.Context) *AutoscalersDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "autoscaler.autoscalers.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AutoscalersDeleteCall) Do() (*Operation, error) {
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
	//   "description": "Deletes the specified Autoscaler resource.",
	//   "httpMethod": "DELETE",
	//   "id": "autoscaler.autoscalers.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "autoscaler.autoscalers.get":

type AutoscalersGetCall struct {
	s            *Service
	project      string
	zone         string
	autoscaler   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified Autoscaler resource.
func (r *AutoscalersService) Get(project string, zone string, autoscaler string) *AutoscalersGetCall {
	c := &AutoscalersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersGetCall) QuotaUser(quotaUser string) *AutoscalersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersGetCall) UserIP(userIP string) *AutoscalersGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersGetCall) Fields(s ...googleapi.Field) *AutoscalersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AutoscalersGetCall) IfNoneMatch(entityTag string) *AutoscalersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersGetCall) Context(ctx context.Context) *AutoscalersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
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

// Do executes the "autoscaler.autoscalers.get" call.
// Exactly one of *Autoscaler or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Autoscaler.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AutoscalersGetCall) Do() (*Autoscaler, error) {
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
	ret := &Autoscaler{
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
	//   "description": "Gets the specified Autoscaler resource.",
	//   "httpMethod": "GET",
	//   "id": "autoscaler.autoscalers.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "response": {
	//     "$ref": "Autoscaler"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "autoscaler.autoscalers.insert":

type AutoscalersInsertCall struct {
	s          *Service
	project    string
	zone       string
	autoscaler *Autoscaler
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Adds new Autoscaler resource.
func (r *AutoscalersService) Insert(project string, zone string, autoscaler *Autoscaler) *AutoscalersInsertCall {
	c := &AutoscalersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersInsertCall) QuotaUser(quotaUser string) *AutoscalersInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersInsertCall) UserIP(userIP string) *AutoscalersInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersInsertCall) Fields(s ...googleapi.Field) *AutoscalersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersInsertCall) Context(ctx context.Context) *AutoscalersInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "autoscaler.autoscalers.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AutoscalersInsertCall) Do() (*Operation, error) {
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
	//   "description": "Adds new Autoscaler resource.",
	//   "httpMethod": "POST",
	//   "id": "autoscaler.autoscalers.insert",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "autoscaler.autoscalers.list":

type AutoscalersListCall struct {
	s            *Service
	project      string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists all Autoscaler resources in this zone.
func (r *AutoscalersService) List(project string, zone string) *AutoscalersListCall {
	c := &AutoscalersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter":
func (c *AutoscalersListCall) Filter(filter string) *AutoscalersListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults":
func (c *AutoscalersListCall) MaxResults(maxResults int64) *AutoscalersListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken":
func (c *AutoscalersListCall) PageToken(pageToken string) *AutoscalersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersListCall) QuotaUser(quotaUser string) *AutoscalersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersListCall) UserIP(userIP string) *AutoscalersListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersListCall) Fields(s ...googleapi.Field) *AutoscalersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AutoscalersListCall) IfNoneMatch(entityTag string) *AutoscalersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersListCall) Context(ctx context.Context) *AutoscalersListCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
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

// Do executes the "autoscaler.autoscalers.list" call.
// Exactly one of *AutoscalerListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AutoscalerListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AutoscalersListCall) Do() (*AutoscalerListResponse, error) {
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
	ret := &AutoscalerListResponse{
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
	//   "description": "Lists all Autoscaler resources in this zone.",
	//   "httpMethod": "GET",
	//   "id": "autoscaler.autoscalers.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers",
	//   "response": {
	//     "$ref": "AutoscalerListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "autoscaler.autoscalers.patch":

type AutoscalersPatchCall struct {
	s           *Service
	project     string
	zone        string
	autoscaler  string
	autoscaler2 *Autoscaler
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Patch: Update the entire content of the Autoscaler resource. This
// method supports patch semantics.
func (r *AutoscalersService) Patch(project string, zone string, autoscaler string, autoscaler2 *Autoscaler) *AutoscalersPatchCall {
	c := &AutoscalersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	c.autoscaler2 = autoscaler2
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersPatchCall) QuotaUser(quotaUser string) *AutoscalersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersPatchCall) UserIP(userIP string) *AutoscalersPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersPatchCall) Fields(s ...googleapi.Field) *AutoscalersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersPatchCall) Context(ctx context.Context) *AutoscalersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "autoscaler.autoscalers.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AutoscalersPatchCall) Do() (*Operation, error) {
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
	//   "description": "Update the entire content of the Autoscaler resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "autoscaler.autoscalers.patch",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "autoscaler.autoscalers.update":

type AutoscalersUpdateCall struct {
	s           *Service
	project     string
	zone        string
	autoscaler  string
	autoscaler2 *Autoscaler
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Update: Update the entire content of the Autoscaler resource.
func (r *AutoscalersService) Update(project string, zone string, autoscaler string, autoscaler2 *Autoscaler) *AutoscalersUpdateCall {
	c := &AutoscalersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.autoscaler = autoscaler
	c.autoscaler2 = autoscaler2
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *AutoscalersUpdateCall) QuotaUser(quotaUser string) *AutoscalersUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *AutoscalersUpdateCall) UserIP(userIP string) *AutoscalersUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AutoscalersUpdateCall) Fields(s ...googleapi.Field) *AutoscalersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AutoscalersUpdateCall) Context(ctx context.Context) *AutoscalersUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *AutoscalersUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.autoscaler2)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/zones/{zone}/autoscalers/{autoscaler}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":    c.project,
		"zone":       c.zone,
		"autoscaler": c.autoscaler,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "autoscaler.autoscalers.update" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AutoscalersUpdateCall) Do() (*Operation, error) {
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
	//   "description": "Update the entire content of the Autoscaler resource.",
	//   "httpMethod": "PUT",
	//   "id": "autoscaler.autoscalers.update",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "autoscaler"
	//   ],
	//   "parameters": {
	//     "autoscaler": {
	//       "description": "Name of the Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "Zone name of Autoscaler resource.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/zones/{zone}/autoscalers/{autoscaler}",
	//   "request": {
	//     "$ref": "Autoscaler"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "autoscaler.zoneOperations.delete":

type ZoneOperationsDeleteCall struct {
	s          *Service
	project    string
	zone       string
	operation  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes the specified zone-specific operation resource.
func (r *ZoneOperationsService) Delete(project string, zone string, operation string) *ZoneOperationsDeleteCall {
	c := &ZoneOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ZoneOperationsDeleteCall) QuotaUser(quotaUser string) *ZoneOperationsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ZoneOperationsDeleteCall) UserIP(userIP string) *ZoneOperationsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZoneOperationsDeleteCall) Fields(s ...googleapi.Field) *ZoneOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZoneOperationsDeleteCall) Context(ctx context.Context) *ZoneOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ZoneOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"zone":      c.zone,
		"operation": c.operation,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "autoscaler.zoneOperations.delete" call.
func (c *ZoneOperationsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified zone-specific operation resource.",
	//   "httpMethod": "DELETE",
	//   "id": "autoscaler.zoneOperations.delete",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones/{zone}/operations/{operation}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute"
	//   ]
	// }

}

// method id "autoscaler.zoneOperations.get":

type ZoneOperationsGetCall struct {
	s            *Service
	project      string
	zone         string
	operation    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieves the specified zone-specific operation resource.
func (r *ZoneOperationsService) Get(project string, zone string, operation string) *ZoneOperationsGetCall {
	c := &ZoneOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	c.operation = operation
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ZoneOperationsGetCall) QuotaUser(quotaUser string) *ZoneOperationsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ZoneOperationsGetCall) UserIP(userIP string) *ZoneOperationsGetCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ZoneOperationsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"zone":      c.zone,
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

// Do executes the "autoscaler.zoneOperations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ZoneOperationsGetCall) Do() (*Operation, error) {
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
	//   "description": "Retrieves the specified zone-specific operation resource.",
	//   "httpMethod": "GET",
	//   "id": "autoscaler.zoneOperations.get",
	//   "parameterOrder": [
	//     "project",
	//     "zone",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "operation": {
	//       "location": "path",
	//       "pattern": "[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
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
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "autoscaler.zoneOperations.list":

type ZoneOperationsListCall struct {
	s            *Service
	project      string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves the list of operation resources contained within the
// specified zone.
func (r *ZoneOperationsService) List(project string, zone string) *ZoneOperationsListCall {
	c := &ZoneOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.zone = zone
	return c
}

// Filter sets the optional parameter "filter":
func (c *ZoneOperationsListCall) Filter(filter string) *ZoneOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults":
func (c *ZoneOperationsListCall) MaxResults(maxResults int64) *ZoneOperationsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken":
func (c *ZoneOperationsListCall) PageToken(pageToken string) *ZoneOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ZoneOperationsListCall) QuotaUser(quotaUser string) *ZoneOperationsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ZoneOperationsListCall) UserIP(userIP string) *ZoneOperationsListCall {
	c.urlParams_.Set("userIp", userIP)
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

func (c *ZoneOperationsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones/{zone}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"zone":    c.zone,
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

// Do executes the "autoscaler.zoneOperations.list" call.
// Exactly one of *OperationList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *OperationList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ZoneOperationsListCall) Do() (*OperationList, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of operation resources contained within the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "autoscaler.zoneOperations.list",
	//   "parameterOrder": [
	//     "project",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
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
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}

// method id "autoscaler.zones.list":

type ZonesListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List:
func (r *ZonesService) List(project string) *ZonesListCall {
	c := &ZonesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter":
func (c *ZonesListCall) Filter(filter string) *ZonesListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults":
func (c *ZonesListCall) MaxResults(maxResults int64) *ZonesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken":
func (c *ZonesListCall) PageToken(pageToken string) *ZonesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ZonesListCall) QuotaUser(quotaUser string) *ZonesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ZonesListCall) UserIP(userIP string) *ZonesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ZonesListCall) Fields(s ...googleapi.Field) *ZonesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ZonesListCall) IfNoneMatch(entityTag string) *ZonesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ZonesListCall) Context(ctx context.Context) *ZonesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ZonesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/zones")
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

// Do executes the "autoscaler.zones.list" call.
// Exactly one of *ZoneList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *ZoneList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ZonesListCall) Do() (*ZoneList, error) {
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
	ret := &ZoneList{
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
	//   "description": "",
	//   "httpMethod": "GET",
	//   "id": "autoscaler.zones.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "500",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "location": "path",
	//       "pattern": "(?:(?:[-a-z0-9]{1,63}\\.)*(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?):)?(?:[0-9]{1,19}|(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?))",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/zones",
	//   "response": {
	//     "$ref": "ZoneList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/compute.readonly"
	//   ]
	// }

}
