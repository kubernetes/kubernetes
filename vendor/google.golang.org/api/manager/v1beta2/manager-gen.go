// Package manager provides access to the Deployment Manager API.
//
// See https://developers.google.com/deployment-manager/
//
// Usage example:
//
//   import "google.golang.org/api/manager/v1beta2"
//   ...
//   managerService, err := manager.New(oauthHttpClient)
package manager // import "google.golang.org/api/manager/v1beta2"

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

const apiId = "manager:v1beta2"
const apiName = "manager"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/manager/v1beta2/projects/"

// OAuth2 scopes used by this API.
const (
	// View and manage your applications deployed on Google App Engine
	AppengineAdminScope = "https://www.googleapis.com/auth/appengine.admin"

	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View your data across Google Cloud Platform services
	CloudPlatformReadOnlyScope = "https://www.googleapis.com/auth/cloud-platform.read-only"

	// View and manage your Google Compute Engine resources
	ComputeScope = "https://www.googleapis.com/auth/compute"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"

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
	s.Templates = NewTemplatesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Deployments *DeploymentsService

	Templates *TemplatesService
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

func NewTemplatesService(s *Service) *TemplatesService {
	rs := &TemplatesService{s: s}
	return rs
}

type TemplatesService struct {
	s *Service
}

// AccessConfig: A Compute Engine network accessConfig. Identical to the
// accessConfig on corresponding Compute Engine resource.
type AccessConfig struct {
	// Name: Name of this access configuration.
	Name string `json:"name,omitempty"`

	// NatIp: An external IP address associated with this instance.
	NatIp string `json:"natIp,omitempty"`

	// Type: Type of this access configuration file. (Currently only
	// ONE_TO_ONE_NAT is legal.)
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AccessConfig) MarshalJSON() ([]byte, error) {
	type noMethod AccessConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Action: An Action encapsulates a set of commands as a single runnable
// module with additional information needed during run-time.
type Action struct {
	// Commands: A list of commands to run sequentially for this action.
	Commands []string `json:"commands,omitempty"`

	// TimeoutMs: The timeout in milliseconds for this action to run.
	TimeoutMs int64 `json:"timeoutMs,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Commands") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Action) MarshalJSON() ([]byte, error) {
	type noMethod Action
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AllowedRule: An allowed port resource.
type AllowedRule struct {
	// IPProtocol: ?tcp?, ?udp? or ?icmp?
	IPProtocol string `json:"IPProtocol,omitempty"`

	// Ports: List of ports or port ranges (Example inputs include: ["22"],
	// [?33?, "12345-12349"].
	Ports []string `json:"ports,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IPProtocol") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AllowedRule) MarshalJSON() ([]byte, error) {
	type noMethod AllowedRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AutoscalingModule struct {
	CoolDownPeriodSec int64 `json:"coolDownPeriodSec,omitempty"`

	Description string `json:"description,omitempty"`

	MaxNumReplicas int64 `json:"maxNumReplicas,omitempty"`

	MinNumReplicas int64 `json:"minNumReplicas,omitempty"`

	SignalType string `json:"signalType,omitempty"`

	TargetModule string `json:"targetModule,omitempty"`

	// TargetUtilization: target_utilization should be in range [0,1].
	TargetUtilization float64 `json:"targetUtilization,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CoolDownPeriodSec")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingModule) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AutoscalingModuleStatus struct {
	// AutoscalingConfigUrl: [Output Only] The URL of the corresponding
	// Autoscaling configuration.
	AutoscalingConfigUrl string `json:"autoscalingConfigUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AutoscalingConfigUrl") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AutoscalingModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod AutoscalingModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DeployState: [Output Only] The current state of a replica or module.
type DeployState struct {
	// Details: [Output Only] Human readable details about the current
	// state.
	Details string `json:"details,omitempty"`

	// Status: [Output Only] The status of the deployment. Possible values
	// include:
	// - UNKNOWN
	// - DEPLOYING
	// - DEPLOYED
	// - DEPLOYMENT_FAILED
	// - DELETING
	// - DELETED
	// - DELETE_FAILED
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeployState) MarshalJSON() ([]byte, error) {
	type noMethod DeployState
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Deployment: A deployment represents a physical instantiation of a
// Template.
type Deployment struct {
	// CreationDate: [Output Only] The time when this deployment was
	// created.
	CreationDate string `json:"creationDate,omitempty"`

	// Description: A user-supplied description of this Deployment.
	Description string `json:"description,omitempty"`

	// Modules: [Output Only] List of status for the modules in this
	// deployment.
	Modules map[string]ModuleStatus `json:"modules,omitempty"`

	// Name: Name of this deployment. The name must conform to the following
	// regular expression: [a-zA-Z0-9-_]{1,64}
	Name string `json:"name,omitempty"`

	// Overrides: The set of parameter overrides to apply to the
	// corresponding Template before deploying.
	Overrides []*ParamOverride `json:"overrides,omitempty"`

	// State: [Output Only] Current status of this deployment.
	State *DeployState `json:"state,omitempty"`

	// TemplateName: The name of the Template on which this deployment is
	// based.
	TemplateName string `json:"templateName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreationDate") to
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

type DeploymentsListResponse struct {
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Deployment `json:"resources,omitempty"`

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

func (s *DeploymentsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeploymentsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DiskAttachment: How to attach a disk to a Replica.
type DiskAttachment struct {
	// DeviceName: The device name of this disk.
	DeviceName string `json:"deviceName,omitempty"`

	// Index: A zero-based index to assign to this disk, where 0 is reserved
	// for the boot disk. If not specified, this is assigned by the server.
	Index int64 `json:"index,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeviceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DiskAttachment) MarshalJSON() ([]byte, error) {
	type noMethod DiskAttachment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// EnvVariable: An environment variable.
type EnvVariable struct {
	// Hidden: Whether this variable is hidden or visible.
	Hidden bool `json:"hidden,omitempty"`

	// Value: Value of the environment variable.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Hidden") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EnvVariable) MarshalJSON() ([]byte, error) {
	type noMethod EnvVariable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ExistingDisk: A pre-existing persistent disk that will be attached to
// every Replica in the Pool.
type ExistingDisk struct {
	// Attachment: Optional. How the disk will be attached to the Replica.
	Attachment *DiskAttachment `json:"attachment,omitempty"`

	// Source: The fully-qualified URL of the Persistent Disk resource. It
	// must be in the same zone as the Pool.
	Source string `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Attachment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ExistingDisk) MarshalJSON() ([]byte, error) {
	type noMethod ExistingDisk
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// FirewallModule: A Firewall resource
type FirewallModule struct {
	// Allowed: The allowed ports or port ranges.
	Allowed []*AllowedRule `json:"allowed,omitempty"`

	// Description: The description of the firewall (optional)
	Description string `json:"description,omitempty"`

	// Network: The NetworkModule to which this firewall should apply. If
	// not specified, or if specified as 'default', this firewall will be
	// applied to the 'default' network.
	Network string `json:"network,omitempty"`

	// SourceRanges: Source IP ranges to apply this firewall to, see the GCE
	// Spec for details on syntax
	SourceRanges []string `json:"sourceRanges,omitempty"`

	// SourceTags: Source Tags to apply this firewall to, see the GCE Spec
	// for details on syntax
	SourceTags []string `json:"sourceTags,omitempty"`

	// TargetTags: Target Tags to apply this firewall to, see the GCE Spec
	// for details on syntax
	TargetTags []string `json:"targetTags,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Allowed") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FirewallModule) MarshalJSON() ([]byte, error) {
	type noMethod FirewallModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type FirewallModuleStatus struct {
	// FirewallUrl: [Output Only] The URL of the corresponding Firewall
	// resource.
	FirewallUrl string `json:"firewallUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FirewallUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *FirewallModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod FirewallModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HealthCheckModule struct {
	CheckIntervalSec int64 `json:"checkIntervalSec,omitempty"`

	Description string `json:"description,omitempty"`

	HealthyThreshold int64 `json:"healthyThreshold,omitempty"`

	Host string `json:"host,omitempty"`

	Path string `json:"path,omitempty"`

	Port int64 `json:"port,omitempty"`

	TimeoutSec int64 `json:"timeoutSec,omitempty"`

	UnhealthyThreshold int64 `json:"unhealthyThreshold,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CheckIntervalSec") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HealthCheckModule) MarshalJSON() ([]byte, error) {
	type noMethod HealthCheckModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HealthCheckModuleStatus struct {
	// HealthCheckUrl: [Output Only] The HealthCheck URL.
	HealthCheckUrl string `json:"healthCheckUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HealthCheckUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HealthCheckModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod HealthCheckModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type LbModule struct {
	Description string `json:"description,omitempty"`

	HealthChecks []string `json:"healthChecks,omitempty"`

	IpAddress string `json:"ipAddress,omitempty"`

	IpProtocol string `json:"ipProtocol,omitempty"`

	PortRange string `json:"portRange,omitempty"`

	SessionAffinity string `json:"sessionAffinity,omitempty"`

	TargetModules []string `json:"targetModules,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LbModule) MarshalJSON() ([]byte, error) {
	type noMethod LbModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type LbModuleStatus struct {
	// ForwardingRuleUrl: [Output Only] The URL of the corresponding
	// ForwardingRule in GCE.
	ForwardingRuleUrl string `json:"forwardingRuleUrl,omitempty"`

	// TargetPoolUrl: [Output Only] The URL of the corresponding TargetPool
	// resource in GCE.
	TargetPoolUrl string `json:"targetPoolUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ForwardingRuleUrl")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LbModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod LbModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Metadata: A Compute Engine metadata entry. Identical to the metadata
// on the corresponding Compute Engine resource.
type Metadata struct {
	// FingerPrint: The fingerprint of the metadata.
	FingerPrint string `json:"fingerPrint,omitempty"`

	// Items: A list of metadata items.
	Items []*MetadataItem `json:"items,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FingerPrint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Metadata) MarshalJSON() ([]byte, error) {
	type noMethod Metadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MetadataItem: A Compute Engine metadata item, defined as a key:value
// pair. Identical to the metadata on the corresponding Compute Engine
// resource.
type MetadataItem struct {
	// Key: A metadata key.
	Key string `json:"key,omitempty"`

	// Value: A metadata value.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MetadataItem) MarshalJSON() ([]byte, error) {
	type noMethod MetadataItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Module: A module in a configuration. A module represents a single
// homogeneous, possibly replicated task.
type Module struct {
	AutoscalingModule *AutoscalingModule `json:"autoscalingModule,omitempty"`

	FirewallModule *FirewallModule `json:"firewallModule,omitempty"`

	HealthCheckModule *HealthCheckModule `json:"healthCheckModule,omitempty"`

	LbModule *LbModule `json:"lbModule,omitempty"`

	NetworkModule *NetworkModule `json:"networkModule,omitempty"`

	ReplicaPoolModule *ReplicaPoolModule `json:"replicaPoolModule,omitempty"`

	// Type: The type of this module. Valid values ("AUTOSCALING",
	// "FIREWALL", "HEALTH_CHECK", "LOAD_BALANCING", "NETWORK",
	// "REPLICA_POOL")
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AutoscalingModule")
	// to unconditionally include in API requests. By default, fields with
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

// ModuleStatus: [Output Only] Aggregate status for a module.
type ModuleStatus struct {
	// AutoscalingModuleStatus: [Output Only] The status of the
	// AutoscalingModule, set for type AUTOSCALING.
	AutoscalingModuleStatus *AutoscalingModuleStatus `json:"autoscalingModuleStatus,omitempty"`

	// FirewallModuleStatus: [Output Only] The status of the FirewallModule,
	// set for type FIREWALL.
	FirewallModuleStatus *FirewallModuleStatus `json:"firewallModuleStatus,omitempty"`

	// HealthCheckModuleStatus: [Output Only] The status of the
	// HealthCheckModule, set for type HEALTH_CHECK.
	HealthCheckModuleStatus *HealthCheckModuleStatus `json:"healthCheckModuleStatus,omitempty"`

	// LbModuleStatus: [Output Only] The status of the LbModule, set for
	// type LOAD_BALANCING.
	LbModuleStatus *LbModuleStatus `json:"lbModuleStatus,omitempty"`

	// NetworkModuleStatus: [Output Only] The status of the NetworkModule,
	// set for type NETWORK.
	NetworkModuleStatus *NetworkModuleStatus `json:"networkModuleStatus,omitempty"`

	// ReplicaPoolModuleStatus: [Output Only] The status of the
	// ReplicaPoolModule, set for type VM.
	ReplicaPoolModuleStatus *ReplicaPoolModuleStatus `json:"replicaPoolModuleStatus,omitempty"`

	// State: [Output Only] The current state of the module.
	State *DeployState `json:"state,omitempty"`

	// Type: [Output Only] The type of the module.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "AutoscalingModuleStatus") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod ModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// NetworkInterface: A Compute Engine NetworkInterface resource.
// Identical to the NetworkInterface on the corresponding Compute Engine
// resource.
type NetworkInterface struct {
	// AccessConfigs: An array of configurations for this interface. This
	// specifies how this interface is configured to interact with other
	// network services
	AccessConfigs []*AccessConfig `json:"accessConfigs,omitempty"`

	// Name: Name of the interface.
	Name string `json:"name,omitempty"`

	// Network: The name of the NetworkModule to which this interface
	// applies. If not specified, or specified as 'default', this will use
	// the 'default' network.
	Network string `json:"network,omitempty"`

	// NetworkIp: An optional IPV4 internal network address to assign to the
	// instance for this network interface.
	NetworkIp string `json:"networkIp,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccessConfigs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NetworkInterface) MarshalJSON() ([]byte, error) {
	type noMethod NetworkInterface
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type NetworkModule struct {
	// IPv4Range: Required; The range of internal addresses that are legal
	// on this network. This range is a CIDR specification, for example:
	// 192.168.0.0/16.
	IPv4Range string `json:"IPv4Range,omitempty"`

	// Description: The description of the network.
	Description string `json:"description,omitempty"`

	// GatewayIPv4: An optional address that is used for default routing to
	// other networks. This must be within the range specified by IPv4Range,
	// and is typicall the first usable address in that range. If not
	// specified, the default value is the first usable address in
	// IPv4Range.
	GatewayIPv4 string `json:"gatewayIPv4,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IPv4Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NetworkModule) MarshalJSON() ([]byte, error) {
	type noMethod NetworkModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type NetworkModuleStatus struct {
	// NetworkUrl: [Output Only] The URL of the corresponding Network
	// resource.
	NetworkUrl string `json:"networkUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NetworkUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NetworkModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod NetworkModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// NewDisk: A Persistent Disk resource that will be created and attached
// to each Replica in the Pool. Each Replica will have a unique
// persistent disk that is created and attached to that Replica.
type NewDisk struct {
	// Attachment: How the disk will be attached to the Replica.
	Attachment *DiskAttachment `json:"attachment,omitempty"`

	// AutoDelete: If true, then this disk will be deleted when the instance
	// is deleted.
	AutoDelete bool `json:"autoDelete,omitempty"`

	// Boot: If true, indicates that this is the root persistent disk.
	Boot bool `json:"boot,omitempty"`

	// InitializeParams: Create the new disk using these parameters. The
	// name of the disk will be <instance_name>-<five_random_charactersgt;.
	InitializeParams *NewDiskInitializeParams `json:"initializeParams,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Attachment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NewDisk) MarshalJSON() ([]byte, error) {
	type noMethod NewDisk
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// NewDiskInitializeParams: Initialization parameters for creating a new
// disk.
type NewDiskInitializeParams struct {
	// DiskSizeGb: The size of the created disk in gigabytes.
	DiskSizeGb int64 `json:"diskSizeGb,omitempty,string"`

	// DiskType: Name of the disk type resource describing which disk type
	// to use to create the disk. For example 'pd-ssd' or 'pd-standard'.
	// Default is 'pd-standard'
	DiskType string `json:"diskType,omitempty"`

	// SourceImage: The fully-qualified URL of a source image to use to
	// create this disk.
	SourceImage string `json:"sourceImage,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DiskSizeGb") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NewDiskInitializeParams) MarshalJSON() ([]byte, error) {
	type noMethod NewDiskInitializeParams
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ParamOverride: A specification for overriding parameters in a
// Template that corresponds to the Deployment.
type ParamOverride struct {
	// Path: A JSON Path expression that specifies which parameter should be
	// overridden.
	Path string `json:"path,omitempty"`

	// Value: The new value to assign to the overridden parameter.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Path") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ParamOverride) MarshalJSON() ([]byte, error) {
	type noMethod ParamOverride
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ReplicaPoolModule struct {
	// EnvVariables: A list of environment variables.
	EnvVariables map[string]EnvVariable `json:"envVariables,omitempty"`

	// HealthChecks: The Health Checks to configure for the
	// ReplicaPoolModule
	HealthChecks []string `json:"healthChecks,omitempty"`

	// NumReplicas: Number of replicas in this module.
	NumReplicas int64 `json:"numReplicas,omitempty"`

	// ReplicaPoolParams: Information for a ReplicaPoolModule.
	ReplicaPoolParams *ReplicaPoolParams `json:"replicaPoolParams,omitempty"`

	// ResourceView: [Output Only] The name of the Resource View associated
	// with a ReplicaPoolModule. This field will be generated by the
	// service.
	ResourceView string `json:"resourceView,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EnvVariables") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReplicaPoolModule) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaPoolModule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ReplicaPoolModuleStatus struct {
	// ReplicaPoolUrl: [Output Only] The URL of the associated ReplicaPool
	// resource.
	ReplicaPoolUrl string `json:"replicaPoolUrl,omitempty"`

	// ResourceViewUrl: [Output Only] The URL of the Resource Group
	// associated with this ReplicaPool.
	ResourceViewUrl string `json:"resourceViewUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReplicaPoolUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReplicaPoolModuleStatus) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaPoolModuleStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReplicaPoolParams: Configuration information for a ReplicaPools
// resource. Specifying an item within will determine the ReplicaPools
// API version used for a ReplicaPoolModule. Only one may be specified.
type ReplicaPoolParams struct {
	// V1beta1: ReplicaPoolParams specifications for use with ReplicaPools
	// v1beta1.
	V1beta1 *ReplicaPoolParamsV1Beta1 `json:"v1beta1,omitempty"`

	// ForceSendFields is a list of field names (e.g. "V1beta1") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReplicaPoolParams) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaPoolParams
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReplicaPoolParamsV1Beta1: Configuration information for a
// ReplicaPools v1beta1 API resource. Directly maps to ReplicaPool
// InitTemplate.
type ReplicaPoolParamsV1Beta1 struct {
	// AutoRestart: Whether these replicas should be restarted if they
	// experience a failure. The default value is true.
	AutoRestart bool `json:"autoRestart,omitempty"`

	// BaseInstanceName: The base name for instances within this
	// ReplicaPool.
	BaseInstanceName string `json:"baseInstanceName,omitempty"`

	// CanIpForward: Enables IP Forwarding
	CanIpForward bool `json:"canIpForward,omitempty"`

	// Description: An optional textual description of the resource.
	Description string `json:"description,omitempty"`

	// DisksToAttach: A list of existing Persistent Disk resources to attach
	// to each replica in the pool. Each disk will be attached in read-only
	// mode to every replica.
	DisksToAttach []*ExistingDisk `json:"disksToAttach,omitempty"`

	// DisksToCreate: A list of Disk resources to create and attach to each
	// Replica in the Pool. Currently, you can only define one disk and it
	// must be a root persistent disk. Note that Replica Pool will create a
	// root persistent disk for each replica.
	DisksToCreate []*NewDisk `json:"disksToCreate,omitempty"`

	// InitAction: Name of the Action to be run during initialization of a
	// ReplicaPoolModule.
	InitAction string `json:"initAction,omitempty"`

	// MachineType: The machine type for this instance. Either a complete
	// URL, or the resource name (e.g. n1-standard-1).
	MachineType string `json:"machineType,omitempty"`

	// Metadata: The metadata key/value pairs assigned to this instance.
	Metadata *Metadata `json:"metadata,omitempty"`

	// NetworkInterfaces: A list of network interfaces for the instance.
	// Currently only one interface is supported by Google Compute Engine.
	NetworkInterfaces []*NetworkInterface `json:"networkInterfaces,omitempty"`

	OnHostMaintenance string `json:"onHostMaintenance,omitempty"`

	// ServiceAccounts: A list of Service Accounts to enable for this
	// instance.
	ServiceAccounts []*ServiceAccount `json:"serviceAccounts,omitempty"`

	// Tags: A list of tags to apply to the Google Compute Engine instance
	// to identify resources.
	Tags *Tag `json:"tags,omitempty"`

	// Zone: The zone for this ReplicaPool.
	Zone string `json:"zone,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AutoRestart") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReplicaPoolParamsV1Beta1) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaPoolParamsV1Beta1
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ServiceAccount: A Compute Engine service account, identical to the
// Compute Engine resource.
type ServiceAccount struct {
	// Email: Service account email address.
	Email string `json:"email,omitempty"`

	// Scopes: List of OAuth2 scopes to obtain for the service account.
	Scopes []string `json:"scopes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Email") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ServiceAccount) MarshalJSON() ([]byte, error) {
	type noMethod ServiceAccount
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Tag: A Compute Engine Instance tag, identical to the tags on the
// corresponding Compute Engine Instance resource.
type Tag struct {
	// FingerPrint: The fingerprint of the tag.
	FingerPrint string `json:"fingerPrint,omitempty"`

	// Items: Items contained in this tag.
	Items []string `json:"items,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FingerPrint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Tag) MarshalJSON() ([]byte, error) {
	type noMethod Tag
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Template: A Template represents a complete configuration for a
// Deployment.
type Template struct {
	// Actions: Action definitions for use in Module intents in this
	// Template.
	Actions map[string]Action `json:"actions,omitempty"`

	// Description: A user-supplied description of this Template.
	Description string `json:"description,omitempty"`

	// Modules: A list of modules for this Template.
	Modules map[string]Module `json:"modules,omitempty"`

	// Name: Name of this Template. The name must conform to the expression:
	// [a-zA-Z0-9-_]{1,64}
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Actions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Template) MarshalJSON() ([]byte, error) {
	type noMethod Template
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type TemplatesListResponse struct {
	NextPageToken string `json:"nextPageToken,omitempty"`

	Resources []*Template `json:"resources,omitempty"`

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

func (s *TemplatesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod TemplatesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "manager.deployments.delete":

type DeploymentsDeleteCall struct {
	s              *Service
	projectId      string
	region         string
	deploymentName string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Delete:
func (r *DeploymentsService) Delete(projectId string, region string, deploymentName string) *DeploymentsDeleteCall {
	c := &DeploymentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.deploymentName = deploymentName
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/regions/{region}/deployments/{deploymentName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":      c.projectId,
		"region":         c.region,
		"deploymentName": c.deploymentName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "manager.deployments.delete" call.
func (c *DeploymentsDeleteCall) Do() error {
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
	//   "description": "",
	//   "httpMethod": "DELETE",
	//   "id": "manager.deployments.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "deploymentName"
	//   ],
	//   "parameters": {
	//     "deploymentName": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/regions/{region}/deployments/{deploymentName}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "manager.deployments.get":

type DeploymentsGetCall struct {
	s              *Service
	projectId      string
	region         string
	deploymentName string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
}

// Get:
func (r *DeploymentsService) Get(projectId string, region string, deploymentName string) *DeploymentsGetCall {
	c := &DeploymentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.deploymentName = deploymentName
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/regions/{region}/deployments/{deploymentName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":      c.projectId,
		"region":         c.region,
		"deploymentName": c.deploymentName,
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

// Do executes the "manager.deployments.get" call.
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
	//   "description": "",
	//   "httpMethod": "GET",
	//   "id": "manager.deployments.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "deploymentName"
	//   ],
	//   "parameters": {
	//     "deploymentName": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/regions/{region}/deployments/{deploymentName}",
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

// method id "manager.deployments.insert":

type DeploymentsInsertCall struct {
	s          *Service
	projectId  string
	region     string
	deployment *Deployment
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert:
func (r *DeploymentsService) Insert(projectId string, region string, deployment *Deployment) *DeploymentsInsertCall {
	c := &DeploymentsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/regions/{region}/deployments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "manager.deployments.insert" call.
// Exactly one of *Deployment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Deployment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DeploymentsInsertCall) Do() (*Deployment, error) {
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
	//   "description": "",
	//   "httpMethod": "POST",
	//   "id": "manager.deployments.insert",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/regions/{region}/deployments",
	//   "request": {
	//     "$ref": "Deployment"
	//   },
	//   "response": {
	//     "$ref": "Deployment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appengine.admin",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/compute",
	//     "https://www.googleapis.com/auth/devstorage.read_write",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "manager.deployments.list":

type DeploymentsListCall struct {
	s            *Service
	projectId    string
	region       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List:
func (r *DeploymentsService) List(projectId string, region string) *DeploymentsListCall {
	c := &DeploymentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 100, inclusive.
// (Default: 50)
func (c *DeploymentsListCall) MaxResults(maxResults int64) *DeploymentsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/regions/{region}/deployments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
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

// Do executes the "manager.deployments.list" call.
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
	//   "description": "",
	//   "httpMethod": "GET",
	//   "id": "manager.deployments.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "50",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/regions/{region}/deployments",
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

// method id "manager.templates.delete":

type TemplatesDeleteCall struct {
	s            *Service
	projectId    string
	templateName string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete:
func (r *TemplatesService) Delete(projectId string, templateName string) *TemplatesDeleteCall {
	c := &TemplatesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.templateName = templateName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TemplatesDeleteCall) QuotaUser(quotaUser string) *TemplatesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TemplatesDeleteCall) UserIP(userIP string) *TemplatesDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TemplatesDeleteCall) Fields(s ...googleapi.Field) *TemplatesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TemplatesDeleteCall) Context(ctx context.Context) *TemplatesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *TemplatesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/templates/{templateName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":    c.projectId,
		"templateName": c.templateName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "manager.templates.delete" call.
func (c *TemplatesDeleteCall) Do() error {
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
	//   "description": "",
	//   "httpMethod": "DELETE",
	//   "id": "manager.templates.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "templateName"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "templateName": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/templates/{templateName}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "manager.templates.get":

type TemplatesGetCall struct {
	s            *Service
	projectId    string
	templateName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get:
func (r *TemplatesService) Get(projectId string, templateName string) *TemplatesGetCall {
	c := &TemplatesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.templateName = templateName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TemplatesGetCall) QuotaUser(quotaUser string) *TemplatesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TemplatesGetCall) UserIP(userIP string) *TemplatesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TemplatesGetCall) Fields(s ...googleapi.Field) *TemplatesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TemplatesGetCall) IfNoneMatch(entityTag string) *TemplatesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TemplatesGetCall) Context(ctx context.Context) *TemplatesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *TemplatesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/templates/{templateName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":    c.projectId,
		"templateName": c.templateName,
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

// Do executes the "manager.templates.get" call.
// Exactly one of *Template or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Template.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TemplatesGetCall) Do() (*Template, error) {
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
	ret := &Template{
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
	//   "id": "manager.templates.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "templateName"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "templateName": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/templates/{templateName}",
	//   "response": {
	//     "$ref": "Template"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}

// method id "manager.templates.insert":

type TemplatesInsertCall struct {
	s          *Service
	projectId  string
	template   *Template
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert:
func (r *TemplatesService) Insert(projectId string, template *Template) *TemplatesInsertCall {
	c := &TemplatesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.template = template
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TemplatesInsertCall) QuotaUser(quotaUser string) *TemplatesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TemplatesInsertCall) UserIP(userIP string) *TemplatesInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TemplatesInsertCall) Fields(s ...googleapi.Field) *TemplatesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TemplatesInsertCall) Context(ctx context.Context) *TemplatesInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *TemplatesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.template)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/templates")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "manager.templates.insert" call.
// Exactly one of *Template or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Template.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TemplatesInsertCall) Do() (*Template, error) {
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
	ret := &Template{
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
	//   "httpMethod": "POST",
	//   "id": "manager.templates.insert",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/templates",
	//   "request": {
	//     "$ref": "Template"
	//   },
	//   "response": {
	//     "$ref": "Template"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/ndev.cloudman"
	//   ]
	// }

}

// method id "manager.templates.list":

type TemplatesListCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List:
func (r *TemplatesService) List(projectId string) *TemplatesListCall {
	c := &TemplatesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum count of
// results to be returned. Acceptable values are 0 to 100, inclusive.
// (Default: 50)
func (c *TemplatesListCall) MaxResults(maxResults int64) *TemplatesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Specifies a
// nextPageToken returned by a previous list request. This token can be
// used to request the next page of results from a previous list
// request.
func (c *TemplatesListCall) PageToken(pageToken string) *TemplatesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TemplatesListCall) QuotaUser(quotaUser string) *TemplatesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TemplatesListCall) UserIP(userIP string) *TemplatesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TemplatesListCall) Fields(s ...googleapi.Field) *TemplatesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TemplatesListCall) IfNoneMatch(entityTag string) *TemplatesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TemplatesListCall) Context(ctx context.Context) *TemplatesListCall {
	c.ctx_ = ctx
	return c
}

func (c *TemplatesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/templates")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
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

// Do executes the "manager.templates.list" call.
// Exactly one of *TemplatesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TemplatesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TemplatesListCall) Do() (*TemplatesListResponse, error) {
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
	ret := &TemplatesListResponse{
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
	//   "id": "manager.templates.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "50",
	//       "description": "Maximum count of results to be returned. Acceptable values are 0 to 100, inclusive. (Default: 50)",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Specifies a nextPageToken returned by a previous list request. This token can be used to request the next page of results from a previous list request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/templates",
	//   "response": {
	//     "$ref": "TemplatesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-platform.read-only",
	//     "https://www.googleapis.com/auth/ndev.cloudman",
	//     "https://www.googleapis.com/auth/ndev.cloudman.readonly"
	//   ]
	// }

}
