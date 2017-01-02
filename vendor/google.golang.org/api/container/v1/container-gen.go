// Package container provides access to the Google Container Engine API.
//
// See https://cloud.google.com/container-engine/
//
// Usage example:
//
//   import "google.golang.org/api/container/v1"
//   ...
//   containerService, err := container.New(oauthHttpClient)
package container

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

const apiId = "container:v1"
const apiName = "container"
const apiVersion = "v1"
const basePath = "https://container.googleapis.com/"

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
	rs.Zones = NewProjectsZonesService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Zones *ProjectsZonesService
}

func NewProjectsZonesService(s *Service) *ProjectsZonesService {
	rs := &ProjectsZonesService{s: s}
	rs.Clusters = NewProjectsZonesClustersService(s)
	rs.Operations = NewProjectsZonesOperationsService(s)
	return rs
}

type ProjectsZonesService struct {
	s *Service

	Clusters *ProjectsZonesClustersService

	Operations *ProjectsZonesOperationsService
}

func NewProjectsZonesClustersService(s *Service) *ProjectsZonesClustersService {
	rs := &ProjectsZonesClustersService{s: s}
	rs.NodePools = NewProjectsZonesClustersNodePoolsService(s)
	return rs
}

type ProjectsZonesClustersService struct {
	s *Service

	NodePools *ProjectsZonesClustersNodePoolsService
}

func NewProjectsZonesClustersNodePoolsService(s *Service) *ProjectsZonesClustersNodePoolsService {
	rs := &ProjectsZonesClustersNodePoolsService{s: s}
	return rs
}

type ProjectsZonesClustersNodePoolsService struct {
	s *Service
}

func NewProjectsZonesOperationsService(s *Service) *ProjectsZonesOperationsService {
	rs := &ProjectsZonesOperationsService{s: s}
	return rs
}

type ProjectsZonesOperationsService struct {
	s *Service
}

// AddonsConfig: Configuration for the addons that can be automatically
// spun up in the cluster, enabling additional functionality.
type AddonsConfig struct {
	// HorizontalPodAutoscaling: Configuration for the horizontal pod
	// autoscaling feature, which increases or decreases the number of
	// replica pods a replication controller has based on the resource usage
	// of the existing pods.
	HorizontalPodAutoscaling *HorizontalPodAutoscaling `json:"horizontalPodAutoscaling,omitempty"`

	// HttpLoadBalancing: Configuration for the HTTP (L7) load balancing
	// controller addon, which makes it easy to set up HTTP load balancers
	// for services in a cluster.
	HttpLoadBalancing *HttpLoadBalancing `json:"httpLoadBalancing,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "HorizontalPodAutoscaling") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HorizontalPodAutoscaling")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AddonsConfig) MarshalJSON() ([]byte, error) {
	type noMethod AddonsConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Cluster: A Google Container Engine cluster.
type Cluster struct {
	// AddonsConfig: Configurations for the various addons available to run
	// in the cluster.
	AddonsConfig *AddonsConfig `json:"addonsConfig,omitempty"`

	// ClusterIpv4Cidr: The IP address range of the container pods in this
	// cluster, in
	// [CIDR](http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
	// notation (e.g. `10.96.0.0/14`). Leave blank to have one automatically
	// chosen or specify a `/14` block in `10.0.0.0/8`.
	ClusterIpv4Cidr string `json:"clusterIpv4Cidr,omitempty"`

	// CreateTime: [Output only] The time the cluster was created, in
	// [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format.
	CreateTime string `json:"createTime,omitempty"`

	// CurrentMasterVersion: [Output only] The current software version of
	// the master endpoint.
	CurrentMasterVersion string `json:"currentMasterVersion,omitempty"`

	// CurrentNodeCount: [Output only] The number of nodes currently in the
	// cluster.
	CurrentNodeCount int64 `json:"currentNodeCount,omitempty"`

	// CurrentNodeVersion: [Output only] The current version of the node
	// software components. If they are currently at multiple versions
	// because they're in the process of being upgraded, this reflects the
	// minimum version of all nodes.
	CurrentNodeVersion string `json:"currentNodeVersion,omitempty"`

	// Description: An optional description of this cluster.
	Description string `json:"description,omitempty"`

	// Endpoint: [Output only] The IP address of this cluster's master
	// endpoint. The endpoint can be accessed from the internet at
	// `https://username:password@endpoint/`. See the `masterAuth` property
	// of this resource for username and password information.
	Endpoint string `json:"endpoint,omitempty"`

	// InitialClusterVersion: [Output only] The software version of the
	// master endpoint and kubelets used in the cluster when it was first
	// created. The version can be upgraded over time.
	InitialClusterVersion string `json:"initialClusterVersion,omitempty"`

	// InitialNodeCount: The number of nodes to create in this cluster. You
	// must ensure that your Compute Engine resource quota is sufficient for
	// this number of instances. You must also have available firewall and
	// routes quota. For requests, this field should only be used in lieu of
	// a "node_pool" object, since this configuration (along with the
	// "node_config") will be used to create a "NodePool" object with an
	// auto-generated name. Do not use this and a node_pool at the same
	// time.
	InitialNodeCount int64 `json:"initialNodeCount,omitempty"`

	// InstanceGroupUrls: [Output only] The resource URLs of [instance
	// groups](/compute/docs/instance-groups/) associated with this cluster.
	InstanceGroupUrls []string `json:"instanceGroupUrls,omitempty"`

	// Locations: The list of Google Compute Engine
	// [locations](/compute/docs/zones#available) in which the cluster's
	// nodes should be located.
	Locations []string `json:"locations,omitempty"`

	// LoggingService: The logging service the cluster should use to write
	// logs. Currently available options: * `logging.googleapis.com` - the
	// Google Cloud Logging service. * `none` - no logs will be exported
	// from the cluster. * if left as an empty
	// string,`logging.googleapis.com` will be used.
	LoggingService string `json:"loggingService,omitempty"`

	// MasterAuth: The authentication information for accessing the master
	// endpoint.
	MasterAuth *MasterAuth `json:"masterAuth,omitempty"`

	// MonitoringService: The monitoring service the cluster should use to
	// write metrics. Currently available options: *
	// `monitoring.googleapis.com` - the Google Cloud Monitoring service. *
	// `none` - no metrics will be exported from the cluster. * if left as
	// an empty string, `monitoring.googleapis.com` will be used.
	MonitoringService string `json:"monitoringService,omitempty"`

	// Name: The name of this cluster. The name must be unique within this
	// project and zone, and can be up to 40 characters with the following
	// restrictions: * Lowercase letters, numbers, and hyphens only. * Must
	// start with a letter. * Must end with a number or a letter.
	Name string `json:"name,omitempty"`

	// Network: The name of the Google Compute Engine
	// [network](/compute/docs/networks-and-firewalls#networks) to which the
	// cluster is connected. If left unspecified, the `default` network will
	// be used.
	Network string `json:"network,omitempty"`

	// NodeConfig: Parameters used in creating the cluster's nodes. See
	// `nodeConfig` for the description of its properties. For requests,
	// this field should only be used in lieu of a "node_pool" object, since
	// this configuration (along with the "initial_node_count") will be used
	// to create a "NodePool" object with an auto-generated name. Do not use
	// this and a node_pool at the same time. For responses, this field will
	// be populated with the node configuration of the first node pool. If
	// unspecified, the defaults are used.
	NodeConfig *NodeConfig `json:"nodeConfig,omitempty"`

	// NodeIpv4CidrSize: [Output only] The size of the address space on each
	// node for hosting containers. This is provisioned from within the
	// `container_ipv4_cidr` range.
	NodeIpv4CidrSize int64 `json:"nodeIpv4CidrSize,omitempty"`

	// NodePools: The node pools associated with this cluster. When creating
	// a new cluster, only a single node pool should be specified. This
	// field should not be set if "node_config" or "initial_node_count" are
	// specified.
	NodePools []*NodePool `json:"nodePools,omitempty"`

	// SelfLink: [Output only] Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServicesIpv4Cidr: [Output only] The IP address range of the
	// Kubernetes services in this cluster, in
	// [CIDR](http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
	// notation (e.g. `1.2.3.4/29`). Service addresses are typically put in
	// the last `/16` from the container CIDR.
	ServicesIpv4Cidr string `json:"servicesIpv4Cidr,omitempty"`

	// Status: [Output only] The current status of this cluster.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED"
	//   "PROVISIONING"
	//   "RUNNING"
	//   "RECONCILING"
	//   "STOPPING"
	//   "ERROR"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output only] Additional information about the current
	// status of this cluster, if available.
	StatusMessage string `json:"statusMessage,omitempty"`

	// Subnetwork: The name of the Google Compute Engine
	// [subnetwork](/compute/docs/subnetworks) to which the cluster is
	// connected.
	Subnetwork string `json:"subnetwork,omitempty"`

	// Zone: [Output only] The name of the Google Compute Engine
	// [zone](/compute/docs/zones#available) in which the cluster resides.
	Zone string `json:"zone,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AddonsConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AddonsConfig") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Cluster) MarshalJSON() ([]byte, error) {
	type noMethod Cluster
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterUpdate: ClusterUpdate describes an update to the cluster.
// Exactly one update can be applied to a cluster with each request, so
// at most one field can be provided.
type ClusterUpdate struct {
	// DesiredAddonsConfig: Configurations for the various addons available
	// to run in the cluster.
	DesiredAddonsConfig *AddonsConfig `json:"desiredAddonsConfig,omitempty"`

	// DesiredMasterVersion: The Kubernetes version to change the master to.
	// The only valid value is the latest supported version. Use "-" to have
	// the server automatically select the latest version.
	DesiredMasterVersion string `json:"desiredMasterVersion,omitempty"`

	// DesiredMonitoringService: The monitoring service the cluster should
	// use to write metrics. Currently available options: *
	// "monitoring.googleapis.com" - the Google Cloud Monitoring service *
	// "none" - no metrics will be exported from the cluster
	DesiredMonitoringService string `json:"desiredMonitoringService,omitempty"`

	// DesiredNodePoolId: The node pool to be upgraded. This field is
	// mandatory if the "desired_node_version" or "desired_image_family" is
	// specified and there is more than one node pool on the cluster.
	DesiredNodePoolId string `json:"desiredNodePoolId,omitempty"`

	// DesiredNodeVersion: The Kubernetes version to change the nodes to
	// (typically an upgrade). Use `-` to upgrade to the latest version
	// supported by the server.
	DesiredNodeVersion string `json:"desiredNodeVersion,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DesiredAddonsConfig")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DesiredAddonsConfig") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ClusterUpdate) MarshalJSON() ([]byte, error) {
	type noMethod ClusterUpdate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateClusterRequest: CreateClusterRequest creates a cluster.
type CreateClusterRequest struct {
	// Cluster: A [cluster
	// resource](/container-engine/reference/rest/v1/projects.zones.clusters)
	Cluster *Cluster `json:"cluster,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cluster") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cluster") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateClusterRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateClusterRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateNodePoolRequest: CreateNodePoolRequest creates a node pool for
// a cluster.
type CreateNodePoolRequest struct {
	// NodePool: The node pool to create.
	NodePool *NodePool `json:"nodePool,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NodePool") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NodePool") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateNodePoolRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateNodePoolRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HorizontalPodAutoscaling: Configuration options for the horizontal
// pod autoscaling feature, which increases or decreases the number of
// replica pods a replication controller has based on the resource usage
// of the existing pods.
type HorizontalPodAutoscaling struct {
	// Disabled: Whether the Horizontal Pod Autoscaling feature is enabled
	// in the cluster. When enabled, it ensures that a Heapster pod is
	// running in the cluster, which is also used by the Cloud Monitoring
	// service.
	Disabled bool `json:"disabled,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Disabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Disabled") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HorizontalPodAutoscaling) MarshalJSON() ([]byte, error) {
	type noMethod HorizontalPodAutoscaling
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HttpLoadBalancing: Configuration options for the HTTP (L7) load
// balancing controller addon, which makes it easy to set up HTTP load
// balancers for services in a cluster.
type HttpLoadBalancing struct {
	// Disabled: Whether the HTTP Load Balancing controller is enabled in
	// the cluster. When enabled, it runs a small pod in the cluster that
	// manages the load balancers.
	Disabled bool `json:"disabled,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Disabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Disabled") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HttpLoadBalancing) MarshalJSON() ([]byte, error) {
	type noMethod HttpLoadBalancing
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListClustersResponse: ListClustersResponse is the result of
// ListClustersRequest.
type ListClustersResponse struct {
	// Clusters: A list of clusters in the project in the specified zone, or
	// across all ones.
	Clusters []*Cluster `json:"clusters,omitempty"`

	// MissingZones: If any zones are listed here, the list of clusters
	// returned may be missing those zones.
	MissingZones []string `json:"missingZones,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Clusters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Clusters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListClustersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClustersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListNodePoolsResponse: ListNodePoolsResponse is the result of
// ListNodePoolsRequest.
type ListNodePoolsResponse struct {
	// NodePools: A list of node pools for a cluster.
	NodePools []*NodePool `json:"nodePools,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NodePools") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NodePools") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListNodePoolsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListNodePoolsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListOperationsResponse: ListOperationsResponse is the result of
// ListOperationsRequest.
type ListOperationsResponse struct {
	// MissingZones: If any zones are listed here, the list of operations
	// returned may be missing the operations from those zones.
	MissingZones []string `json:"missingZones,omitempty"`

	// Operations: A list of operations in the project in the specified
	// zone.
	Operations []*Operation `json:"operations,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MissingZones") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MissingZones") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MasterAuth: The authentication information for accessing the master
// endpoint. Authentication can be done using HTTP basic auth or using
// client certificates.
type MasterAuth struct {
	// ClientCertificate: [Output only] Base64-encoded public certificate
	// used by clients to authenticate to the cluster endpoint.
	ClientCertificate string `json:"clientCertificate,omitempty"`

	// ClientKey: [Output only] Base64-encoded private key used by clients
	// to authenticate to the cluster endpoint.
	ClientKey string `json:"clientKey,omitempty"`

	// ClusterCaCertificate: [Output only] Base64-encoded public certificate
	// that is the root of trust for the cluster.
	ClusterCaCertificate string `json:"clusterCaCertificate,omitempty"`

	// Password: The password to use for HTTP basic authentication to the
	// master endpoint. Because the master endpoint is open to the Internet,
	// you should create a strong password.
	Password string `json:"password,omitempty"`

	// Username: The username to use for HTTP basic authentication to the
	// master endpoint.
	Username string `json:"username,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClientCertificate")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientCertificate") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MasterAuth) MarshalJSON() ([]byte, error) {
	type noMethod MasterAuth
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NodeConfig: Parameters that describe the nodes in a cluster.
type NodeConfig struct {
	// DiskSizeGb: Size of the disk attached to each node, specified in GB.
	// The smallest allowed disk size is 10GB. If unspecified, the default
	// disk size is 100GB.
	DiskSizeGb int64 `json:"diskSizeGb,omitempty"`

	// MachineType: The name of a Google Compute Engine [machine
	// type](/compute/docs/machine-types) (e.g. `n1-standard-1`). If
	// unspecified, the default machine type is `n1-standard-1`.
	MachineType string `json:"machineType,omitempty"`

	// Metadata: The metadata key/value pairs assigned to instances in the
	// cluster. Keys must conform to the regexp [a-zA-Z0-9-_]+ and be less
	// than 128 bytes in length. These are reflected as part of a URL in the
	// metadata server. Additionally, to avoid ambiguity, keys must not
	// conflict with any other metadata keys for the project or be one of
	// the four reserved keys: "instance-template", "kube-env",
	// "startup-script", and "user-data" Values are free-form strings, and
	// only have meaning as interpreted by the image running in the
	// instance. The only restriction placed on them is that each value's
	// size must be less than or equal to 32 KB. The total size of all keys
	// and values must be less than 512 KB.
	Metadata map[string]string `json:"metadata,omitempty"`

	// OauthScopes: The set of Google API scopes to be made available on all
	// of the node VMs under the "default" service account. The following
	// scopes are recommended, but not required, and by default are not
	// included: * `https://www.googleapis.com/auth/compute` is required for
	// mounting persistent storage on your nodes. *
	// `https://www.googleapis.com/auth/devstorage.read_only` is required
	// for communicating with **gcr.io** (the [Google Container
	// Registry](/container-registry/)). If unspecified, no scopes are
	// added, unless Cloud Logging or Cloud Monitoring are enabled, in which
	// case their required scopes will be added.
	OauthScopes []string `json:"oauthScopes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DiskSizeGb") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DiskSizeGb") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NodeConfig) MarshalJSON() ([]byte, error) {
	type noMethod NodeConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NodePool: NodePool contains the name and configuration for a
// cluster's node pool. Node pools are a set of nodes (i.e. VM's), with
// a common configuration and specification, under the control of the
// cluster master. They may have a set of Kubernetes labels applied to
// them, which may be used to reference them during pod scheduling. They
// may also be resized up or down, to accommodate the workload.
type NodePool struct {
	// Config: The node configuration of the pool.
	Config *NodeConfig `json:"config,omitempty"`

	// InitialNodeCount: The initial node count for the pool. You must
	// ensure that your Compute Engine resource quota is sufficient for this
	// number of instances. You must also have available firewall and routes
	// quota.
	InitialNodeCount int64 `json:"initialNodeCount,omitempty"`

	// InstanceGroupUrls: [Output only] The resource URLs of [instance
	// groups](/compute/docs/instance-groups/) associated with this node
	// pool.
	InstanceGroupUrls []string `json:"instanceGroupUrls,omitempty"`

	// Name: The name of the node pool.
	Name string `json:"name,omitempty"`

	// SelfLink: Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The status of the nodes in this pool instance.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED"
	//   "PROVISIONING"
	//   "RUNNING"
	//   "RUNNING_WITH_ERROR"
	//   "RECONCILING"
	//   "STOPPING"
	//   "ERROR"
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output only] Additional information about the current
	// status of this node pool instance, if available.
	StatusMessage string `json:"statusMessage,omitempty"`

	// Version: The version of the Kubernetes of this node.
	Version string `json:"version,omitempty"`

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

	// NullFields is a list of field names (e.g. "Config") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NodePool) MarshalJSON() ([]byte, error) {
	type noMethod NodePool
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This operation resource represents operations that may
// have happened or are happening on the cluster. All fields are output
// only.
type Operation struct {
	// Detail: Detailed operation progress, if available.
	Detail string `json:"detail,omitempty"`

	// Name: The server-assigned ID for the operation.
	Name string `json:"name,omitempty"`

	// OperationType: The operation type.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "CREATE_CLUSTER"
	//   "DELETE_CLUSTER"
	//   "UPGRADE_MASTER"
	//   "UPGRADE_NODES"
	//   "REPAIR_CLUSTER"
	//   "UPDATE_CLUSTER"
	//   "CREATE_NODE_POOL"
	//   "DELETE_NODE_POOL"
	OperationType string `json:"operationType,omitempty"`

	// SelfLink: Server-defined URL for the resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The current status of the operation.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
	Status string `json:"status,omitempty"`

	// StatusMessage: If an error has occurred, a textual description of the
	// error.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetLink: Server-defined URL for the target of the operation.
	TargetLink string `json:"targetLink,omitempty"`

	// Zone: The name of the Google Compute Engine
	// [zone](/compute/docs/zones#available) in which the operation is
	// taking place.
	Zone string `json:"zone,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Detail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Detail") to include in API
	// requests with the JSON null value. By default, fields with empty
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

// ServerConfig: Container Engine service configuration.
type ServerConfig struct {
	// DefaultClusterVersion: Version of Kubernetes the service deploys by
	// default.
	DefaultClusterVersion string `json:"defaultClusterVersion,omitempty"`

	// DefaultImageFamily: Default image family.
	DefaultImageFamily string `json:"defaultImageFamily,omitempty"`

	// ValidImageFamilies: List of valid image families.
	ValidImageFamilies []string `json:"validImageFamilies,omitempty"`

	// ValidNodeVersions: List of valid node upgrade target versions.
	ValidNodeVersions []string `json:"validNodeVersions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "DefaultClusterVersion") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultClusterVersion") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ServerConfig) MarshalJSON() ([]byte, error) {
	type noMethod ServerConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateClusterRequest: UpdateClusterRequest updates the settings of a
// cluster.
type UpdateClusterRequest struct {
	// Update: A description of the update.
	Update *ClusterUpdate `json:"update,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Update") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Update") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateClusterRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateClusterRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "container.projects.zones.getServerconfig":

type ProjectsZonesGetServerconfigCall struct {
	s            *Service
	projectId    string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetServerconfig: Returns configuration info about the Container
// Engine service.
func (r *ProjectsZonesService) GetServerconfig(projectId string, zone string) *ProjectsZonesGetServerconfigCall {
	c := &ProjectsZonesGetServerconfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesGetServerconfigCall) Fields(s ...googleapi.Field) *ProjectsZonesGetServerconfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesGetServerconfigCall) IfNoneMatch(entityTag string) *ProjectsZonesGetServerconfigCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesGetServerconfigCall) Context(ctx context.Context) *ProjectsZonesGetServerconfigCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesGetServerconfigCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesGetServerconfigCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/serverconfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.getServerconfig" call.
// Exactly one of *ServerConfig or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ServerConfig.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesGetServerconfigCall) Do(opts ...googleapi.CallOption) (*ServerConfig, error) {
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
	ret := &ServerConfig{
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
	//   "description": "Returns configuration info about the Container Engine service.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.getServerconfig",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) to return operations for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/serverconfig",
	//   "response": {
	//     "$ref": "ServerConfig"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.create":

type ProjectsZonesClustersCreateCall struct {
	s                    *Service
	projectId            string
	zone                 string
	createclusterrequest *CreateClusterRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Create: Creates a cluster, consisting of the specified number and
// type of Google Compute Engine instances. By default, the cluster is
// created in the project's [default
// network](/compute/docs/networks-and-firewalls#networks). One firewall
// is added for the cluster. After cluster creation, the cluster creates
// routes for each node to allow the containers on that node to
// communicate with all other instances in the cluster. Finally, an
// entry is added to the project's global metadata indicating which CIDR
// range is being used by the cluster.
func (r *ProjectsZonesClustersService) Create(projectId string, zone string, createclusterrequest *CreateClusterRequest) *ProjectsZonesClustersCreateCall {
	c := &ProjectsZonesClustersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.createclusterrequest = createclusterrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersCreateCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersCreateCall) Context(ctx context.Context) *ProjectsZonesClustersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createclusterrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a cluster, consisting of the specified number and type of Google Compute Engine instances. By default, the cluster is created in the project's [default network](/compute/docs/networks-and-firewalls#networks). One firewall is added for the cluster. After cluster creation, the cluster creates routes for each node to allow the containers on that node to communicate with all other instances in the cluster. Finally, an entry is added to the project's global metadata indicating which CIDR range is being used by the cluster.",
	//   "httpMethod": "POST",
	//   "id": "container.projects.zones.clusters.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters",
	//   "request": {
	//     "$ref": "CreateClusterRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.delete":

type ProjectsZonesClustersDeleteCall struct {
	s          *Service
	projectId  string
	zone       string
	clusterId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the cluster, including the Kubernetes endpoint and
// all worker nodes. Firewalls and routes that were configured during
// cluster creation are also deleted. Other Google Compute Engine
// resources that might be in use by the cluster (e.g. load balancer
// resources) will not be deleted if they weren't present at the initial
// create time.
func (r *ProjectsZonesClustersService) Delete(projectId string, zone string, clusterId string) *ProjectsZonesClustersDeleteCall {
	c := &ProjectsZonesClustersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersDeleteCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersDeleteCall) Context(ctx context.Context) *ProjectsZonesClustersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Deletes the cluster, including the Kubernetes endpoint and all worker nodes. Firewalls and routes that were configured during cluster creation are also deleted. Other Google Compute Engine resources that might be in use by the cluster (e.g. load balancer resources) will not be deleted if they weren't present at the initial create time.",
	//   "httpMethod": "DELETE",
	//   "id": "container.projects.zones.clusters.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.get":

type ProjectsZonesClustersGetCall struct {
	s            *Service
	projectId    string
	zone         string
	clusterId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the details of a specific cluster.
func (r *ProjectsZonesClustersService) Get(projectId string, zone string, clusterId string) *ProjectsZonesClustersGetCall {
	c := &ProjectsZonesClustersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersGetCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesClustersGetCall) IfNoneMatch(entityTag string) *ProjectsZonesClustersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersGetCall) Context(ctx context.Context) *ProjectsZonesClustersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.get" call.
// Exactly one of *Cluster or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Cluster.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsZonesClustersGetCall) Do(opts ...googleapi.CallOption) (*Cluster, error) {
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
	ret := &Cluster{
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
	//   "description": "Gets the details of a specific cluster.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.clusters.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}",
	//   "response": {
	//     "$ref": "Cluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.list":

type ProjectsZonesClustersListCall struct {
	s            *Service
	projectId    string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all clusters owned by a project in either the specified
// zone or all zones.
func (r *ProjectsZonesClustersService) List(projectId string, zone string) *ProjectsZonesClustersListCall {
	c := &ProjectsZonesClustersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersListCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesClustersListCall) IfNoneMatch(entityTag string) *ProjectsZonesClustersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersListCall) Context(ctx context.Context) *ProjectsZonesClustersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.list" call.
// Exactly one of *ListClustersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListClustersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsZonesClustersListCall) Do(opts ...googleapi.CallOption) (*ListClustersResponse, error) {
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
	ret := &ListClustersResponse{
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
	//   "description": "Lists all clusters owned by a project in either the specified zone or all zones.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.clusters.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides, or \"-\" for all zones.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters",
	//   "response": {
	//     "$ref": "ListClustersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.update":

type ProjectsZonesClustersUpdateCall struct {
	s                    *Service
	projectId            string
	zone                 string
	clusterId            string
	updateclusterrequest *UpdateClusterRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Update: Updates the settings of a specific cluster.
func (r *ProjectsZonesClustersService) Update(projectId string, zone string, clusterId string, updateclusterrequest *UpdateClusterRequest) *ProjectsZonesClustersUpdateCall {
	c := &ProjectsZonesClustersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	c.updateclusterrequest = updateclusterrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersUpdateCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersUpdateCall) Context(ctx context.Context) *ProjectsZonesClustersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updateclusterrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.update" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersUpdateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Updates the settings of a specific cluster.",
	//   "httpMethod": "PUT",
	//   "id": "container.projects.zones.clusters.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster to upgrade.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}",
	//   "request": {
	//     "$ref": "UpdateClusterRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.nodePools.create":

type ProjectsZonesClustersNodePoolsCreateCall struct {
	s                     *Service
	projectId             string
	zone                  string
	clusterId             string
	createnodepoolrequest *CreateNodePoolRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Create: Creates a node pool for a cluster.
func (r *ProjectsZonesClustersNodePoolsService) Create(projectId string, zone string, clusterId string, createnodepoolrequest *CreateNodePoolRequest) *ProjectsZonesClustersNodePoolsCreateCall {
	c := &ProjectsZonesClustersNodePoolsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	c.createnodepoolrequest = createnodepoolrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersNodePoolsCreateCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersNodePoolsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersNodePoolsCreateCall) Context(ctx context.Context) *ProjectsZonesClustersNodePoolsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersNodePoolsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersNodePoolsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createnodepoolrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.nodePools.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersNodePoolsCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Creates a node pool for a cluster.",
	//   "httpMethod": "POST",
	//   "id": "container.projects.zones.clusters.nodePools.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://developers.google.com/console/help/new/#projectnumber).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools",
	//   "request": {
	//     "$ref": "CreateNodePoolRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.nodePools.delete":

type ProjectsZonesClustersNodePoolsDeleteCall struct {
	s          *Service
	projectId  string
	zone       string
	clusterId  string
	nodePoolId string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a node pool from a cluster.
func (r *ProjectsZonesClustersNodePoolsService) Delete(projectId string, zone string, clusterId string, nodePoolId string) *ProjectsZonesClustersNodePoolsDeleteCall {
	c := &ProjectsZonesClustersNodePoolsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	c.nodePoolId = nodePoolId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersNodePoolsDeleteCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersNodePoolsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersNodePoolsDeleteCall) Context(ctx context.Context) *ProjectsZonesClustersNodePoolsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersNodePoolsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersNodePoolsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":  c.projectId,
		"zone":       c.zone,
		"clusterId":  c.clusterId,
		"nodePoolId": c.nodePoolId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.nodePools.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersNodePoolsDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Deletes a node pool from a cluster.",
	//   "httpMethod": "DELETE",
	//   "id": "container.projects.zones.clusters.nodePools.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId",
	//     "nodePoolId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "nodePoolId": {
	//       "description": "The name of the node pool to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://developers.google.com/console/help/new/#projectnumber).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.nodePools.get":

type ProjectsZonesClustersNodePoolsGetCall struct {
	s            *Service
	projectId    string
	zone         string
	clusterId    string
	nodePoolId   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves the node pool requested.
func (r *ProjectsZonesClustersNodePoolsService) Get(projectId string, zone string, clusterId string, nodePoolId string) *ProjectsZonesClustersNodePoolsGetCall {
	c := &ProjectsZonesClustersNodePoolsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	c.nodePoolId = nodePoolId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersNodePoolsGetCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersNodePoolsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesClustersNodePoolsGetCall) IfNoneMatch(entityTag string) *ProjectsZonesClustersNodePoolsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersNodePoolsGetCall) Context(ctx context.Context) *ProjectsZonesClustersNodePoolsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersNodePoolsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersNodePoolsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":  c.projectId,
		"zone":       c.zone,
		"clusterId":  c.clusterId,
		"nodePoolId": c.nodePoolId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.nodePools.get" call.
// Exactly one of *NodePool or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *NodePool.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesClustersNodePoolsGetCall) Do(opts ...googleapi.CallOption) (*NodePool, error) {
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
	ret := &NodePool{
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
	//   "description": "Retrieves the node pool requested.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.clusters.nodePools.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId",
	//     "nodePoolId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "nodePoolId": {
	//       "description": "The name of the node pool.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://developers.google.com/console/help/new/#projectnumber).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}",
	//   "response": {
	//     "$ref": "NodePool"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.clusters.nodePools.list":

type ProjectsZonesClustersNodePoolsListCall struct {
	s            *Service
	projectId    string
	zone         string
	clusterId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the node pools for a cluster.
func (r *ProjectsZonesClustersNodePoolsService) List(projectId string, zone string, clusterId string) *ProjectsZonesClustersNodePoolsListCall {
	c := &ProjectsZonesClustersNodePoolsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersNodePoolsListCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersNodePoolsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesClustersNodePoolsListCall) IfNoneMatch(entityTag string) *ProjectsZonesClustersNodePoolsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesClustersNodePoolsListCall) Context(ctx context.Context) *ProjectsZonesClustersNodePoolsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesClustersNodePoolsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesClustersNodePoolsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.clusters.nodePools.list" call.
// Exactly one of *ListNodePoolsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListNodePoolsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsZonesClustersNodePoolsListCall) Do(opts ...googleapi.CallOption) (*ListNodePoolsResponse, error) {
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
	ret := &ListNodePoolsResponse{
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
	//   "description": "Lists the node pools for a cluster.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.clusters.nodePools.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "clusterId"
	//   ],
	//   "parameters": {
	//     "clusterId": {
	//       "description": "The name of the cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://developers.google.com/console/help/new/#projectnumber).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools",
	//   "response": {
	//     "$ref": "ListNodePoolsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.operations.get":

type ProjectsZonesOperationsGetCall struct {
	s            *Service
	projectId    string
	zone         string
	operationId  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the specified operation.
func (r *ProjectsZonesOperationsService) Get(projectId string, zone string, operationId string) *ProjectsZonesOperationsGetCall {
	c := &ProjectsZonesOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	c.operationId = operationId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesOperationsGetCall) Fields(s ...googleapi.Field) *ProjectsZonesOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesOperationsGetCall) IfNoneMatch(entityTag string) *ProjectsZonesOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesOperationsGetCall) Context(ctx context.Context) *ProjectsZonesOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/operations/{operationId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"zone":        c.zone,
		"operationId": c.operationId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsZonesOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Gets the specified operation.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.operations.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone",
	//     "operationId"
	//   ],
	//   "parameters": {
	//     "operationId": {
	//       "description": "The server-assigned `name` of the operation.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/operations/{operationId}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.zones.operations.list":

type ProjectsZonesOperationsListCall struct {
	s            *Service
	projectId    string
	zone         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all operations in a project in a specific zone or all
// zones.
func (r *ProjectsZonesOperationsService) List(projectId string, zone string) *ProjectsZonesOperationsListCall {
	c := &ProjectsZonesOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesOperationsListCall) Fields(s ...googleapi.Field) *ProjectsZonesOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsZonesOperationsListCall) IfNoneMatch(entityTag string) *ProjectsZonesOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsZonesOperationsListCall) Context(ctx context.Context) *ProjectsZonesOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsZonesOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsZonesOperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "container.projects.zones.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsZonesOperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	ret := &ListOperationsResponse{
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
	//   "description": "Lists all operations in a project in a specific zone or all zones.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.operations.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID or project number](https://support.google.com/cloud/answer/6158840).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) to return operations for, or `-` for all zones.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/projects/{projectId}/zones/{zone}/operations",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
