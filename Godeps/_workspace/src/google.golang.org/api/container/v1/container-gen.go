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
	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
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
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Background

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
	return rs
}

type ProjectsZonesClustersService struct {
	s *Service
}

func NewProjectsZonesOperationsService(s *Service) *ProjectsZonesOperationsService {
	rs := &ProjectsZonesOperationsService{s: s}
	return rs
}

type ProjectsZonesOperationsService struct {
	s *Service
}

// Cluster: A Google Container Engine cluster.
type Cluster struct {
	// ClusterIpv4Cidr: The IP address range of the container pods in this
	// cluster, in
	// [CIDR](http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
	// notation (e.g. `10.96.0.0/14`). Leave blank to have one automatically
	// chosen or specify a `/14` block in `10.0.0.0/8` or `172.16.0.0/12`.
	ClusterIpv4Cidr string `json:"clusterIpv4Cidr,omitempty"`

	// CreateTime: [Output only] The time the cluster was created, in
	// [RFC3339](href='https://www.ietf.org/rfc/rfc3339.txt) text format.
	// @OutputOnly.
	CreateTime string `json:"createTime,omitempty"`

	// CurrentMasterVersion: [Output only] The current software version of
	// the master endpoint. @OutputOnly.
	CurrentMasterVersion string `json:"currentMasterVersion,omitempty"`

	// CurrentNodeVersion: [Output only] The current version of the node
	// software components. If they are currently at different versions
	// because they're in the process of being upgraded, this reflects the
	// minimum version of any of them. @OutputOnly.
	CurrentNodeVersion string `json:"currentNodeVersion,omitempty"`

	// Description: An optional description of this cluster.
	Description string `json:"description,omitempty"`

	// Endpoint: [Output only] The IP address of this cluster's Kubernetes
	// master endpoint. The endpoint can be accessed from the internet at
	// `https://username:password@endpoint/`. @OutputOnly. See the
	// `masterAuth` property of this resource for username and password
	// information.
	Endpoint string `json:"endpoint,omitempty"`

	// InitialClusterVersion: [Output only] The software version of
	// Kubernetes master and kubelets used in the cluster when it was first
	// created. The version can be upgraded over time. @OutputOnly.
	InitialClusterVersion string `json:"initialClusterVersion,omitempty"`

	// InitialNodeCount: The number of nodes to create in this cluster. You
	// must ensure that your Compute Engine [resource
	// quota](/compute/docs/resource-quotas) is sufficient for this number
	// of instances. You must also have available firewall and routes quota.
	InitialNodeCount int64 `json:"initialNodeCount,omitempty"`

	// InstanceGroupUrls: [Output only] The resource URLs of [instance
	// groups](/compute/docs/instance-groups/) associated with this cluster.
	// @OutputOnly.
	InstanceGroupUrls []string `json:"instanceGroupUrls,omitempty"`

	// LoggingService: The logging service that the cluster should write
	// logs to. Currently available options: * "logging.googleapis.com" -
	// the Google Cloud Logging service * "none" - no logs will be exported
	// from the cluster * "" - default value; the default is
	// "logging.googleapis.com"
	LoggingService string `json:"loggingService,omitempty"`

	// MasterAuth: The authentication information for accessing the master.
	MasterAuth *MasterAuth `json:"masterAuth,omitempty"`

	// MonitoringService: The monitoring service that the cluster should
	// write metrics to. Currently available options: *
	// "monitoring.googleapis.com" - the Google Cloud Monitoring service *
	// "none" - no metrics will be exported from the cluster * "" - default
	// value; the default is "monitoring.googleapis.com"
	MonitoringService string `json:"monitoringService,omitempty"`

	// Name: The name of this cluster. The name must be unique within this
	// project and zone, and can be up to 40 characters with the following
	// restrictions: * Lowercase letters, numbers, and hyphens only. * Must
	// start with a letter. * Must end with a number or a letter.
	Name string `json:"name,omitempty"`

	// Network: The name of the Google Compute Engine
	// [network](/compute/docs/networking#networks_1) to which the cluster
	// is connected. If left unspecified, the "default" network will be
	// used.
	Network string `json:"network,omitempty"`

	// NodeConfig: Parameters used in creating the cluster's nodes. See the
	// descriptions of the child properties of `nodeConfig`.
	NodeConfig *NodeConfig `json:"nodeConfig,omitempty"`

	// NodeIpv4CidrSize: [Output only] The size of the address space on each
	// node for hosting containers. This is provisioned from within the
	// container_ipv4_cidr range. @OutputOnly.
	NodeIpv4CidrSize int64 `json:"nodeIpv4CidrSize,omitempty"`

	// SelfLink: [Output only] Server-defined URL for the resource.
	// @OutputOnly.
	SelfLink string `json:"selfLink,omitempty"`

	// ServicesIpv4Cidr: [Output only] The IP address range of the
	// Kubernetes services in this cluster, in
	// [CIDR](http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
	// notation (e.g. `1.2.3.4/29`). Service addresses are typically put in
	// the last /16 from the container CIDR. @OutputOnly.
	ServicesIpv4Cidr string `json:"servicesIpv4Cidr,omitempty"`

	// Status: [Output only] The current status of this cluster.
	// @OutputOnly.
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
	// status of this cluster, if available. @OutputOnly.
	StatusMessage string `json:"statusMessage,omitempty"`

	// Zone: [Output only] The name of the Google Compute Engine
	// [zone](/compute/docs/zones#available) in which the cluster resides.
	// @OutputOnly.
	Zone string `json:"zone,omitempty"`
}

// ClusterUpdate: ClusterUpdate describes an update to the cluster.
type ClusterUpdate struct {
	// DesiredNodeVersion: The Kubernetes version to change the nodes to
	// (typically an upgrade). Use "-" to upgrade to the latest version
	// supported by the server.
	DesiredNodeVersion string `json:"desiredNodeVersion,omitempty"`
}

// CreateClusterRequest: CreateClusterRequest creates a cluster.
type CreateClusterRequest struct {
	// Cluster: A [cluster
	// resource](/container-engine/docs/v1/projects/zones/clusters)
	Cluster *Cluster `json:"cluster,omitempty"`
}

// ListClustersResponse: ListClustersResponse is the result of
// ListClustersRequest.
type ListClustersResponse struct {
	// Clusters: A list of clusters in the project in the specified zone, or
	// across all ones.
	Clusters []*Cluster `json:"clusters,omitempty"`
}

// ListOperationsResponse: ListOperationsResponse is the result of
// ListOperationsRequest.
type ListOperationsResponse struct {
	// Operations: A list of operations in the project in the specified
	// zone.
	Operations []*Operation `json:"operations,omitempty"`
}

// MasterAuth: The authentication information for accessing the master
// endpoint. Authentication can be done using HTTP basic auth or using
// client certificates.
type MasterAuth struct {
	// ClientCertificate: [Output only] Base64 encoded public certificate
	// used by clients to authenticate to the cluster endpoint. @OutputOnly.
	ClientCertificate string `json:"clientCertificate,omitempty"`

	// ClientKey: [Output only] Base64 encoded private key used by clients
	// to authenticate to the cluster endpoint. @OutputOnly.
	ClientKey string `json:"clientKey,omitempty"`

	// ClusterCaCertificate: [Output only] Base64 encoded public certificate
	// that is the root of trust for the cluster. @OutputOnly.
	ClusterCaCertificate string `json:"clusterCaCertificate,omitempty"`

	// Password: The password to use for HTTP basic authentication when
	// accessing the Kubernetes master endpoint. Because the master endpoint
	// is open to the internet, you should create a strong password.
	Password string `json:"password,omitempty"`

	// Username: The username to use for HTTP basic authentication when
	// accessing the Kubernetes master endpoint.
	Username string `json:"username,omitempty"`
}

// NodeConfig: Per-node parameters.
type NodeConfig struct {
	// DiskSizeGb: Size of the disk attached to each node, specified in GB.
	// The smallest allowed disk size is 10GB, and the default is 100GB.
	DiskSizeGb int64 `json:"diskSizeGb,omitempty"`

	// MachineType: The name of a Google Compute Engine [machine
	// type](/compute/docs/machine-types) (e.g. `n1-standard-1`). If
	// unspecified, the default machine type is `n1-standard-1`.
	MachineType string `json:"machineType,omitempty"`

	// OauthScopes: The set of Google API scopes to be made available on all
	// of the node VMs under the "default" service account. Currently, the
	// following scopes are necessary to ensure the correct functioning of
	// the cluster: * "https://www.googleapis.com/auth/compute" *
	// "https://www.googleapis.com/auth/devstorage.read_only"
	OauthScopes []string `json:"oauthScopes,omitempty"`
}

// Operation: Defines the operation resource. All fields are output
// only.
type Operation struct {
	// Name: The server-assigned ID for the operation. @OutputOnly.
	Name string `json:"name,omitempty"`

	// OperationType: The operation type. @OutputOnly.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "CREATE_CLUSTER"
	//   "DELETE_CLUSTER"
	//   "UPGRADE_MASTER"
	//   "UPGRADE_NODES"
	//   "REPAIR_CLUSTER"
	OperationType string `json:"operationType,omitempty"`

	// SelfLink: Server-defined URL for the resource. @OutputOnly.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The current status of the operation. @OutputOnly.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
	Status string `json:"status,omitempty"`

	// StatusMessage: If an error has occurred, a textual description of the
	// error. @OutputOnly.
	StatusMessage string `json:"statusMessage,omitempty"`

	// TargetLink: Server-defined URL for the target of the operation.
	// @OutputOnly.
	TargetLink string `json:"targetLink,omitempty"`

	// Zone: The name of the Google Compute Engine
	// [zone](/compute/docs/zones#available) in which the operation is
	// taking place. @OutputOnly.
	Zone string `json:"zone,omitempty"`
}

// UpdateClusterRequest: UpdateClusterRequest updates a cluster.
type UpdateClusterRequest struct {
	// Update: A description of the update.
	Update *ClusterUpdate `json:"update,omitempty"`
}

// method id "container.projects.zones.clusters.create":

type ProjectsZonesClustersCreateCall struct {
	s                    *Service
	projectId            string
	zone                 string
	createclusterrequest *CreateClusterRequest
	opt_                 map[string]interface{}
}

// Create: Creates a cluster, consisting of the specified number and
// type of Google Compute Engine instances, plus a Kubernetes master
// endpoint. By default, the cluster is created in the project's
// [default network]('/compute/docs/networking#networks_1'). One
// firewall is added for the cluster. After cluster creation, the
// cluster creates routes for each node to allow the containers on that
// node to communicate with all other instances in the cluster. Finally,
// an entry is added to the project's global metadata indicating which
// CIDR range is being used by the cluster.
func (r *ProjectsZonesClustersService) Create(projectId string, zone string, createclusterrequest *CreateClusterRequest) *ProjectsZonesClustersCreateCall {
	c := &ProjectsZonesClustersCreateCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	c.createclusterrequest = createclusterrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersCreateCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersCreateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesClustersCreateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createclusterrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a cluster, consisting of the specified number and type of Google Compute Engine instances, plus a Kubernetes master endpoint. By default, the cluster is created in the project's [default network]('/compute/docs/networking#networks_1'). One firewall is added for the cluster. After cluster creation, the cluster creates routes for each node to allow the containers on that node to communicate with all other instances in the cluster. Finally, an entry is added to the project's global metadata indicating which CIDR range is being used by the cluster.",
	//   "httpMethod": "POST",
	//   "id": "container.projects.zones.clusters.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "zone"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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
	s         *Service
	projectId string
	zone      string
	clusterId string
	opt_      map[string]interface{}
}

// Delete: Deletes the cluster, including the Kubernetes endpoint and
// all worker nodes. Firewalls and routes that were configured during
// cluster creation are also deleted.
func (r *ProjectsZonesClustersService) Delete(projectId string, zone string, clusterId string) *ProjectsZonesClustersDeleteCall {
	c := &ProjectsZonesClustersDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersDeleteCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersDeleteCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesClustersDeleteCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the cluster, including the Kubernetes endpoint and all worker nodes. Firewalls and routes that were configured during cluster creation are also deleted.",
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
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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
	s         *Service
	projectId string
	zone      string
	clusterId string
	opt_      map[string]interface{}
}

// Get: Gets a specific cluster.
func (r *ProjectsZonesClustersService) Get(projectId string, zone string, clusterId string) *ProjectsZonesClustersGetCall {
	c := &ProjectsZonesClustersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersGetCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesClustersGetCall) Do() (*Cluster, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Cluster
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a specific cluster.",
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
	//       "description": "The Google Developers Console A [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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
	s         *Service
	projectId string
	zone      string
	opt_      map[string]interface{}
}

// List: Lists all clusters owned by a project in either the specified
// zone or all zones.
func (r *ProjectsZonesClustersService) List(projectId string, zone string) *ProjectsZonesClustersListCall {
	c := &ProjectsZonesClustersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersListCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesClustersListCall) Do() (*ListClustersResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ListClustersResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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
	opt_                 map[string]interface{}
}

// Update: Update settings of a specific cluster.
func (r *ProjectsZonesClustersService) Update(projectId string, zone string, clusterId string, updateclusterrequest *UpdateClusterRequest) *ProjectsZonesClustersUpdateCall {
	c := &ProjectsZonesClustersUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	c.clusterId = clusterId
	c.updateclusterrequest = updateclusterrequest
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesClustersUpdateCall) Fields(s ...googleapi.Field) *ProjectsZonesClustersUpdateCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesClustersUpdateCall) Do() (*Operation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updateclusterrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
		"clusterId": c.clusterId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update settings of a specific cluster.",
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
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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

// method id "container.projects.zones.operations.get":

type ProjectsZonesOperationsGetCall struct {
	s           *Service
	projectId   string
	zone        string
	operationId string
	opt_        map[string]interface{}
}

// Get: Gets the specified operation.
func (r *ProjectsZonesOperationsService) Get(projectId string, zone string, operationId string) *ProjectsZonesOperationsGetCall {
	c := &ProjectsZonesOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	c.operationId = operationId
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesOperationsGetCall) Fields(s ...googleapi.Field) *ProjectsZonesOperationsGetCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesOperationsGetCall) Do() (*Operation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/operations/{operationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"zone":        c.zone,
		"operationId": c.operationId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *Operation
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
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
	s         *Service
	projectId string
	zone      string
	opt_      map[string]interface{}
}

// List: Lists all operations in a project in a specific zone or all
// zones.
func (r *ProjectsZonesOperationsService) List(projectId string, zone string) *ProjectsZonesOperationsListCall {
	c := &ProjectsZonesOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zone = zone
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsZonesOperationsListCall) Fields(s ...googleapi.Field) *ProjectsZonesOperationsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsZonesOperationsListCall) Do() (*ListOperationsResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/projects/{projectId}/zones/{zone}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zone":      c.zone,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ListOperationsResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
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
	//       "description": "The Google Developers Console [project ID](https://console.developers.google.com/project) or [project number](https://developers.google.com/console/help/project-number)",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zone": {
	//       "description": "The name of the Google Compute Engine [zone](/compute/docs/zones#available) to return operations for, or \"-\" for all zones.",
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
