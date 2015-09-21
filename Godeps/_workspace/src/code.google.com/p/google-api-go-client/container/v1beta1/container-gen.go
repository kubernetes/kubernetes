// Package container provides access to the Google Container Engine API.
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/container/v1beta1"
//   ...
//   containerService, err := container.New(oauthHttpClient)
package container

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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

const apiId = "container:v1beta1"
const apiName = "container"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/container/v1beta1/projects/"

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Projects *ProjectsService
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Clusters = NewProjectsClustersService(s)
	rs.Operations = NewProjectsOperationsService(s)
	rs.Zones = NewProjectsZonesService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Clusters *ProjectsClustersService

	Operations *ProjectsOperationsService

	Zones *ProjectsZonesService
}

func NewProjectsClustersService(s *Service) *ProjectsClustersService {
	rs := &ProjectsClustersService{s: s}
	return rs
}

type ProjectsClustersService struct {
	s *Service
}

func NewProjectsOperationsService(s *Service) *ProjectsOperationsService {
	rs := &ProjectsOperationsService{s: s}
	return rs
}

type ProjectsOperationsService struct {
	s *Service
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

type Cluster struct {
	// ClusterApiVersion: The API version of the Kubernetes master and
	// kubelets running in this cluster. Allowed value is 0.4.2, or leave
	// blank to pick up the latest stable release.
	ClusterApiVersion string `json:"clusterApiVersion,omitempty"`

	// ContainerIpv4Cidr: [Output only] The IP addresses of the container
	// pods in this cluster, in  CIDR notation (e.g. 1.2.3.4/29).
	ContainerIpv4Cidr string `json:"containerIpv4Cidr,omitempty"`

	// CreationTimestamp: [Output only] The time the cluster was created, in
	// RFC3339 text format.
	CreationTimestamp string `json:"creationTimestamp,omitempty"`

	// Description: An optional description of this cluster.
	Description string `json:"description,omitempty"`

	// Endpoint: [Output only] The IP address of this cluster's Kubernetes
	// master. The endpoint can be accessed from the internet at
	// https://username:password@endpoint/.
	//
	// See the masterAuth property of
	// this resource for username and password information.
	Endpoint string `json:"endpoint,omitempty"`

	// MasterAuth: The HTTP basic authentication information for accessing
	// the master. Because the master endpoint is open to the internet, you
	// should create a strong password.
	MasterAuth *MasterAuth `json:"masterAuth,omitempty"`

	// Name: The name of this cluster. The name must be unique within this
	// project and zone, and can be up to 40 characters with the following
	// restrictions:
	// - Lowercase letters, numbers, and hyphens only.
	// -
	// Must start with a letter.
	// - Must end with a number or a letter.
	Name string `json:"name,omitempty"`

	// NodeConfig: The machine type and image to use for all nodes in this
	// cluster. See the descriptions of the child properties of nodeConfig.
	NodeConfig *NodeConfig `json:"nodeConfig,omitempty"`

	// NodeRoutingPrefixSize: [Output only] The size of the address space on
	// each node for hosting containers.
	NodeRoutingPrefixSize int64 `json:"nodeRoutingPrefixSize,omitempty"`

	// NumNodes: The number of nodes to create in this cluster. You must
	// ensure that your Compute Engine resource quota is sufficient for this
	// number of instances plus one (to include the master). You must also
	// have available firewall and routes quota.
	NumNodes int64 `json:"numNodes,omitempty"`

	// ServicesIpv4Cidr: [Output only] The IP addresses of the Kubernetes
	// services in this cluster, in  CIDR notation (e.g. 1.2.3.4/29).
	// Service addresses are always in the 10.0.0.0/16 range.
	ServicesIpv4Cidr string `json:"servicesIpv4Cidr,omitempty"`

	// Status: [Output only] The current status of this cluster.
	Status string `json:"status,omitempty"`

	// StatusMessage: [Output only] Additional information about the current
	// status of this cluster, if available.
	StatusMessage string `json:"statusMessage,omitempty"`

	// Zone: [Output only] The name of the Google Compute Engine zone in
	// which the cluster resides.
	Zone string `json:"zone,omitempty"`
}

type CreateClusterRequest struct {
	// Cluster: A cluster resource.
	Cluster *Cluster `json:"cluster,omitempty"`
}

type ListAggregatedClustersResponse struct {
	// Clusters: A list of clusters in the project, across all zones.
	Clusters []*Cluster `json:"clusters,omitempty"`
}

type ListAggregatedOperationsResponse struct {
	// Operations: A list of operations in the project, across all zones.
	Operations []*Operation `json:"operations,omitempty"`
}

type ListClustersResponse struct {
	// Clusters: A list of clusters in the project in the specified zone.
	Clusters []*Cluster `json:"clusters,omitempty"`
}

type ListOperationsResponse struct {
	// Operations: A list of operations in the project in the specified
	// zone.
	Operations []*Operation `json:"operations,omitempty"`
}

type MasterAuth struct {
	// Password: The password to use when accessing the Kubernetes master
	// endpoint.
	Password string `json:"password,omitempty"`

	// User: The username to use when accessing the Kubernetes master
	// endpoint.
	User string `json:"user,omitempty"`
}

type NodeConfig struct {
	// MachineType: The name of a Google Compute Engine machine type (e.g.
	// n1-standard-1).
	//
	// If unspecified, the default machine type is
	// n1-standard-1.
	MachineType string `json:"machineType,omitempty"`

	// SourceImage: The fully-specified name of a Google Compute Engine
	// image. For example:
	// https://www.googleapis.com/compute/v1/projects/debian-cloud/global/ima
	// ges/backports-debian-7-wheezy-vYYYYMMDD (where YYYMMDD is the version
	// date).
	//
	// If specifying an image, you are responsible for ensuring its
	// compatibility with the Debian 7 backports image. We recommend leaving
	// this field blank to accept the default backports-debian-7-wheezy
	// value.
	SourceImage string `json:"sourceImage,omitempty"`
}

type Operation struct {
	// ErrorMessage: If an error has occurred, a textual description of the
	// error.
	ErrorMessage string `json:"errorMessage,omitempty"`

	// Name: The server-assigned ID for this operation. If the operation is
	// fulfilled upfront, it may not have a resource name.
	Name string `json:"name,omitempty"`

	// OperationType: The operation type.
	OperationType string `json:"operationType,omitempty"`

	// Status: The current status of the operation.
	Status string `json:"status,omitempty"`

	// Target: [Optional] The URL of the cluster resource that this
	// operation is associated with.
	Target string `json:"target,omitempty"`

	// Zone: The name of the Google Compute Engine zone in which the
	// operation is taking place.
	Zone string `json:"zone,omitempty"`
}

// method id "container.projects.clusters.list":

type ProjectsClustersListCall struct {
	s         *Service
	projectId string
	opt_      map[string]interface{}
}

// List: Lists all clusters owned by a project across all zones.
func (r *ProjectsClustersService) List(projectId string) *ProjectsClustersListCall {
	c := &ProjectsClustersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersListCall) Fields(s ...googleapi.Field) *ProjectsClustersListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsClustersListCall) Do() (*ListAggregatedClustersResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/clusters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ListAggregatedClustersResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all clusters owned by a project across all zones.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.clusters.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/clusters",
	//   "response": {
	//     "$ref": "ListAggregatedClustersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "container.projects.operations.list":

type ProjectsOperationsListCall struct {
	s         *Service
	projectId string
	opt_      map[string]interface{}
}

// List: Lists all operations in a project, across all zones.
func (r *ProjectsOperationsService) List(projectId string) *ProjectsOperationsListCall {
	c := &ProjectsOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	return c
}

// Fields allows partial responses to be retrieved.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsOperationsListCall) Fields(s ...googleapi.Field) *ProjectsOperationsListCall {
	c.opt_["fields"] = googleapi.CombineFields(s)
	return c
}

func (c *ProjectsOperationsListCall) Do() (*ListAggregatedOperationsResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fields"]; ok {
		params.Set("fields", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var ret *ListAggregatedOperationsResponse
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all operations in a project, across all zones.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.operations.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/operations",
	//   "response": {
	//     "$ref": "ListAggregatedOperationsResponse"
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
	zoneId               string
	createclusterrequest *CreateClusterRequest
	opt_                 map[string]interface{}
}

// Create: Creates a cluster, consisting of the specified number and
// type of Google Compute Engine instances, plus a Kubernetes master
// instance.
//
// The cluster is created in the project's default
// network.
//
// A firewall is added that allows traffic into port 443 on
// the master, which enables HTTPS. A firewall and a route is added for
// each node to allow the containers on that node to communicate with
// all other instances in the cluster.
//
// Finally, a route named
// k8s-iproute-10-xx-0-0 is created to track that the cluster's
// 10.xx.0.0/16 CIDR has been assigned.
func (r *ProjectsZonesClustersService) Create(projectId string, zoneId string, createclusterrequest *CreateClusterRequest) *ProjectsZonesClustersCreateCall {
	c := &ProjectsZonesClustersCreateCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/clusters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zoneId":    c.zoneId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//   "description": "Creates a cluster, consisting of the specified number and type of Google Compute Engine instances, plus a Kubernetes master instance.\n\nThe cluster is created in the project's default network.\n\nA firewall is added that allows traffic into port 443 on the master, which enables HTTPS. A firewall and a route is added for each node to allow the containers on that node to communicate with all other instances in the cluster.\n\nFinally, a route named k8s-iproute-10-xx-0-0 is created to track that the cluster's 10.xx.0.0/16 CIDR has been assigned.",
	//   "httpMethod": "POST",
	//   "id": "container.projects.zones.clusters.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "zoneId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/clusters",
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
	zoneId    string
	clusterId string
	opt_      map[string]interface{}
}

// Delete: Deletes the cluster, including the Kubernetes master and all
// worker nodes.
//
// Firewalls and routes that were configured at cluster
// creation are also deleted.
func (r *ProjectsZonesClustersService) Delete(projectId string, zoneId string, clusterId string) *ProjectsZonesClustersDeleteCall {
	c := &ProjectsZonesClustersDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/clusters/{clusterId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zoneId":    c.zoneId,
		"clusterId": c.clusterId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//   "description": "Deletes the cluster, including the Kubernetes master and all worker nodes.\n\nFirewalls and routes that were configured at cluster creation are also deleted.",
	//   "httpMethod": "DELETE",
	//   "id": "container.projects.zones.clusters.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "zoneId",
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
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/clusters/{clusterId}",
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
	zoneId    string
	clusterId string
	opt_      map[string]interface{}
}

// Get: Gets a specific cluster.
func (r *ProjectsZonesClustersService) Get(projectId string, zoneId string, clusterId string) *ProjectsZonesClustersGetCall {
	c := &ProjectsZonesClustersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/clusters/{clusterId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zoneId":    c.zoneId,
		"clusterId": c.clusterId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//     "zoneId",
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
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/clusters/{clusterId}",
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
	zoneId    string
	opt_      map[string]interface{}
}

// List: Lists all clusters owned by a project in the specified zone.
func (r *ProjectsZonesClustersService) List(projectId string, zoneId string) *ProjectsZonesClustersListCall {
	c := &ProjectsZonesClustersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/clusters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zoneId":    c.zoneId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//   "description": "Lists all clusters owned by a project in the specified zone.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.clusters.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "zoneId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone in which the cluster resides.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/clusters",
	//   "response": {
	//     "$ref": "ListClustersResponse"
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
	zoneId      string
	operationId string
	opt_        map[string]interface{}
}

// Get: Gets the specified operation.
func (r *ProjectsZonesOperationsService) Get(projectId string, zoneId string, operationId string) *ProjectsZonesOperationsGetCall {
	c := &ProjectsZonesOperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/operations/{operationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"zoneId":      c.zoneId,
		"operationId": c.operationId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//     "zoneId",
	//     "operationId"
	//   ],
	//   "parameters": {
	//     "operationId": {
	//       "description": "The server-assigned name of the operation.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone in which the operation resides. This is always the same zone as the cluster with which the operation is associated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/operations/{operationId}",
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
	zoneId    string
	opt_      map[string]interface{}
}

// List: Lists all operations in a project in a specific zone.
func (r *ProjectsZonesOperationsService) List(projectId string, zoneId string) *ProjectsZonesOperationsListCall {
	c := &ProjectsZonesOperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.zoneId = zoneId
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{projectId}/zones/{zoneId}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"zoneId":    c.zoneId,
	})
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
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
	//   "description": "Lists all operations in a project in a specific zone.",
	//   "httpMethod": "GET",
	//   "id": "container.projects.zones.operations.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "zoneId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "The Google Developers Console project ID or  project number.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "zoneId": {
	//       "description": "The name of the Google Compute Engine zone to return operations for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{projectId}/zones/{zoneId}/operations",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
