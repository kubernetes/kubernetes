/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package http_cloud

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"path"
	"net/url"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	HTTPProviderTimeout          = 90 * time.Second
	CloudProviderPath            = "/cloudprovider/v1alpha1"
	LocalePath                   = CloudProviderPath + "/locales"
	CloudProviderName            = "providername"
	InstanceName                 = "instances"
	SSHKeyToAllName              = "SSHKeyToAll"
	HostName                     = "hostnames"
	ClusterName                  = "clusters"
	RouterName                   = "routers"
	ZoneName                     = "zones"
	TCPLoadBalancerName          = "tcploadbalancernames"
)

type httpCloud struct {
	clientURL string

	// A cloud instance running this program must be in some 'region' or 'zone'    
	locale    string
}

// The current provider name
type ProviderName struct {
	Name string `json:"providerName"`
}

type InstanceSpec struct {
	ID        string            `json:"instanceID"`
	Addresses []api.NodeAddress `json:"nodeAddresses"`
}

type Instance struct {
	Type api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Object api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec InstanceSpec `json:"spec,omitempty"`
}

type InstanceList struct {
	Type api.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	List api.ListMeta `json:"metadata,omitempty"`

	// List of nodes
	Items []Instance `json:"items"`
}

// Zone
type ZoneSpec struct {
	Domain string `json:"failureDomain"`
	Region string `json:"region"`
}

type Zone struct {
	Type api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Object api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec ZoneSpec `json:"spec,omitempty"`
}

// Cluster
type ClusterSpec struct {
	MasterAddr string `json:"masterAddress"`
}

type Cluster struct {
	//Name string `json:"clusterName"`
	Type api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Object api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec ClusterSpec `json:"spec,omitempty"`
}

type ClusterList struct {
	//ClusterNames []ClusterName `json:"clusters"`
	Type api.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	List api.ListMeta `json:"metadata,omitempty"`

	// List of nodes
	Items []Cluster `json:"items"`
}

// Route
type RouteSpec struct {
	NameHint        string `json:"nameHint"`
	Target          string `json:"targetInstance"`
	DestinationCIDR string `json:"destinationCIDR"`
}

type Route struct {
	Type api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Object api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec RouteSpec `json:"spec,omitempty"`
}

type RouteList struct {
	Type api.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	List api.ListMeta `json:"metadata,omitempty"`

	// List of nodes
	Items []Route `json:"items"`
}

// TCPLoadBalancer
type TCPLoadBalancerStatus struct {
	Ingress []api.LoadBalancerIngress `json:"ingress"`
	Exists  bool                      `json:"exists"`
}

type TCPLoadBalancerSpec struct {
	Region          string            `json:"region"`
	ExternalIP      string            `json:"externalIP"`
	Ports           []api.ServicePort `json:"ports"`
	Hosts           []string          `json:"hosts"`
	ServiceAffinity string            `json:"sessionAffinity"`
}

type TCPLoadBalancer struct {
	Type api.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	Object api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	// http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status
	Spec TCPLoadBalancerSpec `json:"spec,omitempty"`
	
	Status TCPLoadBalancerStatus `json:"status,omitempty"`
}

// This does not need to have object name, because we don't delete it
// In addition, a user might have multiple keys
type InstanceSSHKey struct {
	User string `json:"user"`
	Key  []byte `json:"keyData"`
}

type Hostnames struct {
	Name string `json:"hostname"`
}

type Config struct {
	ClientURL string `json:"clientURL"`
}

func init() {
	cloudprovider.RegisterCloudProvider("http", func(config io.Reader) (cloudprovider.Interface, error) { return newHTTPCloud(config) })
}

// Creates a new instance of httpCloud through the config file sent.
func newHTTPCloud(config io.Reader) (*httpCloud, error) {
	if config != nil {
		stream := streamToByte(config)
		var conf Config
		if err := json.Unmarshal(stream, &conf); err != nil {
			return nil, fmt.Errorf("Unmarshalling JSON Config file Error: %s", err)
		}
		urlAddress := conf.ClientURL
		// Validate URL
		_, err := url.ParseRequestURI(urlAddress)
		if err != nil {
			return nil, fmt.Errorf("Can't parse the clientURL provided: %s", err)
		}
		// Handle Trailing slashes
		urlAddress = strings.TrimRight(urlAddress, "/")
	
		hc := &httpCloud{
			clientURL: urlAddress,
		}
		
		// A cloud instance running this program must be in some 'region' or 'zone'	
		zone, err := hc.GetZone();
		if err != nil {
			return nil, err
		}
		
		hc.locale = zone.Region
		return hc, nil
	}
	return nil, fmt.Errorf("Config file is empty or is not provided")
}

// Returns the cloud provider ID.
func (h *httpCloud) ProviderName() string {
	path := path.Join(CloudProviderPath, CloudProviderName)
	var resp ProviderName

	if err := h.get(path, &resp); err != nil {
		return ""
	} else {
		return resp.Name
	}
}

// Returns an implementation of Instances for HTTP cloud.
func (h *httpCloud) Instances() (cloudprovider.Instances, bool) {
	if h.supported(InstanceName) {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Clusters for HTTP cloud.
func (h *httpCloud) Clusters() (cloudprovider.Clusters, bool) {
	if h.supported(ClusterName) {
		return h, true
	}
	return nil, false
}

// Returns an implementation of TCPLoadBalancer for HTTP cloud.
func (h *httpCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	if h.supported(TCPLoadBalancerName) {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Zones for HTTP cloud.
func (h *httpCloud) Zones() (cloudprovider.Zones, bool) {
	if h.supported(ZoneName) {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Routes for HTTP cloud.
func (h *httpCloud) Routes() (cloudprovider.Routes, bool) {
	if h.supported(RouterName) {
		return h, true
	}
	return nil, false
}

// Returns the NodeAddresses of a particular machine instance.
func (h *httpCloud) Instance(instance string) (*Instance, error) {
	path := path.Join(LocalePath, h.locale, InstanceName, instance)
	var resp Instance
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Returns the NodeAddresses of a particular machine instance.
func (h *httpCloud) NodeAddresses(instance string) ([]api.NodeAddress, error) {

	instObj, err := h.Instance(instance)
	if err != nil {
		return nil, err
	}

	if ok, err := checkNodeAddressesValid(instObj.Spec.Addresses); !ok {
		return nil, err
	}
	
	return instObj.Spec.Addresses, nil
}

// Returns the cloud provider ID of the specified instance (deprecated).
func (h *httpCloud) ExternalID(instance string) (string, error) {
	return "", errors.New("unimplemented")
}

// Returns the cloud provider ID of the specified instance.
func (h *httpCloud) InstanceID(instance string) (string, error) {
	
	instObj, err := h.Instance(instance)
	if err != nil {
		return "", err
	}
	
	if instObj.Spec.ID == "" {
		return "", fmt.Errorf("Instance ID field is required and cannot be empty")
	}
	return instObj.Spec.ID, nil
}

// Enumerates the set of minions instances known by the cloud provider.
func (h *httpCloud) List(filter string) ([]string, error) {
	path := path.Join(LocalePath, h.locale, InstanceName, filter)
	var resp InstanceList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	
	var result []string
	for _, instance := range resp.Items {
		if instance.Object.Name == "" {
			return nil, fmt.Errorf("Instance Name field is required and cannot be empty")
		}
		result = append(result, instance.Object.Name)
	}
	return result, nil
}

// Adds an SSH public key as a legal identity for all instances.
// Expected format for the key is standard ssh-keygen format: <protocol> <blob>.
func (h *httpCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	path := path.Join(LocalePath, h.locale, SSHKeyToAllName)
	requestType := "POST"
	key := &InstanceSSHKey{
		User: user,
		Key:  keyData,
	}
	Body, _ := json.Marshal(key)
	body := bytes.NewReader(Body)
	JSONResp, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}

	var resp api.Status
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, resp)
	}

	if resp.Status != api.StatusSuccess {
		return fmt.Errorf("Error %s in adding SSH Key:%s for User:%s", resp.Reason, keyData, user)
	}
	return nil
}

// Returns the name of the node we are currently running on given the hostname.
func (h *httpCloud) CurrentNodeName(hostname string) (string, error) {
	path := path.Join(LocalePath, h.locale, HostName, hostname)
	var resp Instance
	if err := h.get(path, &resp); err != nil {
		return "", err
	}
	if resp.Object.Name == "" {
		return "", fmt.Errorf("Node Name field is required and cannot be empty")
	}
	return resp.Object.Name, nil
}

// Returns the Zone containing the current failure zone and locality region that the program is running in.
func (h *httpCloud) GetZone() (cloudprovider.Zone, error) {
	path := path.Join(CloudProviderPath, ZoneName)
	var resp Zone
	if err := h.get(path, &resp); err != nil {
		return cloudprovider.Zone{}, err
	}
	if resp.Spec.Domain == "" {
		return cloudprovider.Zone{}, fmt.Errorf("zone:failureDomain field is required and should not be empty")
	}
	if resp.Spec.Region == "" {
		return cloudprovider.Zone{}, fmt.Errorf("zone:region field is required and should not be empty")
	}
	return cloudprovider.Zone{
		FailureDomain: resp.Spec.Domain,
		Region:        resp.Spec.Region,
	}, nil
}

// Returns a list of clusters currently running.
func (h *httpCloud) ListClusters() ([]string, error) {
	path := path.Join(LocalePath, h.locale, ClusterName)
	var resp ClusterList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	if len(resp.Items) == 0 {
		return nil, fmt.Errorf("No cluster names provided")
	}
	var result []string
	for _, cluster := range resp.Items {
		if cluster.Object.Name == "" {
			return nil, fmt.Errorf("Cluster name is requiread and cannot be empty")
		}
		result = append(result, cluster.Object.Name)
	}
	return result, nil
}

// Returns the address of the master of the cluster.
func (h *httpCloud) Master(clusterName string) (string, error) {
	path := path.Join(LocalePath, h.locale, ClusterName, clusterName)
	var resp Cluster
	if err := h.get(path, &resp); err != nil {
		return "", err
	}
	if resp.Spec.MasterAddr == "" {
		return "", fmt.Errorf("Master address value is required and cannot be empty")
	}
	return resp.Spec.MasterAddr, nil
}

// Returns a list of all the routes that belong to the specific clusterName.
func (h *httpCloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	path := path.Join(LocalePath, h.locale, ClusterName, clusterName, RouterName)
	var resp RouteList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	if ok, err := checkRouteValid(resp.Items); !ok {
		return nil, err
	}
	var routes []*cloudprovider.Route
	for _, route := range resp.Items {
		routes = append(routes, &cloudprovider.Route{
								Name: route.Object.Name,
								TargetInstance: route.Spec.Target,
								DestinationCIDR: route.Spec.DestinationCIDR})
	}

	return routes, nil
}

// Creates a route inside the cloud provider and returns whether the route was created or not.
func (h *httpCloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	path := path.Join(LocalePath, h.locale, ClusterName, clusterName, RouterName)
	requestType := "POST"
	routeCreated := &Route{
		Object: api.ObjectMeta{
			Name: route.Name,
		},
		Spec: RouteSpec{
			NameHint:        nameHint,
			Target:          route.TargetInstance,
			DestinationCIDR: route.DestinationCIDR,
		},
	}

	Body, _ := json.Marshal(routeCreated)
	body := bytes.NewReader(Body)
	JSONResp, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}

	var resp api.Status
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, resp)
	}

	if resp.Status != api.StatusSuccess {
		return fmt.Errorf("Error %s in creating route: %s", resp.Reason, route.Name)
	}
	return nil
}

// Delete the requested route.
func (h *httpCloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	path := path.Join(LocalePath, h.locale, ClusterName, clusterName, RouterName, route.Name)
	requestType := "DELETE"
	JSONResp, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}

	var resp api.Status
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, resp)
	}

	if resp.Status != api.StatusSuccess {
		return fmt.Errorf("Error %s in deleting route: %s", resp.Reason, route.Name)
	}
	return nil
}

// Returns the TCP Load Balancer Status if it exists in the region with the particular name.
func (h *httpCloud) GetTCPLoadBalancer(name, region string) (status *api.LoadBalancerStatus, exists bool, err error) {
	path := path.Join(LocalePath, region, TCPLoadBalancerName, name)
	var resp TCPLoadBalancer
	if err := h.get(path, &resp); err != nil {
		return nil, false, err
	}
	if resp.Status.Exists {
		for _, ingress := range resp.Status.Ingress {
			if ingress.Hostname == "" && ingress.IP == "" {
				return nil, false, fmt.Errorf("Either ingress:IP or ingress:hostname is required and cannot be empty")
			}
		}
		return &api.LoadBalancerStatus{Ingress: resp.Status.Ingress}, true, nil
	}
	return nil, false, nil
}

// Creates a new tcp load balancer and return the status of the balancer.
func (h *httpCloud) EnsureTCPLoadBalancer(name, region string, externalIP net.IP, ports []*api.ServicePort, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error) {
	path := path.Join(LocalePath, region, TCPLoadBalancerName)
	requestType := "POST"
	
	var Ports []api.ServicePort
	for _, servicePort := range ports {
		Ports = append(Ports, *servicePort)
	}
	
	balancer := &TCPLoadBalancer{
		Object: api.ObjectMeta{
			Name: name,
		},
		Spec: TCPLoadBalancerSpec{
			Region: region, 
			ExternalIP: string(externalIP), 
			Ports: Ports,
			Hosts: hosts, 
			ServiceAffinity: string(affinityType),
		},
	}
	
	Body, _ := json.Marshal(balancer)
	body := bytes.NewReader(Body)
	JSONResp, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	
	var resp TCPLoadBalancer
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return nil, fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, body)
	}
	
	for _, ingress := range resp.Status.Ingress {
		if ingress.Hostname == "" && ingress.IP == "" {
			return nil, fmt.Errorf("Either ingress:IP or ingress:hostname is required and cannot be empty")
		}
	}
	
	return &api.LoadBalancerStatus{Ingress: resp.Status.Ingress}, nil
}

// Updates the hosts given in the Load Balancer specified.
// Since we do partial update, we have to get, modify, and put it back
func (h *httpCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	path := path.Join(LocalePath, region, TCPLoadBalancerName, name)
	var resp TCPLoadBalancer
	if err := h.get(path, &resp); err != nil {
		return fmt.Errorf("HTTP get to cloudprovider failed: %v", err)
	}
	resp.Spec.Hosts = hosts
	requestType := "PUT"
	Body, _ := json.Marshal(resp)
	body := bytes.NewReader(Body)
	JSONResp, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}

	var resp_status api.Status
	if err := json.Unmarshal(JSONResp, &resp_status); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, resp)
	}

	if resp_status.Status != api.StatusSuccess {
		return fmt.Errorf("Error %s in update hosts for tcploadbalancer: %s", resp_status.Reason, name)
	}

	return nil
}

// Deletes the specified Load Balancer.
func (h *httpCloud) EnsureTCPLoadBalancerDeleted(name, region string) error {
	path := path.Join(LocalePath, region, TCPLoadBalancerName, name)
	requestType := "DELETE"
	JSONResp, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}

	var resp api.Status
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, resp)
	}

	if resp.Status != api.StatusSuccess {
		return fmt.Errorf("Error %s in deleting tcploadbalancer: %s", resp.Reason, name)
	}

	return nil
}

// Sends HTTP requests for all interfaces support methods.
func (h *httpCloud) supported(iface string) bool {
	path := path.Join(CloudProviderPath, iface)
	requestType := "OPTIONS"
	_, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		glog.Errorf("%s not supported", iface)
		return false
	}
	return true
}

// Creates a request for the specified type, path and body and then receives the reply back from the receiver.
// If response code is not 200 or 204(for PUT or DELETE), then an error is sent out.
func (h *httpCloud) sendHTTPRequest(requestType string, requestPath string, requestBody io.Reader) ([]byte, error) {
	url := h.clientURL + requestPath
	req, err := http.NewRequest(requestType, url, requestBody)
	if err != nil {
		return nil, err
	}

	client := &http.Client{
		Timeout:   HTTPProviderTimeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if resp.StatusCode == 200 || (requestType == "DELETE" && resp.StatusCode == 204) || (requestType == "PUT" && resp.StatusCode == 204) {
		if err != nil {
			return nil, err
		} else {
			return body, nil
		}
	} else if resp.StatusCode == 501 && requestType == "OPTIONS" {
		// Interface supported or not
		return nil, fmt.Errorf("Interface not supported")
	} else {
		var response api.Status
		if err := json.Unmarshal(body, &response); err != nil {
			return nil, fmt.Errorf("Error in reading JSON response from the cloudprovider for HTTP %d Error for %s %s: %v\n", resp.StatusCode, requestType, requestPath, err)
		} else {
			return nil, fmt.Errorf("Error from Cloudprovider for %s %s: HTTP %d Error with the current status as %s, the reason for failure is %s\n.", requestType, requestPath, resp.StatusCode, response.Message, response.Reason)
		}
	}
}

// Sends a HTTP Get Request and Unmarshals the JSON Response.
func (h *httpCloud) get(path string, resp interface{}) error {
	requestType := "GET"
	body, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	if body != nil {
		if err := json.Unmarshal(body, resp); err != nil {
			return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v\n The response was: %s", requestType, path, err, body)
		}
	}
	return nil
}

// Checks whether the NodeAddress struct has all the valid values or not.
func checkNodeAddressesValid(Nodes []api.NodeAddress) (bool, error) {
	types := []string{"InternalIP", "ExternalIP", "Hostname", "LegacyHostIP"}
	for _, node := range Nodes {
		if node.Address == "" {
			return false, fmt.Errorf("nodeAdresses:address value is required and cannot be empty.")
		}

		match := false
		for _, Type := range types {
			if string(node.Type) == Type {
				match = true
			}
		}
		if !match {
			return false, fmt.Errorf("nodeAdresses:type expected to be either InternalIP, ExternalIP, Hostname or LegacyHostIP. Got %s", node.Type)
		}
	}
	return true, nil
}

// Checks whether the RouteList struct has valid values or not.
func checkRouteValid(RouteList []Route) (bool, error) {
	for _, route := range RouteList {
		if route.Object.Name == "" {
			return false, fmt.Errorf("routeName is required and cannot be empty.")
		}
		if route.Spec.Target == "" {
			return false, fmt.Errorf("targetInstance is required and cannot be empty.")
		}
		if route.Spec.DestinationCIDR == "" {
			return false, fmt.Errorf("destinationCIDR is required and cannot be empty.")
		}
	}
	return true, nil
}

// Coverts the io.Reader stream to bytes
func streamToByte(stream io.Reader) []byte {
	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)
	return buf.Bytes()
}
