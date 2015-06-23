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
	"net/url"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type httpCloud struct {
	clientURL string
}

// The current provider name
type ProviderName struct {
	Name string `json:"providerName"`
}

type InstanceResources struct {
	Resources []NodeResource `json:"resources"`
}

type NodeResource struct {
	ResourceName string           `json:"resourceName"`
	Quantity     ResourceQuantity `json:"quantity"`
}

type ResourceQuantity struct {
	Amount int64  `json:"amount"`
	Format string `json:"format"`
}

type InstanceNodeAddress struct {
	Node []api.NodeAddress `json:"nodeAddresses"`
}

type InstanceID struct {
	ID string `json:"instanceID"`
}

type InstanceList struct {
	List []InstanceName `json:"instances"`
}

type InstanceName struct {
	Name string `json:"instanceName"`
}

type InstanceSSHKey struct {
	User string `json:"user"`
	Key  []byte `json:"keyData"`
}

type InstanceNodeName struct {
	Name string `json:"nodeName"`
}

type InstanceSSHKeyResponse struct {
	KeyAdded bool `json:"SSHKeyAdded"`
}

type Zone struct {
	Domain string `json:"failureDomain"`
	Region string `json:"region"`
}

type ClusterList struct {
	ClusterNames []ClusterName `json:"clusters"`
}

type ClusterName struct {
	Name string `json:"clusterName"`
}

type ClusterMaster struct {
	Address string `json:"masterAddress"`
}

type RouteList struct {
	List []Route `json:"routes"`
}

type CreateRoute struct {
	ClusterName string `json:"clusterName"`
	NameHint    string `json:"nameHint"`
	NewRoute    Route  `json:"route"`
}

type Route struct {
	Name            string `json:"routeName"`
	Target          string `json:"targetInstance"`
	DestinationCIDR string `json:"destinationCIDR"`
}

type RouteCreatedStatus struct {
	RouteCreated bool `json:"routeCreated"`
}

type TCPLoadBalancerStatus struct {
	Ingress []api.LoadBalancerIngress `json:"ingress"`
	Exists  bool                      `json:"exists"`
}

type TCPLoadBalancer struct {
	Name            string            `json:"loadBalancerName"`
	Region          string            `json:"region"`
	IP              string            `json:"externalIP"`
	Ports           []api.ServicePort `json:"ports"`
	Hosts           []Hostnames       `json:"hosts"`
	ServiceAffinity string            `json:"sessionAffinity"`
}

type Hostnames struct {
	Name string `json:"hostname"`
}

type HostTCPLoadBalancer struct {
	Hosts []Hostnames `json:"hosts"`
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
		return &httpCloud{
			clientURL: urlAddress,
		}, nil
	}
	return nil, fmt.Errorf("Config file is empty or is not provided")
}

// Returns the cloud provider ID.
func (h *httpCloud) ProviderName() string {
	path := "/cloudprovider/v1aplha1/providerName"
	var resp ProviderName

	if err := h.get(path, &resp); err != nil {
		return ""
	} else {
		return resp.Name
	}
}

// Returns an implementation of Instances for HTTP cloud.
func (h *httpCloud) Instances() (cloudprovider.Instances, bool) {
	if h.supported("instances") {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Clusters for HTTP cloud.
func (h *httpCloud) Clusters() (cloudprovider.Clusters, bool) {
	if h.supported("clusters") {
		return h, true
	}
	return nil, false
}

// Returns an implementation of TCPLoadBalancer for HTTP cloud.
func (h *httpCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	if h.supported("tcpLoadBalancers") {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Zones for HTTP cloud.
func (h *httpCloud) Zones() (cloudprovider.Zones, bool) {
	if h.supported("zones") {
		return h, true
	}
	return nil, false
}

// Returns an implementation of Routes for HTTP cloud.
func (h *httpCloud) Routes() (cloudprovider.Routes, bool) {
	if h.supported("routes") {
		return h, true
	}
	return nil, false
}

// Returns the NodeAddresses of a particular machine instance.
func (h *httpCloud) NodeAddresses(instance string) ([]api.NodeAddress, error) {
	path := "/cloudprovider/v1aplha1/instances/" + instance + "/nodeAddresses"
	var resp InstanceNodeAddress
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	if ok, err := checkNodeAddressesValid(resp.Node); !ok {
		return nil, err
	}
	return resp.Node, nil
}

// Returns the cloud provider ID of the specified instance (deprecated).
func (h *httpCloud) ExternalID(instance string) (string, error) {
	return "", errors.New("unimplemented")
}

// Returns the cloud provider ID of the specified instance.
func (h *httpCloud) InstanceID(instance string) (string, error) {
	path := "/cloudprovider/v1aplha1/instances/" + instance + "/ID"
	var resp InstanceID
	if err := h.get(path, &resp); err != nil {
		return "", err
	}
	if resp.ID == "" {
		return "", fmt.Errorf("Instance ID field is required and cannot be empty")
	}
	return resp.ID, nil
}

// Enumerates the set of minions instances known by the cloud provider.
func (h *httpCloud) List(filter string) ([]string, error) {
	path := "/cloudprovider/v1aplha1/instances/" + filter
	var resp InstanceList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	var result []string
	for _, instance := range resp.List {
		if instance.Name == "" {
			return nil, fmt.Errorf("Instance Name field is required and cannot be empty")
		}
		result = append(result, instance.Name)
	}
	return result, nil
}

// Adds an SSH public key as a legal identity for all instances.
// Expected format for the key is standard ssh-keygen format: <protocol> <blob>.
func (h *httpCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	path := "/cloudprovider/v1aplha1/instances/sshKey"
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
	var resp InstanceSSHKeyResponse
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, body)
	}
	// If key is not added for the particular user, then return an error or else return nil.
	if !resp.KeyAdded {
		return fmt.Errorf("Error in adding SSH Key:%s for User:%s", keyData, user)
	}
	return nil
}

// Returns the name of the node we are currently running on given the hostname.
func (h *httpCloud) CurrentNodeName(hostname string) (string, error) {
	path := "/cloudprovider/v1aplha1/instances/node/" + hostname
	var resp InstanceNodeName
	if err := h.get(path, &resp); err != nil {
		return "", err
	}
	if resp.Name == "" {
		return "", fmt.Errorf("Node Name field is required and cannot be empty")
	}
	return resp.Name, nil
}

// Returns the Zone containing the current failure zone and locality region that the program is running in.
func (h *httpCloud) GetZone() (cloudprovider.Zone, error) {
	path := "/cloudprovider/v1aplha1/zones"
	var resp Zone
	if err := h.get(path, &resp); err != nil {
		return cloudprovider.Zone{}, err
	}
	if resp.Domain == "" {
		return cloudprovider.Zone{}, fmt.Errorf("zone:failureDomain field is required and should not be empty")
	}
	if resp.Region == "" {
		return cloudprovider.Zone{}, fmt.Errorf("zone:region field is required and should not be empty")
	}
	return cloudprovider.Zone{
		FailureDomain: resp.Domain,
		Region:        resp.Region,
	}, nil
}

// Returns a list of clusters currently running.
func (h *httpCloud) ListClusters() ([]string, error) {
	path := "/cloudprovider/v1aplha1/clusters"
	var resp ClusterList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	if len(resp.ClusterNames) == 0 {
		return nil, fmt.Errorf("No cluster names provided")
	}
	var result []string
	for _, cluster := range resp.ClusterNames {
		if cluster.Name == "" {
			return nil, fmt.Errorf("Cluster name is requiread and cannot be empty")
		}
		result = append(result, cluster.Name)
	}
	return result, nil
}

// Returns the address of the master of the cluster.
func (h *httpCloud) Master(clusterName string) (string, error) {
	path := "/cloudprovider/v1aplha1/clusters/" + clusterName + "/master"
	var resp ClusterMaster
	if err := h.get(path, &resp); err != nil {
		return "", err
	}
	if resp.Address == "" {
		return "", fmt.Errorf("Master address value is required and cannot be empty")
	}
	return resp.Address, nil
}

// Returns a list of all the routes that belong to the specific clusterName.
func (h *httpCloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	path := "/cloudprovider/v1aplha1/routes/" + clusterName
	var resp RouteList
	if err := h.get(path, &resp); err != nil {
		return nil, err
	}
	if ok, err := checkRouteValid(resp.List); !ok {
		return nil, err
	}
	var routes []*cloudprovider.Route
	for _, route := range resp.List {
		routes = append(routes, &cloudprovider.Route{route.Name, route.Target, route.DestinationCIDR})
	}

	return routes, nil
}

// Creates a route inside the cloud provider and returns whether the route was created or not.
func (h *httpCloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	path := "/cloudprovider/v1aplha1/routes"
	requestType := "POST"
	routeCreated := &CreateRoute{
		ClusterName: clusterName,
		NameHint:    nameHint,
		NewRoute: Route{
			Name:            route.Name,
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
	var resp RouteCreatedStatus
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, body)
	}
	// If key is not added for the particular user, then return an error or else return nil.
	if !resp.RouteCreated {
		return fmt.Errorf("Error in creating route: %s", route.Name)
	}
	return nil
}

// Delete the requested route.
func (h *httpCloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	path := "/cloudprovider/v1aplha1/routes/" + clusterName + "/" + route.Name
	requestType := "DELETE"
	_, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	return nil
}

// Returns the TCP Load Balancer Status if it exists in the region with the particular name.
func (h *httpCloud) GetTCPLoadBalancer(name, region string) (status *api.LoadBalancerStatus, exists bool, err error) {
	path := "/cloudprovider/v1aplha1/tcpLoadBalancers/" + region + "/" + name
	var resp TCPLoadBalancerStatus
	if err := h.get(path, &resp); err != nil {
		return nil, false, err
	}
	if resp.Exists {
		for _, ingress := range resp.Ingress {
			if ingress.Hostname == "" && ingress.IP == "" {
				return nil, false, fmt.Errorf("Either ingress:IP or ingress:hostname is required and cannot be empty")
			}
		}
		return &api.LoadBalancerStatus{Ingress: resp.Ingress}, true, nil
	}
	return nil, false, nil
}

// Creates a new tcp load balancer and return the status of the balancer.
func (h *httpCloud) EnsureTCPLoadBalancer(name, region string, externalIP net.IP, ports []*api.ServicePort, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error) {
	path := "/cloudprovider/v1aplha1/tcpLoadBalancers"
	requestType := "POST"
	var Ports []api.ServicePort
	for _, servicePort := range ports {
		Ports = append(Ports, *servicePort)
	}
	var Hosts []Hostnames
	for _, host := range hosts {
		Hosts = append(Hosts, Hostnames{Name: host})
	}
	balancer := &TCPLoadBalancer{
		Name: name, Region: region, IP: string(externalIP), Ports: Ports, Hosts: Hosts, ServiceAffinity: string(affinityType),
	}
	Body, _ := json.Marshal(balancer)
	body := bytes.NewReader(Body)
	JSONResp, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return nil, fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	var resp api.LoadBalancerStatus
	if err := json.Unmarshal(JSONResp, &resp); err != nil {
		return nil, fmt.Errorf("Error in reading JSON response from the cloudprovider for %s %s: %v The response was: %s", requestType, path, err, body)
	}
	for _, ingress := range resp.Ingress {
		if ingress.Hostname == "" && ingress.IP == "" {
			return nil, fmt.Errorf("Either ingress:IP or ingress:hostname is required and cannot be empty")
		}
	}
	return &api.LoadBalancerStatus{Ingress: resp.Ingress}, nil
}

// Updates the hosts given in the Load Balancer specified.
func (h *httpCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	path := "/cloudprovider/v1aplha1/tcpLoadBalancers/" + region + "/" + name
	requestType := "PUT"
	var TCPHosts HostTCPLoadBalancer
	for _, host := range hosts {
		TCPHosts.Hosts = append(TCPHosts.Hosts, Hostnames{Name: host})
	}
	Body, _ := json.Marshal(TCPHosts)
	body := bytes.NewReader(Body)
	_, err := h.sendHTTPRequest(requestType, path, body)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	return nil
}

// Deletes the specified Load Balancer.
func (h *httpCloud) EnsureTCPLoadBalancerDeleted(name, region string) error {
	path := "/cloudprovider/v1aplha1/tcpLoadBalancers/" + region + "/" + name
	requestType := "DELETE"
	_, err := h.sendHTTPRequest(requestType, path, nil)
	if err != nil {
		return fmt.Errorf("HTTP request to cloudprovider failed: %v", err)
	}
	return nil
}

// Sends HTTP requests for all interfaces support methods.
func (h *httpCloud) supported(iface string) bool {
	path := "/cloudprovider/v1aplha1/" + iface
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

	client := &http.Client{}
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

// Checks whether the NodeResource struct has all the valid values or not.
func checkResourcesValid(Resources []NodeResource) (bool, error) {
	formats := []string{"DecimalExponent", "BinarySI", "DecimalSI"}
	resources := []string{"cpu", "memory", "storage", "pods", "services", "replicationcontrollers", "resourcequotas", "secrets", "persistentvolumeclaims"}
	for _, Resource := range Resources {
		if Resource.Quantity.Amount == 0.0 {
			return false, fmt.Errorf("resources:quantity:format value is required and cannot be 0.")
		}
		match := false
		for _, Format := range formats {
			if string(Resource.Quantity.Format) == Format {
				match = true
			}
		}
		if !match {
			return false, fmt.Errorf("resources:quantity:format expected to be either DecimalExponent, BinarySI or DecimalSI. Got %s", Resource.Quantity.Format)
		}
		match = false
		for _, Name := range resources {
			if Resource.ResourceName == Name {
				match = true
			}
		}
		if !match {
			return false, fmt.Errorf("resources:resourceName expected to be either cpu, memory, storage, pods, services, replicationcontrollers, resourcequotas, secrets or persistentvolumeclaims. Got %s", Resource.Quantity.Format)
		}
	}
	return true, nil
}

// Checks whether the RouteList struct has valid values or not.
func checkRouteValid(RouteList []Route) (bool, error) {
	for _, route := range RouteList {
		if route.Name == "" {
			return false, fmt.Errorf("routeName is required and cannot be empty.")
		}
		if route.Target == "" {
			return false, fmt.Errorf("targetInstance is required and cannot be empty.")
		}
		if route.DestinationCIDR == "" {
			return false, fmt.Errorf("destinationCIDR is required and cannot be empty.")
		}
	}
	return true, nil
}

// Creates the api.NodeResources Struct from a NodeResource struct which is received from a JSON response.
func makeNodeResources(Resources []NodeResource) *api.NodeResources {
	resourceList := api.ResourceList{}
	for _, Resource := range Resources {
		resourceList[api.ResourceName(Resource.ResourceName)] = *resource.NewQuantity(Resource.Quantity.Amount, resource.Format(Resource.Quantity.Format))
	}
	return &api.NodeResources{
		Capacity: resourceList,
	}
}

// Coverts the io.Reader stream to bytes
func streamToByte(stream io.Reader) []byte {
	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)
	return buf.Bytes()
}
