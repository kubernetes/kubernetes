/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// ConcertoAPIService is an abstraction for Flexiant Concerto API.
type ConcertoAPIService interface {
	// Retrieves the info related to the instance which name is passed
	GetInstanceByName(name string) (ConcertoInstance, error)
	// Retrieves all instances
	GetInstanceList() ([]ConcertoInstance, error)
	// Creates a LB with the specified name
	CreateLoadBalancer(name string, port int, nodePort int) (*ConcertoLoadBalancer, error)
	// Retrieves a LB with the specified name
	GetLoadBalancerByName(name string) (*ConcertoLoadBalancer, error)
	// Deletes Load Balancer with given Id
	DeleteLoadBalancerById(id string) error
	// Gets the nodes registered with the load balancer
	GetLoadBalancerNodes(loadBalancerId string) ([]ConcertoLoadBalancerNode, error)
	// Gets the IPs of the nodes registered with the load balancer
	GetLoadBalancerNodesAsIPs(loadBalancerId string) ([]string, error)
	// Registers the instances with the load balancer
	RegisterInstancesWithLoadBalancer(loadBalancerId string, nodesIPs []string) error
	// Deregisters the instances from the load balancer
	DeregisterInstancesFromLoadBalancer(loadBalancerId string, nodesIPs []string) error
}

// ConcertoInstance is an abstraction for a Concerto cloud instance
type ConcertoInstance struct {
	Id       string  // Unique identifier for the instance in Concerto
	Name     string  // Hostname for the instance
	PublicIP string  // Public IP for the instance
	CPUs     float64 // Number of cores
	Memory   int64   // Amount of RAM (in MiB)
	Storage  int64   // Amount of disk (in GiB)
}

// Ship is used for deserializing
type Ship struct {
	Id             string
	Fqdn           string
	Public_ip      string
	Server_plan_id string
	Cpus           float64 // Number of cores
	Memory         int64   // Amount of RAM (in MB)
	Storage        int64   // Amount of disk (in GiB)
}

// ConcertoLoadBalancer abstracts a Concerto Load Balancer
type ConcertoLoadBalancer struct {
	Id       string `json:"id"`   // Unique identifier for the LB in Concerto
	Name     string `json:"name"` // Name of the LB in concerto
	FQDN     string `json:"fqdn"` // Fully Qualified domain name
	Port     int    `json:"port"`
	NodePort int    `json:"nodeport"`
	Protocol string `json:"protocol"`
}

// ConcertoLoadBalancer abstracts a Concerto Load Balancer
type ConcertoLoadBalancerNode struct {
	ID string
	IP string `json:"public_ip"`
	// Port int    `json:"port"`
}

// Concerto REST API client implementation
type concertoAPIServiceREST struct {
	// Pre-configured HTTP client
	client concertoRESTService
}

type concertoRESTService interface {
	// Get resource at 'path', returning body, status code, and error if any
	Get(path string) ([]byte, int, error)
	// Post resource to 'path', returning body, status code, and error if any
	Post(path string, json []byte) ([]byte, int, error)
	// Delete resource at 'path', returning body, status code, and error if any
	Delete(path string) ([]byte, int, error)
}

// BuildConcertoRESTClient Factory for 'concertoAPIServiceREST' objects
func buildConcertoRESTClient(config ConcertoConfig) (ConcertoAPIService, error) {
	rs, err := newRestService(config)
	if err != nil {
		return nil, fmt.Errorf("error building Concerto REST client: %v", err)
	}

	return &concertoAPIServiceREST{client: rs}, nil
}

func (c *concertoAPIServiceREST) GetInstanceList() ([]ConcertoInstance, error) {
	var ships []Ship
	instances := []ConcertoInstance{}

	data, status, err := c.client.Get("/kaas/ships")
	if err != nil {
		return nil, fmt.Errorf("error on 'GET /kaas/ships' http request: %v", err)
	}

	if status == 404 {
		return instances, nil
	}

	err = json.Unmarshal(data, &ships)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling http response: %v (received data was: %s)", err, string(data))
	}

	for _, s := range ships {
		concertoInstance := ConcertoInstance{
			Id:       s.Id,
			Name:     s.Fqdn,
			PublicIP: s.Public_ip,
			CPUs:     s.Cpus,
			Memory:   s.Memory,
			Storage:  s.Storage,
		}
		instances = append(instances, concertoInstance)
	}

	return instances, nil
}

func (c *concertoAPIServiceREST) GetInstanceByName(name string) (ConcertoInstance, error) {
	concertoInstances, err := c.GetInstanceList()
	if err != nil {
		return ConcertoInstance{}, fmt.Errorf("error getting list of instances from Concerto: %v", err)
	}

	for _, instance := range concertoInstances {
		if instance.Name == name {
			return instance, nil
		}
	}

	return ConcertoInstance{}, cloudprovider.InstanceNotFound
}

func (c *concertoAPIServiceREST) GetLoadBalancerList() ([]ConcertoLoadBalancer, error) {
	var lbs []ConcertoLoadBalancer

	data, status, err := c.client.Get("/kaas/load_balancers")
	if err != nil {
		return nil, fmt.Errorf("error on 'GET /kaas/load_balancers' http request: %v", err)
	}

	if status >= 400 {
		return nil, fmt.Errorf("HTTP status %v when getting '/kaas/load_balancers'", status)
	}

	err = json.Unmarshal(data, &lbs)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling http response: %v (received data was: %s)", err, string(data))
	}

	return lbs, nil
}

func (c *concertoAPIServiceREST) GetLoadBalancerByName(name string) (*ConcertoLoadBalancer, error) {
	concertoLBs, err := c.GetLoadBalancerList()
	if err != nil {
		return nil, fmt.Errorf("error getting load balancer list from Concerto: %v", err)
	}

	for _, lb := range concertoLBs {
		if lb.Name == name {
			return &lb, nil
		}
	}

	glog.Warningf("Could not find load balancer '%s' on Concerto", name)
	return nil, nil
}

func (c *concertoAPIServiceREST) DeleteLoadBalancerById(id string) error {
	_, status, err := c.client.Delete("/kaas/load_balancers/" + id)
	if err != nil {
		return fmt.Errorf("error on http request 'DELETE /kaas/load_balancers/%s': %v", id, err)
	}
	if status != 200 && status != 204 {
		return fmt.Errorf("http status %v on request 'DELETE /kaas/load_balancers/%s' (expected 204 or 200)", status, id)
	}
	return nil
}

func (c *concertoAPIServiceREST) RegisterInstancesWithLoadBalancer(loadBalancerId string, ips []string) error {
	for _, ip := range ips {
		err := c.registerInstanceWithLoadBalancer(loadBalancerId, ip)
		if err != nil {
			return fmt.Errorf("error registering instance '%s' with load balancer '%s': %v", ip, loadBalancerId, err)
		}
	}
	return nil
}

func (c *concertoAPIServiceREST) registerInstanceWithLoadBalancer(loadBalancerId string, ip string) error {
	instance, err := c.GetInstanceByIP(ip)
	if err != nil {
		return fmt.Errorf("error getting Concerto instance by IP '%s': %v", ip, err)
	}
	jsonNode := instance.toNode().toJson()
	url := fmt.Sprintf("/kaas/load_balancers/%s/nodes", loadBalancerId)
	_, status, err := c.client.Post(url, jsonNode)
	if err != nil {
		return fmt.Errorf("error on http request 'POST %s': %v", url, err)
	}
	if status != 201 {
		return fmt.Errorf("http status %v on request 'POST %s' (expected 201)", status, url)
	}
	return nil
}

func (c *concertoAPIServiceREST) DeregisterInstancesFromLoadBalancer(loadBalancerId string, ips []string) error {
	for _, ip := range ips {
		err := c.deregisterInstanceFromLoadBalancer(loadBalancerId, ip)
		if err != nil {
			return fmt.Errorf("error deregistering instance '%s' from load balancer '%s': %v", ip, loadBalancerId, err)
		}
	}
	return nil
}

func (c *concertoAPIServiceREST) deregisterInstanceFromLoadBalancer(loadBalancerId string, ip string) error {
	node, err := c.GetNodeByIP(loadBalancerId, ip)
	if err != nil {
		return fmt.Errorf("error getting node by ip '%s' from load balancer '%s': %v", ip, loadBalancerId, err)
	}
	path := fmt.Sprintf("/kaas/load_balancers/%s/nodes/%s", loadBalancerId, node.ID)
	_, status, err := c.client.Delete(path)
	if err != nil {
		return fmt.Errorf("error on http request 'DELETE %s': %v", path, err)
	}
	if status != 200 && status != 204 {
		return fmt.Errorf("http status %v on request 'DELETE %s' (expected 204 or 200)", status, path)
	}
	return nil
}

func (c *concertoAPIServiceREST) GetLoadBalancerNodes(loadBalancerId string) ([]ConcertoLoadBalancerNode, error) {
	var nodes []ConcertoLoadBalancerNode

	path := fmt.Sprintf("/kaas/load_balancers/%s/nodes", loadBalancerId)
	data, status, err := c.client.Get(path)
	if err != nil {
		return nil, fmt.Errorf("error on http request 'GET %s': %v", path, err)
	}

	if status == 404 {
		return nil, fmt.Errorf("load balancer not found: %v", loadBalancerId)
	}

	err = json.Unmarshal(data, &nodes)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling http response: %v (received data was: %s)", err, string(data))
	}

	return nodes, nil
}

func (c *concertoAPIServiceREST) GetLoadBalancerNodesAsIPs(loadBalancerId string) (nodeips []string, e error) {
	nodes, err := c.GetLoadBalancerNodes(loadBalancerId)
	if err != nil {
		return nil, fmt.Errorf("error getting nodes from load balancer '%s' : %v", loadBalancerId, err)
	}

	for _, node := range nodes {
		nodeips = append(nodeips, node.IP)
	}

	return nodeips, nil
}

func (c *concertoAPIServiceREST) CreateLoadBalancer(name string, port int, nodePort int) (*ConcertoLoadBalancer, error) {
	lb := ConcertoLoadBalancer{
		Name:     name,
		FQDN:     name,
		Port:     port,
		NodePort: nodePort,
		Protocol: "tcp",
	}
	data, status, err := c.client.Post("/kaas/load_balancers", lb.toJson())
	if err != nil {
		return nil, fmt.Errorf("error on http request 'POST /kaas/load_balancers': %v", err)
	}
	if status != 201 {
		return nil, fmt.Errorf("HTTP %v when creating load balancer %s", status, name)
	}

	err = json.Unmarshal(data, &lb) // So that we get the Id
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling http response: %v (received data was: %s)", err, string(data))
	}

	return &lb, nil
}

func (c *concertoAPIServiceREST) GetInstanceByIP(ip string) (ConcertoInstance, error) {
	concertoInstances, err := c.GetInstanceList()
	if err != nil {
		return ConcertoInstance{}, fmt.Errorf("error getting instance list from Concerto: %v", err)
	}

	for _, instance := range concertoInstances {
		if instance.PublicIP == ip {
			return instance, nil
		}
	}

	return ConcertoInstance{}, cloudprovider.InstanceNotFound
}

func (c *concertoAPIServiceREST) GetNodeByIP(loadBalancerId, ip string) (ConcertoLoadBalancerNode, error) {
	lbNodes, err := c.GetLoadBalancerNodes(loadBalancerId)
	if err != nil {
		return ConcertoLoadBalancerNode{}, fmt.Errorf("error getting nodes from load balancer '%s' : %v", loadBalancerId, err)
	}

	for _, node := range lbNodes {
		if node.IP == ip {
			return node, nil
		}
	}

	return ConcertoLoadBalancerNode{}, fmt.Errorf("Node '%s' not found in load balancer '%s'", ip, loadBalancerId)
}

func (ci ConcertoInstance) toNode() ConcertoLoadBalancerNode {
	var node ConcertoLoadBalancerNode
	node.IP = ci.PublicIP
	return node
}

func (cn ConcertoLoadBalancerNode) toJson() []byte {
	b, err := json.Marshal(cn)
	if err != nil {
		return nil
	}
	return b
}

func (lb ConcertoLoadBalancer) toJson() []byte {
	b, err := json.Marshal(lb)
	if err != nil {
		return nil
	}
	return b
}
