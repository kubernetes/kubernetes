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
	"fmt"
	"net"

	"k8s.io/kubernetes/pkg/api"
)

// GetLoadBalancer implementation for Flexiant Concerto.
func (c *ConcertoCloud) GetLoadBalancer(name, _region string) (status *api.LoadBalancerStatus, exists bool, err error) {
	lb, err := c.service.GetLoadBalancerByName(name)
	if err != nil {
		return nil, false, fmt.Errorf("error getting Concerto load balancer by name '%s' : %v", name, err)
	}

	if lb == nil {
		return nil, false, nil
	}

	status = toStatus(lb)
	return status, true, nil
}

func toStatus(lb *ConcertoLoadBalancer) *api.LoadBalancerStatus {
	status := &api.LoadBalancerStatus{}

	var ingress api.LoadBalancerIngress
	ingress.Hostname = lb.FQDN
	status.Ingress = []api.LoadBalancerIngress{ingress}

	return status
}

// EnsureLoadBalancer implementation for Flexiant Concerto.
func (c *ConcertoCloud) EnsureLoadBalancer(name, region string, loadBalancerIP net.IP, ports []*api.ServicePort, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error) {
	// Concerto LB does not support session affinity
	if affinityType != api.ServiceAffinityNone {
		return nil, fmt.Errorf("Unsupported load balancer affinity: %v", affinityType)
	}
	// Can not specify a public IP for the LB
	if loadBalancerIP != nil {
		return nil, fmt.Errorf("can not specify an IP address for a Concerto load balancer")
	}
	// Dont support multi-port
	if len(ports) != 1 {
		return nil, fmt.Errorf("Concerto load balancer only supports one single port")
	}

	// Check previous existence
	lb, err := c.service.GetLoadBalancerByName(name)
	if err != nil {
		return nil, fmt.Errorf("error checking existence of load balancer '%s' in Concerto: %v", name, err)
	}

	if lb == nil {
		// It does not exist: create it
		lb, err = c.createLoadBalancer(name, ports, hosts)
		if err != nil {
			return nil, fmt.Errorf("error creating load balancer '%s' in Concerto: %v", name, err)
		}
	} else {
		// It already exists: update it
		err = c.UpdateLoadBalancer(name, region, hosts)
		if err != nil {
			return nil, fmt.Errorf("error updating load balancer '%s' in Concerto: %v", name, err)
		}
	}

	return toStatus(lb), nil
}

func (c *ConcertoCloud) createLoadBalancer(name string, ports []*api.ServicePort, hosts []string) (*ConcertoLoadBalancer, error) {
	// Create the LB
	port := ports[0].Port // The port that will be exposed on the service.
	// targetPort := ports[0].TargetPort // Optional: The target port on pods selected by this service
	nodePort := ports[0].NodePort // The port on each node on which this service is exposed.
	lb, err := c.service.CreateLoadBalancer(name, port, nodePort)
	if err != nil {
		return nil, fmt.Errorf("error creating Concerto load balancer (%v/%v/%v): %v", name, port, nodePort, err)
	}

	// Add the corresponding nodes
	if len(hosts) > 0 {
		ipAddresses, err := c.hostsNamesToIPs(hosts)
		if err != nil {
			return nil, fmt.Errorf("error converting hostnames to IP addresses: %v", err)
		}
		err = c.service.RegisterInstancesWithLoadBalancer(lb.Id, ipAddresses)
		if err != nil {
			return nil, fmt.Errorf("error registering instances with load balancer: %v", err)
		}
	}

	return lb, nil
}

// UpdateLoadBalancer implementation for Flexiant Concerto.
func (c *ConcertoCloud) UpdateLoadBalancer(name, region string, hosts []string) error {
	// Get the load balancer
	lb, err := c.service.GetLoadBalancerByName(name)
	if err != nil {
		return fmt.Errorf("error getting Concerto load balancer by name '%s': %v", name, err)
	}
	// Get the LB nodes
	currentNodes, err := c.service.GetLoadBalancerNodesAsIPs(lb.Id)
	if err != nil {
		return fmt.Errorf("error getting nodes (as IP addresses) of Concerto load balancer '%s': %v", lb.Id, err)
	}

	// Calculate nodes to deregister
	wantedNodes, err := c.hostsNamesToIPs(hosts)
	if err != nil {
		return fmt.Errorf("error converting hostnames to IP addresses: %v", err)
	}
	nodesToRemove := subtractStringArrays(currentNodes, wantedNodes)
	// Calculate nodes to be registered
	nodesToAdd := subtractStringArrays(wantedNodes, currentNodes)
	// Do the thing
	err = c.service.DeregisterInstancesFromLoadBalancer(lb.Id, nodesToRemove)
	if err != nil {
		return fmt.Errorf("error deregistering instances (%v) from load balancer (%v) : %v", nodesToRemove, lb.Id, err)
	}
	err = c.service.RegisterInstancesWithLoadBalancer(lb.Id, nodesToAdd)
	if err != nil {
		return fmt.Errorf("error registering instances (%v) with load balancer (%v) : %v", nodesToAdd, lb.Id, err)
	}

	return nil
}

// EnsureLoadBalancerDeleted implementation for Flexiant Concerto.
func (c *ConcertoCloud) EnsureLoadBalancerDeleted(name, region string) error {
	// Get the LB
	lb, err := c.service.GetLoadBalancerByName(name)
	if err != nil {
		return fmt.Errorf("error getting Concerto load balancer by name '%s': %v", name, err)
	}
	if lb == nil {
		return nil
	}
	return c.service.DeleteLoadBalancerById(lb.Id)
}

func (c *ConcertoCloud) hostsNamesToIPs(hosts []string) ([]string, error) {
	var ips []string
	instances, err := c.service.GetInstanceList()
	if err != nil {
		return nil, fmt.Errorf("error getting Concerto instance list : %v", err)
	}
	for _, name := range hosts {
		found := false
		for _, instance := range instances {
			if instance.Name == name {
				ips = append(ips, instance.PublicIP)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("could not find instance: %s", name)
		}
	}
	return ips, nil
}
