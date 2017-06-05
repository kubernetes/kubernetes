/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package aliyun

import (
	"fmt"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/slb"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/sets"
)

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (aly *Aliyun) GetLoadBalancer(clusterName string, service *api.Service) (status *api.LoadBalancerStatus, exists bool, err error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(clusterName, service)

	loadbalancer, exists, err := aly.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return nil, false, fmt.Errorf("Couldn't get load balancer by name '%s' in region '%s': %v", loadBalancerName, aly.regionID, err)
	}

	glog.V(4).Infof("GetLoadBalancer(%s, %s): %v", clusterName, service, loadbalancer)

	if !exists {
		glog.Infof("Couldn't find the loadbalancer {clusterName: '%s', loadBalancerName: '%s', loadbalancer: '%v', region: '%v'}", clusterName, loadBalancerName, loadbalancer, aly.regionID)
		return nil, false, nil
	}

	status = &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: loadbalancer.Address}}

	return status, true, nil
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
// To create a LoadBalancer for kubernetes, we do the following:
// 1. create a aliyun SLB loadbalancer;
// 2. create listeners for the new loadbalancer, number of listeners = number of service ports;
// 3. add backends to the new loadbalancer.
func (aly *Aliyun) EnsureLoadBalancer(clusterName string, apiService *api.Service, hosts []string) (*api.LoadBalancerStatus, error) {
	glog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)", clusterName, apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, hosts, apiService.Annotations)

	if apiService.Spec.SessionAffinity != api.ServiceAffinityNone {
		// Aliyun supports sticky sessions, but only when configured for HTTP/HTTPS (cookies based).
		// But Kubernetes Services support TCP and UDP for protocols.
		// Although session affinity is calculated in kube-proxy, where it determines which pod to
		// response a request, we still need to hit the same kube-proxy (the node). Other kube-proxy
		// do not have the knowledge.
		return nil, fmt.Errorf("Unsupported load balancer affinity: %v", apiService.Spec.SessionAffinity)
	}

	if len(apiService.Spec.Ports) == 0 {
		return nil, fmt.Errorf("requested load balancer with no ports")
	}

	// Aliyun does not support user-specified ip addr for LB. We just
	// print some log and ignore the public ip.
	if apiService.Spec.LoadBalancerIP != "" {
		glog.Warning("Public IP cannot be specified for aliyun SLB")
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(clusterName, apiService)

	glog.V(2).Infof("Checking if aliyun load balancer already exists: %s", loadBalancerName)
	_, exists, err := aly.GetLoadBalancer(clusterName, apiService)
	if err != nil {
		return nil, fmt.Errorf("Error checking if aliyun load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		err := aly.EnsureLoadBalancerDeleted(clusterName, apiService)
		if err != nil {
			return nil, fmt.Errorf("Error deleting existing aliyun load balancer: %v", err)
		}

		glog.V(2).Infof("Deleted loadbalancer '%s' before creating in region '%s'", loadBalancerName, aly.regionID)
	}

	lb_response, err := aly.createLoadBalancer(loadBalancerName)
	if err != nil {
		glog.Errorf("Error creating loadbalancer '%s': %v", loadBalancerName, err)
		return nil, err
	}

	glog.Infof("Create loadbalancer '%s' in region '%s'", loadBalancerName, aly.regionID)

	// For the public network instance charged per fixed bandwidth
	// the sum of bandwidth peaks allocated to different Listeners
	// cannot exceed the Bandwidth value set when creating the
	// Server Load Balancer instance, and the Bandwidth value on Listener
	// cannot be set to -1
	//
	// For the public network instance charged per traffic consumed,
	// the Bandwidth on Listener can be set to -1, indicating the
	// bandwidth peak is unlimited.
	bandwidth := -1
	if len(apiService.Spec.Ports) > 0 && aly.lbOpts.AddressType == slb.InternetAddressType && aly.lbOpts.InternetChargeType == common.InternetChargeType("paybybandwidth") {
		bandwidth = aly.lbOpts.Bandwidth / len(apiService.Spec.Ports)
	}

	// For every port, we need a listener.
	for _, port := range apiService.Spec.Ports {
		glog.V(4).Infof("Create a listener for port: %v", port)

		if port.Protocol == api.ProtocolTCP {
			err := aly.createLoadBalancerTCPListener(lb_response.LoadBalancerId, port, bandwidth)
			if err != nil {
				glog.Errorf("Error create loadbalancer TCP listener (LoadBalancerId:'%s', Port: '%v', Bandwidth: '%d'): %v", lb_response.LoadBalancerId, port, bandwidth, err)
				return nil, err
			}
			glog.Infof("Created LoadBalancerTCPListener (LoadBalancerId:'%s', Port: '%v', Bandwidth: '%d')", lb_response.LoadBalancerId, port, bandwidth)
		} else if port.Protocol == api.ProtocolUDP {
			err := aly.createLoadBalancerUDPListener(lb_response.LoadBalancerId, port, bandwidth)
			if err != nil {
				glog.Errorf("Error create loadbalancer UDP listener (LoadBalancerId:'%s', Port: '%v', Bandwidth: '%d'): %v", lb_response.LoadBalancerId, port, bandwidth, err)
				return nil, err
			}
			glog.Infof("Created LoadBalancerUDPListener (LoadBalancerId:'%s', Port: '%v', Bandwidth: '%d')", lb_response.LoadBalancerId, port, bandwidth)
		}
	}

	instanceIDs := []string{}
	for _, hostname := range hosts {
		instanceID := nameToInstanceId(hostname)
		instanceIDs = append(instanceIDs, instanceID)
	}

	err = aly.addBackendServers(lb_response.LoadBalancerId, instanceIDs)
	if err != nil {
		glog.Errorf("Couldn't add backend servers '%v' to loadbalancer '%v': %v", instanceIDs, loadBalancerName, err)
		return nil, err
	}

	glog.V(4).Infof("Added backend servers '%v' to loadbalancer '%s'", instanceIDs, loadBalancerName)

	err = aly.setLoadBalancerStatus(lb_response.LoadBalancerId, slb.ActiveStatus)
	if err != nil {
		glog.Errorf("Couldn't activate loadbalancer '%v'", lb_response.LoadBalancerId)
		return nil, err
	}

	glog.Infof("Activated loadbalancer '%v', ingress ip '%v'", loadBalancerName, lb_response.Address)

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: lb_response.Address}}

	return status, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (aly *Aliyun) UpdateLoadBalancer(clusterName string, service *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(clusterName, service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v)", clusterName, loadBalancerName, hosts)

	loadbalancer, exists, err := aly.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return fmt.Errorf("Couldn't get load balancer by name '%s' in region '%s': %v", loadBalancerName, aly.regionID, err)
	}

	if !exists {
		return fmt.Errorf("Couldn't find load balancer by name '%s' in region '%s'", loadBalancerName, aly.regionID)
	}

	// Expected instances for the load balancer.
	expected := sets.NewString()
	for _, hostname := range hosts {
		instanceID := nameToInstanceId(hostname)
		expected.Insert(instanceID)
	}

	// Actual instances of the load balancer.
	actual := sets.NewString()
	lb_attribute, err := aly.getLoadBalancerAttribute(loadbalancer.LoadBalancerId)
	if err != nil {
		glog.Errorf("Couldn't get loadbalancer '%v' attribute: %v", loadBalancerName, err)
		return err
	}
	for _, backendserver := range lb_attribute.BackendServers.BackendServer {
		actual.Insert(backendserver.ServerId)
	}

	addInstances := expected.Difference(actual)
	removeInstances := actual.Difference(expected)

	glog.V(4).Infof("For the loadbalancer, expected instances: %v, actual instances: %v, need to remove instances: %v, need to add instances: %v", expected, actual, removeInstances, addInstances)

	if len(addInstances) > 0 {
		instanceIDs := addInstances.List()
		err := aly.addBackendServers(loadbalancer.LoadBalancerId, instanceIDs)
		if err != nil {
			glog.Errorf("Couldn't add backend servers '%v' to loadbalancer '%v': %v", instanceIDs, loadBalancerName)
			return err
		}
		glog.V(1).Infof("Instances '%v' added to loadbalancer %s", instanceIDs, loadBalancerName)
	}

	if len(removeInstances) > 0 {
		instanceIDs := removeInstances.List()
		err := aly.removeBackendServers(loadbalancer.LoadBalancerId, instanceIDs)
		if err != nil {
			glog.Errorf("Couldn't remove backend servers '%v' from loadbalancer '%v': %v", instanceIDs, loadBalancerName)
			return err
		}
		glog.V(1).Infof("Instances '%v' removed from loadbalancer %s", instanceIDs, loadBalancerName)
	}

	return nil
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (aly *Aliyun) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(clusterName, service)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v, %v)", clusterName, loadBalancerName)

	loadbalancer, exists, err := aly.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return fmt.Errorf("Couldn't get load balancer by name '%s' in region '%s': %v", loadBalancerName, aly.regionID, err)
	}

	if !exists {
		glog.Infof(" Loadbalancer '%s', already deleted in region '%s'", loadBalancerName, aly.regionID)
		return nil
	}

	err = aly.deleteLoadBalancer(loadbalancer.LoadBalancerId)
	if err != nil {
		return fmt.Errorf("Error deleting load balancer by name '%s' in region '%s': %v", loadBalancerName, aly.regionID, err)
	}

	glog.Infof("Delete loadbalancer '%s' in region '%s'", loadBalancerName, aly.regionID)

	return nil
}

func (aly *Aliyun) createLoadBalancer(name string) (response *slb.CreateLoadBalancerResponse, err error) {
	args := slb.CreateLoadBalancerArgs{
		RegionId:           common.Region(aly.regionID),
		LoadBalancerName:   name,
		AddressType:        aly.lbOpts.AddressType,
		InternetChargeType: aly.lbOpts.InternetChargeType,
		Bandwidth:          aly.lbOpts.Bandwidth,
	}
	response, err = aly.slbClient.CreateLoadBalancer(&args)
	if err != nil {
		glog.Errorf("Couldn't CreateLoadBalancer(%v): %v", args, err)
		return nil, err
	}

	glog.V(4).Infof("CreateLoadBalancer(%v): %v", args, response)

	return response, nil
}

func (aly *Aliyun) deleteLoadBalancer(loadBalancerID string) error {
	return aly.slbClient.DeleteLoadBalancer(loadBalancerID)
}

// Add backend servers to the specified load balancer.
func (aly *Aliyun) addBackendServers(loadbalancerID string, instanceIDs []string) error {
	backendServers := []slb.BackendServerType{}
	for index, instanceID := range instanceIDs {
		backendServers = append(backendServers,
			slb.BackendServerType{
				ServerId: instanceID,
				Weight:   100,
			},
		)

		// For AddBackendServer, The maximum number of elements in backendServers List is 20.
		if index%20 == 19 {
			_, err := aly.slbClient.AddBackendServers(loadbalancerID, backendServers)
			if err != nil {
				glog.Errorf("Couldn't AddBackendServers(%v, %v): %v", loadbalancerID, backendServers, err)
				return err
			}
			backendServers = backendServers[0:0]
		}
	}

	_, err := aly.slbClient.AddBackendServers(loadbalancerID, backendServers)
	if err != nil {
		glog.Errorf("Couldn't AddBackendServers(%v, %v): %v", loadbalancerID, backendServers, err)
		return err
	}

	glog.V(4).Infof("AddBackendServers(%v, %v)", loadbalancerID, backendServers)

	return nil
}

// Remove backend servers from the specified load balancer.
func (aly *Aliyun) removeBackendServers(loadBalancerID string, instanceIDs []string) error {
	_, err := aly.slbClient.RemoveBackendServers(loadBalancerID, instanceIDs)
	if err != nil {
		glog.Errorf("Couldn't RemoveBackendServers(%v, %v): %v", loadBalancerID, instanceIDs, err)
		return err
	}

	return nil
}

func (aly *Aliyun) createLoadBalancerTCPListener(loadBalancerID string, port api.ServicePort, bandwidth int) error {
	args := slb.CreateLoadBalancerTCPListenerArgs{
		LoadBalancerId:    loadBalancerID,
		ListenerPort:      int(port.Port),
		BackendServerPort: int(port.NodePort),
		// Bandwidth peak of Listener Value: -1 | 1 - 1000 Mbps, default is -1.
		Bandwidth: bandwidth,
	}
	return aly.slbClient.CreateLoadBalancerTCPListener(&args)
}

func (aly *Aliyun) createLoadBalancerUDPListener(loadBalancerID string, port api.ServicePort, bandwidth int) error {
	args := slb.CreateLoadBalancerUDPListenerArgs{
		LoadBalancerId:    loadBalancerID,
		ListenerPort:      int(port.Port),
		BackendServerPort: int(port.NodePort),
		Bandwidth:         bandwidth,
	}
	return aly.slbClient.CreateLoadBalancerUDPListener(&args)
}

func (aly *Aliyun) getLoadBalancerByName(name string) (loadbalancer *slb.LoadBalancerType, exists bool, err error) {
	// Find all the loadbalancers in the current region.
	args := slb.DescribeLoadBalancersArgs{
		RegionId: common.Region(aly.regionID),
	}
	loadbalancers, err := aly.slbClient.DescribeLoadBalancers(&args)
	if err != nil {
		glog.Errorf("Couldn't DescribeLoadBalancers(%v): %v", args, err)
		return nil, false, err
	}
	glog.V(4).Infof("getLoadBalancerByName(%s) in region '%s': %v", name, aly.regionID, loadbalancers)

	// Find the specified load balancer with the matching name
	for _, lb := range loadbalancers {
		if lb.LoadBalancerName == name {
			glog.V(4).Infof("Find loadbalancer(%s) in region '%s'", name, aly.regionID)
			return &lb, true, nil
		}
	}

	glog.Infof("Couldn't find loadbalancer by name '%s'", name)

	return nil, false, nil
}

func (aly *Aliyun) getLoadBalancerAttribute(loadBalancerID string) (loadbalancer *slb.LoadBalancerType, err error) {
	loadbalancer, err = aly.slbClient.DescribeLoadBalancerAttribute(loadBalancerID)
	if err != nil {
		glog.Errorf("Couldn't DescribeLoadBalancerAttribute(%s): %v", loadBalancerID, err)
		return nil, err
	}

	return loadbalancer, nil
}

func (aly *Aliyun) setLoadBalancerStatus(loadBalancerID string, status slb.Status) (err error) {
	return aly.slbClient.SetLoadBalancerStatus(loadBalancerID, status)
}
