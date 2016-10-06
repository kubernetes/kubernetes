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

package qingcloud

// See https://docs.qingcloud.com/api/lb/index.html
// qingcloud loadbalancer and instance have a default strict Security Group(firewall),
// its only allow SSH and PING. So, first of all, you shoud manually add correct rules
// for all nodes and loadbalancers. You can simply add a rule by pass all tcp port traffic.
// The loadbalancers also need at least one EIP before create it, please allocate some EIPs,
// and set them in service annotation ServiceAnnotationLoadBalancerEipIds.

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/magicshui/qingcloud-go"
	lb "github.com/magicshui/qingcloud-go/loadbalancer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	// ServiceAnnotationLoadBalancerEipIds is the annotation used on the
	// service to indicate that we want a qingcloud loadbalancer allocate by eip id.
	// The value is the EIP ID list split by ',' like "eip-j38f2h3h,eip-ornz2xq7", you must set it.
	ServiceAnnotationLoadBalancerEipIds = "service.beta.kubernetes.io/qingcloud-load-balancer-eip-ids"

	// ServiceAnnotationLoadBalancerType is the annotation used on the
	// service to indicate that we want a qingcloud loadbalancer type.
	// value "0" means the LB can max support 5000 concurrency connections, it's default type.
	// value "1" means the LB can max support 20000 concurrency connections.
	// value "2" means the LB can max support 40000 concurrency connections.
	// value "3" means the LB can max support 100000 concurrency connections.
	ServiceAnnotationLoadBalancerType = "service.beta.kubernetes.io/qingcloud-load-balancer-type"
)

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (qc *Qingcloud) GetLoadBalancer(clusterName string, service *api.Service) (status *api.LoadBalancerStatus, exists bool, err error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(3).Infof("GetLoadBalancer(%v, %v)", clusterName, loadBalancerName)

	loadbalancer, exists, err := qc.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return nil, false, err
	}
	if !exists {
		return nil, false, nil
	}

	status = &api.LoadBalancerStatus{}
	for _, ip := range loadbalancer.Eips {
		status.Ingress = append(status.Ingress, api.LoadBalancerIngress{IP: ip.EipAddr})
	}

	return status, true, nil
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
// To create a LoadBalancer for kubernetes, we do the following:
// 1. create a qingcloud loadbalancer;
// 2. create listeners for the new loadbalancer, number of listeners = number of service ports;
// 3. add backends to the new loadbalancer.
func (qc *Qingcloud) EnsureLoadBalancer(clusterName string, apiService *api.Service, hosts []string) (*api.LoadBalancerStatus, error) {
	glog.V(3).Infof("EnsureLoadBalancer(%v, %v, %v)", clusterName, apiService, hosts)

	tcpPortNum := 0
	for _, port := range apiService.Spec.Ports {
		if port.Protocol == api.ProtocolUDP {
			glog.Warningf("qingcloud not support udp port, skip [%v]", port.Port)
		} else {
			tcpPortNum++
		}
	}
	if tcpPortNum == 0 {
		return nil, fmt.Errorf("requested load balancer with no tcp ports")
	}

	// Qingcloud does not support user-specified ip addr for LB. We just
	// print some log and ignore the public ip.
	if apiService.Spec.LoadBalancerIP != "" {
		glog.Warningf("Public IP[%v] cannot be specified for qingcloud LB", apiService.Spec.LoadBalancerIP)
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)

	glog.V(2).Infof("Checking if qingcloud load balancer already exists: %s", loadBalancerName)
	loadbalancer, exists, err := qc.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return nil, fmt.Errorf("Error checking if qingcloud load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		if err = qc.deleteLoadBalancer(loadbalancer.LoadbalancerID); err != nil {
			glog.V(1).Infof("Deleted loadbalancer '%s' error before creating: %v", loadBalancerName, err)
			return nil, err
		}

		glog.V(2).Infof("Starting delete loadbalancer '%s' before creating", loadBalancerName)
		err = qc.waitLoadBalancerDelete(loadbalancer.LoadbalancerID, waitActiveTimeout)
		if err != nil {
			glog.Error(err)
			return nil, err
		}
		glog.V(2).Infof("Done, deleted loadbalancer '%s'", loadBalancerName)
	}

	lbType := apiService.Annotations[ServiceAnnotationLoadBalancerType]
	if lbType != "0" && lbType != "1" && lbType != "2" && lbType != "3" {
		lbType = "0"
	}
	loadBalancerType, _ := strconv.Atoi(lbType)

	lbEipIds := apiService.Annotations[ServiceAnnotationLoadBalancerEipIds]
	if lbEipIds == "" {
		return nil, fmt.Errorf("you must set %v", ServiceAnnotationLoadBalancerEipIds)
	}
	loadBalancerEipIds := strings.Split(lbEipIds, ",")

	loadBalancerId, err := qc.createLoadBalancer(loadBalancerName, loadBalancerType, loadBalancerEipIds)
	if err != nil {
		glog.Errorf("Error creating loadbalancer '%s': %v", loadBalancerName, err)
		return nil, err
	}

	glog.Infof("Create loadbalancer '%s' in zone '%s'", loadBalancerName, qc.zone)

	balanceMode := "roundrobin"
	if apiService.Spec.SessionAffinity == api.ServiceAffinityClientIP {
		balanceMode = "source"
	}

	instances := []string{}
	for _, hostname := range hosts {
		instances = append(instances, nodeNameToInstanceId(types.NodeName(hostname)))
	}

	// For every port(qingcloud only support tcp), we need a listener.
	for _, port := range apiService.Spec.Ports {
		if port.Protocol == api.ProtocolUDP {
			continue
		}

		glog.V(3).Infof("Create a listener for tcp port: %v", port)

		listenerID, err := qc.addLoadBalancerListener(loadBalancerId, int64(port.Port), balanceMode)
		if err != nil {
			glog.Errorf("Error create loadbalancer TCP listener (LoadBalancerId:'%s', Port: '%v'): %v", loadBalancerId, port, err)
			return nil, err
		}
		glog.Infof("Created LoadBalancerTCPListener (LoadBalancerId:'%s', Port: '%v')", loadBalancerId, port)

		err = qc.addLoadBalancerBackends(map[string][]int64{listenerID: {int64(port.NodePort)}}, instances)
		if err != nil {
			glog.Errorf("Couldn't add backend servers '%v' to loadbalancer '%v': %v", instances, loadBalancerName, err)
			return nil, err
		}
		glog.V(3).Infof("Added backend servers '%v' to loadbalancer '%s'", instances, loadBalancerName)
	}

	eips, err := qc.waitLoadBalancerActive(loadBalancerId, waitActiveTimeout)
	if err != nil {
		glog.Error(err)
		return nil, err
	}

	// enforce the loadbalancer config
	err = qc.updateLoadBalancer(loadBalancerId)
	if err != nil {
		glog.Errorf("Couldn't update loadbalancer '%v'", loadBalancerId)
		return nil, err
	}

	status := &api.LoadBalancerStatus{}
	for _, ip := range eips {
		status.Ingress = append(status.Ingress, api.LoadBalancerIngress{IP: ip})
	}
	glog.Infof("Start loadbalancer '%v', ingress ip '%v'", loadBalancerName, status.Ingress)

	return status, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (qc *Qingcloud) UpdateLoadBalancer(clusterName string, service *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(3).Infof("UpdateLoadBalancer(%v, %v, %v)", clusterName, loadBalancerName, hosts)

	loadbalancer, exists, err := qc.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("Couldn't find load balancer by name '%s' in zone '%s'", loadBalancerName, qc.zone)
	}

	// Expected instances for the load balancer.
	expected := sets.NewString()
	for _, hostname := range hosts {
		instanceID := nodeNameToInstanceId(types.NodeName(hostname))
		expected.Insert(instanceID)
	}

	listenerBackendPorts := map[string][]int64{}
	instanceBackendIDs := map[string][]string{}
	loadbalancerListeners, err := qc.getLoadBalancerListeners(loadbalancer.LoadbalancerID)
	if err != nil {
		glog.Errorf("Couldn't get loadbalancer '%v' err: %v", loadBalancerName, err)
		return err
	}
	for _, listener := range loadbalancerListeners {
		nodePort, found := getNodePort(service, int32(listener.ListenerPort), api.ProtocolTCP)
		if !found {
			continue
		}
		listenerBackendPorts[listener.LoadbalancerListenerID] = append(listenerBackendPorts[listener.LoadbalancerListenerID], int64(nodePort))

		for _, backend := range listener.Backends {
			instanceBackendIDs[backend.ResourceID] = append(instanceBackendIDs[backend.ResourceID], backend.LoadbalancerBackendID)
		}
	}

	// Actual instances of the load balancer.
	actual := sets.StringKeySet(instanceBackendIDs)
	addInstances := expected.Difference(actual)
	removeInstances := actual.Difference(expected)

	var needUpdate bool

	glog.V(3).Infof("For the loadbalancer, expected instances: %v, actual instances: %v, need to remove instances: %v, need to add instances: %v", expected, actual, removeInstances, addInstances)

	if len(addInstances) > 0 {
		instances := addInstances.List()
		err := qc.addLoadBalancerBackends(listenerBackendPorts, instances)
		if err != nil {
			glog.Errorf("Couldn't add backend servers '%v' to loadbalancer '%v': %v", instances, loadBalancerName)
			return err
		}
		needUpdate = true
		glog.V(1).Infof("Instances '%v' added to loadbalancer %s", instances, loadBalancerName)
	}

	if len(removeInstances) > 0 {
		instances := removeInstances.List()

		backendIDs := make([]string, 0, len(instances))
		for _, instance := range instances {
			backendIDs = append(backendIDs, instanceBackendIDs[instance]...)
		}
		err := qc.deleteLoadBalancerBackends(backendIDs)
		if err != nil {
			glog.Errorf("Couldn't remove backend servers '%v' from loadbalancer '%v': %v", instances, loadBalancerName)
			return err
		}
		needUpdate = true
		glog.V(1).Infof("Instances '%v' removed from loadbalancer %s", instances, loadBalancerName)
	}

	if needUpdate {
		glog.V(1).Info("Enforce the loadbalancer update backends config")
		return qc.updateLoadBalancer(loadbalancer.LoadbalancerID)
	}

	glog.V(3).Info("Skip update loadbalancer backends")

	return nil
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (qc *Qingcloud) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(3).Infof("EnsureLoadBalancerDeleted(%v, %v)", clusterName, loadBalancerName)

	loadbalancer, exists, err := qc.getLoadBalancerByName(loadBalancerName)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}

	err = qc.deleteLoadBalancer(loadbalancer.LoadbalancerID)
	if err != nil {
		return err
	}

	glog.Infof("Delete loadbalancer '%s' in zone '%s'", loadBalancerName, qc.zone)

	return nil
}

func (qc *Qingcloud) createLoadBalancer(lbName string, lbType int, lbEipIds []string) (string, error) {
	eipsN := qingcloud.NumberedString{}
	eipsN.Add(lbEipIds...)
	loadbalancerType := qingcloud.Integer{}
	loadbalancerType.Set(lbType)
	loadbalancerName := qingcloud.String{}
	loadbalancerName.Set(lbName)
	resp, err := qc.lbClient.CreateLoadBalancer(lb.CreateLoadBalancerRequest{
		EipsN:            eipsN,
		LoadbalancerType: loadbalancerType,
		LoadbalancerName: loadbalancerName,
	})
	if err != nil {
		return "", err
	}

	return resp.LoadbalancerId, nil
}

func (qc *Qingcloud) deleteLoadBalancer(loadBalancerID string) error {
	loadbalancersN := qingcloud.NumberedString{}
	loadbalancersN.Add(loadBalancerID)
	_, err := qc.lbClient.DeleteLoadBalancers(lb.DeleteLoadBalancersRequest{
		LoadbalancersN: loadbalancersN,
	})

	return err
}

func (qc *Qingcloud) addLoadBalancerBackends(listenerBackendPorts map[string][]int64, instanceIDs []string) error {
	backendsNResourceId := qingcloud.NumberedString{}
	backendsNResourceId.Add(instanceIDs...)
	backendsNWeight := qingcloud.NumberedInteger{}
	backendsNWeight.Add(1)
	for listenerID, backendPorts := range listenerBackendPorts {
		loadbalancerListener := qingcloud.String{}
		loadbalancerListener.Set(listenerID)
		backendsNPort := qingcloud.NumberedInteger{}
		backendsNPort.Add(backendPorts...)

		_, err := qc.lbClient.AddLoadBalancerBackends(lb.AddLoadBalancerBackendsRequest{
			LoadbalancerListener: loadbalancerListener,
			BackendsNResourceId:  backendsNResourceId,
			BackendsNPort:        backendsNPort,
			BackendsNWeight:      backendsNWeight,
		})
		if err != nil {
			return err
		}
	}

	return nil
}

func (qc *Qingcloud) deleteLoadBalancerBackends(loadbalancerBackends []string) error {
	loadbalancerBackendsN := qingcloud.NumberedString{}
	loadbalancerBackendsN.Add(loadbalancerBackends...)
	_, err := qc.lbClient.DeleteLoadBalancerBackends(lb.DeleteLoadBalancerBackendsRequest{
		LoadbalancerBackendsN: loadbalancerBackendsN,
	})

	return err
}

func (qc *Qingcloud) addLoadBalancerListener(loadBalancerID string, listenerPort int64, balanceMode string) (string, error) {
	loadbalancer := qingcloud.String{}
	loadbalancer.Set(loadBalancerID)
	listenersNListenerPort := qingcloud.NumberedInteger{}
	listenersNListenerPort.Add(listenerPort)
	protocol := qingcloud.NumberedString{}
	protocol.Add("tcp")
	listenersNBalanceMode := qingcloud.NumberedString{}
	listenersNBalanceMode.Add(balanceMode)
	resp, err := qc.lbClient.AddLoadBalancerListeners(lb.AddLoadBalancerListenersRequest{
		Loadbalancer:               loadbalancer,
		ListenersNListenerPort:     listenersNListenerPort,
		ListenersNListenerProtocol: protocol,
		ListenersNBackendProtocol:  protocol,
		ListenersNBalanceMode:      listenersNBalanceMode,
	})
	if err != nil {
		return "", err
	}

	return resp.LoadbalancerListeners[0], nil
}

func (qc *Qingcloud) getLoadBalancerByName(name string) (*lb.Loadbalancer, bool, error) {
	statusN := qingcloud.NumberedString{}
	statusN.Add("pending", "active", "stopped")
	searchWord := qingcloud.String{}
	searchWord.Set(name)
	resp, err := qc.lbClient.DescribeLoadBalancers(lb.DescribeLoadBalancersRequest{
		StatusN:    statusN,
		SearchWord: searchWord,
	})
	if err != nil {
		return nil, false, err
	}
	if len(resp.LoadbalancerSet) == 0 {
		return nil, false, nil
	}

	return &resp.LoadbalancerSet[0], true, nil
}

func (qc *Qingcloud) getLoadBalancerByID(id string) (*lb.Loadbalancer, bool, error) {
	loadbalancersN := qingcloud.NumberedString{}
	loadbalancersN.Add(id)
	statusN := qingcloud.NumberedString{}
	statusN.Add("pending", "active", "stopped")
	resp, err := qc.lbClient.DescribeLoadBalancers(lb.DescribeLoadBalancersRequest{
		LoadbalancersN: loadbalancersN,
		StatusN:        statusN,
	})
	if err != nil {
		return nil, false, err
	}
	if len(resp.LoadbalancerSet) == 0 {
		return nil, false, nil
	}

	return &resp.LoadbalancerSet[0], true, nil
}

func (qc *Qingcloud) getLoadBalancerListeners(loadBalancerID string) ([]lb.LoadbalancerListener, error) {
	loadbalancer := qingcloud.String{}
	loadbalancer.Set(loadBalancerID)
	verbose := qingcloud.Integer{}
	verbose.Set(1)
	limit := qingcloud.Integer{}
	limit.Set(100)

	loadbalancerListeners := []lb.LoadbalancerListener{}

	for i := 0; ; i += 100 {
		offset := qingcloud.Integer{}
		offset.Set(i)
		resp, err := qc.lbClient.DescribeLoadBalancerListeners(lb.DescribeLoadBalancerListenersRequest{
			Loadbalancer: loadbalancer,
			Verbose:      verbose,
			Offset:       offset,
			Limit:        limit,
		})
		if err != nil {
			return nil, err
		}
		if len(resp.LoadbalancerListenerSet) == 0 {
			break
		}

		loadbalancerListeners = append(loadbalancerListeners, resp.LoadbalancerListenerSet...)
		if len(loadbalancerListeners) >= resp.TotalCount {
			break
		}
	}

	return loadbalancerListeners, nil
}

// enforce the loadbalancer config
func (qc *Qingcloud) updateLoadBalancer(loadBalancerID string) error {
	loadbalancersN := qingcloud.NumberedString{}
	loadbalancersN.Add(loadBalancerID)
	_, err := qc.lbClient.UpdateLoadBalancers(lb.UpdateLoadBalancersRequest{
		LoadbalancersN: loadbalancersN,
	})

	return err
}

func (qc *Qingcloud) waitLoadBalancerActive(loadBalancerID string, timeout time.Duration) ([]string, error) {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			loadbalancer, exist, err := qc.getLoadBalancerByID(loadBalancerID)
			if err != nil {
				return nil, err
			}
			if !exist {
				return nil, fmt.Errorf("Couldn't found %v while waiting it active", loadBalancerID)
			}
			if loadbalancer.Status == "active" && loadbalancer.TransitionStatus == "" {
				eips := []string{}
				for _, eip := range loadbalancer.Eips {
					eips = append(eips, eip.EipAddr)
				}
				return eips, nil
			}
		case <-timer.C:
			return nil, fmt.Errorf("Waiting loadbalancer '%v' timeout", loadBalancerID)
		}
	}
}

func (qc *Qingcloud) waitLoadBalancerDelete(loadBalancerID string, timeout time.Duration) error {
	ticker := time.NewTicker(checkSleepDuration)
	defer ticker.Stop()
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			_, exist, err := qc.getLoadBalancerByID(loadBalancerID)
			if err != nil {
				return err
			}
			if !exist {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("Deleting loadbalancer '%v' timeout", loadBalancerID)
		}
	}
}
