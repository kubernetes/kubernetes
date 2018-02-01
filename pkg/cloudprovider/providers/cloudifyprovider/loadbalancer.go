/*
Copyright (c) 2017 GigaSpaces Technologies Ltd. All rights reserved

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

package cloudifyprovider

import (
	"fmt"
	cloudify "github.com/cloudify-incubator/cloudify-rest-go-client/cloudify"
	"github.com/golang/glog"
	api "k8s.io/api/core/v1"
)

// Balancer - struct with connection settings
type Balancer struct {
	deployment string
	scaleGroup string
	client     *cloudify.Client
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (r *Balancer) UpdateLoadBalancer(clusterName string, service *api.Service, nodes []*api.Node) error {
	glog.Errorf("UpdateLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	var finalState, err = r.updateNode(clusterName, service, nodes)
	glog.Errorf("%v: Final state: ", service.Name, finalState)

	return err
}

func (r *Balancer) getLoadBalancerNode(clusterName string, service *api.Service) *cloudify.NodeInstance {
	var name = ""
	var nameSpace = ""
	if service != nil {
		name = string(service.Name)
		nameSpace = string(service.Namespace)
	}
	var params = map[string]string{}
	// Add filter by deployment
	params["deployment_id"] = r.deployment

	instances, err := r.client.GetLoadBalancerInstances(params, clusterName, nameSpace, name,
		"cloudify.nodes.ApplicationServer.kubernetes.LoadBalancer")
	if err != nil {
		glog.Infof("Not found instances: %+v", err)
		return nil
	}
	if len(instances.Items) > 0 {
		node := instances.Items[0]
		glog.Errorf("Found '%v' for '%v'", node.ID, name)
		return &node
	}
	return nil
}

func (r *Balancer) nodeToState(nodeInstance *cloudify.NodeInstance) (status *api.LoadBalancerStatus) {
	if nodeInstance.RuntimeProperties != nil {
		// look as we found something
		ingress := []api.LoadBalancerIngress{}

		ingressNode := api.LoadBalancerIngress{}

		// hostname
		if v, ok := nodeInstance.RuntimeProperties["hostname"]; ok == true {
			switch v.(type) {
			case string:
				{
					ingressNode.Hostname = v.(string)
				}
			}
		}

		// ip
		if v, ok := nodeInstance.RuntimeProperties["public_ip"]; ok == true {
			switch v.(type) {
			case string:
				{
					ingressNode.IP = v.(string)
				}
			}
		}

		ingress = append(ingress, ingressNode)

		glog.Errorf("Status %v for return: %+v", nodeInstance.ID, ingress)
		return &api.LoadBalancerStatus{ingress}
	}
	return nil
}

// GetLoadBalancer returns whether the specified load balancer exists, and if so, what its status is.
func (r *Balancer) GetLoadBalancer(clusterName string, service *api.Service) (status *api.LoadBalancerStatus, exists bool, retErr error) {
	glog.Errorf("GetLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	nodeInstance := r.getLoadBalancerNode(clusterName, service)
	if nodeInstance != nil {
		status := r.nodeToState(nodeInstance)
		if status != nil {
			return status, true, nil
		}
	}

	glog.Errorf("No such loadbalancer: (%v, %v, %v)", clusterName, service.Namespace, service.Name)

	return nil, false, nil
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it exists, returning
// nil if the load balancer specified either didn't exist or was successfully deleted.
func (r *Balancer) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	glog.Errorf("EnsureLoadBalancerDeleted(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	nodeInstance := r.getLoadBalancerNode(clusterName, service)

	glog.Errorf("Delete for node %+v", nodeInstance.ID)
	if nodeInstance != nil {
		err := r.client.WaitBeforeRunExecution(r.deployment)
		if err != nil {
			return err
		}
		var exec cloudify.ExecutionPost
		exec.WorkflowID = "execute_operation"
		exec.DeploymentID = r.deployment
		exec.Parameters = map[string]interface{}{}
		exec.Parameters["operation"] = "maintenance.delete"
		exec.Parameters["node_ids"] = []string{}
		exec.Parameters["type_names"] = []string{}
		exec.Parameters["run_by_dependency_order"] = false
		exec.Parameters["allow_kwargs_override"] = nil
		exec.Parameters["node_instance_ids"] = []string{nodeInstance.ID}
		exec.Parameters["operation_kwargs"] = map[string]interface{}{}
		execution, err := r.client.RunExecution(exec, true)
		if err != nil {
			return err
		}

		glog.Errorf("%v: Final status for delete(%v), last status: %v", service.Name, execution.ID, execution.Status)

		if execution.Status == "failed" {
			return fmt.Errorf(execution.ErrorMessage)
		}
		return nil
	}

	glog.Errorf("No such loadbalancer: (%v, %v, %v)", clusterName, service.Namespace, service.Name)
	return nil
}

func (r *Balancer) createOrGetLoadBalancer(clusterName string, service *api.Service) (*cloudify.NodeInstance, error) {
	// already have some loadbalancer with same name
	var nodeInstance *cloudify.NodeInstance
	nodeInstance = r.getLoadBalancerNode(clusterName, service)
	if nodeInstance != nil {
		return nodeInstance, nil
	}
	glog.Errorf("No precreated nodes for %s", service.Name)

	// No such loadbalancers
	nodeInstance = r.getLoadBalancerNode("", nil)
	if nodeInstance != nil {
		return nodeInstance, nil
	}
	glog.Errorf("No empty nodes for %s", service.Name)

	var exec cloudify.ExecutionPost
	exec.WorkflowID = "scale"
	exec.DeploymentID = r.deployment
	exec.Parameters = map[string]interface{}{}
	exec.Parameters["scalable_entity_name"] = r.scaleGroup // Use scale group instead real node id
	execution, err := r.client.RunExecution(exec, false)
	if err != nil {
		return nil, err
	}

	glog.Errorf("%v: Final status for scale(%v), last status: %v", service.Name, execution.ID, execution.Status)
	if execution.Status == "failed" {
		return nil, fmt.Errorf(execution.ErrorMessage)
	}

	// try one more time
	return r.getLoadBalancerNode("", nil), nil
}

func (r *Balancer) updateNode(clusterName string, service *api.Service, nodes []*api.Node) (*api.LoadBalancerStatus, error) {
	if len(service.Spec.Ports) == 0 {
		return nil, fmt.Errorf("requested load balancer with no ports")
	}

	ports := []map[string]string{}

	for _, port := range service.Spec.Ports {
		lbPort := map[string]string{}
		lbPort["protocol"] = string(port.Protocol)
		lbPort["port"] = fmt.Sprintf("%d", port.Port)
		lbPort["nodePort"] = fmt.Sprintf("%d", port.NodePort)
		ports = append(ports, lbPort)
	}

	nodeAddresses := []map[string]string{}
	for _, node := range nodes {
		nodeAddress := map[string]string{}
		for _, address := range node.Status.Addresses {
			// hostname address
			if address.Type == api.NodeHostName {
				nodeAddress["hostname"] = address.Address
			}

			// internal address
			if address.Type == api.NodeInternalIP {
				nodeAddress["ip"] = address.Address

			}

			// external address
			if address.Type == api.NodeInternalIP {
				nodeAddress["external_ip"] = address.Address
			}
		}
		nodeAddresses = append(nodeAddresses, nodeAddress)
	}

	params := map[string]interface{}{}
	params["cluster"] = clusterName
	params["name"] = service.Name
	params["namespace"] = service.Namespace
	params["ports"] = ports
	if len(nodeAddresses) > 0 {
		params["nodes"] = nodeAddresses
	}

	// search possible host for loadbalancer
	nodeInstance, err := r.createOrGetLoadBalancer(clusterName, service)
	if err != nil {
		return nil, err
	}
	if nodeInstance != nil {
		err := r.client.WaitBeforeRunExecution(r.deployment)
		if err != nil {
			return nil, err
		}
		var exec cloudify.ExecutionPost
		exec.WorkflowID = "execute_operation"
		exec.DeploymentID = r.deployment
		exec.Parameters = map[string]interface{}{}
		exec.Parameters["operation"] = "maintenance.init"
		exec.Parameters["node_ids"] = []string{}
		exec.Parameters["type_names"] = []string{}
		exec.Parameters["run_by_dependency_order"] = false
		exec.Parameters["allow_kwargs_override"] = nil
		exec.Parameters["node_instance_ids"] = []string{nodeInstance.ID}
		exec.Parameters["operation_kwargs"] = params
		execution, err := r.client.RunExecution(exec, true)
		if err != nil {
			return nil, err
		}

		glog.Errorf("%v: Final status for init(%v), last status: %v", service.Name, execution.ID, execution.Status)
		if execution.Status == "failed" {
			return nil, fmt.Errorf(execution.ErrorMessage)
		}

		status := r.nodeToState(nodeInstance)
		if status != nil {
			return status, nil
		}
	}

	return nil, fmt.Errorf("Can't create loadbalancer %s", service.Name)

}

// EnsureLoadBalancer creates a new load balancer, or updates the existing one. Returns the status of the balancer.
func (r *Balancer) EnsureLoadBalancer(clusterName string, service *api.Service, nodes []*api.Node) (*api.LoadBalancerStatus, error) {
	glog.Errorf("EnsureLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	return r.updateNode(clusterName, service, nodes)
}

// NewBalancer - create instance with support kubernetes balancer interface.
func NewBalancer(client *cloudify.Client, deployment, scaleGroup string) *Balancer {
	return &Balancer{
		client:     client,
		deployment: deployment,
		scaleGroup: scaleGroup,
	}
}
