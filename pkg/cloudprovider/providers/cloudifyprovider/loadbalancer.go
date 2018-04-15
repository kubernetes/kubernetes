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
	"github.com/cloudify-incubator/cloudify-rest-go-client/cloudify"
	"github.com/golang/glog"
	api "k8s.io/api/core/v1"
	"errors"
)

// Balancer - struct with connection settings
type Balancer struct {
	deployment string
	client     *cloudify.Client
}

func (r *Balancer) GetDeploymentBalancerInfo() (map[string]string, error) {
	deploymentInfo := make(map[string]string)

	data, err := cloudify.ParseDeploymentFile(r.deployment)
	if err != nil {
		fmt.Errorf("Error While trying to parse deployment file")
		return nil, err
	}

	for _, deployment := range data.Deployments {
		dep := deployment.(map[string]interface{})

		if dep["deployment_type"] == "load" {
			deploymentInfo["id"] = dep["id"].(string)
			deploymentInfo["node_data_type"] = dep["node_data_type"].(string)
			return deploymentInfo, nil
		}
	}

	return deploymentInfo, nil
}

func (r *Balancer) GetDeploymentBalancerID() (map[string]string, error) {
	deploymentInfo, err := r.GetDeploymentBalancerInfo()
	if err != nil {
		glog.Errorf("Error: %+v", err)
		return nil, err
	}

	if deploymentInfo == nil {
		errorMessage := "cloudify deployment info is empty"
		glog.Errorf(errorMessage)
		return nil, errors.New(errorMessage)
	}

	return deploymentInfo, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (r *Balancer) UpdateLoadBalancer(clusterName string, service *api.Service, nodes []*api.Node) error {
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	var finalState, err = r.updateNode(clusterName, service, nodes)
	glog.Infof("%v: Final state: ", service.Name, finalState)

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
	deploymentInfo, err := r.GetDeploymentBalancerID()
	if err != nil || deploymentInfo == nil {
		return nil
	}

	// Add filter by deployment
	params["deployment_id"] = deploymentInfo["id"]

	instances, err := r.client.GetLoadBalancerInstances(params, clusterName, nameSpace, name,
		deploymentInfo["node_data_type"])
	if err != nil {
		glog.Infof("Not found instances: %+v", err)
		return nil
	}
	if len(instances.Items) > 0 {
		node := instances.Items[0]
		glog.Infof("Found '%v' for '%v'", node.ID, name)
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
		hostName := nodeInstance.GetStringProperty("hostname")
		if hostName != "" {
			ingressNode.Hostname = hostName
		}

		// ip
		hostPublicIP := nodeInstance.GetStringProperty("public_ip")
		if hostPublicIP != "" {
			ingressNode.IP = hostPublicIP
		}

		ingress = append(ingress, ingressNode)

		glog.Infof("Status %v for return: %+v", nodeInstance.ID, ingress)
		return &api.LoadBalancerStatus{ingress}
	}
	return nil
}

// GetLoadBalancer returns whether the specified load balancer exists, and if so, what its status is.
func (r *Balancer) GetLoadBalancer(clusterName string, service *api.Service) (status *api.LoadBalancerStatus, exists bool, retErr error) {
	glog.V(4).Infof("GetLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	nodeInstance := r.getLoadBalancerNode(clusterName, service)
	if nodeInstance != nil {
		status := r.nodeToState(nodeInstance)
		if status != nil {
			return status, true, nil
		}
	}

	glog.Infof("No such loadbalancer: (%v, %v, %v)", clusterName, service.Namespace, service.Name)

	return nil, false, nil
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it exists, returning
// nil if the load balancer specified either didn't exist or was successfully deleted.
func (r *Balancer) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	nodeInstance := r.getLoadBalancerNode(clusterName, service)

	glog.Infof("Delete for node %+v", nodeInstance.ID)

	// Get The deployment ID For Load Balancer node
	deploymentInfo, err := r.GetDeploymentBalancerID()
	if err != nil {
		return err
	}

	if deploymentInfo == nil {
		return errors.New("cannot find the deployment info")
	}

	if nodeInstance != nil {
		err := r.client.WaitBeforeRunExecution(deploymentInfo["id"])
		if err != nil {
			return err
		}

		var exec cloudify.ExecutionPost
		exec.WorkflowID = "execute_operation"
		exec.DeploymentID = deploymentInfo["id"]
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

		glog.Infof("%v: Final status for delete(%v), last status: %v", service.Name, execution.ID, execution.Status)

		if execution.Status == "failed" {
			return fmt.Errorf(execution.ErrorMessage)
		}
		return nil
	}

	glog.Infof("No such loadbalancer: (%v, %v, %v)", clusterName, service.Namespace, service.Name)
	return nil
}

func (r *Balancer) getLoadbalancerScaleGroup() (string, error) {
	var params = map[string]string{}
	// Add filter by deployment
	deploymentInfo, err := r.GetDeploymentBalancerID()
	if err != nil {
		return "", err
	}

	if deploymentInfo == nil {
		return "", errors.New("cannot find the deployment info")
	}
	params["deployment_id"] = deploymentInfo["id"]
	nodes, err := r.client.GetNodesFull(params)
	if err != nil {
		return "", err
	}
	for _, node := range nodes.Items {
		if node.Type == deploymentInfo["node_data_type"] {
			if node.ScalingGroupName != "" {
				return node.ScalingGroupName, nil
			}
		}
	}

	return "", fmt.Errorf("no groups for scale")
}

func (r *Balancer) createOrGetLoadBalancer(clusterName string, service *api.Service) (*cloudify.NodeInstance, error) {
	// already have some loadbalancer with same name
	var nodeInstance *cloudify.NodeInstance
	nodeInstance = r.getLoadBalancerNode(clusterName, service)
	if nodeInstance != nil {
		return nodeInstance, nil
	}
	glog.Infof("No precreated nodes for '%s'", service.Name)

	// No such loadbalancers
	nodeInstance = r.getLoadBalancerNode("", nil)
	if nodeInstance != nil {
		return nodeInstance, nil
	}
	glog.Infof("No empty nodes for %s", service.Name)

	loadScalingGroup, err := r.getLoadbalancerScaleGroup()
	if err != nil {
		return nil, err
	}

	deploymentInfo, err := r.GetDeploymentBalancerID()
	if err != nil {
		return nil, err
	}

	if deploymentInfo == nil {
		return nil, errors.New("cannot find the deployment info")
	}

	var exec cloudify.ExecutionPost
	exec.WorkflowID = "scale"
	exec.DeploymentID = deploymentInfo["id"]
	exec.Parameters = map[string]interface{}{}
	exec.Parameters["scalable_entity_name"] = loadScalingGroup
	execution, err := r.client.RunExecution(exec, false)
	if err != nil {
		return nil, err
	}

	glog.Infof("%v: Final status for scale(%v), last status: %v", service.Name, execution.ID, execution.Status)
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

	deploymentInfo, err := r.GetDeploymentBalancerID()
	if err != nil {
		return nil, err
	}

	if deploymentInfo == nil {
		return nil, errors.New("cannot find the deployment info")
	}

	if nodeInstance != nil {
		err := r.client.WaitBeforeRunExecution(deploymentInfo["id"])
		if err != nil {
			return nil, err
		}
		var exec cloudify.ExecutionPost
		exec.WorkflowID = "execute_operation"
		exec.DeploymentID = deploymentInfo["id"]
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

		glog.Infof("%v: Final status for init(%v), last status: %v", service.Name, execution.ID, execution.Status)
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
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v)", clusterName, service.Namespace, service.Name)

	return r.updateNode(clusterName, service, nodes)
}

// NewBalancer - create instance with support kubernetes balancer interface.
func NewBalancer(client *cloudify.Client, deployment string) *Balancer {
	return &Balancer{
		client:     client,
		deployment: deployment,
	}
}
