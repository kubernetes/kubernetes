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

// See https://docs.qingcloud.com/api/instance/index.html

import (
	"errors"

	"github.com/golang/glog"
	"github.com/magicshui/qingcloud-go"
	"github.com/magicshui/qingcloud-go/instance"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
)

// NodeAddresses returns the addresses of the specified instance.
func (qc *Qingcloud) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	glog.V(3).Infof("NodeAddresses(%v) called", name)

	ins, err := qc.getInstanceById(nodeNameToInstanceId(name))
	if err != nil {
		glog.Errorf("error getting instance '%s': %v", name, err)
		return nil, err
	}

	addrs := []api.NodeAddress{}
	for _, vxnet := range ins.Vxnets {
		if vxnet.PrivateIP != "" {
			addrs = append(addrs, api.NodeAddress{Type: api.NodeInternalIP, Address: vxnet.PrivateIP})
		}
	}
	if ins.Eip.EipAddr != "" {
		addrs = append(addrs, api.NodeAddress{Type: api.NodeExternalIP, Address: ins.Eip.EipAddr})
	}

	glog.V(3).Infof("NodeAddresses: %v, %v", name, addrs)
	return addrs, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (qc *Qingcloud) ExternalID(nodeName types.NodeName) (string, error) {
	glog.V(3).Infof("ExternalID(%v) called", nodeName)

	ins, err := qc.getInstanceById(nodeNameToInstanceId(nodeName))
	if err != nil {
		return "", nil
	}

	return ins.InstanceID, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (qc *Qingcloud) InstanceID(nodeName types.NodeName) (string, error) {
	glog.V(3).Infof("InstanceID(%v) called", nodeName)

	ins, err := qc.getInstanceById(nodeNameToInstanceId(nodeName))
	if err != nil {
		return "", nil
	}

	return ins.InstanceID, nil
}

// InstanceType returns the type of the specified instance.
func (qc *Qingcloud) InstanceType(name types.NodeName) (string, error) {
	glog.V(3).Infof("InstanceType(%v) called", name)

	ins, err := qc.getInstanceById(nodeNameToInstanceId(name))
	if err != nil {
		return "", err
	}

	return ins.InstanceType, nil
}

// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
func (qc *Qingcloud) List(filter string) ([]types.NodeName, error) {
	glog.V(3).Infof("List(%v) called", filter)

	instances, err := qc.getInstancesByFilter(filter)
	if err != nil {
		glog.Errorf("error getting instances by filter '%s': %v", filter, err)
		return nil, err
	}
	result := []types.NodeName{}
	for _, ins := range instances {
		result = append(result, types.NodeName(ins.InstanceID))
	}

	glog.V(3).Infof("List instances: %v", result)

	return result, nil
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances.
// The method is currently only used in gce.
func (qc *Qingcloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("Unimplemented")
}

// CurrentNodeName returns the name of the node we are currently running on
// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
func (qc *Qingcloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

func (qc *Qingcloud) getInstanceById(instanceId string) (*instance.Instance, error) {
	instanceN := qingcloud.NumberedString{}
	instanceN.Add(instanceId)
	statusN := qingcloud.NumberedString{}
	statusN.Add("pending", "running", "stopped")
	verbose := qingcloud.Integer{}
	verbose.Set(1)
	resp, err := qc.instanceClient.DescribeInstances(instance.DescribeInstanceRequest{
		InstancesN: instanceN,
		StatusN:    statusN,
		Verbose:    verbose,
	})
	if err != nil {
		return nil, err
	}
	if len(resp.InstanceSet) == 0 {
		return nil, cloudprovider.InstanceNotFound
	}

	return &resp.InstanceSet[0], nil
}

// List instances that match the filter
func (qc *Qingcloud) getInstancesByFilter(filter string) ([]instance.Instance, error) {
	searchWord := qingcloud.String{}
	searchWord.Set(filter)
	statusN := qingcloud.NumberedString{}
	statusN.Add("running", "stopped")
	verbose := qingcloud.Integer{}
	verbose.Set(1)
	limit := qingcloud.Integer{}
	limit.Set(100)

	instances := []instance.Instance{}

	for i := 0; ; i += 100 {
		offset := qingcloud.Integer{}
		offset.Set(i)
		resp, err := qc.instanceClient.DescribeInstances(instance.DescribeInstanceRequest{
			SearchWord: searchWord,
			StatusN:    statusN,
			Verbose:    verbose,
			Offset:     offset,
			Limit:      limit,
		})
		if err != nil {
			return nil, err
		}
		if len(resp.InstanceSet) == 0 {
			break
		}

		instances = append(instances, resp.InstanceSet...)
		if len(instances) >= resp.TotalCount {
			break
		}
	}

	return instances, nil
}
