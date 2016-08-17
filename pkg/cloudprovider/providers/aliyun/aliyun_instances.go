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
	"errors"
	"fmt"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/ecs"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

// NodeAddresses returns the addresses of the specified instance.
func (aly *Aliyun) NodeAddresses(name string) ([]api.NodeAddress, error) {
	glog.V(4).Infof("NodeAddresses(%v) called", name)

	addrs, err := aly.getAddressesByInstanceId(nameToInstanceId(name))
	if err != nil {
		glog.Errorf("Error getting node address by name '%s': %v", name, err)
		return nil, err
	}

	glog.V(4).Infof("NodeAddresses(%v) => %v", name, addrs)
	return addrs, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (aly *Aliyun) ExternalID(name string) (string, error) {
	return nameToInstanceId(name), nil
}

// InstanceID returns the cloud provider ID of the specified instance.
// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
func (aly *Aliyun) InstanceID(name string) (string, error) {
	return nameToInstanceId(name), nil
}

// InstanceType returns the type of the specified instance.
func (aly *Aliyun) InstanceType(name string) (string, error) {
	return "", nil
}

// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
func (aly *Aliyun) List(name_filter string) ([]string, error) {
	instances, err := aly.getInstancesByNameFilter(name_filter)
	if err != nil {
		glog.Errorf("Error getting instances by name_filter '%s': %v", name_filter, err)
		return nil, err
	}
	result := []string{}
	for _, instance := range instances {
		result = append(result, instance.InstanceId)
	}

	glog.V(4).Infof("List instances: %s => %v", name_filter, result)

	return result, nil
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances.
// The method is currently only used in gce.
func (aly *Aliyun) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("Unimplemented")
}

// CurrentNodeName returns the name of the node we are currently running on
// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
func (aly *Aliyun) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

// getAddressesByName return an instance address slice by it's name.
func (aly *Aliyun) getAddressesByInstanceId(instanceId string) ([]api.NodeAddress, error) {
	instance, err := aly.getInstanceByInstanceId(instanceId)
	if err != nil {
		glog.Errorf("Error getting instance by InstanceId '%s': %v", instanceId, err)
		return nil, err
	}

	addrs := []api.NodeAddress{}

	if len(instance.PublicIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.PublicIpAddress.IpAddress {
			addrs = append(addrs, api.NodeAddress{Type: api.NodeExternalIP, Address: ipaddr})
		}
	}

	if instance.EipAddress.IpAddress != "" {
		addrs = append(addrs, api.NodeAddress{Type: api.NodeExternalIP, Address: instance.EipAddress.IpAddress})
	}

	if len(instance.InnerIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.InnerIpAddress.IpAddress {
			addrs = append(addrs, api.NodeAddress{Type: api.NodeInternalIP, Address: ipaddr})
		}
	}

	if len(instance.VpcAttributes.PrivateIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.VpcAttributes.PrivateIpAddress.IpAddress {
			addrs = append(addrs, api.NodeAddress{Type: api.NodeInternalIP, Address: ipaddr})
		}
	}

	if instance.VpcAttributes.NatIpAddress != "" {
		addrs = append(addrs, api.NodeAddress{Type: api.NodeInternalIP, Address: instance.VpcAttributes.NatIpAddress})
	}

	return addrs, nil
}

func (aly *Aliyun) getInstanceByInstanceId(instanceId string) (ecs.InstanceAttributesType, error) {
	args := ecs.DescribeInstancesArgs{
		RegionId:    common.Region(aly.regionID),
		InstanceIds: fmt.Sprintf("[\"%s\"]", instanceId),
	}

	instances, _, err := aly.ecsClient.DescribeInstances(&args)
	if err != nil {
		glog.Errorf("Couldn't DescribeInstances(%v): %v", args, err)
		return ecs.InstanceAttributesType{}, err
	}

	if len(instances) == 0 {
		return ecs.InstanceAttributesType{}, fmt.Errorf("Couldn't get Instances by args '%v'", args)
	}

	return instances[0], nil
}

// List instances that match the InstanceName filter
func (aly *Aliyun) getInstancesByNameFilter(name_filter string) ([]ecs.InstanceAttributesType, error) {
	args := ecs.DescribeInstancesArgs{
		RegionId:     common.Region(aly.regionID),
		InstanceName: name_filter,
	}

	instances, _, err := aly.ecsClient.DescribeInstances(&args)
	if err != nil {
		glog.Errorf("Couldn't DescribeInstances(%v): %v", args, err)
		return nil, err
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("Couldn't get Instances by args '%v'", args)
	}

	return instances, nil
}
