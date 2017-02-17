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

package alicloud

import (
	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/ecs"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"time"
)

type Meta struct {
	HostName    string
	InstanceID  string
	PrivateIPV4 []string
	RegionID    common.Region
}

type SDKClientINS struct {
	c        *ecs.Client
	RegionID common.Region
	Instance *ecs.InstanceAttributesType
	NodeName types.NodeName
}

func NewSDKClientINS(region common.Region, access_key_id string, access_key_secret string) *SDKClientINS {
	ins := &SDKClientINS{
		RegionID: region,
		c:        ecs.NewClient(access_key_id, access_key_secret),
	}

	go func(ins *SDKClientINS) {
		for {
			time.Sleep(time.Duration(5 * time.Minute))
			if ins.NodeName == "" {
				continue
			}
			ins.refreshInstance(ins.NodeName)
		}
	}(ins)
	return ins
}

// getAddressesByName return an instance address slice by it's name.
func (s *SDKClientINS) findAddress(nodeName types.NodeName) ([]v1.NodeAddress, error) {

	instance, err := s.findInstanceByNodeName(nodeName)
	if err != nil {
		glog.Errorf("Error getting instance by InstanceId '%s': %v", nodeName, err)
		return nil, err
	}

	addrs := []v1.NodeAddress{}

	if len(instance.PublicIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.PublicIpAddress.IpAddress {
			addrs = append(addrs, v1.NodeAddress{Type: v1.NodeExternalIP, Address: ipaddr})
		}
	}

	if instance.EipAddress.IpAddress != "" {
		addrs = append(addrs, v1.NodeAddress{Type: v1.NodeExternalIP, Address: instance.EipAddress.IpAddress})
	}

	if len(instance.InnerIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.InnerIpAddress.IpAddress {
			addrs = append(addrs, v1.NodeAddress{Type: v1.NodeInternalIP, Address: ipaddr})
		}
	}

	if len(instance.VpcAttributes.PrivateIpAddress.IpAddress) > 0 {
		for _, ipaddr := range instance.VpcAttributes.PrivateIpAddress.IpAddress {
			addrs = append(addrs, v1.NodeAddress{Type: v1.NodeInternalIP, Address: ipaddr})
		}
	}

	if instance.VpcAttributes.NatIpAddress != "" {
		addrs = append(addrs, v1.NodeAddress{Type: v1.NodeInternalIP, Address: instance.VpcAttributes.NatIpAddress})
	}

	return addrs, nil
}

// Returns the instance with the specified node name
// Returns nil if it does not exist
func (s *SDKClientINS) findInstanceByNodeName(nodeName types.NodeName) (*ecs.InstanceAttributesType, error) {
	glog.Infof("Alicloud.findInstanceByNodeName(\"%s\")", nodeName)
	if s.Instance != nil {
		return s.Instance, nil
	}
	return s.refreshInstance(nodeName)
}

func (s *SDKClientINS) refreshInstance(nodeName types.NodeName) (*ecs.InstanceAttributesType, error) {
	args := ecs.DescribeInstancesArgs{
		RegionId:     common.Region(s.RegionID),
		InstanceName: string(nodeName),
	}
	s.NodeName = nodeName
	instances, _, err := s.c.DescribeInstances(&args)
	if err != nil {
		glog.Errorf("DescribeInstances error (%v): %v", args, err)
		return nil, err
	}

	if len(instances) == 0 {
		return nil, cloudprovider.InstanceNotFound
	}
	if len(instances) > 1 {
		glog.Errorf("Warning: Multipul instance found by nodename [%s], the first one will be used. Instance: [%+v]", string(nodeName), instances)
	}
	glog.Infof("Alicloud.refreshInstance(\"%s\") finished. [ %+v ]\n", string(nodeName), instances[0])
	s.Instance = &instances[0]
	return s.Instance, nil
}
