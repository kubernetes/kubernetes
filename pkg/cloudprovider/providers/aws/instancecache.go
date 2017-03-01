/*
Copyright 2017 The Kubernetes Authors.

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

package aws

import (
	"sync"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"github.com/aws/aws-sdk-go/aws"
	"k8s.io/kubernetes/pkg/types"
)

// awsInstanceID is a strongly-typed wrapper around an AWS instance ID
type awsInstanceID string

type instanceCache struct {
	cloud            *Cloud

	mutex            sync.RWMutex
	byInstanceID     map[awsInstanceID]*ec2.Instance
	byNodeName     map[types.NodeName]*ec2.Instance
}

func newInstanceCache(cloud *Cloud) (*instanceCache) {
	return &instanceCache{
		cloud: cloud,
	}
}

func (c*instanceCache) refreshAll() error {
	filters := []*ec2.Filter{
		newEc2Filter("instance-state-name", "running"),
	}

	filters = c.cloud.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.cloud.ec2.DescribeInstances(request)
	if err != nil {
		glog.V(2).Infof("Failed to describe instances %v", err)
		return err
	}

	if len(instances) == 0 {
		glog.Warningf("Failed to find any running instances")
	}


	byInstanceID := make(map[awsInstanceID]*ec2.Instance)
	byNodeName := make(map[types.NodeName]*ec2.Instance)

	for _, i := range instances {
		instanceID := aws.StringValue(i.InstanceId)

		if instanceID == "" {
			glog.Warningf("ignoring instance without id: %v", i)
			continue
		}

		byInstanceID[awsInstanceID(instanceID)] = i

		nodeName := mapInstanceToNodeName(i)
		if nodeName == "" {
			glog.Warningf("ignoring instance with no NodeName: %s", instanceID)
		} else {
			if byNodeName[nodeName] != nil {
				glog.Errorf("detected two instances with the same NodeName: %q", nodeName)
			} else {
				byNodeName[nodeName] = i
			}
		}
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.byInstanceID = byInstanceID
	c.byNodeName = byNodeName

	return nil
}

func (c*instanceCache) GetNodeNames(instanceIDs []awsInstanceID) (map[awsInstanceID]types.NodeName, error) {
	nodeNames := c.getCachedNodeNames(instanceIDs)
	if len(instanceIDs) != len(nodeNames) {
		glog.V(2).Infof("triggering refresh because of missing instance ids")
		err := c.refreshAll()
		if err != nil {
			return nil, err
		}
	}
	nodeNames = c.getCachedNodeNames(instanceIDs)
	return nodeNames, nil
}

func (c*instanceCache) getCachedNodeNames(instanceIDs []awsInstanceID) (map[awsInstanceID]types.NodeName) {
	nodeNames := make(map[awsInstanceID]types.NodeName)

	c.mutex.RLock()
	defer c.mutex.RUnlock()

	for _, id := range instanceIDs {
		instance := c.byInstanceID[id]
		if instance != nil {
			nodeNames[id] = mapInstanceToNodeName(instance)
		}
	}
	return nodeNames
}

func (c*instanceCache) GetInstanceIDs(nodeNames []types.NodeName) (map[types.NodeName]awsInstanceID, error) {
	instanceIDs := c.getCachedInstanceIDs(nodeNames)
	if len(instanceIDs) != len(nodeNames) {
		glog.V(2).Infof("triggering refresh because of missing node names")
		err := c.refreshAll()
		if err != nil {
			return nil, err
		}
	}
	instanceIDs = c.getCachedInstanceIDs(nodeNames)
	return instanceIDs, nil
}

// GetInstanceID is a helper that wraps GetInstanceIDs for the case of a single node name
func (c*instanceCache) GetInstanceID(nodeName types.NodeName) (awsInstanceID, error) {
	nodeNames := []types.NodeName{ nodeName }

	instanceIDs, err := c.GetInstanceIDs(nodeNames)
	if err != nil {
		return "", err
	}

	instanceID  := instanceIDs[nodeName]
	if instanceID == "" {
		glog.Warningf("unable to find instance with node name %s", nodeName)
	}
	return instanceID, nil
}

func (c*instanceCache) getCachedInstanceIDs(nodeNames []types.NodeName) (map[awsInstanceID]types.NodeName) {
	instanceIDs := make(map[types.NodeName]awsInstanceID)

	c.mutex.RLock()
	defer c.mutex.RUnlock()

	for _, nodeName := range nodeNames {
		instance := c.byNodeName[nodeName]
		if instance != nil {
			instanceIDs[nodeName] = awsInstanceID(aws.StringValue(instance.Id))
		}
	}
	return nodeNames
}
