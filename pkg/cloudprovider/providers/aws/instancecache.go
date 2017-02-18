/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"sync"
)

type instanceCache struct {
	cloud *Cloud

	mutex  sync.RWMutex
	latest *instanceCacheSnapshot
}

type awsInstanceID string

func (i awsInstanceID) awsString() *string {
	return aws.String(string(i))
}

type instanceCacheSnapshot struct {
	timestamp    int64
	byInstanceID map[awsInstanceID]*cachedInstance
	byNodeName   map[types.NodeName]*cachedInstance
}

type cachedInstance struct {
	ID awsInstanceID

	// Values that cannot change are exported publicly
	NodeName         types.NodeName
	VpcID            string
	AvailabilityZone string
	InstanceType     string
	SubnetID         string

	cache *instanceCache

	mutex sync.Mutex

	state          *ec2.Instance
	stateTimestamp int64
}

func (i *cachedInstance) DescribeInstance(cachePolicy *CachePolicy) (*ec2.Instance, error) {
	snapshot := i.cache.latestSnapshot()

	if snapshot != nil {
		instance := snapshot.findCachedInstance(i.ID)
		if instance != nil {
			state := i.findLatestState(cachePolicy)
			if state != nil {
				return state, nil
			} else {
				glog.V(2).Infof("triggering instance refresh because instance state expired (instance %s, %s)", i.ID, cachePolicy.Name)
			}
		}
	}

	updated, err := i.cache.refreshAll(snapshot)
	if err != nil {
		return nil, err
	}

	instance := updated.findCachedInstance(i.ID)
	if instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}

	state := i.findLatestState(nil)
	if state != nil {
		return state, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func (i *cachedInstance) findLatestState(cachePolicy *CachePolicy) *ec2.Instance {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	if i.state != nil {
		if cachePolicy == nil || cachePolicy.IsValid(i.stateTimestamp) {
			return i.state
		} else {
			glog.V(2).Infof("triggering refresh because cache policy %s expired", cachePolicy.Name)
		}
	}

	return nil
}

func newInstanceCache(cloud *Cloud) *instanceCache {
	return &instanceCache{
		cloud: cloud,
	}
}

func (c *instanceCache) latestSnapshot() *instanceCacheSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.latest
}

func (c *instanceCache) refreshAll(previous *instanceCacheSnapshot) (*instanceCacheSnapshot, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.latest != nil && c.latest != previous {
		// A concurrent refresh happened, return that
		return c.latest, nil
	}

	filters := []*ec2.Filter{
		newEc2Filter("instance-state-name", "running"),
	}

	filters = c.cloud.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.cloud.ec2.DescribeInstances(request)
	if err != nil {
		glog.V(2).Infof("error querying EC2 instances: %v", err)
		return nil, err
	}

	if len(instances) == 0 {
		glog.Warningf("did not find any running instances")
	}

	byInstanceID := make(map[awsInstanceID]*cachedInstance)
	byNodeName := make(map[types.NodeName]*cachedInstance)

	now := nanoTime()

	latest := c.latest
	if latest == nil {
		// Easier than nil checks everywhere
		latest = &instanceCacheSnapshot{}
	}

	for _, i := range instances {
		instanceID := awsInstanceID(aws.StringValue(i.InstanceId))

		if instanceID == "" {
			glog.Warningf("ignoring instance without id: %v", i)
			continue
		}

		if byInstanceID[instanceID] != nil {
			return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
		}

		ci := latest.byInstanceID[instanceID]
		nodeName := mapInstanceToNodeName(i)

		if ci == nil {
			az := ""
			if i.Placement != nil {
				az = aws.StringValue(i.Placement.AvailabilityZone)
			}

			ci = &cachedInstance{
				ID:               instanceID,
				NodeName:         nodeName,
				VpcID:            aws.StringValue(i.VpcId),
				AvailabilityZone: az,
				InstanceType:     aws.StringValue(i.InstanceType),
				SubnetID:         aws.StringValue(i.SubnetId),

				cache: c,
			}
		}

		byInstanceID[instanceID] = ci
		{
			ci.mutex.Lock()
			ci.state = i
			ci.stateTimestamp = now
			ci.mutex.Unlock()
		}

		if nodeName == "" {
			glog.Warningf("ignoring instance with no NodeName: %s", instanceID)
		} else {
			if byNodeName[nodeName] != nil {
				glog.Errorf("detected two instances with the same NodeName: %q", nodeName)
			} else {
				byNodeName[nodeName] = ci
			}
		}
	}

	next := &instanceCacheSnapshot{
		byInstanceID: byInstanceID,
		byNodeName:   byNodeName,
		timestamp:    now,
	}
	c.latest = next

	return next, nil
}

func (c *instanceCache) FindInstancesByNodeNames(cachePolicy *CachePolicy, nodeNames []types.NodeName) (map[types.NodeName]*cachedInstance, error) {
	snapshot := c.latestSnapshot()

	if snapshot != nil {
		if cachePolicy.IsValid(snapshot.timestamp) {
			instances := snapshot.findInstancesByNodeNames(nodeNames)
			if len(nodeNames) == len(instances) {
				return instances, nil
			} else {
				glog.V(2).Infof("triggering refresh because of missing node names")

			}
		} else {
			glog.V(2).Infof("triggering refresh because cache policy %s expired", cachePolicy.Name)
		}
	}

	updated, err := c.refreshAll(snapshot)
	if err != nil {
		return nil, err
	}
	instances := updated.findInstancesByNodeNames(nodeNames)
	return instances, nil
}

func (c *instanceCache) GetInstancesByNodeNames(cachePolicy *CachePolicy, nodeNames []types.NodeName) (map[types.NodeName]*cachedInstance, error) {
	instances, err := c.FindInstancesByNodeNames(cachePolicy, nodeNames)
	if err != nil {
		return nil, err
	}

	if len(instances) != len(nodeNames) {
		for _, nodeName := range nodeNames {
			if instances[nodeName] == nil {
				return nil, cloudprovider.InstanceNotFound
			}
		}
	}
	return instances, nil
}

func (c *instanceCache) GetInstanceByNodeName(cachePolicy *CachePolicy, nodeName types.NodeName) (*cachedInstance, error) {
	instances, err := c.FindInstancesByNodeNames(cachePolicy, []types.NodeName{nodeName})
	if err != nil {
		return nil, err
	}

	instance := instances[nodeName]
	if instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}
	return instance, nil
}

func (c *instanceCacheSnapshot) findInstancesByNodeNames(nodeNames []types.NodeName) map[types.NodeName]*cachedInstance {
	instances := make(map[types.NodeName]*cachedInstance)
	for _, nodeName := range nodeNames {
		instance := c.byNodeName[nodeName]
		if instance != nil {
			instances[nodeName] = instance
		}
	}
	return instances
}

func (s *instanceCacheSnapshot) findCachedInstance(instanceID awsInstanceID) *cachedInstance {
	instance := s.byInstanceID[instanceID]
	return instance
}

func (s *instanceCacheSnapshot) findCachedInstances(instanceIDs []awsInstanceID) map[awsInstanceID]*cachedInstance {
	instances := make(map[awsInstanceID]*cachedInstance)

	for _, id := range instanceIDs {
		instance := s.byInstanceID[id]
		if instance != nil {
			instances[id] = instance
		}
	}
	return instances
}
