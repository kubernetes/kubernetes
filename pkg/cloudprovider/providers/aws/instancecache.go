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

// instanceCache is a cache of instances, that minimizes the number of AWS requests.
//
// On AWS, rate limits are based primarily on the number of requests, not their complexity;
// even on other platforms typically the overhead of a call is fairly large and thus the
// same cache pattern might well be usable for other clouds also.
//
// The cache has a two level structure: we cache all instances (with a timestamp).  That cache
// can be queried with an expiration time.  Typically if looking up by ID, it is safe to query
// without any expiration; a cache miss (currently) automatically triggers a full refresh.
//
// For each instance there is some immutable data stored (AZ, NodeName etc).
//
// In addition, there is a per-instance state cache, where we can get the latest state for a
// particular instance.  Currently a cache refresh for an instance triggers a full cache refresh
// for all instances, because a single-instance query counts the same against quota as a full refresh.
//
// Although we optimize for quota request, we do so at the expense of bytes transferred (for
// example).  We introduce a CachePolicy object that can express other strategies as and when
// we need them - for example only refreshing a particular instance.

type instanceCache struct {
	cloud *Cloud

	mutex  sync.RWMutex
	latest *instanceCacheSnapshot
}

type awsInstanceID string

func (i awsInstanceID) awsString() *string {
	return aws.String(string(i))
}

// instanceCacheSnapshot holds a view at a moment in time of all the instances
type instanceCacheSnapshot struct {
	timestamp    int64
	byInstanceID map[awsInstanceID]*cachedInstance
	byNodeName   map[types.NodeName]*cachedInstance
}

// cachedInstance is a reusable instance wrapper, holding some immutable state and caching access to mutable state
type cachedInstance struct {
	ID awsInstanceID

	// Values that cannot change are exported publicly
	NodeName         types.NodeName
	VpcID            string
	AvailabilityZone string
	InstanceType     string
	SubnetID         string

	cache *instanceCache

	// mutex protects the mutable elements below
	mutex sync.Mutex

	// state is the state in EC2 as observed last
	state *ec2.Instance

	// stateTimestamp is the timestamp at which we last updated state
	stateTimestamp int64
}

// DescribeInstance gets the state for the instance, subject to the cache policy.
// Currently a cache-miss will cause a refresh of all instances, because this is the same "price" in terms of quota.
func (i *cachedInstance) DescribeInstance(cachePolicy *CachePolicy) (*ec2.Instance, error) {
	snapshot := i.cache.latestSnapshot()

	if snapshot != nil {
		instance := snapshot.findCachedInstance(i.ID)
		if instance != nil {
			state := i.findStateIfCached(cachePolicy)
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

	state := i.findStateIfCached(nil)
	if state != nil {
		return state, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

// findStateIfCached returns the last state, if it is valid under the cache policy.
// If there is no state, or the state has expired, it returns nil.
func (i *cachedInstance) findStateIfCached(cachePolicy *CachePolicy) *ec2.Instance {
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

// newInstanceCache is a constructor for an instanceCache
func newInstanceCache(cloud *Cloud) *instanceCache {
	return &instanceCache{
		cloud: cloud,
	}
}

// latestSnapshot returns the last view of all instances (or nil if no snapshot)
func (c *instanceCache) latestSnapshot() *instanceCacheSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.latest
}

// refreshAll builds an updated snapshot by calling DescribeInstances.
// It accepts the previous snapshot, and will not refresh if a concurrent request has triggered a concurrent refresh.
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

		// CachedInstances are long-lived - we reuse them
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

// findInstancesByNodeNames is a helper that looks up instances against the snapshot.  If the snapshot is invalid,
// or any are not found, it will trigger a full refresh.
func (c *instanceCache) findInstancesByNodeNames(cachePolicy *CachePolicy, nodeNames []types.NodeName) (map[types.NodeName]*cachedInstance, error) {
	snapshot := c.latestSnapshot()

	if snapshot != nil {
		if cachePolicy.IsValid(snapshot.timestamp) {
			instances := snapshot.findCachedInstancesByNodeNames(nodeNames)
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
	instances := updated.findCachedInstancesByNodeNames(nodeNames)
	return instances, nil
}

// GetInstanceByNodeName looks up the instance in the current cached view of instances.  If the cached
// view is invalid according to cachePolicy, or if the node name is not found, a full refresh will be triggered.
func (c *instanceCache) GetInstanceByNodeName(cachePolicy *CachePolicy, nodeName types.NodeName) (*cachedInstance, error) {
	if cachePolicy.Validity == 0 {
		// In general, this is not safe because a new node can be replaced
		glog.Warningf("Doing instance cached NodeName lookup without cache expiry")
	}

	instances, err := c.findInstancesByNodeNames(cachePolicy, []types.NodeName{nodeName})
	if err != nil {
		return nil, err
	}

	instance := instances[nodeName]
	if instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}
	return instance, nil
}

// findCachedInstancesByNodeNames does a lookup of the instances by node name, returning any that are found.
// It does not trigger a refresh.
func (c *instanceCacheSnapshot) findCachedInstancesByNodeNames(nodeNames []types.NodeName) map[types.NodeName]*cachedInstance {
	instances := make(map[types.NodeName]*cachedInstance)
	for _, nodeName := range nodeNames {
		instance := c.byNodeName[nodeName]
		if instance != nil {
			instances[nodeName] = instance
		}
	}
	return instances
}

// findCachedInstance does a lookup of the instance by id, returning nil if not found.
// It does not trigger a cache refresh.
func (s *instanceCacheSnapshot) findCachedInstance(instanceID awsInstanceID) *cachedInstance {
	instance := s.byInstanceID[instanceID]
	return instance
}
