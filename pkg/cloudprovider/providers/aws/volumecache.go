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
	"k8s.io/kubernetes/pkg/cloudprovider"
	"sync"
)

type volumeCache struct {
	ec2 EC2

	mutex  sync.RWMutex
	latest *volumeCacheSnapshot
}

type volumeCacheSnapshot struct {
	timestamp  int64
	byVolumeId map[awsVolumeID]*cachedVolume
}

type cachedVolume struct {
	ID awsVolumeID

	// Values that cannot change are exported publicly
	AvailabilityZone string

	cache *volumeCache

	mutex sync.Mutex

	state          *ec2.Volume
	stateTimestamp int64
}

func (i *cachedVolume) DescribeVolume(cachePolicy *CachePolicy) (*ec2.Volume, error) {
	snapshot := i.cache.latestSnapshot()

	if snapshot != nil {
		instance := snapshot.findCachedVolume(i.ID)
		if instance != nil {
			state := i.findLatestState(cachePolicy)
			if state != nil {
				return state, nil
			} else {
				glog.V(2).Infof("triggering volume refresh because state expired (volume %s, %s)", i.ID, cachePolicy.Name)
			}
		}
	}

	updated, err := i.cache.refreshAll(snapshot)
	if err != nil {
		return nil, err
	}

	instance := updated.findCachedVolume(i.ID)
	if instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}

	state := i.findLatestState(nil)
	if state != nil {
		return state, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func (i *cachedVolume) findLatestState(cachePolicy *CachePolicy) *ec2.Volume {
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

func newVolumeCache(ec2 EC2) *volumeCache {
	return &volumeCache{
		ec2: ec2,
	}
}

func (c *volumeCache) latestSnapshot() *volumeCacheSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.latest
}

func (c *volumeCache) refreshAll(previous *volumeCacheSnapshot) (*volumeCacheSnapshot, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.latest != nil && c.latest != previous {
		// A concurrent refresh happened, return that
		return c.latest, nil
	}

	request := &ec2.DescribeVolumesInput{}

	volumes, err := c.ec2.DescribeVolumes(request)
	if err != nil {
		glog.V(2).Infof("error querying EC2 volumes: %v", err)
		return nil, err
	}

	byVolumeId := make(map[awsVolumeID]*cachedVolume)

	now := nanoTime()

	latest := c.latest
	if latest == nil {
		// Easier than nil checks everywhere
		latest = &volumeCacheSnapshot{}
	}

	for _, volumeDetails := range volumes {
		volumeID := awsVolumeID(aws.StringValue(volumeDetails.VolumeId))

		if volumeID == "" {
			glog.Warningf("ignoring volume without id: %v", volumeDetails)
			continue
		}

		if byVolumeId[volumeID] != nil {
			return nil, fmt.Errorf("multiple volumes found for volume: %s", volumeID)
		}

		ci := latest.byVolumeId[volumeID]

		if ci == nil {
			az := ""
			if volumeDetails.AvailabilityZone != nil {
				az = aws.StringValue(volumeDetails.AvailabilityZone)
			}

			ci = &cachedVolume{
				ID:               volumeID,
				AvailabilityZone: az,

				cache: c,
			}
		}

		byVolumeId[volumeID] = ci
		{
			ci.mutex.Lock()
			ci.state = volumeDetails
			ci.stateTimestamp = now
			ci.mutex.Unlock()
		}

	}

	next := &volumeCacheSnapshot{
		byVolumeId: byVolumeId,
		timestamp:  now,
	}
	c.latest = next

	return next, nil
}

func (c *volumeCache) FindVolumesById(cachePolicy *CachePolicy, ids []awsVolumeID) (map[awsVolumeID]*cachedVolume, error) {
	snapshot := c.latestSnapshot()

	if snapshot != nil {
		if cachePolicy.IsValid(snapshot.timestamp) {
			volumes := snapshot.findCachedVolumes(ids)
			if len(ids) == len(volumes) {
				return volumes, nil
			} else {
				glog.V(2).Infof("triggering refresh because of missing volumes")

			}
		} else {
			glog.V(2).Infof("triggering refresh because cache policy %s expired", cachePolicy.Name)
		}
	}

	updated, err := c.refreshAll(snapshot)
	if err != nil {
		return nil, err
	}
	volumes := updated.findCachedVolumes(ids)
	return volumes, nil
}

func (c *volumeCache) FindVolumeById(cachePolicy *CachePolicy, id awsVolumeID) (*cachedVolume, error) {
	volumes, err := c.FindVolumesById(cachePolicy, []awsVolumeID{id})
	if err != nil {
		return nil, err
	}

	volume := volumes[id]
	if volume == nil {
		return nil, nil
	}
	return volume, nil
}

func (s *volumeCacheSnapshot) findCachedVolume(id awsVolumeID) *cachedVolume {
	volume := s.byVolumeId[id]
	return volume
}

func (s *volumeCacheSnapshot) findCachedVolumes(ids []awsVolumeID) map[awsVolumeID]*cachedVolume {
	volumes := make(map[awsVolumeID]*cachedVolume)

	for _, id := range ids {
		volume := s.byVolumeId[id]
		if volume != nil {
			volumes[id] = volume
		}
	}
	return volumes
}
