// +build !providerless

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
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
)

// awsInstanceRegMatch represents Regex Match for AWS instance.
var awsInstanceRegMatch = regexp.MustCompile("^i-[^/]*$")

// InstanceID represents the ID of the instance in the AWS API, e.g. i-12345678
// The "traditional" format is "i-12345678"
// A new longer format is also being introduced: "i-12345678abcdef01"
// We should not assume anything about the length or format, though it seems
// reasonable to assume that instances will continue to start with "i-".
type InstanceID string

func (i InstanceID) awsString() *string {
	return aws.String(string(i))
}

// KubernetesInstanceID represents the id for an instance in the kubernetes API;
// the following form
//  * aws:///<zone>/<awsInstanceId>
//  * aws:////<awsInstanceId>
//  * <awsInstanceId>
type KubernetesInstanceID string

// MapToAWSInstanceID extracts the InstanceID from the KubernetesInstanceID
func (name KubernetesInstanceID) MapToAWSInstanceID() (InstanceID, error) {
	s := string(name)

	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Build a URL with an empty host (AZ)
		s = "aws://" + "/" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		return "", fmt.Errorf("Invalid instance name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS instance (%s)", name)
	}

	awsID := ""
	tokens := strings.Split(strings.Trim(url.Path, "/"), "/")
	if len(tokens) == 1 {
		// instanceId
		awsID = tokens[0]
	} else if len(tokens) == 2 {
		// az/instanceId
		awsID = tokens[1]
	}

	// We sanity check the resulting volume; the two known formats are
	// i-12345678 and i-12345678abcdef01
	if awsID == "" || !awsInstanceRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS instance (%s)", name)
	}

	return InstanceID(awsID), nil
}

// mapToAWSInstanceID extracts the InstanceIDs from the Nodes, returning an error if a Node cannot be mapped
func mapToAWSInstanceIDs(nodes []*v1.Node) ([]InstanceID, error) {
	var instanceIDs []InstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			return nil, fmt.Errorf("node %q did not have ProviderID set", node.Name)
		}
		instanceID, err := KubernetesInstanceID(node.Spec.ProviderID).MapToAWSInstanceID()
		if err != nil {
			return nil, fmt.Errorf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs, nil
}

// mapToAWSInstanceIDsTolerant extracts the InstanceIDs from the Nodes, skipping Nodes that cannot be mapped
func mapToAWSInstanceIDsTolerant(nodes []*v1.Node) []InstanceID {
	var instanceIDs []InstanceID
	for _, node := range nodes {
		if node.Spec.ProviderID == "" {
			klog.Warningf("node %q did not have ProviderID set", node.Name)
			continue
		}
		instanceID, err := KubernetesInstanceID(node.Spec.ProviderID).MapToAWSInstanceID()
		if err != nil {
			klog.Warningf("unable to parse ProviderID %q for node %q", node.Spec.ProviderID, node.Name)
			continue
		}
		instanceIDs = append(instanceIDs, instanceID)
	}

	return instanceIDs
}

// Gets the full information about this instance from the EC2 API
func describeInstance(ec2Client EC2, instanceID InstanceID) (*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	instances, err := ec2Client.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}
	return instances[0], nil
}

// instanceCache manages the cache of DescribeInstances
type instanceCache struct {
	// TODO: Get rid of this field, send all calls through the instanceCache
	cloud *Cloud

	mutex    sync.Mutex
	snapshot *allInstancesSnapshot
}

// Gets the full information about these instance from the EC2 API
func (c *instanceCache) describeAllInstancesUncached() (*allInstancesSnapshot, error) {
	now := time.Now()

	klog.V(4).Infof("EC2 DescribeInstances - fetching all instances")

	var filters []*ec2.Filter
	instances, err := c.cloud.describeInstances(filters)
	if err != nil {
		return nil, err
	}

	m := make(map[InstanceID]*ec2.Instance)
	for _, i := range instances {
		id := InstanceID(aws.StringValue(i.InstanceId))
		m[id] = i
	}

	snapshot := &allInstancesSnapshot{now, m}

	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.snapshot != nil && snapshot.olderThan(c.snapshot) {
		// If this happens a lot, we could run this function in a mutex and only return one result
		klog.Infof("Not caching concurrent AWS DescribeInstances results")
	} else {
		c.snapshot = snapshot
	}

	return snapshot, nil
}

// cacheCriteria holds criteria that must hold to use a cached snapshot
type cacheCriteria struct {
	// MaxAge indicates the maximum age of a cached snapshot we can accept.
	// If set to 0 (i.e. unset), cached values will not time out because of age.
	MaxAge time.Duration

	// HasInstances is a list of InstanceIDs that must be in a cached snapshot for it to be considered valid.
	// If an instance is not found in the cached snapshot, the snapshot be ignored and we will re-fetch.
	HasInstances []InstanceID
}

// describeAllInstancesCached returns all instances, using cached results if applicable
func (c *instanceCache) describeAllInstancesCached(criteria cacheCriteria) (*allInstancesSnapshot, error) {
	var err error
	snapshot := c.getSnapshot()
	if snapshot != nil && !snapshot.MeetsCriteria(criteria) {
		snapshot = nil
	}

	if snapshot == nil {
		snapshot, err = c.describeAllInstancesUncached()
		if err != nil {
			return nil, err
		}
	} else {
		klog.V(6).Infof("EC2 DescribeInstances - using cached results")
	}

	return snapshot, nil
}

// getSnapshot returns a snapshot if one exists
func (c *instanceCache) getSnapshot() *allInstancesSnapshot {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	return c.snapshot
}

// olderThan is a simple helper to encapsulate timestamp comparison
func (s *allInstancesSnapshot) olderThan(other *allInstancesSnapshot) bool {
	// After() is technically broken by time changes until we have monotonic time
	return other.timestamp.After(s.timestamp)
}

// MeetsCriteria returns true if the snapshot meets the criteria in cacheCriteria
func (s *allInstancesSnapshot) MeetsCriteria(criteria cacheCriteria) bool {
	if criteria.MaxAge > 0 {
		// Sub() is technically broken by time changes until we have monotonic time
		now := time.Now()
		if now.Sub(s.timestamp) > criteria.MaxAge {
			klog.V(6).Infof("instanceCache snapshot cannot be used as is older than MaxAge=%s", criteria.MaxAge)
			return false
		}
	}

	if len(criteria.HasInstances) != 0 {
		for _, id := range criteria.HasInstances {
			if nil == s.instances[id] {
				klog.V(6).Infof("instanceCache snapshot cannot be used as does not contain instance %s", id)
				return false
			}
		}
	}

	return true
}

// allInstancesSnapshot holds the results from querying for all instances,
// along with the timestamp for cache-invalidation purposes
type allInstancesSnapshot struct {
	timestamp time.Time
	instances map[InstanceID]*ec2.Instance
}

// FindInstances returns the instances corresponding to the specified ids.  If an id is not found, it is ignored.
func (s *allInstancesSnapshot) FindInstances(ids []InstanceID) map[InstanceID]*ec2.Instance {
	m := make(map[InstanceID]*ec2.Instance)
	for _, id := range ids {
		instance := s.instances[id]
		if instance != nil {
			m[id] = instance
		}
	}
	return m
}
