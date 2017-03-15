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

package gce

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/compute/metadata"
	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (gce *GCECloud) NodeAddresses(_ types.NodeName) ([]v1.NodeAddress, error) {
	internalIP, err := metadata.Get("instance/network-interfaces/0/ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get internal IP: %v", err)
	}
	externalIP, err := metadata.Get("instance/network-interfaces/0/access-configs/0/external-ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get external IP: %v", err)
	}
	return []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: internalIP},
		{Type: v1.NodeExternalIP, Address: externalIP},
	}, nil
}

// ExternalID returns the cloud provider ID of the node with the specified NodeName (deprecated).
func (gce *GCECloud) ExternalID(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			externalInstanceID, err := getCurrentExternalIDViaMetadata()
			if err == nil {
				return externalInstanceID, nil
			}
		}
	}

	// Fallback to GCE API call if metadata server fails to retrieve ID
	inst, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return strconv.FormatUint(inst.ID, 10), nil
}

// InstanceID returns the cloud provider ID of the node with the specified NodeName.
func (gce *GCECloud) InstanceID(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			projectID, zone, err := getProjectAndZone()
			if err == nil {
				return projectID + "/" + zone + "/" + canonicalizeInstanceName(instanceName), nil
			}
		}
	}
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return gce.projectID + "/" + instance.Zone + "/" + instance.Name, nil
}

// InstanceType returns the type of the specified node with the specified NodeName.
func (gce *GCECloud) InstanceType(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			mType, err := getCurrentMachineTypeViaMetadata()
			if err == nil {
				return mType, nil
			}
		}
	}
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return instance.Type, nil
}

func (gce *GCECloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return wait.Poll(2*time.Second, 30*time.Second, func() (bool, error) {
		project, err := gce.service.Projects.Get(gce.projectID).Do()
		if err != nil {
			glog.Errorf("Could not get project: %v", err)
			return false, nil
		}
		keyString := fmt.Sprintf("%s:%s %s@%s", user, strings.TrimSpace(string(keyData)), user, user)
		found := false
		for _, item := range project.CommonInstanceMetadata.Items {
			if item.Key == "sshKeys" {
				if strings.Contains(*item.Value, keyString) {
					// We've already added the key
					glog.Info("SSHKey already in project metadata")
					return true, nil
				}
				value := *item.Value + "\n" + keyString
				item.Value = &value
				found = true
				break
			}
		}
		if !found {
			// This is super unlikely, so log.
			glog.Infof("Failed to find sshKeys metadata, creating a new item")
			project.CommonInstanceMetadata.Items = append(project.CommonInstanceMetadata.Items,
				&compute.MetadataItems{
					Key:   "sshKeys",
					Value: &keyString,
				})
		}
		op, err := gce.service.Projects.SetCommonInstanceMetadata(gce.projectID, project.CommonInstanceMetadata).Do()
		if err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		if err := gce.waitForGlobalOp(op); err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		glog.Infof("Successfully added sshKey to project metadata")
		return true, nil
	})
}

// GetAllZones returns all the zones in which nodes are running
func (gce *GCECloud) GetAllZones() (sets.String, error) {
	// Fast-path for non-multizone
	if len(gce.managedZones) == 1 {
		return sets.NewString(gce.managedZones...), nil
	}

	// TODO: Caching, but this is currently only called when we are creating a volume,
	// which is a relatively infrequent operation, and this is only # zones API calls
	zones := sets.NewString()

	// TODO: Parallelize, although O(zones) so not too bad (N <= 3 typically)
	for _, zone := range gce.managedZones {
		// We only retrieve one page in each zone - we only care about existence
		listCall := gce.service.Instances.List(gce.projectID, zone)

		// No filter: We assume that a zone is either used or unused
		// We could only consider running nodes (like we do in List above),
		// but probably if instances are starting we still want to consider them.
		// I think we should wait until we have a reason to make the
		// call one way or the other; we generally can't guarantee correct
		// volume spreading if the set of zones is changing
		// (and volume spreading is currently only a heuristic).
		// Long term we want to replace GetAllZones (which primarily supports volume
		// spreading) with a scheduler policy that is able to see the global state of
		// volumes and the health of zones.

		// Just a minimal set of fields - we only care about existence
		listCall = listCall.Fields("items(name)")

		res, err := listCall.Do()
		if err != nil {
			return nil, err
		}
		if len(res.Items) != 0 {
			zones.Insert(zone)
		}
	}

	return zones, nil
}

// Implementation of Instances.CurrentNodeName
func (gce *GCECloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// Gets the named instances, returning cloudprovider.InstanceNotFound if any instance is not found
func (gce *GCECloud) getInstancesByNames(names []string) ([]*gceInstance, error) {
	instances := make(map[string]*gceInstance)
	remaining := len(names)

	nodeInstancePrefix := gce.nodeInstancePrefix
	for _, name := range names {
		name = canonicalizeInstanceName(name)
		if !strings.HasPrefix(name, gce.nodeInstancePrefix) {
			glog.Warningf("instance '%s' does not conform to prefix '%s', removing filter", name, gce.nodeInstancePrefix)
			nodeInstancePrefix = ""
		}
		instances[name] = nil
	}

	for _, zone := range gce.managedZones {
		if remaining == 0 {
			break
		}

		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)

			if nodeInstancePrefix != "" {
				// Add the filter for hosts
				listCall = listCall.Filter("name eq " + nodeInstancePrefix + ".*")
			}

			// TODO(zmerlynn): Internal bug 29524655
			// listCall = listCall.Fields("items(name,id,disks,machineType)")
			if pageToken != "" {
				listCall.PageToken(pageToken)
			}

			res, err := listCall.Do()
			if err != nil {
				return nil, err
			}
			pageToken = res.NextPageToken
			for _, i := range res.Items {
				name := i.Name
				if _, ok := instances[name]; !ok {
					continue
				}

				instance := &gceInstance{
					Zone:  zone,
					Name:  name,
					ID:    i.Id,
					Disks: i.Disks,
					Type:  lastComponent(i.MachineType),
				}
				instances[name] = instance
				remaining--
			}
		}
		if page >= maxPages {
			glog.Errorf("getInstancesByNames exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}

	instanceArray := make([]*gceInstance, len(names))
	for i, name := range names {
		name = canonicalizeInstanceName(name)
		instance := instances[name]
		if instance == nil {
			glog.Errorf("Failed to retrieve instance: %q", name)
			return nil, cloudprovider.InstanceNotFound
		}
		instanceArray[i] = instances[name]
	}

	return instanceArray, nil
}

// Gets the named instance, returning cloudprovider.InstanceNotFound if the instance is not found
func (gce *GCECloud) getInstanceByName(name string) (*gceInstance, error) {
	// Avoid changing behaviour when not managing multiple zones
	for _, zone := range gce.managedZones {
		name = canonicalizeInstanceName(name)
		res, err := gce.service.Instances.Get(gce.projectID, zone, name).Do()
		if err != nil {
			glog.Errorf("getInstanceByName: failed to get instance %s; err: %v", name, err)

			if isHTTPErrorCode(err, http.StatusNotFound) {
				continue
			}
			return nil, err
		}
		return &gceInstance{
			Zone:  lastComponent(res.Zone),
			Name:  res.Name,
			ID:    res.Id,
			Disks: res.Disks,
			Type:  lastComponent(res.MachineType),
		}, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func getInstanceIDViaMetadata() (string, error) {
	result, err := metadata.Get("instance/hostname")
	if err != nil {
		return "", err
	}
	parts := strings.Split(result, ".")
	if len(parts) == 0 {
		return "", fmt.Errorf("unexpected response: %s", result)
	}
	return parts[0], nil
}

func getCurrentExternalIDViaMetadata() (string, error) {
	externalID, err := metadata.Get("instance/id")
	if err != nil {
		return "", fmt.Errorf("couldn't get external ID: %v", err)
	}
	return externalID, nil
}

func getCurrentMachineTypeViaMetadata() (string, error) {
	mType, err := metadata.Get("instance/machine-type")
	if err != nil {
		return "", fmt.Errorf("couldn't get machine type: %v", err)
	}
	parts := strings.Split(mType, "/")
	if len(parts) != 4 {
		return "", fmt.Errorf("unexpected response for machine type: %s", mType)
	}

	return parts[3], nil
}

// isCurrentInstance uses metadata server to check if specified
// instanceID matches current machine's instanceID
func (gce *GCECloud) isCurrentInstance(instanceID string) bool {
	currentInstanceID, err := getInstanceIDViaMetadata()
	if err != nil {
		// Log and swallow error
		glog.Errorf("Failed to fetch instanceID via Metadata: %v", err)
		return false
	}

	return currentInstanceID == canonicalizeInstanceName(instanceID)
}
