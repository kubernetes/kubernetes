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
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

type DiskType string

const (
	DiskTypeSSD      = "pd-ssd"
	DiskTypeStandard = "pd-standard"

	diskTypeDefault     = DiskTypeStandard
	diskTypeUriTemplate = "https://www.googleapis.com/compute/v1/projects/%s/zones/%s/diskTypes/%s"
)

// Disks is interface for manipulation with GCE PDs.
type Disks interface {
	// AttachDisk attaches given disk to the node with the specified NodeName.
	// Current instance is used when instanceID is empty string.
	AttachDisk(diskName string, nodeName types.NodeName, readOnly bool) error

	// DetachDisk detaches given disk to the node with the specified NodeName.
	// Current instance is used when nodeName is empty string.
	DetachDisk(devicePath string, nodeName types.NodeName) error

	// DiskIsAttached checks if a disk is attached to the node with the specified NodeName.
	DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error)

	// DisksAreAttached is a batch function to check if a list of disks are attached
	// to the node with the specified NodeName.
	DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error)

	// CreateDisk creates a new PD with given properties. Tags are serialized
	// as JSON into Description field.
	CreateDisk(name string, diskType string, zone string, sizeGb int64, tags map[string]string) error

	// DeleteDisk deletes PD.
	DeleteDisk(diskToDelete string) error

	// GetAutoLabelsForPD returns labels to apply to PersistentVolume
	// representing this PD, namely failure domain and zone.
	// zone can be provided to specify the zone for the PD,
	// if empty all managed zones will be searched.
	GetAutoLabelsForPD(name string, zone string) (map[string]string, error)
}

// GCECloud implements Disks.
var _ Disks = (*GCECloud)(nil)

type gceDisk struct {
	Zone string
	Name string
	Kind string
}

func (gce *GCECloud) AttachDisk(diskName string, nodeName types.NodeName, readOnly bool) error {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return fmt.Errorf("error getting instance %q", instanceName)
	}
	disk, err := gce.getDiskByName(diskName, instance.Zone)
	if err != nil {
		return err
	}
	readWrite := "READ_WRITE"
	if readOnly {
		readWrite = "READ_ONLY"
	}
	attachedDisk := gce.convertDiskToAttachedDisk(disk, readWrite)

	attachOp, err := gce.service.Instances.AttachDisk(gce.projectID, disk.Zone, instance.Name, attachedDisk).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(attachOp, disk.Zone)
}

func (gce *GCECloud) DetachDisk(devicePath string, nodeName types.NodeName) error {
	instanceName := mapNodeNameToInstanceName(nodeName)
	inst, err := gce.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Instance %q does not exist. DetachDisk will assume PD %q is not attached to it.",
				instanceName,
				devicePath)
			return nil
		}

		return fmt.Errorf("error getting instance %q", instanceName)
	}

	detachOp, err := gce.service.Instances.DetachDisk(gce.projectID, inst.Zone, inst.Name, devicePath).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(detachOp, inst.Zone)
}

func (gce *GCECloud) DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Instance %q does not exist. DiskIsAttached will assume PD %q is not attached to it.",
				instanceName,
				diskName)
			return false, nil
		}

		return false, err
	}

	for _, disk := range instance.Disks {
		if disk.DeviceName == diskName {
			// Disk is still attached to node
			return true, nil
		}
	}

	return false, nil
}

func (gce *GCECloud) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Instance %q does not exist. DisksAreAttached will assume PD %v are not attached to it.",
				instanceName,
				diskNames)
			return attached, nil
		}

		return attached, err
	}

	for _, instanceDisk := range instance.Disks {
		for _, diskName := range diskNames {
			if instanceDisk.DeviceName == diskName {
				// Disk is still attached to node
				attached[diskName] = true
			}
		}
	}

	return attached, nil
}

// CreateDisk creates a new Persistent Disk, with the specified name &
// size, in the specified zone. It stores specified tags encoded in
// JSON in Description field.
func (gce *GCECloud) CreateDisk(name string, diskType string, zone string, sizeGb int64, tags map[string]string) error {
	// Do not allow creation of PDs in zones that are not managed. Such PDs
	// then cannot be deleted by DeleteDisk.
	isManaged := false
	for _, managedZone := range gce.managedZones {
		if zone == managedZone {
			isManaged = true
			break
		}
	}
	if !isManaged {
		return fmt.Errorf("kubernetes does not manage zone %q", zone)
	}

	tagsStr, err := gce.encodeDiskTags(tags)
	if err != nil {
		return err
	}

	switch diskType {
	case DiskTypeSSD, DiskTypeStandard:
		// noop
	case "":
		diskType = diskTypeDefault
	default:
		return fmt.Errorf("invalid GCE disk type %q", diskType)
	}
	diskTypeUri := fmt.Sprintf(diskTypeUriTemplate, gce.projectID, zone, diskType)

	diskToCreate := &compute.Disk{
		Name:        name,
		SizeGb:      sizeGb,
		Description: tagsStr,
		Type:        diskTypeUri,
	}

	createOp, err := gce.service.Disks.Insert(gce.projectID, zone, diskToCreate).Do()
	if err != nil {
		return err
	}

	err = gce.waitForZoneOp(createOp, zone)
	if isGCEError(err, "alreadyExists") {
		glog.Warningf("GCE PD %q already exists, reusing", name)
		return nil
	}
	return err
}

func (gce *GCECloud) DeleteDisk(diskToDelete string) error {
	err := gce.doDeleteDisk(diskToDelete)
	if isGCEError(err, "resourceInUseByAnotherResource") {
		return volume.NewDeletedVolumeInUseError(err.Error())
	}

	if err == cloudprovider.DiskNotFound {
		return nil
	}
	return err
}

// Builds the labels that should be automatically added to a PersistentVolume backed by a GCE PD
// Specifically, this builds FailureDomain (zone) and Region labels.
// The PersistentVolumeLabel admission controller calls this and adds the labels when a PV is created.
// If zone is specified, the volume will only be found in the specified zone,
// otherwise all managed zones will be searched.
func (gce *GCECloud) GetAutoLabelsForPD(name string, zone string) (map[string]string, error) {
	var disk *gceDisk
	var err error
	if zone == "" {
		// We would like as far as possible to avoid this case,
		// because GCE doesn't guarantee that volumes are uniquely named per region,
		// just per zone.  However, creation of GCE PDs was originally done only
		// by name, so we have to continue to support that.
		// However, wherever possible the zone should be passed (and it is passed
		// for most cases that we can control, e.g. dynamic volume provisioning)
		disk, err = gce.getDiskByNameUnknownZone(name)
		if err != nil {
			return nil, err
		}
		zone = disk.Zone
	} else {
		// We could assume the disks exists; we have all the information we need
		// However it is more consistent to ensure the disk exists,
		// and in future we may gather addition information (e.g. disk type, IOPS etc)
		disk, err = gce.getDiskByName(name, zone)
		if err != nil {
			return nil, err
		}
	}

	region, err := GetGCERegion(zone)
	if err != nil {
		return nil, err
	}

	if zone == "" || region == "" {
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("PD did not have zone/region information: %q", disk.Name)
	}

	labels := make(map[string]string)
	labels[metav1.LabelZoneFailureDomain] = zone
	labels[metav1.LabelZoneRegion] = region

	return labels, nil
}

// Returns a gceDisk for the disk, if it is found in the specified zone.
// If not found, returns (nil, nil)
func (gce *GCECloud) findDiskByName(diskName string, zone string) (*gceDisk, error) {
	disk, err := gce.service.Disks.Get(gce.projectID, zone, diskName).Do()
	if err == nil {
		d := &gceDisk{
			Zone: lastComponent(disk.Zone),
			Name: disk.Name,
			Kind: disk.Kind,
		}
		return d, nil
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, err
	}
	return nil, nil
}

// Like findDiskByName, but returns an error if the disk is not found
func (gce *GCECloud) getDiskByName(diskName string, zone string) (*gceDisk, error) {
	disk, err := gce.findDiskByName(diskName, zone)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE persistent disk not found: diskName=%q zone=%q", diskName, zone)
	}
	return disk, err
}

// Scans all managed zones to return the GCE PD
// Prefer getDiskByName, if the zone can be established
// Return cloudprovider.DiskNotFound if the given disk cannot be found in any zone
func (gce *GCECloud) getDiskByNameUnknownZone(diskName string) (*gceDisk, error) {
	// Note: this is the gotcha right now with GCE PD support:
	// disk names are not unique per-region.
	// (I can create two volumes with name "myvol" in e.g. us-central1-b & us-central1-f)
	// For now, this is simply undefined behvaiour.
	//
	// In future, we will have to require users to qualify their disk
	// "us-central1-a/mydisk".  We could do this for them as part of
	// admission control, but that might be a little weird (values changing
	// on create)

	var found *gceDisk
	for _, zone := range gce.managedZones {
		disk, err := gce.findDiskByName(diskName, zone)
		if err != nil {
			return nil, err
		}
		// findDiskByName returns (nil,nil) if the disk doesn't exist, so we can't
		// assume that a disk was found unless disk is non-nil.
		if disk == nil {
			continue
		}
		if found != nil {
			return nil, fmt.Errorf("GCE persistent disk name was found in multiple zones: %q", diskName)
		}
		found = disk
	}
	if found != nil {
		return found, nil
	}
	glog.Warningf("GCE persistent disk %q not found in managed zones (%s)",
		diskName, strings.Join(gce.managedZones, ","))

	return nil, cloudprovider.DiskNotFound
}

// encodeDiskTags encodes requested volume tags into JSON string, as GCE does
// not support tags on GCE PDs and we use Description field as fallback.
func (gce *GCECloud) encodeDiskTags(tags map[string]string) (string, error) {
	if len(tags) == 0 {
		// No tags -> empty JSON
		return "", nil
	}

	enc, err := json.Marshal(tags)
	if err != nil {
		return "", err
	}
	return string(enc), nil
}

func (gce *GCECloud) doDeleteDisk(diskToDelete string) error {
	disk, err := gce.getDiskByNameUnknownZone(diskToDelete)
	if err != nil {
		return err
	}

	deleteOp, err := gce.service.Disks.Delete(gce.projectID, disk.Zone, disk.Name).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(deleteOp, disk.Zone)
}

// Converts a Disk resource to an AttachedDisk resource.
func (gce *GCECloud) convertDiskToAttachedDisk(disk *gceDisk, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		DeviceName: disk.Name,
		Kind:       disk.Kind,
		Mode:       readWrite,
		Source: "https://" + path.Join(
			"www.googleapis.com/compute/v1/projects/",
			gce.projectID, "zones", disk.Zone, "disks", disk.Name),
		Type: "PERSISTENT",
	}
}

// isGCEError returns true if given error is a googleapi.Error with given
// reason (e.g. "resourceInUseByAnotherResource")
func isGCEError(err error, reason string) bool {
	apiErr, ok := err.(*googleapi.Error)
	if !ok {
		return false
	}

	for _, e := range apiErr.Errors {
		if e.Reason == reason {
			return true
		}
	}
	return false
}
