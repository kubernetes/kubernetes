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
	"strings"

	"k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"

	"github.com/golang/glog"
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

type DiskType string

const (
	DiskTypeSSD      = "pd-ssd"
	DiskTypeStandard = "pd-standard"

	diskTypeDefault               = DiskTypeStandard
	diskTypeURITemplateSingleZone = "%s/zones/%s/diskTypes/%s"   // {gce.projectID}/zones/{disk.Zone}/diskTypes/{disk.Type}"
	diskTypeURITemplateRegional   = "%s/regions/%s/diskTypes/%s" // {gce.projectID}/regions/{disk.Region}/diskTypes/{disk.Type}"
	diskTypePersistent            = "PERSISTENT"

	diskSourceURITemplateSingleZone = "%s/zones/%s/disks/%s"   // {gce.projectID}/zones/{disk.Zone}/disks/{disk.Name}"
	diskSourceURITemplateRegional   = "%s/regions/%s/disks/%s" //{gce.projectID}/regions/{disk.Region}/disks/repd"

	replicaZoneURITemplateSingleZone = "%s/zones/%s" // {gce.projectID}/zones/{disk.Zone}
)

type diskServiceManager interface {
	// Creates a new persistent disk on GCE with the given disk spec.
	CreateDiskOnCloudProvider(
		name string,
		sizeGb int64,
		tagsStr string,
		diskType string,
		zone string) (gceObject, error)

	// Creates a new regional persistent disk on GCE with the given disk spec.
	CreateRegionalDiskOnCloudProvider(
		name string,
		sizeGb int64,
		tagsStr string,
		diskType string,
		zones sets.String) (gceObject, error)

	// Deletes the persistent disk from GCE with the given diskName.
	DeleteDiskOnCloudProvider(zone string, disk string) (gceObject, error)

	// Deletes the regional persistent disk from GCE with the given diskName.
	DeleteRegionalDiskOnCloudProvider(diskName string) (gceObject, error)

	// Attach a persistent disk on GCE with the given disk spec to the specified instance.
	AttachDiskOnCloudProvider(
		disk *GCEDisk,
		readWrite string,
		instanceZone string,
		instanceName string) (gceObject, error)

	// Detach a persistent disk on GCE with the given disk spec from the specified instance.
	DetachDiskOnCloudProvider(
		instanceZone string,
		instanceName string,
		devicePath string) (gceObject, error)

	// Gets the persistent disk from GCE with the given diskName.
	GetDiskFromCloudProvider(zone string, diskName string) (*GCEDisk, error)

	// Gets the regional persistent disk from GCE with the given diskName.
	GetRegionalDiskFromCloudProvider(diskName string) (*GCEDisk, error)

	// Waits until GCE reports the given operation in the given zone as done.
	WaitForZoneOp(op gceObject, zone string, mc *metricContext) error

	// Waits until GCE reports the given operation in the given region is done.
	WaitForRegionalOp(op gceObject, mc *metricContext) error
}

type gceServiceManager struct {
	gce *GCECloud
}

func (manager *gceServiceManager) CreateDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	zone string) (gceObject, error) {
	diskTypeURI, err := manager.getDiskTypeURI(
		manager.gce.region /* diskRegion */, singleZone{zone}, diskType)
	if err != nil {
		return nil, err
	}

	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		diskToCreateAlpha := &computealpha.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}

		return manager.gce.serviceAlpha.Disks.Insert(
			manager.gce.projectID, zone, diskToCreateAlpha).Do()
	}

	diskToCreateV1 := &compute.Disk{
		Name:        name,
		SizeGb:      sizeGb,
		Description: tagsStr,
		Type:        diskTypeURI,
	}
	return manager.gce.service.Disks.Insert(
		manager.gce.projectID, zone, diskToCreateV1).Do()
}

func (manager *gceServiceManager) CreateRegionalDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	replicaZones sets.String) (gceObject, error) {
	diskTypeURI, err := manager.getDiskTypeURI(
		manager.gce.region /* diskRegion */, multiZone{replicaZones}, diskType)
	if err != nil {
		return nil, err
	}
	fullyQualifiedReplicaZones := []string{}
	for _, replicaZone := range replicaZones.UnsortedList() {
		fullyQualifiedReplicaZones = append(
			fullyQualifiedReplicaZones, manager.getReplicaZoneURI(replicaZone))
	}
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		diskToCreateAlpha := &computealpha.Disk{
			Name:         name,
			SizeGb:       sizeGb,
			Description:  tagsStr,
			Type:         diskTypeURI,
			ReplicaZones: fullyQualifiedReplicaZones,
		}
		return manager.gce.serviceAlpha.RegionDisks.Insert(
			manager.gce.projectID, manager.gce.region, diskToCreateAlpha).Do()
	}

	return nil, fmt.Errorf("The regional PD feature is only available via the GCE Alpha API. Enable \"GCEDiskAlphaAPI\" in the list of \"alpha-features\" in \"gce.conf\" to use the feature.")
}

func (manager *gceServiceManager) AttachDiskOnCloudProvider(
	disk *GCEDisk,
	readWrite string,
	instanceZone string,
	instanceName string) (gceObject, error) {
	source, err := manager.getDiskSourceURI(disk)
	if err != nil {
		return nil, err
	}

	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		attachedDiskAlpha := &computealpha.AttachedDisk{
			DeviceName: disk.Name,
			Kind:       disk.Kind,
			Mode:       readWrite,
			Source:     source,
			Type:       diskTypePersistent,
		}
		return manager.gce.serviceAlpha.Instances.AttachDisk(
			manager.gce.projectID, instanceZone, instanceName, attachedDiskAlpha).Do()
	}

	attachedDiskV1 := &compute.AttachedDisk{
		DeviceName: disk.Name,
		Kind:       disk.Kind,
		Mode:       readWrite,
		Source:     source,
		Type:       disk.Type,
	}
	return manager.gce.service.Instances.AttachDisk(
		manager.gce.projectID, instanceZone, instanceName, attachedDiskV1).Do()
}

func (manager *gceServiceManager) DetachDiskOnCloudProvider(
	instanceZone string,
	instanceName string,
	devicePath string) (gceObject, error) {
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		manager.gce.serviceAlpha.Instances.DetachDisk(
			manager.gce.projectID, instanceZone, instanceName, devicePath).Do()
	}

	return manager.gce.service.Instances.DetachDisk(
		manager.gce.projectID, instanceZone, instanceName, devicePath).Do()
}

func (manager *gceServiceManager) GetDiskFromCloudProvider(
	zone string,
	diskName string) (*GCEDisk, error) {
	if zone == "" {
		return nil, fmt.Errorf("Can not fetch disk %q. Zone is empty.", diskName)
	}

	if diskName == "" {
		return nil, fmt.Errorf("Can not fetch disk. Zone is specified (%q). But disk name is empty.", zone)
	}

	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		diskAlpha, err := manager.gce.serviceAlpha.Disks.Get(
			manager.gce.projectID, zone, diskName).Do()
		if err != nil {
			return nil, err
		}

		var zoneInfo zoneType
		if len(diskAlpha.ReplicaZones) > 1 {
			zones := sets.NewString()
			for _, zoneURI := range diskAlpha.ReplicaZones {
				zones.Insert(lastComponent(zoneURI))
			}
			zoneInfo = multiZone{zones}
		} else {
			zoneInfo = singleZone{lastComponent(diskAlpha.Zone)}
			if diskAlpha.Zone == "" {
				zoneInfo = singleZone{lastComponent(zone)}
			}
		}

		region := strings.TrimSpace(lastComponent(diskAlpha.Region))
		if region == "" {
			region, err = manager.getRegionFromZone(zoneInfo)
			if err != nil {
				return nil, fmt.Errorf("failed to extract region from zone for %q/%q err=%v", zone, diskName, err)
			}
		}

		return &GCEDisk{
			ZoneInfo: zoneInfo,
			Region:   region,
			Name:     diskAlpha.Name,
			Kind:     diskAlpha.Kind,
			Type:     diskAlpha.Type,
		}, nil
	}

	diskStable, err := manager.gce.service.Disks.Get(
		manager.gce.projectID, zone, diskName).Do()
	if err != nil {
		return nil, err
	}

	zoneInfo := singleZone{strings.TrimSpace(lastComponent(diskStable.Zone))}
	if zoneInfo.zone == "" {
		zoneInfo = singleZone{zone}
	}

	region, err := manager.getRegionFromZone(zoneInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to extract region from zone for %q/%q err=%v", zone, diskName, err)
	}

	return &GCEDisk{
		ZoneInfo: zoneInfo,
		Region:   region,
		Name:     diskStable.Name,
		Kind:     diskStable.Kind,
		Type:     diskStable.Type,
	}, nil
}

func (manager *gceServiceManager) GetRegionalDiskFromCloudProvider(
	diskName string) (*GCEDisk, error) {

	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		diskAlpha, err := manager.gce.serviceAlpha.RegionDisks.Get(
			manager.gce.projectID, manager.gce.region, diskName).Do()
		if err != nil {
			return nil, err
		}

		zones := sets.NewString()
		for _, zoneURI := range diskAlpha.ReplicaZones {
			zones.Insert(lastComponent(zoneURI))
		}

		return &GCEDisk{
			ZoneInfo: multiZone{zones},
			Region:   lastComponent(diskAlpha.Region),
			Name:     diskAlpha.Name,
			Kind:     diskAlpha.Kind,
			Type:     diskAlpha.Type,
		}, nil
	}

	return nil, fmt.Errorf("The regional PD feature is only available via the GCE Alpha API. Enable \"GCEDiskAlphaAPI\" in the list of \"alpha-features\" in \"gce.conf\" to use the feature.")
}

func (manager *gceServiceManager) DeleteDiskOnCloudProvider(
	zone string,
	diskName string) (gceObject, error) {

	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		return manager.gce.serviceAlpha.Disks.Delete(
			manager.gce.projectID, zone, diskName).Do()
	}

	return manager.gce.service.Disks.Delete(
		manager.gce.projectID, zone, diskName).Do()
}

func (manager *gceServiceManager) DeleteRegionalDiskOnCloudProvider(
	diskName string) (gceObject, error) {
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		return manager.gce.serviceAlpha.RegionDisks.Delete(
			manager.gce.projectID, manager.gce.region, diskName).Do()
	}

	return nil, fmt.Errorf("DeleteRegionalDiskOnCloudProvider is a regional PD feature and is only available via the GCE Alpha API. Enable \"GCEDiskAlphaAPI\" in the list of \"alpha-features\" in \"gce.conf\" to use the feature.")
}

func (manager *gceServiceManager) WaitForZoneOp(
	op gceObject, zone string, mc *metricContext) error {
	return manager.gce.waitForZoneOp(op, zone, mc)
}

func (manager *gceServiceManager) WaitForRegionalOp(
	op gceObject, mc *metricContext) error {
	return manager.gce.waitForRegionOp(op, manager.gce.region, mc)
}

func (manager *gceServiceManager) getDiskSourceURI(disk *GCEDisk) (string, error) {
	getProjectsAPIEndpoint := manager.getProjectsAPIEndpoint()
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		getProjectsAPIEndpoint = manager.getProjectsAPIEndpointAlpha()
	}

	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		if zoneInfo.zone == "" || disk.Region == "" {
			// Unexpected, but sanity-check
			return "", fmt.Errorf("PD does not have zone/region information: %#v", disk)
		}

		return getProjectsAPIEndpoint + fmt.Sprintf(
			diskSourceURITemplateSingleZone,
			manager.gce.projectID,
			zoneInfo.zone,
			disk.Name), nil
	case multiZone:
		if zoneInfo.replicaZones == nil || zoneInfo.replicaZones.Len() <= 0 {
			// Unexpected, but sanity-check
			return "", fmt.Errorf("PD is regional but does not have any replicaZones specified: %v", disk)
		}
		return getProjectsAPIEndpoint + fmt.Sprintf(
			diskSourceURITemplateRegional,
			manager.gce.projectID,
			disk.Region,
			disk.Name), nil
	case nil:
		// Unexpected, but sanity-check
		return "", fmt.Errorf("PD did not have ZoneInfo: %v", disk)
	default:
		// Unexpected, but sanity-check
		return "", fmt.Errorf("disk.ZoneInfo has unexpected type %T", zoneInfo)
	}
}

func (manager *gceServiceManager) getDiskTypeURI(
	diskRegion string, diskZoneInfo zoneType, diskType string) (string, error) {
	getProjectsAPIEndpoint := manager.getProjectsAPIEndpoint()
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		getProjectsAPIEndpoint = manager.getProjectsAPIEndpointAlpha()
	}

	switch zoneInfo := diskZoneInfo.(type) {
	case singleZone:
		if zoneInfo.zone == "" {
			return "", fmt.Errorf("zone is empty: %v", zoneInfo)
		}

		return getProjectsAPIEndpoint + fmt.Sprintf(
			diskTypeURITemplateSingleZone,
			manager.gce.projectID,
			zoneInfo.zone,
			diskType), nil
	case multiZone:
		if zoneInfo.replicaZones == nil || zoneInfo.replicaZones.Len() <= 0 {
			return "", fmt.Errorf("zoneInfo is regional but does not have any replicaZones specified: %v", zoneInfo)
		}
		return getProjectsAPIEndpoint + fmt.Sprintf(
			diskTypeURITemplateRegional,
			manager.gce.projectID,
			diskRegion,
			diskType), nil
	case nil:
		return "", fmt.Errorf("zoneInfo nil")
	default:
		return "", fmt.Errorf("zoneInfo has unexpected type %T", zoneInfo)
	}
}

func (manager *gceServiceManager) getReplicaZoneURI(zone string) string {
	getProjectsAPIEndpoint := manager.getProjectsAPIEndpoint()
	if manager.gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		getProjectsAPIEndpoint = manager.getProjectsAPIEndpointAlpha()
	}

	return getProjectsAPIEndpoint + fmt.Sprintf(
		replicaZoneURITemplateSingleZone,
		manager.gce.projectID,
		zone)
}

func (manager *gceServiceManager) getRegionFromZone(zoneInfo zoneType) (string, error) {
	var zone string
	switch zoneInfo := zoneInfo.(type) {
	case singleZone:
		if zoneInfo.zone == "" {
			// Unexpected, but sanity-check
			return "", fmt.Errorf("PD is single zone, but zone is not specified: %#v", zoneInfo)
		}

		zone = zoneInfo.zone
	case multiZone:
		if zoneInfo.replicaZones == nil || zoneInfo.replicaZones.Len() <= 0 {
			// Unexpected, but sanity-check
			return "", fmt.Errorf("PD is regional but does not have any replicaZones specified: %v", zoneInfo)
		}

		zone = zoneInfo.replicaZones.UnsortedList()[0]
	case nil:
		// Unexpected, but sanity-check
		return "", fmt.Errorf("zoneInfo is nil")
	default:
		// Unexpected, but sanity-check
		return "", fmt.Errorf("zoneInfo has unexpected type %T", zoneInfo)
	}

	region, err := GetGCERegion(zone)
	if err != nil {
		glog.Warningf("failed to parse GCE region from zone %q: %v", zone, err)
		region = manager.gce.region
	}

	return region, nil
}

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

	// CreateRegionalDisk creates a new Regional Persistent Disk, with the
	// specified properties, replicated to the specified zones. Tags are
	// serialized as JSON into Description field.
	CreateRegionalDisk(name string, diskType string, replicaZones sets.String, sizeGb int64, tags map[string]string) error

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

// GCECloud implements PVLabeler.
var _ cloudprovider.PVLabeler = (*GCECloud)(nil)

type GCEDisk struct {
	ZoneInfo zoneType
	Region   string
	Name     string
	Kind     string
	Type     string
}

type zoneType interface {
	isZoneType()
}

type multiZone struct {
	replicaZones sets.String
}

type singleZone struct {
	zone string
}

func (m multiZone) isZoneType()  {}
func (s singleZone) isZoneType() {}

func newDiskMetricContextZonal(request, region, zone string) *metricContext {
	return newGenericMetricContext("disk", request, region, zone, computeV1Version)
}

func newDiskMetricContextRegional(request, region string) *metricContext {
	return newGenericMetricContext("disk", request, region, unusedMetricLabel, computeV1Version)
}

func (gce *GCECloud) GetLabelsForVolume(pv *v1.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if pv.Spec.GCEPersistentDisk.PDName == volume.ProvisionedVolumeName {
		return nil, nil
	}

	// If the zone is already labeled, honor the hint
	zone := pv.Labels[kubeletapis.LabelZoneFailureDomain]

	labels, err := gce.GetAutoLabelsForPD(pv.Spec.GCEPersistentDisk.PDName, zone)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

func (gce *GCECloud) AttachDisk(diskName string, nodeName types.NodeName, readOnly bool) error {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return fmt.Errorf("error getting instance %q", instanceName)
	}

	// Try fetching as regional PD
	var disk *GCEDisk
	var mc *metricContext
	if gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		disk, err = gce.getRegionalDiskByName(diskName)
		if err != nil {
			glog.V(5).Infof("Could not find regional PD named %q to Attach. Will look for a zonal PD", diskName)
			err = nil
		} else {
			mc = newDiskMetricContextRegional("attach", gce.region)
		}
	}

	if disk == nil {
		disk, err = gce.getDiskByName(diskName, instance.Zone)
		if err != nil {
			return err
		}
		mc = newDiskMetricContextZonal("attach", gce.region, instance.Zone)
	}

	readWrite := "READ_WRITE"
	if readOnly {
		readWrite = "READ_ONLY"
	}

	attachOp, err := gce.manager.AttachDiskOnCloudProvider(
		disk, readWrite, instance.Zone, instance.Name)

	if err != nil {
		return mc.Observe(err)
	}

	return gce.manager.WaitForZoneOp(attachOp, instance.Zone, mc)
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

	mc := newDiskMetricContextZonal("detach", gce.region, inst.Zone)
	detachOp, err := gce.manager.DetachDiskOnCloudProvider(inst.Zone, inst.Name, devicePath)
	if err != nil {
		return mc.Observe(err)
	}

	return gce.manager.WaitForZoneOp(detachOp, inst.Zone, mc)
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
func (gce *GCECloud) CreateDisk(
	name string, diskType string, zone string, sizeGb int64, tags map[string]string) error {

	// Do not allow creation of PDs in zones that are not managed. Such PDs
	// then cannot be deleted by DeleteDisk.
	if isManaged := gce.verifyZoneIsManaged(zone); !isManaged {
		return fmt.Errorf("kubernetes does not manage zone %q", zone)
	}

	tagsStr, err := gce.encodeDiskTags(tags)
	if err != nil {
		return err
	}

	diskType, err = getDiskType(diskType)
	if err != nil {
		return err
	}

	mc := newDiskMetricContextZonal("create", gce.region, zone)

	createOp, err := gce.manager.CreateDiskOnCloudProvider(
		name, sizeGb, tagsStr, diskType, zone)

	if isGCEError(err, "alreadyExists") {
		glog.Warningf("GCE PD %q already exists, reusing", name)
		return nil
	} else if err != nil {
		return mc.Observe(err)
	}

	err = gce.manager.WaitForZoneOp(createOp, zone, mc)
	if isGCEError(err, "alreadyExists") {
		glog.Warningf("GCE PD %q already exists, reusing", name)
		return nil
	}
	return err
}

// CreateRegionalDisk creates a new Regional Persistent Disk, with the specified
// name & size, replicated to the specified zones. It stores specified tags
// encoded in JSON in Description field.
func (gce *GCECloud) CreateRegionalDisk(
	name string, diskType string, replicaZones sets.String, sizeGb int64, tags map[string]string) error {

	// Do not allow creation of PDs in zones that are not managed. Such PDs
	// then cannot be deleted by DeleteDisk.
	unmanagedZones := []string{}
	for _, zone := range replicaZones.UnsortedList() {
		if isManaged := gce.verifyZoneIsManaged(zone); !isManaged {
			unmanagedZones = append(unmanagedZones, zone)
		}
	}

	if len(unmanagedZones) > 0 {
		return fmt.Errorf("kubernetes does not manage specified zones: %q. Managed Zones: %q", unmanagedZones, gce.managedZones)
	}

	tagsStr, err := gce.encodeDiskTags(tags)
	if err != nil {
		return err
	}

	diskType, err = getDiskType(diskType)
	if err != nil {
		return err
	}

	mc := newDiskMetricContextRegional("create", gce.region)

	createOp, err := gce.manager.CreateRegionalDiskOnCloudProvider(
		name, sizeGb, tagsStr, diskType, replicaZones)

	if isGCEError(err, "alreadyExists") {
		glog.Warningf("GCE PD %q already exists, reusing", name)
		return nil
	} else if err != nil {
		return mc.Observe(err)
	}

	err = gce.manager.WaitForRegionalOp(createOp, mc)
	if isGCEError(err, "alreadyExists") {
		glog.Warningf("GCE PD %q already exists, reusing", name)
		return nil
	}
	return err
}

func (gce *GCECloud) verifyZoneIsManaged(zone string) bool {
	for _, managedZone := range gce.managedZones {
		if zone == managedZone {
			return true
		}
	}

	return false
}

func getDiskType(diskType string) (string, error) {
	switch diskType {
	case DiskTypeSSD, DiskTypeStandard:
		return diskType, nil
	case "":
		return diskTypeDefault, nil
	default:
		return "", fmt.Errorf("invalid GCE disk type %q", diskType)
	}
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
	var disk *GCEDisk
	var err error
	if zone == "" {
		// For regional PDs this is fine, but for zonal PDs we would like as far
		// as possible to avoid this case, because GCE doesn't guarantee that
		// volumes are uniquely named per region, just per zone. However,
		// creation of GCE PDs was originally done only by name, so we have to
		// continue to support that.
		// However, wherever possible the zone should be passed (and it is
		// passed for most cases that we can control, e.g. dynamic volume
		// provisioning).
		disk, err = gce.GetDiskByNameUnknownZone(name)
		if err != nil {
			return nil, err
		}
	} else {
		// We could assume the disks exists; we have all the information we need
		// However it is more consistent to ensure the disk exists,
		// and in future we may gather addition information (e.g. disk type, IOPS etc)
		zoneSet, err := volumeutil.LabelZonesToSet(zone)
		if err != nil {
			glog.Warningf("Failed to parse zone field: %q. Will use raw field.", zone)
		}

		if len(zoneSet) > 1 {
			// Regional PD
			disk, err = gce.getRegionalDiskByName(name)
			if err != nil {
				return nil, err
			}
		} else {
			// Zonal PD
			disk, err = gce.getDiskByName(name, zone)
			if err != nil {
				return nil, err
			}
		}
	}

	labels := make(map[string]string)
	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		if zoneInfo.zone == "" || disk.Region == "" {
			// Unexpected, but sanity-check
			return nil, fmt.Errorf("PD did not have zone/region information: %v", disk)
		}
		labels[kubeletapis.LabelZoneFailureDomain] = zoneInfo.zone
		labels[kubeletapis.LabelZoneRegion] = disk.Region
	case multiZone:
		if zoneInfo.replicaZones == nil || zoneInfo.replicaZones.Len() <= 0 {
			// Unexpected, but sanity-check
			return nil, fmt.Errorf("PD is regional but does not have any replicaZones specified: %v", disk)
		}
		labels[kubeletapis.LabelZoneFailureDomain] =
			volumeutil.ZonesSetToLabelValue(zoneInfo.replicaZones)
		labels[kubeletapis.LabelZoneRegion] = disk.Region
	case nil:
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("PD did not have ZoneInfo: %v", disk)
	default:
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("disk.ZoneInfo has unexpected type %T", zoneInfo)
	}

	return labels, nil
}

// Returns a GCEDisk for the disk, if it is found in the specified zone.
// If not found, returns (nil, nil)
func (gce *GCECloud) findDiskByName(diskName string, zone string) (*GCEDisk, error) {
	mc := newDiskMetricContextZonal("get", gce.region, zone)
	disk, err := gce.manager.GetDiskFromCloudProvider(zone, diskName)
	if err == nil {
		return disk, mc.Observe(nil)
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, mc.Observe(err)
	}
	return nil, mc.Observe(nil)
}

// Like findDiskByName, but returns an error if the disk is not found
func (gce *GCECloud) getDiskByName(diskName string, zone string) (*GCEDisk, error) {
	disk, err := gce.findDiskByName(diskName, zone)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE persistent disk not found: diskName=%q zone=%q", diskName, zone)
	}
	return disk, err
}

// Returns a GCEDisk for the regional disk, if it is found.
// If not found, returns (nil, nil)
func (gce *GCECloud) findRegionalDiskByName(diskName string) (*GCEDisk, error) {
	mc := newDiskMetricContextRegional("get", gce.region)
	disk, err := gce.manager.GetRegionalDiskFromCloudProvider(diskName)
	if err == nil {
		return disk, mc.Observe(nil)
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, mc.Observe(err)
	}
	return nil, mc.Observe(nil)
}

// Like findRegionalDiskByName, but returns an error if the disk is not found
func (gce *GCECloud) getRegionalDiskByName(diskName string) (*GCEDisk, error) {
	disk, err := gce.findRegionalDiskByName(diskName)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE regional persistent disk not found: diskName=%q", diskName)
	}
	return disk, err
}

// Scans all managed zones to return the GCE PD
// Prefer getDiskByName, if the zone can be established
// Return cloudprovider.DiskNotFound if the given disk cannot be found in any zone
func (gce *GCECloud) GetDiskByNameUnknownZone(diskName string) (*GCEDisk, error) {
	if gce.AlphaFeatureGate.Enabled(AlphaFeatureGCEDisk) {
		regionalDisk, err := gce.getRegionalDiskByName(diskName)
		if err == nil {
			return regionalDisk, err
		}
	}

	// Note: this is the gotcha right now with GCE PD support:
	// disk names are not unique per-region.
	// (I can create two volumes with name "myvol" in e.g. us-central1-b & us-central1-f)
	// For now, this is simply undefined behvaiour.
	//
	// In future, we will have to require users to qualify their disk
	// "us-central1-a/mydisk".  We could do this for them as part of
	// admission control, but that might be a little weird (values changing
	// on create)

	var found *GCEDisk
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
			switch zoneInfo := disk.ZoneInfo.(type) {
			case multiZone:
				if zoneInfo.replicaZones.Has(zone) {
					glog.Warningf("GCE PD name (%q) was found in multiple zones (%q), but ok because it is a RegionalDisk.",
						diskName, zoneInfo.replicaZones)
					continue
				}
				return nil, fmt.Errorf("GCE PD name was found in multiple zones: %q", diskName)
			default:
				return nil, fmt.Errorf("GCE PD name was found in multiple zones: %q", diskName)
			}
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
	disk, err := gce.GetDiskByNameUnknownZone(diskToDelete)
	if err != nil {
		return err
	}

	var mc *metricContext

	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		mc = newDiskMetricContextZonal("delete", disk.Region, zoneInfo.zone)
		deleteOp, err := gce.manager.DeleteDiskOnCloudProvider(zoneInfo.zone, disk.Name)
		if err != nil {
			return mc.Observe(err)
		}
		return gce.manager.WaitForZoneOp(deleteOp, zoneInfo.zone, mc)
	case multiZone:
		mc = newDiskMetricContextRegional("delete", disk.Region)
		deleteOp, err := gce.manager.DeleteRegionalDiskOnCloudProvider(disk.Name)
		if err != nil {
			return mc.Observe(err)
		}
		return gce.manager.WaitForRegionalOp(deleteOp, mc)
	case nil:
		return fmt.Errorf("PD has nil ZoneInfo: %v", disk)
	default:
		return fmt.Errorf("disk.ZoneInfo has unexpected type %T", zoneInfo)
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
