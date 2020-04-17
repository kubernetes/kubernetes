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

package gce

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	cloudvolume "k8s.io/cloud-provider/volume"
	volerr "k8s.io/cloud-provider/volume/errors"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/klog"
)

// DiskType defines a specific type for holding disk types (eg. pd-ssd)
type DiskType string

const (
	// DiskTypeSSD the type for persistent SSD storage
	DiskTypeSSD = "pd-ssd"

	// DiskTypeStandard the type for standard persistent storage
	DiskTypeStandard = "pd-standard"

	diskTypeDefault               = DiskTypeStandard
	diskTypeURITemplateSingleZone = "%s/zones/%s/diskTypes/%s"   // {gce.projectID}/zones/{disk.Zone}/diskTypes/{disk.Type}"
	diskTypeURITemplateRegional   = "%s/regions/%s/diskTypes/%s" // {gce.projectID}/regions/{disk.Region}/diskTypes/{disk.Type}"
	diskTypePersistent            = "PERSISTENT"

	diskSourceURITemplateSingleZone = "%s/zones/%s/disks/%s"   // {gce.projectID}/zones/{disk.Zone}/disks/{disk.Name}"
	diskSourceURITemplateRegional   = "%s/regions/%s/disks/%s" //{gce.projectID}/regions/{disk.Region}/disks/repd"

	replicaZoneURITemplateSingleZone = "%s/zones/%s" // {gce.projectID}/zones/{disk.Zone}

	diskKind = "compute#disk"
)

type diskServiceManager interface {
	// Creates a new persistent disk on GCE with the given disk spec.
	CreateDiskOnCloudProvider(
		name string,
		sizeGb int64,
		tagsStr string,
		diskType string,
		zone string) (*Disk, error)

	// Creates a new regional persistent disk on GCE with the given disk spec.
	CreateRegionalDiskOnCloudProvider(
		name string,
		sizeGb int64,
		tagsStr string,
		diskType string,
		zones sets.String) (*Disk, error)

	// Deletes the persistent disk from GCE with the given diskName.
	DeleteDiskOnCloudProvider(zone string, disk string) error

	// Deletes the regional persistent disk from GCE with the given diskName.
	DeleteRegionalDiskOnCloudProvider(diskName string) error

	// Attach a persistent disk on GCE with the given disk spec to the specified instance.
	AttachDiskOnCloudProvider(
		disk *Disk,
		readWrite string,
		instanceZone string,
		instanceName string) error

	// Detach a persistent disk on GCE with the given disk spec from the specified instance.
	DetachDiskOnCloudProvider(
		instanceZone string,
		instanceName string,
		devicePath string) error

	ResizeDiskOnCloudProvider(disk *Disk, sizeGb int64, zone string) error
	RegionalResizeDiskOnCloudProvider(disk *Disk, sizeGb int64) error

	// Gets the persistent disk from GCE with the given diskName.
	GetDiskFromCloudProvider(zone string, diskName string) (*Disk, error)

	// Gets the regional persistent disk from GCE with the given diskName.
	GetRegionalDiskFromCloudProvider(diskName string) (*Disk, error)
}

type gceServiceManager struct {
	gce *Cloud
}

var _ diskServiceManager = &gceServiceManager{}

func (manager *gceServiceManager) CreateDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	zone string) (*Disk, error) {
	diskTypeURI, err := manager.getDiskTypeURI(
		manager.gce.region /* diskRegion */, singleZone{zone}, diskType)
	if err != nil {
		return nil, err
	}

	diskToCreateV1 := &compute.Disk{
		Name:        name,
		SizeGb:      sizeGb,
		Description: tagsStr,
		Type:        diskTypeURI,
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	disk := &Disk{
		ZoneInfo: singleZone{zone},
		Region:   manager.gce.region,
		Kind:     diskKind,
		Type:     diskTypeURI,
		SizeGb:   sizeGb,
	}
	return disk, manager.gce.c.Disks().Insert(ctx, meta.ZonalKey(name, zone), diskToCreateV1)
}

func (manager *gceServiceManager) CreateRegionalDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	replicaZones sets.String) (*Disk, error) {

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

	diskToCreate := &compute.Disk{
		Name:         name,
		SizeGb:       sizeGb,
		Description:  tagsStr,
		Type:         diskTypeURI,
		ReplicaZones: fullyQualifiedReplicaZones,
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	disk := &Disk{
		ZoneInfo: multiZone{replicaZones},
		Region:   manager.gce.region,
		Name:     name,
		Kind:     diskKind,
		Type:     diskTypeURI,
		SizeGb:   sizeGb,
	}
	return disk, manager.gce.c.RegionDisks().Insert(ctx, meta.RegionalKey(name, manager.gce.region), diskToCreate)
}

func (manager *gceServiceManager) AttachDiskOnCloudProvider(
	disk *Disk,
	readWrite string,
	instanceZone string,
	instanceName string) error {
	source, err := manager.getDiskSourceURI(disk)
	if err != nil {
		return err
	}

	attachedDiskV1 := &compute.AttachedDisk{
		DeviceName: disk.Name,
		Kind:       disk.Kind,
		Mode:       readWrite,
		Source:     source,
		Type:       diskTypePersistent,
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.Instances().AttachDisk(ctx, meta.ZonalKey(instanceName, instanceZone), attachedDiskV1)
}

func (manager *gceServiceManager) DetachDiskOnCloudProvider(
	instanceZone string,
	instanceName string,
	devicePath string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.Instances().DetachDisk(ctx, meta.ZonalKey(instanceName, instanceZone), devicePath)
}

func (manager *gceServiceManager) GetDiskFromCloudProvider(
	zone string,
	diskName string) (*Disk, error) {
	if zone == "" {
		return nil, fmt.Errorf("can not fetch disk %q, zone is empty", diskName)
	}

	if diskName == "" {
		return nil, fmt.Errorf("can not fetch disk, zone is specified (%q), but disk name is empty", zone)
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	diskStable, err := manager.gce.c.Disks().Get(ctx, meta.ZonalKey(diskName, zone))
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

	return &Disk{
		ZoneInfo: zoneInfo,
		Region:   region,
		Name:     diskStable.Name,
		Kind:     diskStable.Kind,
		Type:     diskStable.Type,
		SizeGb:   diskStable.SizeGb,
	}, nil
}

func (manager *gceServiceManager) GetRegionalDiskFromCloudProvider(
	diskName string) (*Disk, error) {

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	diskBeta, err := manager.gce.c.RegionDisks().Get(ctx, meta.RegionalKey(diskName, manager.gce.region))
	if err != nil {
		return nil, err
	}

	zones := sets.NewString()
	for _, zoneURI := range diskBeta.ReplicaZones {
		zones.Insert(lastComponent(zoneURI))
	}

	return &Disk{
		ZoneInfo: multiZone{zones},
		Region:   lastComponent(diskBeta.Region),
		Name:     diskBeta.Name,
		Kind:     diskBeta.Kind,
		Type:     diskBeta.Type,
		SizeGb:   diskBeta.SizeGb,
	}, nil
}

func (manager *gceServiceManager) DeleteDiskOnCloudProvider(
	zone string,
	diskName string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.Disks().Delete(ctx, meta.ZonalKey(diskName, zone))
}

func (manager *gceServiceManager) DeleteRegionalDiskOnCloudProvider(
	diskName string) error {

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.RegionDisks().Delete(ctx, meta.RegionalKey(diskName, manager.gce.region))
}

func (manager *gceServiceManager) getDiskSourceURI(disk *Disk) (string, error) {
	getProjectsAPIEndpoint := manager.getProjectsAPIEndpoint()

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
	return manager.getProjectsAPIEndpoint() + fmt.Sprintf(
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
		klog.Warningf("failed to parse GCE region from zone %q: %v", zone, err)
		region = manager.gce.region
	}

	return region, nil
}

func (manager *gceServiceManager) ResizeDiskOnCloudProvider(disk *Disk, sizeGb int64, zone string) error {
	resizeServiceRequest := &compute.DisksResizeRequest{
		SizeGb: sizeGb,
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.Disks().Resize(ctx, meta.ZonalKey(disk.Name, zone), resizeServiceRequest)
}

func (manager *gceServiceManager) RegionalResizeDiskOnCloudProvider(disk *Disk, sizeGb int64) error {

	resizeServiceRequest := &compute.RegionDisksResizeRequest{
		SizeGb: sizeGb,
	}

	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()
	return manager.gce.c.RegionDisks().Resize(ctx, meta.RegionalKey(disk.Name, disk.Region), resizeServiceRequest)
}

// Disks is interface for manipulation with GCE PDs.
type Disks interface {
	// AttachDisk attaches given disk to the node with the specified NodeName.
	// Current instance is used when instanceID is empty string.
	AttachDisk(diskName string, nodeName types.NodeName, readOnly bool, regional bool) error

	// DetachDisk detaches given disk to the node with the specified NodeName.
	// Current instance is used when nodeName is empty string.
	DetachDisk(devicePath string, nodeName types.NodeName) error

	// DiskIsAttached checks if a disk is attached to the node with the specified NodeName.
	DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error)

	// DisksAreAttached is a batch function to check if a list of disks are attached
	// to the node with the specified NodeName.
	DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error)

	// BulkDisksAreAttached is a batch function to check if all corresponding disks are attached to the
	// nodes specified with nodeName.
	BulkDisksAreAttached(diskByNodes map[types.NodeName][]string) (map[types.NodeName]map[string]bool, error)

	// CreateDisk creates a new PD with given properties. Tags are serialized
	// as JSON into Description field.
	CreateDisk(name string, diskType string, zone string, sizeGb int64, tags map[string]string) (*Disk, error)

	// CreateRegionalDisk creates a new Regional Persistent Disk, with the
	// specified properties, replicated to the specified zones. Tags are
	// serialized as JSON into Description field.
	CreateRegionalDisk(name string, diskType string, replicaZones sets.String, sizeGb int64, tags map[string]string) (*Disk, error)

	// DeleteDisk deletes PD.
	DeleteDisk(diskToDelete string) error

	// ResizeDisk resizes PD and returns new disk size
	ResizeDisk(diskToResize string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error)

	// GetAutoLabelsForPD returns labels to apply to PersistentVolume
	// representing this PD, namely failure domain and zone.
	GetAutoLabelsForPD(disk *Disk) (map[string]string, error)
}

// Cloud implements Disks.
var _ Disks = (*Cloud)(nil)

// Cloud implements PVLabeler.
var _ cloudprovider.PVLabeler = (*Cloud)(nil)

// Disk holds all relevant data about an instance of GCE storage
type Disk struct {
	ZoneInfo zoneType
	Region   string
	Name     string
	Kind     string
	Type     string
	SizeGb   int64
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

// GetLabelsForVolume retrieved the label info for the provided volume
func (g *Cloud) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	// Ignore any volumes that are being provisioned
	if pv.Spec.GCEPersistentDisk.PDName == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}

	// If the zone is already labeled, honor the hint
	name := pv.Spec.GCEPersistentDisk.PDName
	zone := pv.Labels[v1.LabelZoneFailureDomain]

	disk, err := g.getDiskByNameAndOptionalLabelZones(name, zone)
	if err != nil {
		return nil, err
	}
	labels, err := g.GetAutoLabelsForPD(disk)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

// getDiskByNameAndOptionalZone returns a Disk object for a disk (zonal or regional) for given name and (optional) zone(s) label.
func (g *Cloud) getDiskByNameAndOptionalLabelZones(name, labelZone string) (*Disk, error) {
	if labelZone == "" {
		return g.GetDiskByNameUnknownZone(name)
	}
	zoneSet, err := volumehelpers.LabelZonesToSet(labelZone)
	if err != nil {
		return nil, err
	}
	if len(zoneSet) > 1 {
		// Regional PD
		return g.getRegionalDiskByName(name)
	}
	return g.getDiskByName(name, labelZone)
}

// AttachDisk attaches given disk to the node with the specified NodeName.
// Current instance is used when instanceID is empty string.
func (g *Cloud) AttachDisk(diskName string, nodeName types.NodeName, readOnly bool, regional bool) error {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := g.getInstanceByName(instanceName)
	if err != nil {
		return fmt.Errorf("error getting instance %q", instanceName)
	}

	// Try fetching as regional PD
	var disk *Disk
	var mc *metricContext
	if regional {
		disk, err = g.getRegionalDiskByName(diskName)
		if err != nil {
			return err
		}
		mc = newDiskMetricContextRegional("attach", g.region)
	} else {
		disk, err = g.getDiskByName(diskName, instance.Zone)
		if err != nil {
			return err
		}
		mc = newDiskMetricContextZonal("attach", g.region, instance.Zone)
	}

	readWrite := "READ_WRITE"
	if readOnly {
		readWrite = "READ_ONLY"
	}

	return mc.Observe(g.manager.AttachDiskOnCloudProvider(disk, readWrite, instance.Zone, instance.Name))
}

// DetachDisk detaches given disk to the node with the specified NodeName.
// Current instance is used when nodeName is empty string.
func (g *Cloud) DetachDisk(devicePath string, nodeName types.NodeName) error {
	instanceName := mapNodeNameToInstanceName(nodeName)
	inst, err := g.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			klog.Warningf(
				"Instance %q does not exist. DetachDisk will assume PD %q is not attached to it.",
				instanceName,
				devicePath)
			return nil
		}

		return fmt.Errorf("error getting instance %q", instanceName)
	}

	mc := newDiskMetricContextZonal("detach", g.region, inst.Zone)
	return mc.Observe(g.manager.DetachDiskOnCloudProvider(inst.Zone, inst.Name, devicePath))
}

// DiskIsAttached checks if a disk is attached to the node with the specified NodeName.
func (g *Cloud) DiskIsAttached(diskName string, nodeName types.NodeName) (bool, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := g.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			klog.Warningf(
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

// DisksAreAttached is a batch function to check if a list of disks are attached
// to the node with the specified NodeName.
func (g *Cloud) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	instanceName := mapNodeNameToInstanceName(nodeName)
	instance, err := g.getInstanceByName(instanceName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			klog.Warningf(
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

// BulkDisksAreAttached is a batch function to check if all corresponding disks are attached to the
// nodes specified with nodeName.
func (g *Cloud) BulkDisksAreAttached(diskByNodes map[types.NodeName][]string) (map[types.NodeName]map[string]bool, error) {
	instanceNames := []string{}
	for nodeName := range diskByNodes {
		instanceNames = append(instanceNames, mapNodeNameToInstanceName(nodeName))
	}

	// List all instances with the given instance names
	// Then for each instance listed, add the disks attached to that instance to a map
	listedInstances, err := g.getFoundInstanceByNames(instanceNames)
	if err != nil {
		return nil, fmt.Errorf("error listing instances: %v", err)
	}
	listedInstanceNamesToDisks := make(map[string][]*compute.AttachedDisk)
	for _, instance := range listedInstances {
		listedInstanceNamesToDisks[instance.Name] = instance.Disks
	}

	verifyDisksAttached := make(map[types.NodeName]map[string]bool)

	// For each node and its desired attached disks that needs to be verified
	for nodeName, disksToVerify := range diskByNodes {
		instanceName := canonicalizeInstanceName(mapNodeNameToInstanceName(nodeName))
		disksActuallyAttached := listedInstanceNamesToDisks[instanceName]
		verifyDisksAttached[nodeName] = verifyDisksAttachedToNode(disksToVerify, disksActuallyAttached)
	}

	return verifyDisksAttached, nil
}

// CreateDisk creates a new Persistent Disk, with the specified name &
// size, in the specified zone. It stores specified tags encoded in
// JSON in Description field.
func (g *Cloud) CreateDisk(
	name string, diskType string, zone string, sizeGb int64, tags map[string]string) (*Disk, error) {
	// Do not allow creation of PDs in zones that are do not have nodes. Such PDs
	// are not currently usable.
	curZones, err := g.GetAllCurrentZones()
	if err != nil {
		return nil, err
	}
	if !curZones.Has(zone) {
		return nil, fmt.Errorf("kubernetes does not have a node in zone %q", zone)
	}

	tagsStr, err := g.encodeDiskTags(tags)
	if err != nil {
		return nil, err
	}

	diskType, err = getDiskType(diskType)
	if err != nil {
		return nil, err
	}

	mc := newDiskMetricContextZonal("create", g.region, zone)
	disk, err := g.manager.CreateDiskOnCloudProvider(
		name, sizeGb, tagsStr, diskType, zone)

	mc.Observe(err)
	if err != nil {
		if isGCEError(err, "alreadyExists") {
			klog.Warningf("GCE PD %q already exists, reusing", name)
			return g.manager.GetDiskFromCloudProvider(zone, name)
		}
		return nil, err
	}
	return disk, nil
}

// CreateRegionalDisk creates a new Regional Persistent Disk, with the specified
// name & size, replicated to the specified zones. It stores specified tags
// encoded in JSON in Description field.
func (g *Cloud) CreateRegionalDisk(
	name string, diskType string, replicaZones sets.String, sizeGb int64, tags map[string]string) (*Disk, error) {

	// Do not allow creation of PDs in zones that are do not have nodes. Such PDs
	// are not currently usable. This functionality should be reverted to checking
	// against managed zones if we want users to be able to create RegionalDisks
	// in zones where there are no nodes
	curZones, err := g.GetAllCurrentZones()
	if err != nil {
		return nil, err
	}
	if !curZones.IsSuperset(replicaZones) {
		return nil, fmt.Errorf("kubernetes does not have nodes in specified zones: %q. Zones that contain nodes: %q", replicaZones.Difference(curZones), curZones)
	}

	tagsStr, err := g.encodeDiskTags(tags)
	if err != nil {
		return nil, err
	}

	diskType, err = getDiskType(diskType)
	if err != nil {
		return nil, err
	}

	mc := newDiskMetricContextRegional("create", g.region)

	disk, err := g.manager.CreateRegionalDiskOnCloudProvider(
		name, sizeGb, tagsStr, diskType, replicaZones)

	mc.Observe(err)
	if err != nil {
		if isGCEError(err, "alreadyExists") {
			klog.Warningf("GCE PD %q already exists, reusing", name)
			return g.manager.GetRegionalDiskFromCloudProvider(name)
		}
		return nil, err
	}
	return disk, nil
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

// DeleteDisk deletes rgw referenced persistent disk.
func (g *Cloud) DeleteDisk(diskToDelete string) error {
	err := g.doDeleteDisk(diskToDelete)
	if isGCEError(err, "resourceInUseByAnotherResource") {
		return volerr.NewDeletedVolumeInUseError(err.Error())
	}

	if err == cloudprovider.DiskNotFound {
		return nil
	}
	return err
}

// ResizeDisk expands given disk and returns new disk size
func (g *Cloud) ResizeDisk(diskToResize string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	disk, err := g.GetDiskByNameUnknownZone(diskToResize)
	if err != nil {
		return oldSize, err
	}

	// GCE resizes in chunks of GiBs
	requestGIB := volumehelpers.RoundUpToGiB(newSize)
	newSizeQuant := resource.MustParse(fmt.Sprintf("%dGi", requestGIB))

	// If disk is already of size equal or greater than requested size, we simply return
	if disk.SizeGb >= requestGIB {
		return newSizeQuant, nil
	}

	var mc *metricContext

	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		mc = newDiskMetricContextZonal("resize", disk.Region, zoneInfo.zone)
		err := g.manager.ResizeDiskOnCloudProvider(disk, requestGIB, zoneInfo.zone)

		if err != nil {
			return oldSize, mc.Observe(err)
		}
		return newSizeQuant, mc.Observe(err)
	case multiZone:
		mc = newDiskMetricContextRegional("resize", disk.Region)
		err := g.manager.RegionalResizeDiskOnCloudProvider(disk, requestGIB)

		if err != nil {
			return oldSize, mc.Observe(err)
		}
		return newSizeQuant, mc.Observe(err)
	case nil:
		return oldSize, fmt.Errorf("PD has nil ZoneInfo: %v", disk)
	default:
		return oldSize, fmt.Errorf("disk.ZoneInfo has unexpected type %T", zoneInfo)
	}
}

// GetAutoLabelsForPD builds the labels that should be automatically added to a PersistentVolume backed by a GCE PD
// Specifically, this builds FailureDomain (zone) and Region labels.
// The PersistentVolumeLabel admission controller calls this and adds the labels when a PV is created.
func (g *Cloud) GetAutoLabelsForPD(disk *Disk) (map[string]string, error) {
	labels := make(map[string]string)
	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		if zoneInfo.zone == "" || disk.Region == "" {
			// Unexpected, but sanity-check
			return nil, fmt.Errorf("PD did not have zone/region information: %v", disk)
		}
		labels[v1.LabelZoneFailureDomain] = zoneInfo.zone
		labels[v1.LabelZoneRegion] = disk.Region
	case multiZone:
		if zoneInfo.replicaZones == nil || zoneInfo.replicaZones.Len() <= 0 {
			// Unexpected, but sanity-check
			return nil, fmt.Errorf("PD is regional but does not have any replicaZones specified: %v", disk)
		}
		labels[v1.LabelZoneFailureDomain] =
			volumehelpers.ZonesSetToLabelValue(zoneInfo.replicaZones)
		labels[v1.LabelZoneRegion] = disk.Region
	case nil:
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("PD did not have ZoneInfo: %v", disk)
	default:
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("disk.ZoneInfo has unexpected type %T", zoneInfo)
	}

	return labels, nil
}

// Returns a Disk for the disk, if it is found in the specified zone.
// If not found, returns (nil, nil)
func (g *Cloud) findDiskByName(diskName string, zone string) (*Disk, error) {
	mc := newDiskMetricContextZonal("get", g.region, zone)
	disk, err := g.manager.GetDiskFromCloudProvider(zone, diskName)
	if err == nil {
		return disk, mc.Observe(nil)
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, mc.Observe(err)
	}
	return nil, mc.Observe(nil)
}

// Like findDiskByName, but returns an error if the disk is not found
func (g *Cloud) getDiskByName(diskName string, zone string) (*Disk, error) {
	disk, err := g.findDiskByName(diskName, zone)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE persistent disk not found: diskName=%q zone=%q", diskName, zone)
	}
	return disk, err
}

// Returns a Disk for the regional disk, if it is found.
// If not found, returns (nil, nil)
func (g *Cloud) findRegionalDiskByName(diskName string) (*Disk, error) {
	mc := newDiskMetricContextRegional("get", g.region)
	disk, err := g.manager.GetRegionalDiskFromCloudProvider(diskName)
	if err == nil {
		return disk, mc.Observe(nil)
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, mc.Observe(err)
	}
	return nil, mc.Observe(nil)
}

// Like findRegionalDiskByName, but returns an error if the disk is not found
func (g *Cloud) getRegionalDiskByName(diskName string) (*Disk, error) {
	disk, err := g.findRegionalDiskByName(diskName)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE regional persistent disk not found: diskName=%q", diskName)
	}
	return disk, err
}

// GetDiskByNameUnknownZone scans all managed zones to return the GCE PD
// Prefer getDiskByName, if the zone can be established
// Return cloudprovider.DiskNotFound if the given disk cannot be found in any zone
func (g *Cloud) GetDiskByNameUnknownZone(diskName string) (*Disk, error) {
	regionalDisk, err := g.getRegionalDiskByName(diskName)
	if err == nil {
		return regionalDisk, err
	}

	// Note: this is the gotcha right now with GCE PD support:
	// disk names are not unique per-region.
	// (I can create two volumes with name "myvol" in e.g. us-central1-b & us-central1-f)
	// For now, this is simply undefined behaviour.
	//
	// In future, we will have to require users to qualify their disk
	// "us-central1-a/mydisk".  We could do this for them as part of
	// admission control, but that might be a little weird (values changing
	// on create)

	var found *Disk
	for _, zone := range g.managedZones {
		disk, err := g.findDiskByName(diskName, zone)
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
					klog.Warningf("GCE PD name (%q) was found in multiple zones (%q), but ok because it is a RegionalDisk.",
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
	klog.Warningf("GCE persistent disk %q not found in managed zones (%s)",
		diskName, strings.Join(g.managedZones, ","))

	return nil, cloudprovider.DiskNotFound
}

// encodeDiskTags encodes requested volume tags into JSON string, as GCE does
// not support tags on GCE PDs and we use Description field as fallback.
func (g *Cloud) encodeDiskTags(tags map[string]string) (string, error) {
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

func (g *Cloud) doDeleteDisk(diskToDelete string) error {
	disk, err := g.GetDiskByNameUnknownZone(diskToDelete)
	if err != nil {
		return err
	}

	var mc *metricContext

	switch zoneInfo := disk.ZoneInfo.(type) {
	case singleZone:
		mc = newDiskMetricContextZonal("delete", disk.Region, zoneInfo.zone)
		return mc.Observe(g.manager.DeleteDiskOnCloudProvider(zoneInfo.zone, disk.Name))
	case multiZone:
		mc = newDiskMetricContextRegional("delete", disk.Region)
		return mc.Observe(g.manager.DeleteRegionalDiskOnCloudProvider(disk.Name))
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

// verifyDisksAttachedToNode takes in an slice of disks that should be attached to an instance, and the
// slice of disks actually attached to it. It returns a map verifying if the disks are actually attached.
func verifyDisksAttachedToNode(disksToVerify []string, disksActuallyAttached []*compute.AttachedDisk) map[string]bool {
	verifiedDisks := make(map[string]bool)
	diskNamesActuallyAttached := sets.NewString()
	for _, disk := range disksActuallyAttached {
		diskNamesActuallyAttached.Insert(disk.DeviceName)
	}

	// For every disk that's supposed to be attached, verify that it is
	for _, diskName := range disksToVerify {
		verifiedDisks[diskName] = diskNamesActuallyAttached.Has(diskName)
	}

	return verifiedDisks
}
