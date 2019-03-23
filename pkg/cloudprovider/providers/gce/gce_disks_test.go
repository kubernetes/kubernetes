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
	"testing"

	"fmt"

	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
)

// TODO TODO write a test for GetDiskByNameUnknownZone and make sure casting logic works
// TODO TODO verify that RegionDisks.Get does not return non-replica disks

func TestCreateDisk_Basic(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       []string{"zone1"},
		projectID:          gceProjectID,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128
	tags := make(map[string]string)
	tags["test-tag"] = "test-value"

	expectedDiskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(
		diskTypeURITemplateSingleZone, gceProjectID, zone, diskType)
	expectedDescription := "{\"test-tag\":\"test-value\"}"

	/* Act */
	err := gce.CreateDisk(diskName, diskType, zone, sizeGb, tags)

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if !fakeManager.createDiskCalled {
		t.Error("Never called GCE disk create.")
	}

	// Partial check of equality between disk description sent to GCE and parameters of method.
	diskToCreate := fakeManager.diskToCreateStable
	if diskToCreate.Name != diskName {
		t.Errorf("Expected disk name: %s; Actual: %s", diskName, diskToCreate.Name)
	}

	if diskToCreate.Type != expectedDiskTypeURI {
		t.Errorf("Expected disk type: %s; Actual: %s", expectedDiskTypeURI, diskToCreate.Type)
	}
	if diskToCreate.SizeGb != sizeGb {
		t.Errorf("Expected disk size: %d; Actual: %d", sizeGb, diskToCreate.SizeGb)
	}
	if diskToCreate.Description != expectedDescription {
		t.Errorf("Expected tag string: %s; Actual: %s", expectedDescription, diskToCreate.Description)
	}
}

func TestCreateRegionalDisk_Basic(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1", "zone3", "zone2"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)

	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		projectID:          gceProjectID,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	diskName := "disk"
	diskType := DiskTypeSSD
	replicaZones := sets.NewString("zone1", "zone2")
	const sizeGb int64 = 128
	tags := make(map[string]string)
	tags["test-tag"] = "test-value"

	expectedDiskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(
		diskTypeURITemplateRegional, gceProjectID, gceRegion, diskType)
	expectedDescription := "{\"test-tag\":\"test-value\"}"

	/* Act */
	err := gce.CreateRegionalDisk(diskName, diskType, replicaZones, sizeGb, tags)

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if !fakeManager.createDiskCalled {
		t.Error("Never called GCE disk create.")
	}

	// Partial check of equality between disk description sent to GCE and parameters of method.
	diskToCreate := fakeManager.diskToCreateStable
	if diskToCreate.Name != diskName {
		t.Errorf("Expected disk name: %s; Actual: %s", diskName, diskToCreate.Name)
	}

	if diskToCreate.Type != expectedDiskTypeURI {
		t.Errorf("Expected disk type: %s; Actual: %s", expectedDiskTypeURI, diskToCreate.Type)
	}
	if diskToCreate.SizeGb != sizeGb {
		t.Errorf("Expected disk size: %d; Actual: %d", sizeGb, diskToCreate.SizeGb)
	}
	if diskToCreate.Description != expectedDescription {
		t.Errorf("Expected tag string: %s; Actual: %s", expectedDescription, diskToCreate.Description)
	}
}

func TestCreateDisk_DiskAlreadyExists(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	// Inject disk AlreadyExists error.
	alreadyExistsError := googleapi.ErrorItem{Reason: "alreadyExists"}
	fakeManager.opError = &googleapi.Error{
		Errors: []googleapi.ErrorItem{alreadyExistsError},
	}

	/* Act */
	err := gce.CreateDisk("disk", DiskTypeSSD, "zone1", 128, nil)

	/* Assert */
	if err != nil {
		t.Error(
			"Expected success when a disk with the given name already exists, but an error is returned.")
	}
}

func TestCreateDisk_WrongZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true }}

	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	/* Act */
	err := gce.CreateDisk(diskName, diskType, "zone2", sizeGb, nil)

	/* Assert */
	if err == nil {
		t.Error("Expected error when zone is not managed, but none returned.")
	}
}

func TestCreateDisk_NoManagedZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true }}

	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	/* Act */
	err := gce.CreateDisk(diskName, diskType, "zone1", sizeGb, nil)

	/* Assert */
	if err == nil {
		t.Error("Expected error when managedZones is empty, but none returned.")
	}
}

func TestCreateDisk_BadDiskType(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	gce := Cloud{manager: fakeManager,
		managedZones:       zonesWithNodes,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true }}

	diskName := "disk"
	diskType := "arbitrary-disk"
	zone := "zone1"
	const sizeGb int64 = 128

	/* Act */
	err := gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Assert */
	if err == nil {
		t.Error("Expected error when disk type is not supported, but none returned.")
	}
}

func TestCreateDisk_MultiZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1", "zone2", "zone3"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128

	/* Act & Assert */
	for _, zone := range gce.managedZones {
		diskName = zone + "disk"
		err := gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
		if err != nil {
			t.Errorf("Error creating disk in zone '%v'; error: \"%v\"", zone, err)
		}
	}
}

func TestDeleteDisk_Basic(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128

	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Act */
	err := gce.DeleteDisk(diskName)

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if !fakeManager.deleteDiskCalled {
		t.Error("Never called GCE disk delete.")
	}

}

func TestDeleteDisk_NotFound(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	diskName := "disk"

	/* Act */
	err := gce.DeleteDisk(diskName)

	/* Assert */
	if err != nil {
		t.Error("Expected successful operation when disk is not found, but an error is returned.")
	}
}

func TestDeleteDisk_ResourceBeingUsed(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128

	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	fakeManager.resourceInUse = true

	/* Act */
	err := gce.DeleteDisk(diskName)

	/* Assert */
	if err == nil {
		t.Error("Expected error when disk is in use, but none returned.")
	}
}

func TestDeleteDisk_SameDiskMultiZone(t *testing.T) {
	/* Assert */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1", "zone2", "zone3"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	/* Act */
	// DeleteDisk will call FakeServiceManager.GetDiskFromCloudProvider() with all zones,
	// and FakeServiceManager.GetDiskFromCloudProvider() always returns a disk,
	// so DeleteDisk thinks a disk with diskName exists in all zones.
	err := gce.DeleteDisk(diskName)

	/* Assert */
	if err == nil {
		t.Error("Expected error when disk is found in multiple zones, but none returned.")
	}
}

func TestDeleteDisk_DiffDiskMultiZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"zone1"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	for _, zone := range gce.managedZones {
		diskName = zone + "disk"
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	/* Act & Assert */
	var err error
	for _, zone := range gce.managedZones {
		diskName = zone + "disk"
		err = gce.DeleteDisk(diskName)
		if err != nil {
			t.Errorf("Error deleting disk in zone '%v'; error: \"%v\"", zone, err)
		}
	}
}

func TestGetAutoLabelsForPD_Basic(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "us-central1"
	zone := "us-central1-c"
	zonesWithNodes := []string{zone}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Act */
	labels, err := gce.GetAutoLabelsForPD(diskName, zone)

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if labels[v1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[v1.LabelZoneFailureDomain], zone)
	}
	if labels[v1.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'us-central1'", labels[v1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_NoZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "europe-west1"
	zone := "europe-west1-d"
	zonesWithNodes := []string{zone}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Act */
	labels, err := gce.GetAutoLabelsForPD(diskName, "")

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if labels[v1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[v1.LabelZoneFailureDomain], zone)
	}
	if labels[v1.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'europe-west1'", labels[v1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DiskNotFound(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zone := "asia-northeast1-a"
	zonesWithNodes := []string{zone}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	gce := Cloud{manager: fakeManager,
		managedZones:       zonesWithNodes,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true }}

	/* Act */
	_, err := gce.GetAutoLabelsForPD(diskName, zone)

	/* Assert */
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DiskNotFoundAndNoZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}

	/* Act */
	_, err := gce.GetAutoLabelsForPD(diskName, "")

	/* Assert */
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DupDisk(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "us-west1"
	zonesWithNodes := []string{"us-west1-b", "asia-southeast1-a"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "us-west1-b"
	const sizeGb int64 = 128

	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	/* Act */
	labels, err := gce.GetAutoLabelsForPD(diskName, zone)

	/* Assert */
	if err != nil {
		t.Error("Disk name and zone uniquely identifies a disk, yet an error is returned.")
	}
	if labels[v1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[v1.LabelZoneFailureDomain], zone)
	}
	if labels[v1.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'us-west1'", labels[v1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DupDiskNoZone(t *testing.T) {
	/* Arrange */
	gceProjectID := "test-project"
	gceRegion := "fake-region"
	zonesWithNodes := []string{"us-west1-b", "asia-southeast1-a"}
	fakeManager := newFakeManager(gceProjectID, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128

	alphaFeatureGate := NewAlphaFeatureGate([]string{})
	gce := Cloud{
		manager:            fakeManager,
		managedZones:       zonesWithNodes,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          createNodeZones(zonesWithNodes),
		nodeInformerSynced: func() bool { return true },
	}
	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	/* Act */
	_, err := gce.GetAutoLabelsForPD(diskName, "")

	/* Assert */
	if err == nil {
		t.Error("Expected error when the disk is duplicated and zone is not specified, but none returned.")
	}
}

type targetClientAPI int

const (
	targetStable targetClientAPI = iota
	targetBeta
	targetAlpha
)

type FakeServiceManager struct {
	// Common fields shared among tests
	targetAPI     targetClientAPI
	gceProjectID  string
	gceRegion     string
	zonalDisks    map[string]string      // zone: diskName
	regionalDisks map[string]sets.String // diskName: zones
	opError       error

	// Fields for TestCreateDisk
	createDiskCalled   bool
	diskToCreateAlpha  *computealpha.Disk
	diskToCreateBeta   *computebeta.Disk
	diskToCreateStable *compute.Disk

	// Fields for TestDeleteDisk
	deleteDiskCalled bool
	resourceInUse    bool // Marks the disk as in-use
}

func newFakeManager(gceProjectID string, gceRegion string) *FakeServiceManager {
	return &FakeServiceManager{
		zonalDisks:    make(map[string]string),
		regionalDisks: make(map[string]sets.String),
		gceProjectID:  gceProjectID,
		gceRegion:     gceRegion,
	}
}

/**
 * Upon disk creation, disk info is stored in FakeServiceManager
 * to be used by other tested methods.
 */
func (manager *FakeServiceManager) CreateDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	zone string) error {
	manager.createDiskCalled = true

	switch t := manager.targetAPI; t {
	case targetStable:
		diskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateV1 := &compute.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateStable = diskToCreateV1
		manager.zonalDisks[zone] = diskToCreateV1.Name
		return nil
	case targetBeta:
		diskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateBeta := &computebeta.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateBeta = diskToCreateBeta
		manager.zonalDisks[zone] = diskToCreateBeta.Name
		return nil
	case targetAlpha:
		diskTypeURI := gceComputeAPIEndpointBeta + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateAlpha := &computealpha.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateAlpha = diskToCreateAlpha
		manager.zonalDisks[zone] = diskToCreateAlpha.Name
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

/**
 * Upon disk creation, disk info is stored in FakeServiceManager
 * to be used by other tested methods.
 */
func (manager *FakeServiceManager) CreateRegionalDiskOnCloudProvider(
	name string,
	sizeGb int64,
	tagsStr string,
	diskType string,
	zones sets.String) error {

	manager.createDiskCalled = true
	diskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeURITemplateRegional, manager.gceProjectID, manager.gceRegion, diskType)
	switch t := manager.targetAPI; t {
	case targetStable:
		diskToCreateV1 := &compute.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateStable = diskToCreateV1
		manager.regionalDisks[diskToCreateV1.Name] = zones
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) AttachDiskOnCloudProvider(
	disk *Disk,
	readWrite string,
	instanceZone string,
	instanceName string) error {

	switch t := manager.targetAPI; t {
	case targetStable:
		return nil
	case targetBeta:
		return nil
	case targetAlpha:
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) DetachDiskOnCloudProvider(
	instanceZone string,
	instanceName string,
	devicePath string) error {
	switch t := manager.targetAPI; t {
	case targetStable:
		return nil
	case targetBeta:
		return nil
	case targetAlpha:
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

/**
 * Gets disk info stored in the FakeServiceManager.
 */
func (manager *FakeServiceManager) GetDiskFromCloudProvider(
	zone string, diskName string) (*Disk, error) {

	if manager.zonalDisks[zone] == "" {
		return nil, cloudprovider.DiskNotFound
	}

	if manager.resourceInUse {
		errorItem := googleapi.ErrorItem{Reason: "resourceInUseByAnotherResource"}
		err := &googleapi.Error{Errors: []googleapi.ErrorItem{errorItem}}
		return nil, err
	}

	return &Disk{
		Region:   manager.gceRegion,
		ZoneInfo: singleZone{lastComponent(zone)},
		Name:     diskName,
		Kind:     "compute#disk",
		Type:     "type",
	}, nil
}

/**
 * Gets disk info stored in the FakeServiceManager.
 */
func (manager *FakeServiceManager) GetRegionalDiskFromCloudProvider(
	diskName string) (*Disk, error) {

	if _, ok := manager.regionalDisks[diskName]; !ok {
		return nil, cloudprovider.DiskNotFound
	}

	if manager.resourceInUse {
		errorItem := googleapi.ErrorItem{Reason: "resourceInUseByAnotherResource"}
		err := &googleapi.Error{Errors: []googleapi.ErrorItem{errorItem}}
		return nil, err
	}

	return &Disk{
		Region:   manager.gceRegion,
		ZoneInfo: multiZone{manager.regionalDisks[diskName]},
		Name:     diskName,
		Kind:     "compute#disk",
		Type:     "type",
	}, nil
}

func (manager *FakeServiceManager) ResizeDiskOnCloudProvider(
	disk *Disk,
	size int64,
	zone string) error {
	panic("Not implmented")
}

func (manager *FakeServiceManager) RegionalResizeDiskOnCloudProvider(
	disk *Disk,
	size int64) error {
	panic("Not implemented")
}

/**
 * Disk info is removed from the FakeServiceManager.
 */
func (manager *FakeServiceManager) DeleteDiskOnCloudProvider(
	zone string,
	disk string) error {

	manager.deleteDiskCalled = true
	delete(manager.zonalDisks, zone)

	switch t := manager.targetAPI; t {
	case targetStable:
		return nil
	case targetBeta:
		return nil
	case targetAlpha:
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) DeleteRegionalDiskOnCloudProvider(
	disk string) error {

	manager.deleteDiskCalled = true
	delete(manager.regionalDisks, disk)

	switch t := manager.targetAPI; t {
	case targetStable:
		return nil
	case targetBeta:
		return nil
	case targetAlpha:
		return nil
	default:
		return fmt.Errorf("unexpected type: %T", t)
	}
}

func createNodeZones(zones []string) map[string]sets.String {
	nodeZones := map[string]sets.String{}
	for _, zone := range zones {
		nodeZones[zone] = sets.NewString("dummynode")
	}
	return nodeZones
}
