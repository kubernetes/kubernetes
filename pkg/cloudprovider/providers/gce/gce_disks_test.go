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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

// TODO TODO write a test for GetDiskByNameUnknownZone and make sure casting logic works
// TODO TODO verify that RegionDisks.Get does not return non-replica disks

func TestCreateDisk_Basic(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		projectID:        gceProjectId,
		AlphaFeatureGate: alphaFeatureGate,
	}

	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128
	tags := make(map[string]string)
	tags["test-tag"] = "test-value"

	expectedDiskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(
		diskTypeURITemplateSingleZone, gceProjectId, zone, diskType)
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
	if !fakeManager.doesOpMatch {
		t.Error("Ops used in WaitForZoneOp does not match what's returned by CreateDisk.")
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{AlphaFeatureGCEDisk})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1", "zone3", "zone2"},
		projectID:        gceProjectId,
		AlphaFeatureGate: alphaFeatureGate,
	}

	diskName := "disk"
	diskType := DiskTypeSSD
	replicaZones := sets.NewString("zone1", "zone2")
	const sizeGb int64 = 128
	tags := make(map[string]string)
	tags["test-tag"] = "test-value"

	expectedDiskTypeURI := gceComputeAPIEndpointAlpha + "projects/" + fmt.Sprintf(
		diskTypeURITemplateRegional, gceProjectId, gceRegion, diskType)
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
	if !fakeManager.doesOpMatch {
		t.Error("Ops used in WaitForZoneOp does not match what's returned by CreateDisk.")
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		AlphaFeatureGate: alphaFeatureGate,
	}

	// Inject disk AlreadyExists error.
	alreadyExistsError := googleapi.ErrorItem{Reason: "alreadyExists"}
	fakeManager.waitForOpError = &googleapi.Error{
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	gce := GCECloud{manager: fakeManager, managedZones: []string{}}

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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1", "zone2", "zone3"},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		AlphaFeatureGate: alphaFeatureGate,
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
	if !fakeManager.doesOpMatch {
		t.Error("Ops used in WaitForZoneOp does not match what's returned by DeleteDisk.")
	}

}

func TestDeleteDisk_NotFound(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1", "zone2", "zone3"},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"zone1"},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "us-central1"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "us-central1-c"
	const sizeGb int64 = 128
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{zone},
		AlphaFeatureGate: alphaFeatureGate,
	}

	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Act */
	labels, err := gce.GetAutoLabelsForPD(diskName, zone)

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if labels[kubeletapis.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[kubeletapis.LabelZoneFailureDomain], zone)
	}
	if labels[kubeletapis.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'us-central1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_NoZone(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "europe-west1"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "europe-west1-d"
	const sizeGb int64 = 128
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{zone},
		AlphaFeatureGate: alphaFeatureGate,
	}
	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	/* Act */
	labels, err := gce.GetAutoLabelsForPD(diskName, "")

	/* Assert */
	if err != nil {
		t.Error(err)
	}
	if labels[kubeletapis.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[kubeletapis.LabelZoneFailureDomain], zone)
	}
	if labels[kubeletapis.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'europe-west1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DiskNotFound(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	zone := "asia-northeast1-a"
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}

	/* Act */
	_, err := gce.GetAutoLabelsForPD(diskName, zone)

	/* Assert */
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DiskNotFoundAndNoZone(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{},
		AlphaFeatureGate: alphaFeatureGate,
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
	gceProjectId := "test-project"
	gceRegion := "us-west1"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "us-west1-b"
	const sizeGb int64 = 128

	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"us-west1-b", "asia-southeast1-a"},
		AlphaFeatureGate: alphaFeatureGate,
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
	if labels[kubeletapis.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[kubeletapis.LabelZoneFailureDomain], zone)
	}
	if labels[kubeletapis.LabelZoneRegion] != gceRegion {
		t.Errorf("Region is '%v', but region is 'us-west1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DupDiskNoZone(t *testing.T) {
	/* Arrange */
	gceProjectId := "test-project"
	gceRegion := "fake-region"
	fakeManager := newFakeManager(gceProjectId, gceRegion)
	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128

	alphaFeatureGate, featureGateErr := NewAlphaFeatureGate([]string{})
	if featureGateErr != nil {
		t.Error(featureGateErr)
	}
	gce := GCECloud{
		manager:          fakeManager,
		managedZones:     []string{"us-west1-b", "asia-southeast1-a"},
		AlphaFeatureGate: alphaFeatureGate,
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
	targetAPI      targetClientAPI
	gceProjectID   string
	gceRegion      string
	opAlpha        *computealpha.Operation // Mocks an operation returned by GCE API calls
	opBeta         *computebeta.Operation  // Mocks an operation returned by GCE API calls
	opStable       *compute.Operation      // Mocks an operation returned by GCE API calls
	doesOpMatch    bool
	zonalDisks     map[string]string      // zone: diskName
	regionalDisks  map[string]sets.String // diskName: zones
	waitForOpError error                  // Error to be returned by WaitForZoneOp or WaitForRegionalOp

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
	zone string) (gceObject, error) {
	manager.createDiskCalled = true

	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		diskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateV1 := &compute.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateStable = diskToCreateV1
		manager.zonalDisks[zone] = diskToCreateV1.Name
		return manager.opStable, nil
	case targetBeta:
		manager.opBeta = &computebeta.Operation{}
		diskTypeURI := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateBeta := &computebeta.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateBeta = diskToCreateBeta
		manager.zonalDisks[zone] = diskToCreateBeta.Name
		return manager.opBeta, nil
	case targetAlpha:
		manager.opAlpha = &computealpha.Operation{}
		diskTypeURI := gceComputeAPIEndpointAlpha + "projects/" + fmt.Sprintf(diskTypeURITemplateSingleZone, manager.gceProjectID, zone, diskType)
		diskToCreateAlpha := &computealpha.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateAlpha = diskToCreateAlpha
		manager.zonalDisks[zone] = diskToCreateAlpha.Name
		return manager.opAlpha, nil
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
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
	zones sets.String) (gceObject, error) {
	manager.createDiskCalled = true
	diskTypeURI := gceComputeAPIEndpointAlpha + "projects/" + fmt.Sprintf(diskTypeURITemplateRegional, manager.gceProjectID, manager.gceRegion, diskType)

	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		diskToCreateV1 := &compute.Disk{
			Name:        name,
			SizeGb:      sizeGb,
			Description: tagsStr,
			Type:        diskTypeURI,
		}
		manager.diskToCreateStable = diskToCreateV1
		manager.regionalDisks[diskToCreateV1.Name] = zones
		return manager.opStable, nil
	case targetBeta:
		return nil, fmt.Errorf("RegionalDisk CreateDisk op not supported in beta.")
	case targetAlpha:
		return nil, fmt.Errorf("RegionalDisk CreateDisk op not supported in alpha.")
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) AttachDiskOnCloudProvider(
	disk *GCEDisk,
	readWrite string,
	instanceZone string,
	instanceName string) (gceObject, error) {

	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		return manager.opStable, nil
	case targetBeta:
		manager.opBeta = &computebeta.Operation{}
		return manager.opBeta, nil
	case targetAlpha:
		manager.opAlpha = &computealpha.Operation{}
		return manager.opAlpha, nil
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) DetachDiskOnCloudProvider(
	instanceZone string,
	instanceName string,
	devicePath string) (gceObject, error) {
	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		return manager.opStable, nil
	case targetBeta:
		manager.opBeta = &computebeta.Operation{}
		return manager.opBeta, nil
	case targetAlpha:
		manager.opAlpha = &computealpha.Operation{}
		return manager.opAlpha, nil
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
	}
}

/**
 * Gets disk info stored in the FakeServiceManager.
 */
func (manager *FakeServiceManager) GetDiskFromCloudProvider(
	zone string, diskName string) (*GCEDisk, error) {

	if manager.zonalDisks[zone] == "" {
		return nil, cloudprovider.DiskNotFound
	}

	if manager.resourceInUse {
		errorItem := googleapi.ErrorItem{Reason: "resourceInUseByAnotherResource"}
		err := &googleapi.Error{Errors: []googleapi.ErrorItem{errorItem}}
		return nil, err
	}

	return &GCEDisk{
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
	diskName string) (*GCEDisk, error) {

	if _, ok := manager.regionalDisks[diskName]; !ok {
		return nil, cloudprovider.DiskNotFound
	}

	if manager.resourceInUse {
		errorItem := googleapi.ErrorItem{Reason: "resourceInUseByAnotherResource"}
		err := &googleapi.Error{Errors: []googleapi.ErrorItem{errorItem}}
		return nil, err
	}

	return &GCEDisk{
		Region:   manager.gceRegion,
		ZoneInfo: multiZone{manager.regionalDisks[diskName]},
		Name:     diskName,
		Kind:     "compute#disk",
		Type:     "type",
	}, nil
}

/**
 * Disk info is removed from the FakeServiceManager.
 */
func (manager *FakeServiceManager) DeleteDiskOnCloudProvider(
	zone string,
	disk string) (gceObject, error) {

	manager.deleteDiskCalled = true
	delete(manager.zonalDisks, zone)

	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		return manager.opStable, nil
	case targetBeta:
		manager.opBeta = &computebeta.Operation{}
		return manager.opBeta, nil
	case targetAlpha:
		manager.opAlpha = &computealpha.Operation{}
		return manager.opAlpha, nil
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) DeleteRegionalDiskOnCloudProvider(
	disk string) (gceObject, error) {

	manager.deleteDiskCalled = true
	delete(manager.regionalDisks, disk)

	switch t := manager.targetAPI; t {
	case targetStable:
		manager.opStable = &compute.Operation{}
		return manager.opStable, nil
	case targetBeta:
		manager.opBeta = &computebeta.Operation{}
		return manager.opBeta, nil
	case targetAlpha:
		manager.opAlpha = &computealpha.Operation{}
		return manager.opAlpha, nil
	default:
		return nil, fmt.Errorf("unexpected type: %T", t)
	}
}

func (manager *FakeServiceManager) WaitForZoneOp(
	op gceObject,
	zone string,
	mc *metricContext) error {
	switch v := op.(type) {
	case *computealpha.Operation:
		if op.(*computealpha.Operation) == manager.opAlpha {
			manager.doesOpMatch = true
		}
	case *computebeta.Operation:
		if op.(*computebeta.Operation) == manager.opBeta {
			manager.doesOpMatch = true
		}
	case *compute.Operation:
		if op.(*compute.Operation) == manager.opStable {
			manager.doesOpMatch = true
		}
	default:
		return fmt.Errorf("unexpected type: %T", v)
	}
	return manager.waitForOpError
}

func (manager *FakeServiceManager) WaitForRegionalOp(
	op gceObject, mc *metricContext) error {
	switch v := op.(type) {
	case *computealpha.Operation:
		if op.(*computealpha.Operation) == manager.opAlpha {
			manager.doesOpMatch = true
		}
	case *computebeta.Operation:
		if op.(*computebeta.Operation) == manager.opBeta {
			manager.doesOpMatch = true
		}
	case *compute.Operation:
		if op.(*compute.Operation) == manager.opStable {
			manager.doesOpMatch = true
		}
	default:
		return fmt.Errorf("unexpected type: %T", v)
	}
	return manager.waitForOpError
}
