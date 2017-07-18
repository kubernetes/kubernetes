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
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

func TestCreateDisk_Basic(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
	projectId := "test-project"
	gce := GCECloud{
		manager:      fakeManager,
		managedZones: []string{"zone1"},
		projectID:    projectId,
	}

	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128
	tags := make(map[string]string)
	tags["test-tag"] = "test-value"

	diskTypeUri := gceComputeAPIEndpoint + "projects/" + fmt.Sprintf(diskTypeUriTemplate, projectId, zone, diskType)
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
	diskToCreate := fakeManager.diskToCreate
	if diskToCreate.Name != diskName {
		t.Errorf("Expected disk name: %s; Actual: %s", diskName, diskToCreate.Name)
	}

	if diskToCreate.Type != diskTypeUri {
		t.Errorf("Expected disk type: %s; Actual: %s", diskTypeUri, diskToCreate.Type)
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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

	// Inject disk AlreadyExists error.
	alreadyExistsError := googleapi.ErrorItem{Reason: "alreadyExists"}
	fakeManager.waitForZoneOpError = &googleapi.Error{
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
	fakeManager := newFakeManager()
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
	fakeManager := newFakeManager()
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
	fakeManager := newFakeManager()
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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1", "zone2", "zone3"}}

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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
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
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1", "zone2", "zone3"}}
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	/* Act */
	// DeleteDisk will call FakeServiceManager.GetDisk() with all zones,
	// and FakeServiceManager.GetDisk() always returns a disk,
	// so DeleteDisk thinks a disk with diskName exists in all zones.
	err := gce.DeleteDisk(diskName)

	/* Assert */
	if err == nil {
		t.Error("Expected error when disk is found in multiple zones, but none returned.")
	}
}

func TestDeleteDisk_DiffDiskMultiZone(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
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
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "us-central1-c"
	const sizeGb int64 = 128
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}

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
	if labels[kubeletapis.LabelZoneRegion] != "us-central1" {
		t.Errorf("Region is '%v', but zone is 'us-central1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_NoZone(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "europe-west1-d"
	const sizeGb int64 = 128
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}
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
	if labels[kubeletapis.LabelZoneRegion] != "europe-west1" {
		t.Errorf("Region is '%v', but zone is 'europe-west1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DiskNotFound(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
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
	fakeManager := newFakeManager()
	diskName := "disk"
	gce := GCECloud{manager: fakeManager, managedZones: []string{}}

	/* Act */
	_, err := gce.GetAutoLabelsForPD(diskName, "")

	/* Assert */
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DupDisk(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "us-west1-b"
	const sizeGb int64 = 128

	gce := GCECloud{manager: fakeManager, managedZones: []string{"us-west1-b", "asia-southeast1-a"}}
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
	if labels[kubeletapis.LabelZoneRegion] != "us-west1" {
		t.Errorf("Region is '%v', but zone is 'us-west1'", labels[kubeletapis.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DupDiskNoZone(t *testing.T) {
	/* Arrange */
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128

	gce := GCECloud{manager: fakeManager, managedZones: []string{"us-west1-b", "asia-southeast1-a"}}
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

type FakeServiceManager struct {
	// Common fields shared among tests
	op                 *compute.Operation // Mocks an operation returned by GCE API calls
	doesOpMatch        bool
	disks              map[string]string // zone: diskName
	waitForZoneOpError error             // Error to be returned by WaitForZoneOp

	// Fields for TestCreateDisk
	createDiskCalled bool
	diskToCreate     *compute.Disk

	// Fields for TestDeleteDisk
	deleteDiskCalled bool
	resourceInUse    bool // Marks the disk as in-use
}

func newFakeManager() *FakeServiceManager {
	return &FakeServiceManager{disks: make(map[string]string)}
}

/**
 * Upon disk creation, disk info is stored in FakeServiceManager
 * to be used by other tested methods.
 */
func (manager *FakeServiceManager) CreateDisk(
	project string,
	zone string,
	disk *compute.Disk) (*compute.Operation, error) {

	manager.createDiskCalled = true
	op := &compute.Operation{}
	manager.op = op
	manager.diskToCreate = disk
	manager.disks[zone] = disk.Name
	return op, nil
}

/**
 * Gets disk info stored in the FakeServiceManager.
 */
func (manager *FakeServiceManager) GetDisk(
	project string,
	zone string,
	diskName string) (*compute.Disk, error) {

	if manager.disks[zone] == "" {
		return nil, cloudprovider.DiskNotFound
	}

	if manager.resourceInUse {
		errorItem := googleapi.ErrorItem{Reason: "resourceInUseByAnotherResource"}
		err := &googleapi.Error{Errors: []googleapi.ErrorItem{errorItem}}
		return nil, err
	}

	disk := &compute.Disk{Name: diskName, Zone: zone, Kind: "compute#disk"}
	return disk, nil
}

/**
 * Disk info is removed from the FakeServiceManager.
 */
func (manager *FakeServiceManager) DeleteDisk(
	project string,
	zone string,
	disk string) (*compute.Operation, error) {

	manager.deleteDiskCalled = true
	op := &compute.Operation{}
	manager.op = op
	manager.disks[zone] = ""
	return op, nil
}

func (manager *FakeServiceManager) WaitForZoneOp(
	op *compute.Operation,
	zone string,
	mc *metricContext) error {
	if op == manager.op {
		manager.doesOpMatch = true
	}
	return manager.waitForZoneOpError
}
