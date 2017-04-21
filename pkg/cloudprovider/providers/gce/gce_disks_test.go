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

	"golang.org/x/net/context"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestCreateDisk_Basic(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128
	err := gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
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
	if fakeManager.diskToCreate.Name != diskName || fakeManager.diskToCreate.SizeGb != sizeGb {
		t.Error("Disk description sent to GCE does not match parameters of the method in test.")
	}

	if fakeManager.context.Value(apiContextKey).(apiWithNamespace).apiCall != "gce_disk_insert" {
		t.Error("API call to GCE is incorrect.")
	}
}

func TestCreateDisk_WrongZone(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128
	err := gce.CreateDisk(diskName, diskType, "zone2", sizeGb, nil)
	if err == nil {
		t.Error("Expected error when zone is not managed, but none returned.")
	}
}

func TestCreateDisk_NoManagedZone(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{}}

	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128
	err := gce.CreateDisk(diskName, diskType, "zone1", sizeGb, nil)
	if err == nil {
		t.Error("Expected error when managedZones is empty, but none returned.")
	}
}

func TestCreateDisk_BadDiskType(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}

	diskName := "disk"
	diskType := "arbitrary-disk"
	zone := "zone1"
	const sizeGb int64 = 128
	err := gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	if err == nil {
		t.Error("Expected error when disk type is not supported, but none returned.")
	}
}

func TestCreateDisk_MultiZone(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1", "zone2", "zone3"}}

	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128
	for _, zone := range gce.managedZones {
		diskName = zone + "disk"
		err := gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
		if err != nil {
			t.Errorf("Error creating disk in zone '%v'; error: \"%v\"", zone, err)
		}
	}
}

func TestDeleteDisk_Basic(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128

	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	err := gce.DeleteDisk(diskName)
	if err != nil {
		t.Error(err)
	}
	if !fakeManager.deleteDiskCalled {
		t.Error("Never called GCE disk delete.")
	}
	if !fakeManager.doesOpMatch {
		t.Error("Ops used in WaitForZoneOp does not match what's returned by DeleteDisk.")
	}
	if fakeManager.context.Value(apiContextKey).(apiWithNamespace).apiCall != "gce_disk_delete" {
		t.Error("API call to GCE is incorrect.")
	}

}

func TestDeleteDisk_NotFound(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
	diskName := "disk"

	// If disk is not found, no error should be returned.
	err := gce.DeleteDisk(diskName)
	if err != nil {
		t.Error("Expected successful operation when disk is not found, but an error is returned.")
	}
}

func TestDeleteDisk_ResourceBeingUsed(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "zone1"
	const sizeGb int64 = 128

	// Error when resource is being used.
	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	fakeManager.resourceInUse = true
	err := gce.DeleteDisk(diskName)
	if err == nil {
		t.Error("Expected error when disk is in use, but none returned.")
	}
}

func TestDeleteDisk_SameDiskMultiZone(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1", "zone2", "zone3"}}
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	// Error when disk is found in multiple zones
	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}
	// FakeServiceManager.GetDisk always returns a disk by default,
	// so given the same diskName, the same disk is returned for all zones.
	err := gce.DeleteDisk(diskName)
	if err == nil {
		t.Error("Expected error when disk is found in multiple zones, but none returned.")
	}
}

func TestDeleteDisk_DiffDiskMultiZone(t *testing.T) {
	fakeManager := newFakeManager()
	gce := GCECloud{manager: fakeManager, managedZones: []string{"zone1"}}
	diskName := "disk"
	diskType := DiskTypeSSD
	const sizeGb int64 = 128

	for _, zone := range gce.managedZones {
		diskName = zone + "disk"
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}
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
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeSSD
	zone := "us-central1-c"
	const sizeGb int64 = 128
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}
	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	labels, err := gce.GetAutoLabelsForPD(diskName, zone)
	if err != nil {
		t.Error(err)
	}
	if labels[metav1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[metav1.LabelZoneFailureDomain], zone)
	}
	if labels[metav1.LabelZoneRegion] != "us-central1" {
		t.Errorf("Region is '%v', but zone is 'us-central1'", labels[metav1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_NoZone(t *testing.T) {
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "europe-west1-d"
	const sizeGb int64 = 128
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}
	gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)

	labels, err := gce.GetAutoLabelsForPD(diskName, "")
	if err != nil {
		t.Error(err)
	}
	if labels[metav1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[metav1.LabelZoneFailureDomain], zone)
	}
	if labels[metav1.LabelZoneRegion] != "europe-west1" {
		t.Errorf("Region is '%v', but zone is 'europe-west1'", labels[metav1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DiskNotFound(t *testing.T) {
	fakeManager := newFakeManager()
	diskName := "disk"
	zone := "asia-northeast1-a"
	gce := GCECloud{manager: fakeManager, managedZones: []string{zone}}

	_, err := gce.GetAutoLabelsForPD(diskName, zone)
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DiskNotFoundAndNoZone(t *testing.T) {
	fakeManager := newFakeManager()
	diskName := "disk"

	gce := GCECloud{manager: fakeManager, managedZones: []string{}}

	_, err := gce.GetAutoLabelsForPD(diskName, "")
	if err == nil {
		t.Error("Expected error when the specified disk does not exist, but none returned.")
	}
}

func TestGetAutoLabelsForPD_DupDisk(t *testing.T) {
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	zone := "us-west1-b"
	const sizeGb int64 = 128

	gce := GCECloud{manager: fakeManager, managedZones: []string{"us-west1-b", "asia-southeast1-a"}}
	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	labels, err := gce.GetAutoLabelsForPD(diskName, zone)
	if err != nil {
		t.Error("Disk name and zone uniquely identifies a disk, yet an error is returned.")
	}
	if labels[metav1.LabelZoneFailureDomain] != zone {
		t.Errorf("Failure domain is '%v', but zone is '%v'",
			labels[metav1.LabelZoneFailureDomain], zone)
	}
	if labels[metav1.LabelZoneRegion] != "us-west1" {
		t.Errorf("Region is '%v', but zone is 'us-west1'", labels[metav1.LabelZoneRegion])
	}
}

func TestGetAutoLabelsForPD_DupDiskNoZone(t *testing.T) {
	fakeManager := newFakeManager()
	diskName := "disk"
	diskType := DiskTypeStandard
	const sizeGb int64 = 128

	gce := GCECloud{manager: fakeManager, managedZones: []string{"us-west1-b", "asia-southeast1-a"}}
	for _, zone := range gce.managedZones {
		gce.CreateDisk(diskName, diskType, zone, sizeGb, nil)
	}

	_, err := gce.GetAutoLabelsForPD(diskName, "")
	if err == nil {
		t.Error("Expected error when the disk is duplicated and zone is not specified, but none returned.")
	}
}

type FakeServiceManager struct {
	// Common fields shared among tests
	op          *compute.Operation // Mocks an operation returned by GCE API calls
	doesOpMatch bool
	context     context.Context   // Key of the GCE disk create context
	disks       map[string]string // zone: diskName

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
	disk *compute.Disk,
	dc context.Context) (*compute.Operation, error) {

	manager.createDiskCalled = true
	op := &compute.Operation{}
	manager.op = op
	manager.diskToCreate = disk
	manager.context = dc
	manager.disks[zone] = disk.Name
	return op, nil
}

/**
 * Gets disk info stored in the FakeServiceManager.
 */
func (manager *FakeServiceManager) GetDisk(
	project string,
	zone string,
	diskName string,
	dc context.Context) (*compute.Disk, error) {

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
	disk string,
	dc context.Context) (*compute.Operation, error) {

	manager.deleteDiskCalled = true
	op := &compute.Operation{}
	manager.op = op
	manager.context = dc
	manager.disks[zone] = ""
	return op, nil
}

func (manager *FakeServiceManager) WaitForZoneOp(op *compute.Operation, zone string) error {
	if op == manager.op {
		manager.doesOpMatch = true
	}
	return nil
}
