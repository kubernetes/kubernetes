/*
Copyright 2016 The Kubernetes Authors.

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

package vclib

// Volume Constants
const (
	ThinDiskType             = "thin"
	PreallocatedDiskType     = "preallocated"
	EagerZeroedThickDiskType = "eagerZeroedThick"
	ZeroedThickDiskType      = "zeroedThick"
)

// Controller Constants
const (
	SCSIControllerLimit       = 4
	SCSIControllerDeviceLimit = 15
	SCSIDeviceSlots           = 16
	SCSIReservedSlot          = 7

	SCSIControllerType        = "scsi"
	LSILogicControllerType    = "lsiLogic"
	BusLogicControllerType    = "busLogic"
	LSILogicSASControllerType = "lsiLogic-sas"
	PVSCSIControllerType      = "pvscsi"
)

// Other Constants
const (
	LogLevel                   = 4
	DatastoreProperty          = "datastore"
	ResourcePoolProperty       = "resourcePool"
	DatastoreInfoProperty      = "info"
	VirtualMachineType         = "VirtualMachine"
	RoundTripperDefaultCount   = 3
	VSANDatastoreType          = "vsan"
	DummyVMPrefixName          = "vsphere-k8s"
	ActivePowerState           = "poweredOn"
	DatacenterType             = "Datacenter"
	ClusterComputeResourceType = "ClusterComputeResource"
	HostSystemType             = "HostSystem"
)

// Test Constants
const (
	TestDefaultDatacenter = "DC0"
	TestDefaultDatastore  = "LocalDS_0"
	TestDefaultNetwork    = "VM Network"
	testNameNotFound      = "enoent"
)
