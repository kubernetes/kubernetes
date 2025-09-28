//go:build windows
// +build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package mount

import (
	"fmt"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils/pkg/cim"
)

type cimOperations struct {
}

// do the SMB mount with username, password, remotepath
func (mounter *cimOperations) NewSMBMapping(username, password, remotePath string) error {
	if username == "" || password == "" || remotePath == "" {
		return fmt.Errorf("invalid parameter(username: %s, password: %s, remotepath: %s)", username, sensitiveOptionsRemoved, remotePath)
	}

	return cim.WithCOMThread(func() error {
		result, err := cim.NewSmbGlobalMapping(remotePath, username, password, true)
		if err != nil {
			return fmt.Errorf("NewSmbGlobalMapping failed. result: %d, err: %v", result, err)
		}
		return nil
	})
}

// check whether remotePath is already mounted
func (mounter *cimOperations) IsSMBMappingExist(remotePath string) (bool, error) {
	var isMapped bool
	err := cim.WithCOMThread(func() error {
		inst, err := cim.QuerySmbGlobalMappingByRemotePath(remotePath)
		if err != nil {
			return err
		}

		status, err := cim.GetSmbGlobalMappingStatus(inst)
		if err != nil {
			return err
		}

		isMapped = status == cim.SmbMappingStatusOK
		return nil
	})
	return isMapped, cim.IgnoreNotFound(err)
}

// remove SMB mapping
func (mounter *cimOperations) RemoveSMBMapping(remotePath string) error {
	return cim.WithCOMThread(func() error {
		err := cim.RemoveSmbGlobalMappingByRemotePath(remotePath)
		if err != nil {
			return fmt.Errorf("error remove smb mapping '%s'. err: %v", remotePath, err)
		}
		return nil
	})
}

func (mounter *cimOperations) PartitionDisk(diskNumber uint32) error {
	initialized, err := mounter.IsDiskInitialized(diskNumber)
	if err != nil {
		klog.Errorf("IsDiskInitialized failed: %v", err)
	}
	if !initialized {
		klog.V(4).Infof("Initializing disk %d", diskNumber)
		err = mounter.InitializeDisk(diskNumber)
		if err != nil {
			klog.Errorf("failed InitializeDisk %v", err)
			return err
		}
	} else {
		klog.V(4).Infof("Disk %d already initialized", diskNumber)
	}

	klog.V(4).Infof("Checking if disk %d has basic partitions", diskNumber)
	partitioned, err := mounter.BasicPartitionsExist(diskNumber)
	if err != nil {
		klog.Errorf("failed check BasicPartitionsExist %v", err)
		return err
	}
	if !partitioned {
		klog.V(4).Infof("Creating basic partition on disk %d", diskNumber)
		err = mounter.CreateBasicPartition(diskNumber)
		if err != nil {
			klog.Errorf("failed CreateBasicPartition %v", err)
			return err
		}
	} else {
		klog.V(4).Infof("Disk %d already partitioned", diskNumber)
	}
	return nil
}

func (mounter *cimOperations) IsDiskInitialized(diskNumber uint32) (bool, error) {
	var partitionStyle int32
	err := cim.WithCOMThread(func() error {
		disk, err := cim.QueryDiskByNumber(diskNumber, cim.DiskSelectorListForPartitionStyle)
		if err != nil {
			return fmt.Errorf("error checking initialized status of disk %d: %v", diskNumber, err)
		}

		partitionStyle, err = cim.GetDiskPartitionStyle(disk)
		if err != nil {
			return fmt.Errorf("failed to query partition style of disk %d: %v", diskNumber, err)
		}

		return nil
	})
	return partitionStyle != cim.PartitionStyleUnknown, err
}

func (mounter *cimOperations) InitializeDisk(diskNumber uint32) error {
	return cim.WithCOMThread(func() error {
		disk, err := cim.QueryDiskByNumber(diskNumber, nil)
		if err != nil {
			return fmt.Errorf("failed to initializing disk %d. error: %w", diskNumber, err)
		}

		result, err := cim.InitializeDisk(disk, cim.PartitionStyleGPT)
		if result != 0 || err != nil {
			return fmt.Errorf("failed to initializing disk %d: result %d, error: %w", diskNumber, result, err)
		}

		return nil
	})
}

func (mounter *cimOperations) BasicPartitionsExist(diskNumber uint32) (bool, error) {
	var exist bool
	err := cim.WithCOMThread(func() error {
		partitions, err := cim.ListPartitionsWithFilters(nil, cim.FilterForPartitionOnDisk(diskNumber), cim.FilterForPartitionsOfTypeNormal())
		if cim.IgnoreNotFound(err) != nil {
			return fmt.Errorf("error checking presence of partitions on disk %d:, %v", diskNumber, err)
		}

		exist = len(partitions) > 0
		return nil
	})
	return exist, err
}

func (mounter *cimOperations) CreateBasicPartition(diskNumber uint32) error {
	return cim.WithCOMThread(func() error {
		disk, err := cim.QueryDiskByNumber(diskNumber, nil)
		if err != nil {
			return err
		}

		result, err := cim.CreatePartition(
			disk,
			nil,                           // Size
			true,                          // UseMaximumSize
			nil,                           // Offset
			nil,                           // Alignment
			nil,                           // DriveLetter
			false,                         // AssignDriveLetter
			nil,                           // MbrType,
			cim.GPTPartitionTypeBasicData, // GPT Type
			false,                         // IsHidden
			false,                         // IsActive,
		)
		if (result != 0 && result != cim.ErrorCodeCreatePartitionAccessPathAlreadyInUse) || err != nil {
			return fmt.Errorf("error creating partition on disk %d. result: %d, err: %v", diskNumber, result, err)
		}

		result, _, err = cim.RefreshDisk(disk)
		if result != 0 || err != nil {
			return fmt.Errorf("error rescan disk (%d). result %d, error: %v", diskNumber, result, err)
		}

		partitions, err := cim.ListPartitionsWithFilters(nil, cim.FilterForPartitionOnDisk(diskNumber), cim.FilterForPartitionsOfTypeNormal())
		if err != nil {
			return fmt.Errorf("error query basic partition on disk %d:, %v", diskNumber, err)
		}

		if len(partitions) == 0 {
			return fmt.Errorf("failed to create basic partition on disk %d:, %v", diskNumber, err)
		}

		partition := partitions[0]
		result, status, err := cim.SetPartitionState(partition, true)
		if result != 0 || err != nil {
			return fmt.Errorf("error bring partition %v on disk %d online. result: %d, status %s, err: %v", partition, diskNumber, result, status, err)
		}

		return nil
	})
}

// ListVolumesOnDisk - returns back list of volumes(volumeIDs) in the disk (requested in diskID).
func (mounter *cimOperations) ListVolumesOnDisk(diskNumber uint32) (volumeIDs []string, err error) {
	err = cim.WithCOMThread(func() error {
		partitions, err := cim.ListPartitionsOnDisk(diskNumber, 0, cim.PartitionSelectorListObjectID)
		if err != nil {
			return errors.Wrapf(err, "failed to list partition on disk %d", diskNumber)
		}

		volumes, err := cim.FindVolumesByPartition(partitions)
		if cim.IgnoreNotFound(err) != nil {
			return errors.Wrapf(err, "failed to list volumes on disk %d", diskNumber)
		}

		for _, volume := range volumes {
			uniqueID, err := cim.GetVolumeUniqueID(volume)
			if err != nil {
				return errors.Wrapf(err, "failed to get unique ID for volume %v", volume)
			}
			volumeIDs = append(volumeIDs, uniqueID)
		}

		return nil
	})
	return
}
