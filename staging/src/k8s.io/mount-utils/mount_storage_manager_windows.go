//go:build windows
// +build windows

/*
Copyright The Kubernetes Authors.

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
	"errors"
	"fmt"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils/pkg/wmi"
)

type SMBManager struct {
}

// do the SMB mount with username, password, remotepath
func (mgr SMBManager) NewSMBMapping(username, password, remotePath string) error {
	if username == "" || password == "" || remotePath == "" {
		return fmt.Errorf(
			"invalid parameter(username: %s, password: %s, remotepath: %s)",
			username,
			sensitiveOptionsRemoved,
			remotePath,
		)
	}

	requirePrivacy := true
	return wmi.WithCOMThread(func() error {
		err := wmi.NewSmbGlobalMapping(remotePath, username, password, requirePrivacy)
		if err != nil {
			return fmt.Errorf("create SMB mapping failed for %s: %w", remotePath, err)
		}
		return nil
	})
}

// check whether remotePath is already mounted
func (mgr SMBManager) IsSMBMappingExist(remotePath string) (bool, error) {
	var isMapped bool
	err := wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			inst, err := wmi.QuerySmbGlobalMappingByRemotePath(scope, remotePath)
			if err != nil {
				return err
			}

			status, err := wmi.GetSmbGlobalMappingStatus(inst)
			if err != nil {
				return err
			}

			isMapped = status == wmi.SmbMappingStatusOK
			return nil
		})
	})
	return isMapped, wmi.IgnoreNotFound(err)
}

// remove SMB mapping
func (mgr SMBManager) RemoveSMBMapping(remotePath string) error {
	return wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			err := wmi.RemoveSmbGlobalMappingByRemotePath(scope, remotePath)
			if err != nil {
				return fmt.Errorf("error remove smb mapping '%s'. err: %w", remotePath, err)
			}
			return nil
		})
	})
}

type DefaultStorageManager struct {
}

func NewDefaultStorageManager() *DefaultStorageManager {
	return &DefaultStorageManager{}
}

func (mgr DefaultStorageManager) IsDiskInitialized(disk DiskIdentifier) (bool, error) {
	diskNumber, err := ValidateDiskNumber(disk)
	if err != nil {
		return false, err
	}

	var partitionStyle uint16
	err = wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			disk, err := wmi.QueryDiskByNumber(scope, diskNumber, wmi.DiskSelectorListForPartitionStyle)
			if err != nil {
				return fmt.Errorf("error checking initialized status of disk %d: %w", diskNumber, err)
			}

			partitionStyle, err = wmi.GetDiskPartitionStyle(disk)
			if err != nil {
				return fmt.Errorf("failed to query partition style of disk %d: %w", diskNumber, err)
			}

			return nil
		})
	})
	return partitionStyle != wmi.PartitionStyleUnknown, err
}

func (mgr DefaultStorageManager) InitializeDisk(disk DiskIdentifier) error {
	diskNumber, err := ValidateDiskNumber(disk)
	if err != nil {
		return err
	}

	return wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			disk, err := wmi.QueryDiskByNumber(scope, diskNumber, nil)
			if err != nil {
				return fmt.Errorf("failed to initializing disk %d. error: %w", diskNumber, err)
			}

			err = wmi.InitializeDisk(disk, wmi.PartitionStyleGPT)
			if err != nil {
				return fmt.Errorf("failed to initializing disk %d: error: %w", diskNumber, err)
			}

			return nil
		})
	})
}

func (mgr DefaultStorageManager) BasicPartitionsExist(disk DiskIdentifier) (bool, error) {
	diskNumber, err := ValidateDiskNumber(disk)
	if err != nil {
		return false, err
	}

	var exist bool
	err = wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			partitions, err := wmi.ListPartitionsWithFilters(scope, nil, wmi.FilterForPartitionOnDisk(diskNumber), wmi.FilterForPartitionsOfTypeNormal())
			if err != nil {
				return fmt.Errorf("error checking presence of partitions on disk %d:, %w", diskNumber, err)
			}

			exist = len(partitions) > 0
			return nil
		})
	})
	return exist, err
}

func (mgr DefaultStorageManager) CreateBasicPartition(disk DiskIdentifier) error {
	diskNumber, err := ValidateDiskNumber(disk)
	if err != nil {
		return err
	}

	return wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			disk, err := wmi.QueryDiskByNumber(scope, diskNumber, nil)
			if err != nil {
				return err
			}

			err = wmi.CreatePartition(
				disk,
				nil,                           // Size
				true,                          // UseMaximumSize
				nil,                           // Offset
				nil,                           // Alignment
				nil,                           // DriveLetter
				false,                         // AssignDriveLetter
				nil,                           // MbrType,
				wmi.GPTPartitionTypeBasicData, // GPT Type
				false,                         // IsHidden
				false,                         // IsActive,
			)
			if err != nil {
				var werr *wmi.WMIError
				if !errors.As(err, &werr) || werr.Code != wmi.ErrorCodeCreatePartitionAccessPathAlreadyInUse {
					return fmt.Errorf("error creating partition on disk %d. err: %w", diskNumber, err)
				}
			}

			_, err = wmi.RefreshDisk(disk)
			if err != nil {
				return fmt.Errorf("error rescan disk (%d). error: %w", diskNumber, err)
			}

			partitions, err := wmi.ListPartitionsWithFilters(scope, nil, wmi.FilterForPartitionOnDisk(diskNumber), wmi.FilterForPartitionsOfTypeNormal())
			if err != nil {
				return fmt.Errorf("error query basic partition on disk %d:, %w", diskNumber, err)
			}

			if len(partitions) == 0 {
				return fmt.Errorf("failed to create basic partition on disk %d:, %w", diskNumber, err)
			}

			partition := partitions[0]
			status, err := wmi.SetPartitionState(partition, true)
			if err != nil {
				return fmt.Errorf("error bring partition %v on disk %d online. status %s, err: %w", partition, diskNumber, status, err)
			}

			return nil
		})
	})
}

func (mgr DefaultStorageManager) PartitionDisk(disk DiskIdentifier) error {
	initialized, err := mgr.IsDiskInitialized(disk)
	if err != nil {
		return err
	}

	if !initialized {
		klog.V(4).Infof("Initializing disk %s", disk)
		if err := mgr.InitializeDisk(disk); err != nil {
			return err
		}
	}

	exist, err := mgr.BasicPartitionsExist(disk)
	if err != nil {
		return err
	}

	if !exist {
		klog.V(4).Infof("Creating partition on disk %s", disk)
		if err := mgr.CreateBasicPartition(disk); err != nil {
			return err
		}
	}

	return nil
}

func (mgr DefaultStorageManager) ListVolumesOnDisk(disk DiskIdentifier) ([]VolumeIdentifier, error) {
	diskNumber, err := ValidateDiskNumber(disk)
	if err != nil {
		return nil, err
	}

	var volumeIDs []VolumeIdentifier
	partitionNumber := uint32(0)

	err = wmi.WithCOMThread(func() error {
		return wmi.WithScope(func(scope *wmi.Scope) error {
			partitions, err := wmi.ListPartitionsOnDisk(scope, diskNumber, partitionNumber, wmi.PartitionSelectorListObjectID)
			if err != nil {
				return fmt.Errorf("failed to list partition on disk %d: %w", diskNumber, err)
			}

			volumes, err := wmi.FindVolumesByPartition(scope, partitions)
			if err != nil {
				return fmt.Errorf("failed to list volumes on disk %d: %w", diskNumber, err)
			}
			if volumes == nil {
				return nil
			}

			err = wmi.ForEach(volumes, func(volume *wmi.COMDispatchObject) error {
				uniqueID, err := wmi.GetVolumeUniqueID(volume)
				if err != nil {
					return fmt.Errorf("failed to get unique ID for volume %v: %w", volume, err)
				}
				volumeIDs = append(volumeIDs, VolumeIdentifier(uniqueID))
				return nil
			})
			if err != nil {
				return err
			}

			return nil
		})
	})
	return volumeIDs, err
}
