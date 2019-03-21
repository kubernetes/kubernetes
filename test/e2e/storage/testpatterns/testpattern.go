/*
Copyright 2018 The Kubernetes Authors.

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

package testpatterns

import (
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// MinFileSize represents minimum file size (1 MiB) for testing
	MinFileSize = 1 * framework.MiB

	// FileSizeSmall represents small file size (1 MiB) for testing
	FileSizeSmall = 1 * framework.MiB
	// FileSizeMedium represents medium file size (100 MiB) for testing
	FileSizeMedium = 100 * framework.MiB
	// FileSizeLarge represents large file size (1 GiB) for testing
	FileSizeLarge = 1 * framework.GiB
)

// TestVolType represents a volume type to be tested in a TestSuite
type TestVolType string

var (
	// InlineVolume represents a volume type that is used inline in volumeSource
	InlineVolume TestVolType = "InlineVolume"
	// PreprovisionedPV represents a volume type for pre-provisioned Persistent Volume
	PreprovisionedPV TestVolType = "PreprovisionedPV"
	// DynamicPV represents a volume type for dynamic provisioned Persistent Volume
	DynamicPV TestVolType = "DynamicPV"
)

// TestSnapshotType represents a snapshot type to be tested in a TestSuite
type TestSnapshotType string

var (
	// DynamicCreatedSnapshot represents a snapshot type for dynamic created snapshot
	DynamicCreatedSnapshot TestSnapshotType = "DynamicSnapshot"
)

// TestPattern represents a combination of parameters to be tested in a TestSuite
type TestPattern struct {
	Name         string                  // Name of TestPattern
	FeatureTag   string                  // featureTag for the TestSuite
	VolType      TestVolType             // Volume type of the volume
	FsType       string                  // Fstype of the volume
	VolMode      v1.PersistentVolumeMode // PersistentVolumeMode of the volume
	SnapshotType TestSnapshotType        // Snapshot type of the snapshot
}

var (
	// Definitions for default fsType

	// DefaultFsInlineVolume is TestPattern for "Inline-volume (default fs)"
	DefaultFsInlineVolume = TestPattern{
		Name:    "Inline-volume (default fs)",
		VolType: InlineVolume,
	}
	// DefaultFsPreprovisionedPV is TestPattern for "Pre-provisioned PV (default fs)"
	DefaultFsPreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (default fs)",
		VolType: PreprovisionedPV,
	}
	// DefaultFsDynamicPV is TestPattern for "Dynamic PV (default fs)"
	DefaultFsDynamicPV = TestPattern{
		Name:    "Dynamic PV (default fs)",
		VolType: DynamicPV,
	}

	// Definitions for ext3

	// Ext3InlineVolume is TestPattern for "Inline-volume (ext3)"
	Ext3InlineVolume = TestPattern{
		Name:    "Inline-volume (ext3)",
		VolType: InlineVolume,
		FsType:  "ext3",
	}
	// Ext3PreprovisionedPV is TestPattern for "Pre-provisioned PV (ext3)"
	Ext3PreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (ext3)",
		VolType: PreprovisionedPV,
		FsType:  "ext3",
	}
	// Ext3DynamicPV is TestPattern for "Dynamic PV (ext3)"
	Ext3DynamicPV = TestPattern{
		Name:    "Dynamic PV (ext3)",
		VolType: DynamicPV,
		FsType:  "ext3",
	}

	// Definitions for ext4

	// Ext4InlineVolume is TestPattern for "Inline-volume (ext4)"
	Ext4InlineVolume = TestPattern{
		Name:    "Inline-volume (ext4)",
		VolType: InlineVolume,
		FsType:  "ext4",
	}
	// Ext4PreprovisionedPV is TestPattern for "Pre-provisioned PV (ext4)"
	Ext4PreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (ext4)",
		VolType: PreprovisionedPV,
		FsType:  "ext4",
	}
	// Ext4DynamicPV is TestPattern for "Dynamic PV (ext4)"
	Ext4DynamicPV = TestPattern{
		Name:    "Dynamic PV (ext4)",
		VolType: DynamicPV,
		FsType:  "ext4",
	}

	// Definitions for xfs

	// XfsInlineVolume is TestPattern for "Inline-volume (xfs)"
	XfsInlineVolume = TestPattern{
		Name:    "Inline-volume (xfs)",
		VolType: InlineVolume,
		FsType:  "xfs",
	}
	// XfsPreprovisionedPV is TestPattern for "Pre-provisioned PV (xfs)"
	XfsPreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (xfs)",
		VolType: PreprovisionedPV,
		FsType:  "xfs",
	}
	// XfsDynamicPV is TestPattern for "Dynamic PV (xfs)"
	XfsDynamicPV = TestPattern{
		Name:    "Dynamic PV (xfs)",
		VolType: DynamicPV,
		FsType:  "xfs",
	}

	// Definitions for ntfs

	// NtfsInlineVolume is TestPattern for "Inline-volume (ntfs)"
	NtfsInlineVolume = TestPattern{
		Name:       "Inline-volume (ntfs)",
		VolType:    InlineVolume,
		FsType:     "ntfs",
		FeatureTag: "[sig-windows]",
	}
	// NtfsPreprovisionedPV is TestPattern for "Pre-provisioned PV (ntfs)"
	NtfsPreprovisionedPV = TestPattern{
		Name:       "Pre-provisioned PV (ntfs)",
		VolType:    PreprovisionedPV,
		FsType:     "ntfs",
		FeatureTag: "[sig-windows]",
	}
	// NtfsDynamicPV is TestPattern for "Dynamic PV (xfs)"
	NtfsDynamicPV = TestPattern{
		Name:       "Dynamic PV (ntfs)",
		VolType:    DynamicPV,
		FsType:     "ntfs",
		FeatureTag: "[sig-windows]",
	}

	// Definitions for Filesystem volume mode

	// FsVolModePreprovisionedPV is TestPattern for "Pre-provisioned PV (filesystem)"
	FsVolModePreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (filesystem volmode)",
		VolType: PreprovisionedPV,
		VolMode: v1.PersistentVolumeFilesystem,
	}
	// FsVolModeDynamicPV is TestPattern for "Dynamic PV (filesystem)"
	FsVolModeDynamicPV = TestPattern{
		Name:    "Dynamic PV (filesystem volmode)",
		VolType: DynamicPV,
		VolMode: v1.PersistentVolumeFilesystem,
	}

	// Definitions for block volume mode

	// BlockVolModePreprovisionedPV is TestPattern for "Pre-provisioned PV (block)"
	BlockVolModePreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (block volmode)",
		VolType: PreprovisionedPV,
		VolMode: v1.PersistentVolumeBlock,
	}
	// BlockVolModeDynamicPV is TestPattern for "Dynamic PV (block)(immediate bind)"
	BlockVolModeDynamicPV = TestPattern{
		Name:    "Dynamic PV (block volmode)",
		VolType: DynamicPV,
		VolMode: v1.PersistentVolumeBlock,
	}

	// Definitions for snapshot case

	// DynamicSnapshot is TestPattern for "Dynamic snapshot"
	DynamicSnapshot = TestPattern{
		Name:         "Dynamic Snapshot",
		SnapshotType: DynamicCreatedSnapshot,
	}
)
