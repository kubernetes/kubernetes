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

package framework

import (
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

const (
	// MinFileSize represents minimum file size (1 MiB) for testing
	MinFileSize = 1 * e2evolume.MiB

	// FileSizeSmall represents small file size (1 MiB) for testing
	FileSizeSmall = 1 * e2evolume.MiB
	// FileSizeMedium represents medium file size (100 MiB) for testing
	FileSizeMedium = 100 * e2evolume.MiB
	// FileSizeLarge represents large file size (1 GiB) for testing
	FileSizeLarge = 1 * e2evolume.GiB
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
	// CSIInlineVolume represents a volume type that is defined inline and provided by a CSI driver.
	CSIInlineVolume TestVolType = "CSIInlineVolume"
	// GenericEphemeralVolume represents a volume type that is defined inline and provisioned through a PVC.
	GenericEphemeralVolume TestVolType = "GenericEphemeralVolume"
)

// TestSnapshotType represents a snapshot type to be tested in a TestSuite
type TestSnapshotType string

var (
	// DynamicCreatedSnapshot represents a snapshot type for dynamic created snapshot
	DynamicCreatedSnapshot TestSnapshotType = "DynamicSnapshot"
	// PreprovisionedCreatedSnapshot represents a snapshot type for pre-provisioned snapshot
	PreprovisionedCreatedSnapshot TestSnapshotType = "PreprovisionedSnapshot"
)

// TestSnapshotDeletionPolicy represents the deletion policy of the snapshot class
type TestSnapshotDeletionPolicy string

var (
	// DeleteSnapshot represents delete policy
	DeleteSnapshot TestSnapshotDeletionPolicy = "Delete"
	// RetainSnapshot represents retain policy
	RetainSnapshot TestSnapshotDeletionPolicy = "Retain"
)

func (t TestSnapshotDeletionPolicy) String() string {
	return string(t)
}

// TestPattern represents a combination of parameters to be tested in a TestSuite
type TestPattern struct {
	Name                   string                      // Name of TestPattern
	TestTags               []interface{}               // additional parameters for framework.It, like framework.WithDisruptive()
	VolType                TestVolType                 // Volume type of the volume
	FsType                 string                      // Fstype of the volume
	VolMode                v1.PersistentVolumeMode     // PersistentVolumeMode of the volume
	SnapshotType           TestSnapshotType            // Snapshot type of the snapshot
	SnapshotDeletionPolicy TestSnapshotDeletionPolicy  // Deletion policy of the snapshot class
	BindingMode            storagev1.VolumeBindingMode // VolumeBindingMode of the volume
	AllowExpansion         bool                        // AllowVolumeExpansion flag of the StorageClass
}

var (
	// Definitions for default fsType

	// DefaultFsInlineVolume is TestPattern for "Inline-volume (default fs)"
	DefaultFsInlineVolume = TestPattern{
		Name:    "Inline-volume (default fs)",
		VolType: InlineVolume,
	}
	// DefaultFsCSIEphemeralVolume is TestPattern for "CSI Ephemeral-volume (default fs)"
	DefaultFsCSIEphemeralVolume = TestPattern{
		Name:    "CSI Ephemeral-volume (default fs)",
		VolType: CSIInlineVolume,
	}
	// DefaultFsGenericEphemeralVolume is TestPattern for "Generic Ephemeral-volume (default fs)"
	DefaultFsGenericEphemeralVolume = TestPattern{
		Name:           "Generic Ephemeral-volume (default fs)",
		VolType:        GenericEphemeralVolume,
		AllowExpansion: true,
	}
	// DefaultFsPreprovisionedPV is TestPattern for "Pre-provisioned PV (default fs)"
	DefaultFsPreprovisionedPV = TestPattern{
		Name:    "Pre-provisioned PV (default fs)",
		VolType: PreprovisionedPV,
	}
	// DefaultFsDynamicPV is TestPattern for "Dynamic PV (default fs)"
	DefaultFsDynamicPV = TestPattern{
		Name:                   "Dynamic PV (default fs)",
		VolType:                DynamicPV,
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
	}

	// Definitions for ext3

	// Ext3InlineVolume is TestPattern for "Inline-volume (ext3)"
	Ext3InlineVolume = TestPattern{
		Name:    "Inline-volume (ext3)",
		VolType: InlineVolume,
		FsType:  "ext3",
	}
	// Ext3CSIEphemeralVolume is TestPattern for "CSI Ephemeral-volume (ext3)"
	Ext3CSIEphemeralVolume = TestPattern{
		Name:    "CSI Ephemeral-volume (ext3)",
		VolType: CSIInlineVolume,
		FsType:  "ext3",
	}
	// Ext3GenericEphemeralVolume is TestPattern for "Generic Ephemeral-volume (ext3)"
	Ext3GenericEphemeralVolume = TestPattern{
		Name:    "Generic Ephemeral-volume (ext3)",
		VolType: GenericEphemeralVolume,
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
	// Ext4CSIEphemeralVolume is TestPattern for "CSI Ephemeral-volume (ext4)"
	Ext4CSIEphemeralVolume = TestPattern{
		Name:    "CSI Ephemeral-volume (ext4)",
		VolType: CSIInlineVolume,
		FsType:  "ext4",
	}
	// Ext4GenericEphemeralVolume is TestPattern for "Generic Ephemeral-volume (ext4)"
	Ext4GenericEphemeralVolume = TestPattern{
		Name:    "Generic Ephemeral-volume (ext4)",
		VolType: GenericEphemeralVolume,
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
		Name:                   "Dynamic PV (ext4)",
		VolType:                DynamicPV,
		FsType:                 "ext4",
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
	}

	// Definitions for xfs

	// XfsInlineVolume is TestPattern for "Inline-volume (xfs)"
	XfsInlineVolume = TestPattern{
		Name:     "Inline-volume (xfs)",
		VolType:  InlineVolume,
		FsType:   "xfs",
		TestTags: []interface{}{framework.WithSlow()},
	}
	// XfsCSIEphemeralVolume is TestPattern for "CSI Ephemeral-volume (xfs)"
	XfsCSIEphemeralVolume = TestPattern{
		Name:     "CSI Ephemeral-volume (xfs)",
		VolType:  CSIInlineVolume,
		FsType:   "xfs",
		TestTags: []interface{}{framework.WithSlow()},
	}
	// XfsGenericEphemeralVolume is TestPattern for "Generic Ephemeral-volume (xfs)"
	XfsGenericEphemeralVolume = TestPattern{
		Name:     "Generic Ephemeral-volume (xfs)",
		VolType:  GenericEphemeralVolume,
		FsType:   "xfs",
		TestTags: []interface{}{framework.WithSlow()},
	}
	// XfsPreprovisionedPV is TestPattern for "Pre-provisioned PV (xfs)"
	XfsPreprovisionedPV = TestPattern{
		Name:     "Pre-provisioned PV (xfs)",
		VolType:  PreprovisionedPV,
		FsType:   "xfs",
		TestTags: []interface{}{framework.WithSlow()},
	}
	// XfsDynamicPV is TestPattern for "Dynamic PV (xfs)"
	XfsDynamicPV = TestPattern{
		Name:                   "Dynamic PV (xfs)",
		VolType:                DynamicPV,
		FsType:                 "xfs",
		TestTags:               []interface{}{framework.WithSlow()},
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
	}

	// Definitions for ntfs

	// NtfsInlineVolume is TestPattern for "Inline-volume (ntfs)"
	NtfsInlineVolume = TestPattern{
		Name:     "Inline-volume (ntfs)",
		VolType:  InlineVolume,
		FsType:   "ntfs",
		TestTags: []interface{}{feature.Windows},
	}
	// NtfsCSIEphemeralVolume is TestPattern for "CSI Ephemeral-volume (ntfs)"
	NtfsCSIEphemeralVolume = TestPattern{
		Name:     "CSI Ephemeral-volume (ntfs) [alpha]",
		VolType:  CSIInlineVolume,
		FsType:   "ntfs",
		TestTags: []interface{}{feature.Windows},
	}
	// NtfsGenericEphemeralVolume is TestPattern for "Generic Ephemeral-volume (ntfs)"
	NtfsGenericEphemeralVolume = TestPattern{
		Name:     "Generic Ephemeral-volume (ntfs)",
		VolType:  GenericEphemeralVolume,
		FsType:   "ntfs",
		TestTags: []interface{}{feature.Windows},
	}
	// NtfsPreprovisionedPV is TestPattern for "Pre-provisioned PV (ntfs)"
	NtfsPreprovisionedPV = TestPattern{
		Name:     "Pre-provisioned PV (ntfs)",
		VolType:  PreprovisionedPV,
		FsType:   "ntfs",
		TestTags: []interface{}{feature.Windows},
	}
	// NtfsDynamicPV is TestPattern for "Dynamic PV (ntfs)"
	NtfsDynamicPV = TestPattern{
		Name:                   "Dynamic PV (ntfs)",
		VolType:                DynamicPV,
		FsType:                 "ntfs",
		TestTags:               []interface{}{feature.Windows},
		SnapshotDeletionPolicy: DeleteSnapshot,
		SnapshotType:           DynamicCreatedSnapshot,
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
	// BlockVolModeDynamicPV is TestPattern for "Dynamic PV (block)"
	BlockVolModeDynamicPV = TestPattern{
		Name:                   "Dynamic PV (block volmode)",
		VolType:                DynamicPV,
		VolMode:                v1.PersistentVolumeBlock,
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
	}
	// BlockVolModeGenericEphemeralVolume is for generic ephemeral inline volumes in raw block mode.
	BlockVolModeGenericEphemeralVolume = TestPattern{
		Name:           "Generic Ephemeral-volume (block volmode) (late-binding)",
		VolType:        GenericEphemeralVolume,
		VolMode:        v1.PersistentVolumeBlock,
		BindingMode:    storagev1.VolumeBindingWaitForFirstConsumer,
		AllowExpansion: true,
	}

	// Definitions for snapshot case

	// DynamicSnapshotDelete is TestPattern for "Dynamic snapshot"
	DynamicSnapshotDelete = TestPattern{
		Name:                   "Dynamic Snapshot (delete policy)",
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
		VolType:                DynamicPV,
	}
	// PreprovisionedSnapshotDelete is TestPattern for "Pre-provisioned snapshot"
	PreprovisionedSnapshotDelete = TestPattern{
		Name:                   "Pre-provisioned Snapshot (delete policy)",
		SnapshotType:           PreprovisionedCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
		VolType:                DynamicPV,
	}
	// EphemeralSnapshotDelete is TestPattern for snapshotting of a generic ephemeral volume
	// where snapshots are deleted.
	EphemeralSnapshotDelete = TestPattern{
		Name:                   "Ephemeral Snapshot (delete policy)",
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: DeleteSnapshot,
		VolType:                GenericEphemeralVolume,
	}
	// DynamicSnapshotRetain is TestPattern for "Dynamic snapshot"
	DynamicSnapshotRetain = TestPattern{
		Name:                   "Dynamic Snapshot (retain policy)",
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: RetainSnapshot,
		VolType:                DynamicPV,
	}
	// PreprovisionedSnapshotRetain is TestPattern for "Pre-provisioned snapshot"
	PreprovisionedSnapshotRetain = TestPattern{
		Name:                   "Pre-provisioned Snapshot (retain policy)",
		SnapshotType:           PreprovisionedCreatedSnapshot,
		SnapshotDeletionPolicy: RetainSnapshot,
		VolType:                DynamicPV,
	}
	// EphemeralSnapshotDelete is TestPattern for snapshotting of a generic ephemeral volume
	// where snapshots are preserved.
	EphemeralSnapshotRetain = TestPattern{
		Name:                   "Ephemeral Snapshot (retain policy)",
		SnapshotType:           DynamicCreatedSnapshot,
		SnapshotDeletionPolicy: RetainSnapshot,
		VolType:                GenericEphemeralVolume,
	}

	// Definitions for volume expansion case

	// DefaultFsDynamicPVAllowExpansion is TestPattern for "Dynamic PV (default fs)(allowExpansion)"
	DefaultFsDynamicPVAllowExpansion = TestPattern{
		Name:           "Dynamic PV (default fs)(allowExpansion)",
		VolType:        DynamicPV,
		AllowExpansion: true,
	}

	// NtfsDynamicPVAllowExpansion is TestPattern for "Dynamic PV (default fs)(allowExpansion)"
	NtfsDynamicPVAllowExpansion = TestPattern{
		Name:           "Dynamic PV (ntfs)(allowExpansion)",
		VolType:        DynamicPV,
		AllowExpansion: true,
		FsType:         "ntfs",
		TestTags:       []interface{}{feature.Windows},
	}

	// BlockVolModeDynamicPVAllowExpansion is TestPattern for "Dynamic PV (block volmode)(allowExpansion)"
	BlockVolModeDynamicPVAllowExpansion = TestPattern{
		Name:           "Dynamic PV (block volmode)(allowExpansion)",
		VolType:        DynamicPV,
		VolMode:        v1.PersistentVolumeBlock,
		AllowExpansion: true,
	}

	// Definitions for topology tests

	// TopologyImmediate is TestPattern for immediate binding
	TopologyImmediate = TestPattern{
		Name:        "Dynamic PV (immediate binding)",
		VolType:     DynamicPV,
		BindingMode: storagev1.VolumeBindingImmediate,
	}

	// TopologyDelayed is TestPattern for delayed binding
	TopologyDelayed = TestPattern{
		Name:        "Dynamic PV (delayed binding)",
		VolType:     DynamicPV,
		BindingMode: storagev1.VolumeBindingWaitForFirstConsumer,
	}
)

// NewVolTypeMap creates a map with the given TestVolTypes enabled
func NewVolTypeMap(types ...TestVolType) map[TestVolType]bool {
	m := map[TestVolType]bool{}
	for _, t := range types {
		m[t] = true
	}
	return m
}
