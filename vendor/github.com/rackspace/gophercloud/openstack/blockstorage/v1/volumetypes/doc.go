// Package volumetypes provides information and interaction with volume types
// in the OpenStack Block Storage service. A volume type indicates the type of
// a block storage volume, such as SATA, SCSCI, SSD, etc. These can be
// customized or defined by the OpenStack admin.
//
// You can also define extra_specs associated with your volume types. For
// instance, you could have a VolumeType=SATA, with extra_specs (RPM=10000,
// RAID-Level=5) . Extra_specs are defined and customized by the admin.
package volumetypes
