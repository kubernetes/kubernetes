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

package util

import (
	"crypto/sha1"
	"encoding/hex"
)

// This file is a common place holder for volume limit utility constants
// shared between volume package and scheduler

const (
	// EBSVolumeLimitKey resource name that will store volume limits for EBS
	EBSVolumeLimitKey = "attachable-volumes-aws-ebs"
	// EBSNitroLimitRegex finds nitro instance types with different limit than EBS defaults
	EBSNitroLimitRegex = "^[cmr]5.*|t3|z1d"
	// DefaultMaxEBSVolumes is the limit for volumes attached to an instance.
	// Amazon recommends no more than 40; the system root volume uses at least one.
	// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/volume_limits.html#linux-specific-volume-limits
	DefaultMaxEBSVolumes = 39
	// DefaultMaxEBSNitroVolumeLimit is default EBS volume limit on m5 and c5 instances
	DefaultMaxEBSNitroVolumeLimit = 25
	// AzureVolumeLimitKey stores resource name that will store volume limits for Azure
	AzureVolumeLimitKey = "attachable-volumes-azure-disk"
	// GCEVolumeLimitKey stores resource name that will store volume limits for GCE node
	GCEVolumeLimitKey = "attachable-volumes-gce-pd"

	// CinderVolumeLimitKey contains Volume limit key for Cinder
	CinderVolumeLimitKey = "attachable-volumes-cinder"
	// DefaultMaxCinderVolumes defines the maximum number of PD Volumes for Cinder
	// For Openstack we are keeping this to a high enough value so as depending on backend
	// cluster admins can configure it.
	DefaultMaxCinderVolumes = 256

	// CSIAttachLimitPrefix defines prefix used for CSI volumes
	CSIAttachLimitPrefix = "attachable-volumes-csi-"

	// ResourceNameLengthLimit stores maximum allowed Length for a ResourceName
	ResourceNameLengthLimit = 63
)

// GetCSIAttachLimitKey returns limit key used for CSI volumes
func GetCSIAttachLimitKey(driverName string) string {
	csiPrefixLength := len(CSIAttachLimitPrefix)
	totalkeyLength := csiPrefixLength + len(driverName)
	if totalkeyLength >= ResourceNameLengthLimit {
		charsFromDriverName := driverName[:23]
		hash := sha1.New()
		hash.Write([]byte(driverName))
		hashed := hex.EncodeToString(hash.Sum(nil))
		hashed = hashed[:16]
		return CSIAttachLimitPrefix + charsFromDriverName + hashed
	}
	return CSIAttachLimitPrefix + driverName
}
