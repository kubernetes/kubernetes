/*
Copyright 2019 The Kubernetes Authors.

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

package nodevolumelimits

import (
	"crypto/sha1"
	"encoding/hex"
	"strings"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// NOTE: constants in this file are copied from pkg/volume/util package to avoid external scheduler dependency

const (
	// ebsVolumeLimitKey resource name that will store volume limits for EBS
	ebsVolumeLimitKey = "attachable-volumes-aws-ebs"
	// ebsNitroLimitRegex finds nitro instance types with different limit than EBS defaults
	ebsNitroLimitRegex = "^[cmr]5.*|t3|z1d"
	// defaultMaxEBSVolumes is the limit for volumes attached to an instance.
	// Amazon recommends no more than 40; the system root volume uses at least one.
	// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/volume_limits.html#linux-specific-volume-limits
	defaultMaxEBSVolumes = 39
	// defaultMaxEBSNitroVolumeLimit is default EBS volume limit on m5 and c5 instances
	defaultMaxEBSNitroVolumeLimit = 25
	// azureVolumeLimitKey stores resource name that will store volume limits for Azure
	azureVolumeLimitKey = "attachable-volumes-azure-disk"
	// gceVolumeLimitKey stores resource name that will store volume limits for GCE node
	gceVolumeLimitKey = "attachable-volumes-gce-pd"

	// cinderVolumeLimitKey contains Volume limit key for Cinder
	cinderVolumeLimitKey = "attachable-volumes-cinder"
	// defaultMaxCinderVolumes defines the maximum number of PD Volumes for Cinder
	// For Openstack we are keeping this to a high enough value so as depending on backend
	// cluster admins can configure it.
	defaultMaxCinderVolumes = 256

	// csiAttachLimitPrefix defines prefix used for CSI volumes
	csiAttachLimitPrefix = "attachable-volumes-csi-"

	// resourceNameLengthLimit stores maximum allowed Length for a ResourceName
	resourceNameLengthLimit = 63
)

// isCSIMigrationOn returns a boolean value indicating whether
// the CSI migration has been enabled for a particular storage plugin.
func isCSIMigrationOn(csiNode *storagev1.CSINode, pluginName string) bool {
	if csiNode == nil || len(pluginName) == 0 {
		return false
	}

	// In-tree storage to CSI driver migration feature should be enabled,
	// along with the plugin-specific one
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationAWS) {
			return false
		}
	case csilibplugins.GCEPDInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationGCE) {
			return false
		}
	case csilibplugins.AzureDiskInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationAzureDisk) {
			return false
		}
	case csilibplugins.CinderInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationOpenStack) {
			return false
		}
	default:
		return false
	}

	// The plugin name should be listed in the CSINode object annotation.
	// This indicates that the plugin has been migrated to a CSI driver in the node.
	csiNodeAnn := csiNode.GetAnnotations()
	if csiNodeAnn == nil {
		return false
	}

	var mpaSet sets.String
	mpa := csiNodeAnn[v1.MigratedPluginsAnnotationKey]
	if len(mpa) == 0 {
		mpaSet = sets.NewString()
	} else {
		tok := strings.Split(mpa, ",")
		mpaSet = sets.NewString(tok...)
	}

	return mpaSet.Has(pluginName)
}

// volumeLimits returns volume limits associated with the node.
func volumeLimits(n *framework.NodeInfo) map[v1.ResourceName]int64 {
	volumeLimits := map[v1.ResourceName]int64{}
	for k, v := range n.Allocatable.ScalarResources {
		if v1helper.IsAttachableVolumeResourceName(k) {
			volumeLimits[k] = v
		}
	}
	return volumeLimits
}

// getCSIAttachLimitKey returns limit key used for CSI volumes
// NOTE: This function copied from pkg/volume/util package to avoid external scheduler dependency
func getCSIAttachLimitKey(driverName string) string {
	csiPrefixLength := len(csiAttachLimitPrefix)
	totalkeyLength := csiPrefixLength + len(driverName)
	if totalkeyLength >= resourceNameLengthLimit {
		charsFromDriverName := driverName[:23]
		hash := sha1.New()
		hash.Write([]byte(driverName))
		hashed := hex.EncodeToString(hash.Sum(nil))
		hashed = hashed[:16]
		return csiAttachLimitPrefix + charsFromDriverName + hashed
	}
	return csiAttachLimitPrefix + driverName
}
