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
	"strings"

	v1 "k8s.io/api/core/v1"
	v1helper "k8s.io/api/core/v1/helper"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
)

// isCSIMigrationOn returns a boolean value indicating whether
// the CSI migration has been enabled for a particular storage plugin.
func isCSIMigrationOn(csiNode *storagev1.CSINode, pluginName string, fts feature.Features) bool {
	if csiNode == nil || len(pluginName) == 0 {
		return false
	}

	// In-tree storage to CSI driver migration feature should be enabled,
	// along with the plugin-specific one
	if !fts.EnableCSIMigration {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		if !fts.EnableCSIMigrationAWS {
			return false
		}
	case csilibplugins.GCEPDInTreePluginName:
		if !fts.EnableCSIMigrationGCE {
			return false
		}
	case csilibplugins.AzureDiskInTreePluginName:
		if !fts.EnableCSIMigrationAzureDisk {
			return false
		}
	case csilibplugins.CinderInTreePluginName:
		if !fts.EnableCSIMigrationOpenStack {
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
