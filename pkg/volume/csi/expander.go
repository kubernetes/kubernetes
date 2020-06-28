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

package csi

import (
	"context"
	"errors"
	"fmt"

	api "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

var _ volume.NodeExpandableVolumePlugin = &csiPlugin{}

func (c *csiPlugin) RequiresFSResize() bool {
	// We could check plugin's node capability but we instead are going to rely on
	// NodeExpand to do the right thing and return early if plugin does not have
	// node expansion capability.
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandCSIVolumes) {
		klog.V(4).Infof("Resizing is not enabled for CSI volume")
		return false
	}
	return true
}

func (c *csiPlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
	klog.V(4).Infof(log("Expander.NodeExpand(%s)", resizeOptions.DeviceMountPath))
	csiSource, err := getCSISourceFromSpec(resizeOptions.VolumeSpec)
	if err != nil {
		return false, errors.New(log("Expander.NodeExpand failed to get CSI persistent source: %v", err))
	}

	csClient, err := newCsiDriverClient(csiDriverName(csiSource.Driver))
	if err != nil {
		return false, err
	}
	fsVolume, err := util.CheckVolumeModeFilesystem(resizeOptions.VolumeSpec)
	if err != nil {
		return false, errors.New(log("Expander.NodeExpand failed to check VolumeMode of source: %v", err))
	}

	return c.nodeExpandWithClient(resizeOptions, csiSource, csClient, fsVolume)
}

func (c *csiPlugin) nodeExpandWithClient(
	resizeOptions volume.NodeResizeOptions,
	csiSource *api.CSIPersistentVolumeSource,
	csClient csiClient,
	fsVolume bool) (bool, error) {
	driverName := csiSource.Driver

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	nodeExpandSet, err := csClient.NodeSupportsNodeExpand(ctx)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to check if node supports expansion : %v", err)
	}

	if !nodeExpandSet {
		return false, fmt.Errorf("Expander.NodeExpand found CSI plugin %s/%s to not support node expansion", c.GetPluginName(), driverName)
	}

	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := csClient.NodeSupportsStageUnstage(ctx)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to check if plugins supports stage_unstage %v", err)
	}

	// if plugin does not support STAGE_UNSTAGE but CSI volume path is staged
	// it must mean this was placeholder staging performed by k8s and not CSI staging
	// in which case we should return from here so as volume can be node published
	// before we can resize
	if !stageUnstageSet && resizeOptions.CSIVolumePhase == volume.CSIVolumeStaged {
		return false, nil
	}

	volumeTargetPath := resizeOptions.DeviceMountPath
	if !fsVolume {
		volumeTargetPath = resizeOptions.DevicePath
	}

	_, err = csClient.NodeExpandVolume(ctx, csiSource.VolumeHandle, volumeTargetPath, resizeOptions.NewSize)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to expand the volume : %v", err)
	}
	return true, nil
}
