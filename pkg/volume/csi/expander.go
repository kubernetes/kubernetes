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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

var _ volume.NodeExpandableVolumePlugin = &csiPlugin{}

func (c *csiPlugin) RequiresFSResize() bool {
	// We could check plugin's node capability but we instead are going to rely on
	// NodeExpand to do the right thing and return early if plugin does not have
	// node expansion capability.
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandCSIVolumes) {
		klog.V(4).Infof("Resizing is not enabled for this CSI volume")
		return false
	}
	return true
}

func (c *csiPlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
	klog.V(4).Infof(log("Expander.NodeExpand(%s)", resizeOptions.DeviceMountPath))
	pvSource, err := getCSISourceFromSpec(resizeOptions.VolumeSpec)
	if err != nil {
		return false, err
	}
	k8s := c.host.GetKubeClient()
	if k8s == nil {
		klog.Error(log("failed to get a kubernetes client"))
		return false, errors.New("failed to get a Kubernetes client")
	}

	csiClient, err := newCsiDriverClient(csiDriverName(pvSource.Driver))
	if err != nil {
		return false, err
	}

	csiSource, err := getCSISourceFromSpec(resizeOptions.VolumeSpec)
	if err != nil {
		klog.Error(log("Expander.NodeExpand failed to get CSI persistent source: %v", err))
		return false, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	nodeExpandSet, err := csiClient.NodeSupportsNodeExpand(ctx)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to check if node supports expansion : %v", err)
	}

	if !nodeExpandSet {
		return false, fmt.Errorf("Expander.NodeExpand found CSI plugin %s to not support node expansion", c.GetPluginName())
	}

	// Check whether "STAGE_UNSTAGE_VOLUME" is set
	stageUnstageSet, err := csiClient.NodeSupportsStageUnstage(ctx)
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

	_, err = csiClient.NodeExpandVolume(ctx, csiSource.VolumeHandle, resizeOptions.DeviceMountPath, resizeOptions.NewSize)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to expand the volume : %v", err)
	}
	return true, nil
}
