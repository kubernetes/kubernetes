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
	"errors"
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	api "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

var _ volume.NodeExpandableVolumePlugin = &csiPlugin{}

func (c *csiPlugin) RequiresFSResize() bool {
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
		// Treat the absence of the CSI driver as a transient error
		// See https://github.com/kubernetes/kubernetes/issues/120268
		return false, volumetypes.NewTransientOperationFailure(err.Error())
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

	ctx, cancel := createCSIOperationContext(resizeOptions.VolumeSpec, csiTimeout)
	defer cancel()

	nodeExpandSet, err := csClient.NodeSupportsNodeExpand(ctx)
	if err != nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to check if node supports expansion : %v", err)
	}

	if !nodeExpandSet {
		return false, volumetypes.NewOperationNotSupportedError(fmt.Sprintf("NodeExpand is not supported by the CSI driver %s", driverName))
	}

	pv := resizeOptions.VolumeSpec.PersistentVolume
	if pv == nil {
		return false, fmt.Errorf("Expander.NodeExpand failed to find associated PersistentVolume for plugin %s", c.GetPluginName())
	}
	nodeExpandSecrets := map[string]string{}
	expandClient := c.host.GetKubeClient()
	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeExpandSecret) {
		if csiSource.NodeExpandSecretRef != nil {
			nodeExpandSecrets, err = getCredentialsFromSecret(expandClient, csiSource.NodeExpandSecretRef)
			if err != nil {
				return false, fmt.Errorf("expander.NodeExpand failed to get NodeExpandSecretRef %s/%s: %v",
					csiSource.NodeExpandSecretRef.Namespace, csiSource.NodeExpandSecretRef.Name, err)
			}
		}
	}

	opts := csiResizeOptions{
		volumePath:        resizeOptions.DeviceMountPath,
		stagingTargetPath: resizeOptions.DeviceStagePath,
		volumeID:          csiSource.VolumeHandle,
		newSize:           resizeOptions.NewSize,
		fsType:            csiSource.FSType,
		accessMode:        api.ReadWriteOnce,
		mountOptions:      pv.Spec.MountOptions,
		secrets:           nodeExpandSecrets,
	}

	if !fsVolume {
		// for block volumes the volumePath in CSI NodeExpandvolumeRequest is
		// basically same as DevicePath because block devices are not mounted and hence
		// DeviceMountPath does not get populated in resizeOptions.DeviceMountPath
		opts.volumePath = resizeOptions.DevicePath
		opts.fsType = fsTypeBlockName
	}

	if pv.Spec.AccessModes != nil {
		opts.accessMode = pv.Spec.AccessModes[0]
	}

	_, err = csClient.NodeExpandVolume(ctx, opts)
	if err != nil {
		if inUseError(err) {
			failedConditionErr := fmt.Errorf("Expander.NodeExpand failed to expand the volume : %w", volumetypes.NewFailedPreconditionError(err.Error()))
			return false, failedConditionErr
		}
		return false, fmt.Errorf("Expander.NodeExpand failed to expand the volume : %w", err)
	}
	return true, nil
}

func inUseError(err error) bool {
	st, ok := status.FromError(err)
	if !ok {
		// not a grpc error
		return false
	}
	// if this is a failed precondition error then that means driver does not support expansion
	// of in-use volumes
	// More info - https://github.com/container-storage-interface/spec/blob/master/spec.md#controllerexpandvolume-errors
	return st.Code() == codes.FailedPrecondition
}
