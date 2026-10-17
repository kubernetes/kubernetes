/*
Copyright 2016 The Kubernetes Authors.

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

	api "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"
)

// HealthClient is the subset of CSI node RPCs used for volume and storage health probing.
type HealthClient interface {
	NodeGetVolumeHealth(ctx context.Context, volID, stagingTargetPath, volumePublishPath string) ([]api.VolumeHealthCondition, error)
	NodeGetStorageHealth(ctx context.Context, secrets map[string]string) ([]storagev1.StorageHealthCondition, error)
	NodeSupportsVolumeHealth(ctx context.Context) (bool, error)
	NodeSupportsStorageHealth(ctx context.Context) (bool, error)
}

// NewHealthClient returns a HealthClient for the given registered CSI driver.
func NewHealthClient(driverName string) HealthClient {
	c := &CSIHealthClient{}
	c.csiClientGetter.driverName = csiDriverName(driverName)
	return c
}

var _ HealthClient = &CSIHealthClient{}

// ListRegisteredDrivers returns the names of CSI drivers currently registered with kubelet.
func ListRegisteredDrivers() []string {
	return csiDrivers.List()
}

// GetNodeInfoManager returns the CSI node info manager used to update CSINode objects.
// It may be nil before the CSI plugin is initialized.
func GetNodeInfoManager() nodeinfomanager.Interface {
	return nim
}

// CSIHealthClient wraps a cached CSI client with per-call timeouts.
type CSIHealthClient struct {
	csiClientGetter
}

func (c *CSIHealthClient) NodeSupportsVolumeHealth(ctx context.Context) (bool, error) {
	client, err := c.csiClientGetter.Get()
	if err != nil {
		return false, err
	}
	ctx, cancel := context.WithTimeout(ctx, csiTimeout)
	defer cancel()
	return client.NodeSupportsVolumeHealth(ctx)
}

func (c *CSIHealthClient) NodeSupportsStorageHealth(ctx context.Context) (bool, error) {
	client, err := c.csiClientGetter.Get()
	if err != nil {
		return false, err
	}
	ctx, cancel := context.WithTimeout(ctx, csiTimeout)
	defer cancel()
	return client.NodeSupportsStorageHealth(ctx)
}

func (c *CSIHealthClient) NodeGetVolumeHealth(ctx context.Context, volID, stagingTargetPath, volumePublishPath string) ([]api.VolumeHealthCondition, error) {
	client, err := c.csiClientGetter.Get()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, csiTimeout)
	defer cancel()
	return client.NodeGetVolumeHealth(ctx, volID, stagingTargetPath, volumePublishPath)
}

func (c *CSIHealthClient) NodeGetStorageHealth(ctx context.Context, secrets map[string]string) ([]storagev1.StorageHealthCondition, error) {
	client, err := c.csiClientGetter.Get()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, csiTimeout)
	defer cancel()
	return client.NodeGetStorageHealth(ctx, secrets)
}
