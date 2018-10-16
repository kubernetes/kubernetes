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

package cinder

import (
	"io"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cloud-provider"
	"k8s.io/cloud-provider-openstack/pkg/cloudprovider/providers/openstack"
	cindervolume "k8s.io/cloud-provider-openstack/pkg/volume/cinder"
)

const (
	// ProviderName is the name of the cinder volume provider
	ProviderName = "cinder"
)

func init() {
	openstack.RegisterMetrics()

	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cinder, err := newCinder(config)
		if err != nil {
			return nil, err
		}
		return cinder, nil
	})
}

func newCinder(config io.Reader) (*cinder, error) {
	cfg, err := openstack.ReadConfig(config)
	if err != nil {
		return nil, err
	}
	os, err := openstack.NewOpenStack(cfg)
	if err != nil {
		return nil, err
	}
	return &cinder{
		os: os,
	}, nil
}

type cinder struct {
	os cindervolume.BlockStorageProvider
}

func (cinder cinder) AttachDisk(instanceID, volumeID string) (string, error) {
	return cinder.os.AttachDisk(instanceID, volumeID)
}

func (cinder cinder) DetachDisk(instanceID, volumeID string) error {
	return cinder.os.DetachDisk(instanceID, volumeID)
}

func (cinder cinder) DeleteVolume(volumeID string) error {
	return cinder.os.DeleteVolume(volumeID)
}

func (cinder cinder) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, string, bool, error) {
	return cinder.os.CreateVolume(name, size, vtype, availability, tags)
}

func (cinder cinder) GetDevicePath(volumeID string) (string, error) {
	return cinder.os.GetDevicePath(volumeID)
}

func (cinder cinder) InstanceID() (string, error) {
	return cinder.os.InstanceID()
}

func (cinder cinder) GetAttachmentDiskPath(instanceID, volumeID string) (string, error) {
	return cinder.os.GetAttachmentDiskPath(instanceID, volumeID)
}

func (cinder cinder) OperationPending(diskName string) (bool, string, error) {
	return cinder.os.OperationPending(diskName)
}

func (cinder cinder) DiskIsAttached(instanceID, volumeID string) (bool, error) {
	return cinder.os.DiskIsAttached(instanceID, volumeID)
}

func (cinder cinder) DiskIsAttachedByName(nodeName types.NodeName, volumeID string) (bool, string, error) {
	return cinder.os.DiskIsAttachedByName(nodeName, volumeID)
}

func (cinder cinder) DisksAreAttachedByName(nodeName types.NodeName, volumeIDs []string) (map[string]bool, error) {
	return cinder.os.DisksAreAttachedByName(nodeName, volumeIDs)
}

func (cinder cinder) ShouldTrustDevicePath() bool {
	return cinder.os.ShouldTrustDevicePath()
}

func (cinder cinder) Instances() (cloudprovider.Instances, bool) {
	return cinder.os.Instances()
}

func (cinder cinder) ExpandVolume(volumeID string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	return cinder.os.ExpandVolume(volumeID, oldSize, newSize)
}

func (cinder cinder) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

func (cinder cinder) Initialize(clientBuilder cloudprovider.ControllerClientBuilder) {}

func (cinder cinder) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

func (cinder cinder) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

func (cinder cinder) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (cinder cinder) ProviderName() string {
	return ProviderName
}

func (cinder cinder) HasClusterID() bool {
	return true
}
