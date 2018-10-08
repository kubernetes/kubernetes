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
		cinder, err := NewCinder(config)
		if err != nil {
			return nil, err
		}
		return cinder, nil
	})
}

func NewCinder(config io.Reader) (*Cinder, error) {
	cfg, err := openstack.ReadConfig(config)
	if err != nil {
		return nil, err
	}
	os, err := openstack.NewOpenStack(cfg)
	if err != nil {
		return nil, err
	}
	return &Cinder{
		os: os,
	}, nil
}

type Cinder struct {
	os cindervolume.BlockStorageProvider
}

func (cinder Cinder) AttachDisk(instanceID, volumeID string) (string, error) {
	return cinder.os.AttachDisk(instanceID, volumeID)
}

func (cinder Cinder) DetachDisk(instanceID, volumeID string) error {
	return cinder.os.DetachDisk(instanceID, volumeID)
}

func (cinder Cinder) DeleteVolume(volumeID string) error {
	return cinder.os.DeleteVolume(volumeID)
}

func (cinder Cinder) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, string, bool, error) {
	return cinder.os.CreateVolume(name, size, vtype, availability, tags)
}

func (cinder Cinder) GetDevicePath(volumeID string) (string, error) {
	return cinder.os.GetDevicePath(volumeID)
}

func (cinder Cinder) InstanceID() (string, error) {
	return cinder.os.InstanceID()
}

func (cinder Cinder) GetAttachmentDiskPath(instanceID, volumeID string) (string, error) {
	return cinder.os.GetAttachmentDiskPath(instanceID, volumeID)
}

func (cinder Cinder) OperationPending(diskName string) (bool, string, error) {
	return cinder.os.OperationPending(diskName)
}

func (cinder Cinder) DiskIsAttached(instanceID, volumeID string) (bool, error) {
	return cinder.os.DiskIsAttached(instanceID, volumeID)
}

func (cinder Cinder) DiskIsAttachedByName(nodeName types.NodeName, volumeID string) (bool, string, error) {
	return cinder.os.DiskIsAttachedByName(nodeName, volumeID)
}

func (cinder Cinder) DisksAreAttachedByName(nodeName types.NodeName, volumeIDs []string) (map[string]bool, error) {
	return cinder.os.DisksAreAttachedByName(nodeName, volumeIDs)
}

func (cinder Cinder) ShouldTrustDevicePath() bool {
	return cinder.os.ShouldTrustDevicePath()
}

func (cinder Cinder) Instances() (cloudprovider.Instances, bool) {
	return cinder.os.Instances()
}

func (cinder Cinder) ExpandVolume(volumeID string, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	return cinder.os.ExpandVolume(volumeID, oldSize, newSize)
}

func (cinder Cinder) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

func (cinder Cinder) Initialize(clientBuilder cloudprovider.ControllerClientBuilder) {}

func (cinder Cinder) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

func (cinder Cinder) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

func (cinder Cinder) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (cinder Cinder) ProviderName() string {
	return ProviderName
}

func (cinder Cinder) HasClusterID() bool {
	return true
}
