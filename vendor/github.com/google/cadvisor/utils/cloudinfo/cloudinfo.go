// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Get information about the cloud provider (if any) cAdvisor is running on.

package cloudinfo

import (
	info "github.com/google/cadvisor/info/v1"
)

type CloudInfo interface {
	GetCloudProvider() info.CloudProvider
	GetInstanceType() info.InstanceType
	GetInstanceID() info.InstanceID
}

type realCloudInfo struct {
	cloudProvider info.CloudProvider
	instanceType  info.InstanceType
	instanceID    info.InstanceID
}

func NewRealCloudInfo() CloudInfo {
	cloudProvider := detectCloudProvider()
	instanceType := detectInstanceType(cloudProvider)
	instanceID := detectInstanceID(cloudProvider)
	return &realCloudInfo{
		cloudProvider: cloudProvider,
		instanceType:  instanceType,
		instanceID:    instanceID,
	}
}

func (self *realCloudInfo) GetCloudProvider() info.CloudProvider {
	return self.cloudProvider
}

func (self *realCloudInfo) GetInstanceType() info.InstanceType {
	return self.instanceType
}

func (self *realCloudInfo) GetInstanceID() info.InstanceID {
	return self.instanceID
}

func detectCloudProvider() info.CloudProvider {
	switch {
	case onGCE():
		return info.GCE
	case onAWS():
		return info.AWS
	case onAzure():
		return info.Azure
	case onBaremetal():
		return info.Baremetal
	}
	return info.UnknownProvider
}

func detectInstanceType(cloudProvider info.CloudProvider) info.InstanceType {
	switch cloudProvider {
	case info.GCE:
		return getGceInstanceType()
	case info.AWS:
		return getAwsInstanceType()
	case info.Azure:
		return getAzureInstanceType()
	case info.Baremetal:
		return info.NoInstance
	}
	return info.UnknownInstance
}

func detectInstanceID(cloudProvider info.CloudProvider) info.InstanceID {
	switch cloudProvider {
	case info.GCE:
		return getGceInstanceID()
	case info.AWS:
		return getAwsInstanceID()
	case info.Azure:
		return getAzureInstanceID()
	case info.Baremetal:
		return info.UnNamedInstance
	}
	return info.UnNamedInstance
}

//TODO: Implement method.
func onBaremetal() bool {
	return false
}
