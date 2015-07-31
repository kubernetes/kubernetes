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
}

type realCloudInfo struct {
	cloudProvider info.CloudProvider
	instanceType  info.InstanceType
}

func NewRealCloudInfo() CloudInfo {
	cloudProvider := detectCloudProvider()
	instanceType := detectInstanceType(cloudProvider)
	return &realCloudInfo{
		cloudProvider: cloudProvider,
		instanceType:  instanceType,
	}
}

func (self *realCloudInfo) GetCloudProvider() info.CloudProvider {
	return self.cloudProvider
}

func (self *realCloudInfo) GetInstanceType() info.InstanceType {
	return self.instanceType
}

func detectCloudProvider() info.CloudProvider {
	switch {
	case onGCE():
		return info.GCE
	case onAWS():
		return info.AWS
	case onBaremetal():
		return info.Baremetal
	}
	return info.UnkownProvider
}

func detectInstanceType(cloudProvider info.CloudProvider) info.InstanceType {
	switch cloudProvider {
	case info.GCE:
		return getGceInstanceType()
	case info.AWS:
		return getAwsInstanceType()
	case info.Baremetal:
		return info.NoInstance
	}
	return info.UnknownInstance
}

//TODO: Implement method.
func onAWS() bool {
	return false
}

//TODO: Implement method.
func getAwsInstanceType() info.InstanceType {
	return info.UnknownInstance
}

//TODO: Implement method.
func onBaremetal() bool {
	return false
}
