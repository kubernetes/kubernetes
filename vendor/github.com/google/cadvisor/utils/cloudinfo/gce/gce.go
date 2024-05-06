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

package gce

import (
	"os"
	"strings"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/cloudinfo"

	"cloud.google.com/go/compute/metadata"
	"k8s.io/klog/v2"
)

const (
	gceProductName = "/sys/class/dmi/id/product_name"
	google         = "Google"
)

func init() {
	cloudinfo.RegisterCloudProvider(info.GCE, &provider{})
}

type provider struct{}

var _ cloudinfo.CloudProvider = provider{}

func (provider) IsActiveProvider() bool {
	data, err := os.ReadFile(gceProductName)
	if err != nil {
		klog.V(2).Infof("Error while reading product_name: %v", err)
		return false
	}
	return strings.Contains(string(data), google)
}

func (provider) GetInstanceType() info.InstanceType {
	machineType, err := metadata.Get("instance/machine-type")
	if err != nil {
		return info.UnknownInstance
	}

	responseParts := strings.Split(machineType, "/") // Extract the instance name from the machine type.
	return info.InstanceType(responseParts[len(responseParts)-1])
}

func (provider) GetInstanceID() info.InstanceID {
	instanceID, err := metadata.Get("instance/id")
	if err != nil {
		return info.UnknownInstance
	}
	return info.InstanceID(info.InstanceType(instanceID))
}
