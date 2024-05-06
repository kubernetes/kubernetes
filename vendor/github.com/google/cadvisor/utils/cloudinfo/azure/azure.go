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

package cloudinfo

import (
	"os"
	"strings"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/cloudinfo"
)

const (
	sysVendorFileName    = "/sys/class/dmi/id/sys_vendor"
	biosUUIDFileName     = "/sys/class/dmi/id/product_uuid"
	microsoftCorporation = "Microsoft Corporation"
)

func init() {
	cloudinfo.RegisterCloudProvider(info.Azure, &provider{})
}

type provider struct{}

var _ cloudinfo.CloudProvider = provider{}

func (provider) IsActiveProvider() bool {
	data, err := os.ReadFile(sysVendorFileName)
	if err != nil {
		return false
	}
	return strings.Contains(string(data), microsoftCorporation)
}

// TODO: Implement method.
func (provider) GetInstanceType() info.InstanceType {
	return info.UnknownInstance
}

func (provider) GetInstanceID() info.InstanceID {
	data, err := os.ReadFile(biosUUIDFileName)
	if err != nil {
		return info.UnNamedInstance
	}
	return info.InstanceID(strings.TrimSuffix(string(data), "\n"))
}
