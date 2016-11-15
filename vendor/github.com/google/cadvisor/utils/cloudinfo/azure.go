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
	info "github.com/google/cadvisor/info/v1"
	"io/ioutil"
	"strings"
)

const (
	SysVendorFileName    = "/sys/class/dmi/id/sys_vendor"
	BiosUUIDFileName     = "/sys/class/dmi/id/product_uuid"
	MicrosoftCorporation = "Microsoft Corporation"
)

func onAzure() bool {
	data, err := ioutil.ReadFile(SysVendorFileName)
	if err != nil {
		return false
	}
	return strings.Contains(string(data), MicrosoftCorporation)
}

// TODO: Implement method.
func getAzureInstanceType() info.InstanceType {
	return info.UnknownInstance
}

func getAzureInstanceID() info.InstanceID {
	data, err := ioutil.ReadFile(BiosUUIDFileName)
	if err != nil {
		return info.UnNamedInstance
	}
	return info.InstanceID(strings.TrimSuffix(string(data), "\n"))
}
