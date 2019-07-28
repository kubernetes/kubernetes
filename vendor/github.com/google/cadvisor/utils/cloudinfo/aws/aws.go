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
	"io/ioutil"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/session"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/cloudinfo"
)

const (
	productVerFileName = "/sys/class/dmi/id/product_version"
	biosVerFileName    = "/sys/class/dmi/id/bios_vendor"
	amazon             = "amazon"
)

func init() {
	cloudinfo.RegisterCloudProvider(info.AWS, &provider{})
}

type provider struct{}

var _ cloudinfo.CloudProvider = provider{}

func (provider) IsActiveProvider() bool {
	var dataProduct []byte
	var dataBios []byte
	if _, err := os.Stat(productVerFileName); err == nil {
		dataProduct, err = ioutil.ReadFile(productVerFileName)
		if err != nil {
			return false
		}
	}

	if _, err := os.Stat(biosVerFileName); err == nil {
		dataBios, err = ioutil.ReadFile(biosVerFileName)
		if err != nil {
			return false
		}
	}

	return strings.Contains(string(dataProduct), amazon) || strings.Contains(strings.ToLower(string(dataBios)), amazon)
}

func getAwsMetadata(name string) string {
	client := ec2metadata.New(session.New(&aws.Config{}))
	data, err := client.GetMetadata(name)
	if err != nil {
		return info.UnknownInstance
	}
	return data
}

func (provider) GetInstanceType() info.InstanceType {
	return info.InstanceType(getAwsMetadata("instance-type"))
}

func (provider) GetInstanceID() info.InstanceID {
	return info.InstanceID(getAwsMetadata("instance-id"))
}
