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
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/session"
	"io/ioutil"
	"strings"

	info "github.com/google/cadvisor/info/v1"
)

const (
	ProductVerFileName = "/sys/class/dmi/id/product_version"
	Amazon             = "amazon"
)

func onAWS() bool {
	data, err := ioutil.ReadFile(ProductVerFileName)
	if err != nil {
		return false
	}
	return strings.Contains(string(data), Amazon)
}

func getAwsMetadata(name string) string {
	client := ec2metadata.New(session.New(&aws.Config{}))
	data, err := client.GetMetadata(name)
	if err != nil {
		return info.UnknownInstance
	}
	return data
}

func getAwsInstanceType() info.InstanceType {
	return info.InstanceType(getAwsMetadata("instance-type"))
}

func getAwsInstanceID() info.InstanceID {
	return info.InstanceID(getAwsMetadata("instance-id"))
}
