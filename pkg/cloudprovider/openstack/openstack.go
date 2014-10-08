/*
Copyright 2014 Google Inc. All rights reserved.

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

package openstack

import (
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {}

func init() {
	cloudprovider.RegisterCloudProvider("openstack", func(config io.Reader) (cloudprovider.Interface, error) {
		return newOpenStack(config)
	})
}

func newOpenStack(config io.Reader) (*OpenStack, error) {
	return &OpenStack{}, nil
}

func (os *OpenStack) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	return nil, false
}

func (os *OpenStack) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}
