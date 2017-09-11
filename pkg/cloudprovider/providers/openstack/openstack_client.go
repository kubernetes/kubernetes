/*
Copyright 2017 The Kubernetes Authors.

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
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"

	"github.com/golang/glog"
)

func (os *OpenStack) NewNetworkV2() (*gophercloud.ServiceClient, error) {
	network, err := openstack.NewNetworkV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find network v2 endpoint for region %s: %v", os.region, err)
		return nil, err
	}
	return network, nil
}

func (os *OpenStack) NewComputeV2() (*gophercloud.ServiceClient, error) {
	compute, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute v2 endpoint for region %s: %v", os.region, err)
		return nil, err
	}
	return compute, nil
}

func (os *OpenStack) NewBlockStorageV1() (*gophercloud.ServiceClient, error) {
	storage, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Errorf("Unable to initialize cinder v1 client for region %s: %v", os.region, err)
		return nil, err
	}
	return storage, nil
}

func (os *OpenStack) NewBlockStorageV2() (*gophercloud.ServiceClient, error) {
	storage, err := openstack.NewBlockStorageV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Errorf("Unable to initialize cinder v2 client for region %s: %v", os.region, err)
		return nil, err
	}
	return storage, nil
}
