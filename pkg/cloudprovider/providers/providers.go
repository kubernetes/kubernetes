/*
Copyright 2015 The Kubernetes Authors.

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

package providers

import (
	// Cloud providers
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/cloudstack"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/ovirt"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/rackspace"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
)

// RegisterCloudProviders registers all cloud providers
func RegisterCloudProviders() {
	aws.RegisterCloudProvider()
	azure.RegisterCloudProvider()
	cloudstack.RegisterCloudProvider()
	gce.RegisterCloudProvider()
	mesos.RegisterCloudProvider()
	openstack.RegisterCloudProvider()
	ovirt.RegisterCloudProvider()
	rackspace.RegisterCloudProvider()
	vsphere.RegisterCloudProvider()
}
