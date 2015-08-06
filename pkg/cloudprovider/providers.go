/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package cloudprovider

import (
	// Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/aws"
	_ "k8s.io/kubernetes/pkg/cloudprovider/gce"
	_ "k8s.io/kubernetes/pkg/cloudprovider/mesos"
	_ "k8s.io/kubernetes/pkg/cloudprovider/openstack"
	_ "k8s.io/kubernetes/pkg/cloudprovider/ovirt"
	_ "k8s.io/kubernetes/pkg/cloudprovider/rackspace"
	_ "k8s.io/kubernetes/pkg/cloudprovider/vagrant"
)
