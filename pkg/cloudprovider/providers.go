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
