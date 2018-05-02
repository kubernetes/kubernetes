/*
Copyright 2016 The Kubernetes Authors.

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

package master

import (
	"fmt"
	"net"

	"github.com/golang/glog"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// DefaultServiceIPRange takes a the serviceIPRange flag and returns the defaulted service ip range (if  needed),
// api server service IP, and an error
func DefaultServiceIPRange(passedServiceClusterIPRange net.IPNet) (net.IPNet, net.IP, error) {
	serviceClusterIPRange := passedServiceClusterIPRange
	if passedServiceClusterIPRange.IP == nil {
		glog.Infof("Network range for service cluster IPs is unspecified. Defaulting to %v.", kubeoptions.DefaultServiceIPCIDR)
		serviceClusterIPRange = kubeoptions.DefaultServiceIPCIDR
	}
	if size := ipallocator.RangeSize(&serviceClusterIPRange); size < 8 {
		return net.IPNet{}, net.IP{}, fmt.Errorf("The service cluster IP range must be at least %d IP addresses", 8)
	}

	// Select the first valid IP from ServiceClusterIPRange to use as the GenericAPIServer service IP.
	apiServerServiceIP, err := ipallocator.GetIndexedIP(&serviceClusterIPRange, 1)
	if err != nil {
		return net.IPNet{}, net.IP{}, err
	}
	glog.V(4).Infof("Setting service IP to %q (read-write).", apiServerServiceIP)

	return serviceClusterIPRange, apiServerServiceIP, nil
}
