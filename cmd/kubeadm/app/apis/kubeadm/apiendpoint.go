/*
Copyright 2020 The Kubernetes Authors.

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

package kubeadm

import (
	"net"
	"strconv"

	"github.com/pkg/errors"
)

// APIEndpointFromString returns an APIEndpoint struct based on a "host:port" raw string.
func APIEndpointFromString(apiEndpoint string) (APIEndpoint, error) {
	apiEndpointHost, apiEndpointPortStr, err := net.SplitHostPort(apiEndpoint)
	if err != nil {
		return APIEndpoint{}, errors.Wrapf(err, "invalid advertise address endpoint: %s", apiEndpoint)
	}
	apiEndpointPort, err := net.LookupPort("tcp", apiEndpointPortStr)
	if err != nil {
		return APIEndpoint{}, errors.Wrapf(err, "invalid advertise address endpoint port: %s", apiEndpointPortStr)
	}
	return APIEndpoint{
		AdvertiseAddress: apiEndpointHost,
		BindPort:         int32(apiEndpointPort),
	}, nil
}

func (endpoint *APIEndpoint) String() string {
	return net.JoinHostPort(endpoint.AdvertiseAddress, strconv.FormatInt(int64(endpoint.BindPort), 10))
}
