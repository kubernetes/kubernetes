// Copyright 2016 The rkt Authors
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

package networking

import (
	"fmt"
	"net"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

// ForwardedPort describes a port that will be
// forwarded (mapped) from the host to the pod
type ForwardedPort struct {
	PodPort  types.Port
	HostPort types.ExposedPort
}

// findAppPort looks through the manifest to find a port with a given name.
// If multiple apps expose the same port name, it will fail
func findAppPort(manifest *schema.PodManifest, portName types.ACName) (*types.Port, error) {
	var foundPort *types.Port

	for _, app := range manifest.Apps {
		for _, port := range app.App.Ports {
			if portName == port.Name {
				if foundPort != nil { // error: ambiguous
					return nil, fmt.Errorf("port name %q defined multiple apps", portName)
				}
				p := port // duplicate b/c port gets overwritten
				foundPort = &p
			}
		}
	}
	return foundPort, nil
}

// ForwardedPorts matches up ExposedPorts (host ports) with Ports on the app side.
// By default, it tries to match up by name - apps expose ports, and the podspec
// maps them. The podspec can also map from host to pod, without a corresponding app
// (which is needed for CRI)
// This will error if:
// - a name is ambiguous
// - the same port:proto combination is forwarded
func ForwardedPorts(manifest *schema.PodManifest) ([]ForwardedPort, error) {
	var fps []ForwardedPort
	var err error

	// For every ExposedPort, find its corresponding PodPort
	for _, ep := range manifest.Ports {
		podPort := ep.PodPort

		// If there is no direct mapping, search for the port by name
		if podPort == nil {
			podPort, err = findAppPort(manifest, ep.Name)
			if err != nil {
				return nil, err
			}
			if podPort == nil {
				return nil, fmt.Errorf("port name %q could not be found in any apps", ep.Name)
			}
		}
		fp := ForwardedPort{
			HostPort: ep,
			PodPort:  *podPort,
		}
		fp.HostPort.PodPort = &fp.PodPort
		if fp.HostPort.HostIP == nil {
			fp.HostPort.HostIP = net.IPv4(0, 0, 0, 0)
		}

		// Check all already-existing ports for conflicts
		for idx := range fps {
			if fp.conflicts(&fps[idx]) {
				return nil, fmt.Errorf("port %s-%s:%d already mapped to pod port %d",
					fp.PodPort.Protocol, fp.HostPort.HostIP.String(), fp.HostPort.HostPort, fps[idx].PodPort.Port)
			}
		}

		fps = append(fps, fp)
	}
	return fps, nil
}

// conflicts checks if two ports conflict with each other
func (fp *ForwardedPort) conflicts(fp1 *ForwardedPort) bool {
	if fp.PodPort.Protocol != fp1.PodPort.Protocol {
		return false
	}

	if fp.HostPort.HostPort != fp1.HostPort.HostPort {
		return false
	}

	// If either port has the 0.0.0.0 address, they conflict
	zeroAddr := net.IPv4(0, 0, 0, 0)
	if fp.HostPort.HostIP.Equal(zeroAddr) || fp1.HostPort.HostIP.Equal(zeroAddr) {
		return true
	}

	if fp.HostPort.HostIP.Equal(fp1.HostPort.HostIP) {
		return true
	}

	return false
}
