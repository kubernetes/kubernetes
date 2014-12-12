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

package health

import (
	"fmt"
	"net"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type TCPHealthChecker struct{}

// getTCPAddrParts parses the components of a TCP connection address.  For testability.
func getTCPAddrParts(status api.PodStatus, container api.Container) (string, int, error) {
	params := container.LivenessProbe.TCPSocket
	if params == nil {
		return "", -1, fmt.Errorf("error, no TCP parameters specified: %v", container)
	}
	port := -1
	switch params.Port.Kind {
	case util.IntstrInt:
		port = params.Port.IntVal
	case util.IntstrString:
		port = findPortByName(container, params.Port.StrVal)
		if port == -1 {
			// Last ditch effort - maybe it was an int stored as string?
			var err error
			if port, err = strconv.Atoi(params.Port.StrVal); err != nil {
				return "", -1, err
			}
		}
	}
	if port == -1 {
		return "", -1, fmt.Errorf("unknown port: %v", params.Port)
	}
	if len(status.PodIP) == 0 {
		return "", -1, fmt.Errorf("no host specified.")
	}

	return status.PodIP, port, nil
}

// DoTCPCheck checks that a TCP socket to the address can be opened.
// If the socket can be opened, it returns Healthy.
// If the socket fails to open, it returns Unhealthy.
// This is exported because some other packages may want to do direct TCP checks.
func DoTCPCheck(addr string) (Status, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return Unhealthy, nil
	}
	err = conn.Close()
	if err != nil {
		glog.Errorf("unexpected error closing health check socket: %v (%#v)", err, err)
	}
	return Healthy, nil
}

func (t *TCPHealthChecker) HealthCheck(podFullName, podUUID string, status api.PodStatus, container api.Container) (Status, error) {
	host, port, err := getTCPAddrParts(status, container)
	if err != nil {
		return Unknown, err
	}
	return DoTCPCheck(net.JoinHostPort(host, strconv.Itoa(port)))
}

func (t *TCPHealthChecker) CanCheck(probe *api.LivenessProbe) bool {
	return probe.TCPSocket != nil
}
