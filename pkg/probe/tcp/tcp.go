/*
Copyright 2015 Google Inc. All rights reserved.

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

package tcp

import (
	"net"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"

	"github.com/golang/glog"
)

func New() TCPProber {
	return TCPProber{}
}

type TCPProber struct{}

func (pr TCPProber) Probe(host string, port int) (probe.Status, error) {
	return DoTCPProbe(net.JoinHostPort(host, strconv.Itoa(port)))
}

// DoTCPProbe checks that a TCP socket to the address can be opened.
// If the socket can be opened, it returns Healthy.
// If the socket fails to open, it returns Unhealthy.
// This is exported because some other packages may want to do direct TCP probes.
func DoTCPProbe(addr string) (probe.Status, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return probe.Failure, nil
	}
	err = conn.Close()
	if err != nil {
		glog.Errorf("unexpected error closing health check socket: %v (%#v)", err, err)
	}
	return probe.Success, nil
}
