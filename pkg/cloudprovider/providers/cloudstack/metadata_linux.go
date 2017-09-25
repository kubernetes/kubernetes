// +build linux

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

package cloudstack

import (
	"net"
	"time"

	"github.com/d2g/dhcp4client"
)

func newDHCPClient(nic *net.Interface) (*dhcp4client.Client, error) {
	pktsock, err := dhcp4client.NewPacketSock(nic.Index)
	if err != nil {
		return nil, err
	}

	return dhcp4client.New(
		dhcp4client.HardwareAddr(nic.HardwareAddr),
		dhcp4client.Timeout(2*time.Second),
		dhcp4client.Broadcast(false),
		dhcp4client.Connection(pktsock),
	)
}
