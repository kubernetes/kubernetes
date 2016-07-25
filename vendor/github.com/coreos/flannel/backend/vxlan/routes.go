// Copyright 2015 flannel authors
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

package vxlan

import (
	"net"

	"github.com/coreos/flannel/pkg/ip"
)

type route struct {
	network ip.IP4Net
	vtepMAC net.HardwareAddr
}

type routes []route

func (rts *routes) set(nw ip.IP4Net, vtepMAC net.HardwareAddr) {
	for i, rt := range *rts {
		if rt.network.Equal(nw) {
			(*rts)[i].vtepMAC = vtepMAC
			return
		}
	}
	*rts = append(*rts, route{nw, vtepMAC})
}

func (rts *routes) remove(nw ip.IP4Net) {
	for i, rt := range *rts {
		if rt.network.Equal(nw) {
			(*rts)[i] = (*rts)[len(*rts)-1]
			(*rts) = (*rts)[0 : len(*rts)-1]
			return
		}
	}
}

func (rts routes) findByNetwork(ipAddr ip.IP4) *route {
	for i, rt := range rts {
		if rt.network.Contains(ipAddr) {
			return &rts[i]
		}
	}
	return nil
}
