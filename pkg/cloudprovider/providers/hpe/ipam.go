/*
Copyright 2017 The Kubernetes Authors.

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

package hpe

import (
	"net"

	"github.com/golang/glog"
)

var IpPool []net.IP

type Range struct {
	Start net.IP `json:"start"`
	End   net.IP `json:"end"`
}

func (r Range) Expand() []net.IP {
	ip := dupIP(r.Start)
	out := []net.IP{ip}

	for !ip.Equal(r.End) {
		ip = nextIP(ip)
		out = append(out, ip)
	}
	return out
}

func dupIP(ip net.IP) net.IP {
	if x := ip.To4(); x != nil {
		ip = x
	}
	dup := make(net.IP, len(ip))
	copy(dup, ip)
	return dup
}

func nextIP(ip net.IP) net.IP {
	next := dupIP(ip)
	for j := len(next) - 1; j >= 0; j-- {
		next[j]++
		if next[j] > 0 {
			break
		}
	}
	return next
}

func allocateIp() string {
	glog.V(1).Infof("IpPool is: %v", IpPool)
	if len(IpPool) == 0 {
		glog.V(1).Infof("IP's exhausted, IPAM has no more IP's")
		return ""
	}
	allocatedIp := IpPool[0].String()
	IpPool = append(IpPool[:0], IpPool[1:]...)
	return allocatedIp
}

func releaseIp(releaseIp string) {
	relIp := net.ParseIP(releaseIp)
	relIp = dupIP(relIp)
	IpPool = append(IpPool, relIp)
	glog.V(1).Infof("Released ip: %s", relIp)

}

func initializeIpam(startIp string, endIp string) bool {
	r := Range{
		Start: net.ParseIP(startIp),
		End:   net.ParseIP(endIp),
	}
	if r.Start.To4() == nil {
		glog.Errorf("Invalid START_IP: %s", startIp)
		return false
	}
	if r.End.To4() == nil {
		glog.Errorf("Invalid END_IP: %s", endIp)
		return false
	}
	IpPool = r.Expand()
	return true

}