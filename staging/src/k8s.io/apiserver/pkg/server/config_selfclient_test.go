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

package server

import (
	"net"
	"testing"

	netutils "k8s.io/utils/net"
)

func TestLoopbackHostPortIPv4(t *testing.T) {
	_, ipv6only, err := isIPv6LoopbackSupported()
	if err != nil {
		t.Fatalf("fail to enumerate network interface, %s", err)
	}
	if ipv6only {
		t.Fatalf("no ipv4 loopback interface")
	}

	host, port, err := LoopbackHostPort("1.2.3.4:443")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if host != "1.2.3.4" {
		t.Fatalf("expected 1.2.3.4 as host, got %q", host)
	}
	if port != "443" {
		t.Fatalf("expected 443 as port, got %q", port)
	}

	host, port, err = LoopbackHostPort("0.0.0.0:443")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		t.Fatalf("expected host to be loopback, got %q", host)
	}
	if port != "443" {
		t.Fatalf("expected 443 as port, got %q", port)
	}
}
func TestLoopbackHostPortIPv6(t *testing.T) {
	ipv6, _, err := isIPv6LoopbackSupported()
	if err != nil {
		t.Fatalf("fail to enumerate network interface, %s", err)
	}
	if !ipv6 {
		t.Fatalf("no ipv6 loopback interface")
	}

	host, port, err := LoopbackHostPort("[ff06:0:0:0:0:0:0:c3]:443")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if host != "ff06:0:0:0:0:0:0:c3" {
		t.Fatalf("expected ff06:0:0:0:0:0:0:c3 as host, got %q", host)
	}
	if port != "443" {
		t.Fatalf("expected 443 as port, got %q", port)
	}

	host, port, err = LoopbackHostPort("[::]:443")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() || ip.To4() != nil {
		t.Fatalf("expected IPv6 host to be loopback, got %q", host)
	}
	if port != "443" {
		t.Fatalf("expected 443 as port, got %q", port)
	}
}

func isIPv6LoopbackSupported() (ipv6 bool, ipv6only bool, err error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false, false, err
	}
	ipv4 := false
	for _, address := range addrs {
		ipnet, ok := address.(*net.IPNet)
		if !ok || !ipnet.IP.IsLoopback() {
			continue
		}
		if netutils.IsIPv6(ipnet.IP) {
			ipv6 = true
			continue
		}
		ipv4 = true
	}
	ipv6only = ipv6 && !ipv4
	return ipv6, ipv6only, nil
}
