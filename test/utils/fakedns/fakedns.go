/*
Copyright 2025 The Kubernetes Authors.

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

package fakedns

import (
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	netutils "k8s.io/utils/net"

	"golang.org/x/net/dns/dnsmessage"

	nettesting "k8s.io/kubernetes/third_party/forked/golang/net/testing"
)

// dnsHijackLock is a mutex to ensure that only one test can hijack DNS at a time.
// This is necessary because net.DefaultResolver is a global variable.
var dnsHijackLock sync.Mutex

// Server is a fake DNS server that can be used to override DNS resolution in tests.
// This is particularly useful for testing components that rely on DNS without needing
// to set up an actual DNS server. The server can be configured with a map of domain
// names to IP addresses, and it will respond to DNS queries accordingly.
// To use the fake DNS server, create an instance using NewServer, then call
// Hijack to redirect DNS queries to the fake server.
// This helper does not support running tests in parallel.
type Server struct {
	dnsServer nettesting.FakeDNSServer
}

// NewServer creates a new fake DNS server. The hosts map should contain
// domain names as keys and their corresponding IP addresses as string values.
// It returns an error if any of the IP addresses are invalid.
func NewServer(hosts map[string]string) (*Server, error) {
	parsedHosts := make(map[string]net.IP, len(hosts))
	for domain, ipStr := range hosts {
		ip := netutils.ParseIPSloppy(ipStr)
		if ip == nil {
			return nil, fmt.Errorf("invalid IP address for domain %s: %s", domain, ipStr)
		}
		// Ensure domain is clean without trailing dot for map key
		parsedHosts[strings.TrimSuffix(domain, ".")] = ip
	}
	dnsServer := nettesting.FakeDNSServer{}
	dnsServer.ResponseHandler = func(n, s string, q dnsmessage.Message, t time.Time) (dnsmessage.Message, error) {
		resp := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:            q.Header.ID,
				Response:      true,
				Authoritative: true,
			},
			Questions: q.Questions,
		}

		if len(q.Questions) == 0 {
			return resp, nil
		}

		question := q.Questions[0]
		domain := strings.TrimSuffix(question.Name.String(), ".")

		if ip, ok := parsedHosts[domain]; ok {
			var resource dnsmessage.Resource
			if ipv4 := ip.To4(); ipv4 != nil {
				resource = dnsmessage.Resource{
					Header: dnsmessage.ResourceHeader{Name: question.Name, Type: dnsmessage.TypeA, Class: dnsmessage.ClassINET, TTL: 300},
					Body:   &dnsmessage.AResource{A: [4]byte(ipv4)},
				}
			} else { // Assume IPv6
				resource = dnsmessage.Resource{
					Header: dnsmessage.ResourceHeader{Name: question.Name, Type: dnsmessage.TypeAAAA, Class: dnsmessage.ClassINET, TTL: 300},
					Body:   &dnsmessage.AAAAResource{AAAA: [16]byte(ip)},
				}
			}
			resp.Answers = []dnsmessage.Resource{resource}
		}

		return resp, nil
	}

	return &Server{
		dnsServer: dnsServer,
	}, nil
}

// Hijack takes control of the net.DefaultResolver to redirect DNS queries
// to the fake server. It uses t.Cleanup to automatically restore the original
// resolver after the test.
// This helper does not support running tests in parallel.
//
// Example:
//
//	dnsServer, _ := fakedns.NewServer(hosts)
//	dnsServer.Hijack(t)
func (s *Server) Hijack(t *testing.T) {
	t.Helper()

	dnsHijackLock.Lock()

	originalDial := net.DefaultResolver.Dial
	originalPreferGo := net.DefaultResolver.PreferGo

	net.DefaultResolver.PreferGo = true
	net.DefaultResolver.Dial = s.dnsServer.DialContext

	t.Cleanup(func() {
		net.DefaultResolver.Dial = originalDial
		net.DefaultResolver.PreferGo = originalPreferGo
		dnsHijackLock.Unlock()
	})
}
