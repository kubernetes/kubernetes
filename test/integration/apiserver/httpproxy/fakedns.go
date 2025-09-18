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

package httpproxy

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"golang.org/x/net/dns/dnsmessage"

	netutils "k8s.io/utils/net"
)

// code adapted from https://github.com/golang/go/blob/ef05b66d6115209361dd99ff8f3ab978695fd74a/src/net/dnsclient_unix_test.go

// Server is a fake DNS server that can be used to override DNS resolution in tests.
type Server struct {
	hosts map[string]net.IP
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
	return &Server{hosts: parsedHosts}, nil
}

// Hijack takes control of the net.DefaultResolver to redirect DNS queries
// to the fake server. It returns a cleanup function that must be called to
// restore the original resolver. It is designed to be used with t.Cleanup.
//
// Example:
//
//	dnsServer, _ := fakedns.NewServer(hosts)
//	t.Cleanup(dnsServer.Hijack())
func (s *Server) Hijack() (cleanup func()) {
	originalDial := net.DefaultResolver.Dial
	originalPreferGo := net.DefaultResolver.PreferGo

	net.DefaultResolver.PreferGo = true
	net.DefaultResolver.Dial = s.dialContext

	return func() {
		net.DefaultResolver.Dial = originalDial
		net.DefaultResolver.PreferGo = originalPreferGo
	}
}

// responseHandler creates a DNS response for a given query.
func (s *Server) responseHandler(q dnsmessage.Message) (dnsmessage.Message, error) {
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

	if ip, ok := s.hosts[domain]; ok {
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

// dialContext provides a custom dialer for the net.Resolver.
func (s *Server) dialContext(_ context.Context, n, _ string) (net.Conn, error) {
	if n == "tcp" || n == "tcp4" || n == "tcp6" {
		return &fakeDNSConn{tcp: true, server: s}, nil
	}
	return &fakeDNSPacketConn{fakeDNSConn: fakeDNSConn{tcp: false, server: s}}, nil
}

type fakeDNSConn struct {
	net.Conn
	tcp    bool
	server *Server
	q      dnsmessage.Message
	buf    []byte
}

func (f *fakeDNSConn) Close() error { return nil }

func (f *fakeDNSConn) Read(b []byte) (int, error) {
	if len(f.buf) > 0 {
		n := copy(b, f.buf)
		f.buf = f.buf[n:]
		return n, nil
	}
	resp, err := f.server.responseHandler(f.q)
	if err != nil {
		return 0, err
	}
	bb, err := resp.Pack()
	if err != nil {
		return 0, fmt.Errorf("cannot marshal DNS message: %w", err)
	}
	if f.tcp {
		l := len(bb)
		bb = append([]byte{byte(l >> 8), byte(l)}, bb...)
		f.buf = bb
		return f.Read(b)
	}
	if len(b) < len(bb) {
		return 0, errors.New("read would fragment DNS message")
	}
	copy(b, bb)
	return len(bb), nil
}

func (f *fakeDNSConn) Write(b []byte) (int, error) {
	p := dnsmessage.Parser{}
	if f.tcp && len(b) >= 2 {
		b = b[2:] // strip length prefix
	}
	header, err := p.Start(b)
	if err != nil {
		return 0, err
	}
	f.q.Header = header
	qs, err := p.AllQuestions()
	if err != nil {
		return 0, err
	}
	f.q.Questions = qs
	return len(b), nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error { return nil }

type fakeDNSPacketConn struct {
	net.PacketConn
	fakeDNSConn
}

func (f *fakeDNSPacketConn) SetDeadline(t time.Time) error { return f.fakeDNSConn.SetDeadline(t) }
func (f *fakeDNSPacketConn) Close() error                  { return f.fakeDNSConn.Close() }
