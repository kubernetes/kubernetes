// Copyright 2015 CoreOS, Inc.
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

package client

import (
	"fmt"
	"net"
	"net/url"
)

var (
	// indirection for testing
	lookupSRV = net.LookupSRV
)

type srvDiscover struct{}

// NewSRVDiscover constructs a new Dicoverer that uses the stdlib to lookup SRV records.
func NewSRVDiscover() Discoverer {
	return &srvDiscover{}
}

// Discover looks up the etcd servers for the domain.
func (d *srvDiscover) Discover(domain string) ([]string, error) {
	var urls []*url.URL

	updateURLs := func(service, scheme string) error {
		_, addrs, err := lookupSRV(service, "tcp", domain)
		if err != nil {
			return err
		}
		for _, srv := range addrs {
			urls = append(urls, &url.URL{
				Scheme: scheme,
				Host:   net.JoinHostPort(srv.Target, fmt.Sprintf("%d", srv.Port)),
			})
		}
		return nil
	}

	errHTTPS := updateURLs("etcd-server-ssl", "https")
	errHTTP := updateURLs("etcd-server", "http")

	if errHTTPS != nil && errHTTP != nil {
		return nil, fmt.Errorf("dns lookup errors: %s and %s", errHTTPS, errHTTP)
	}

	endpoints := make([]string, len(urls))
	for i := range urls {
		endpoints[i] = urls[i].String()
	}
	return endpoints, nil
}
