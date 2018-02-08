// Copyright 2015 The etcd Authors
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

// Package srv looks up DNS SRV records.
package srv

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/coreos/etcd/pkg/types"
)

var (
	// indirection for testing
	lookupSRV      = net.LookupSRV // net.DefaultResolver.LookupSRV when ctxs don't conflict
	resolveTCPAddr = net.ResolveTCPAddr
)

// GetCluster gets the cluster information via DNS discovery.
// Also sees each entry as a separate instance.
func GetCluster(service, name, dns string, apurls types.URLs) ([]string, error) {
	tempName := int(0)
	tcp2ap := make(map[string]url.URL)

	// First, resolve the apurls
	for _, url := range apurls {
		tcpAddr, err := resolveTCPAddr("tcp", url.Host)
		if err != nil {
			return nil, err
		}
		tcp2ap[tcpAddr.String()] = url
	}

	stringParts := []string{}
	updateNodeMap := func(service, scheme string) error {
		_, addrs, err := lookupSRV(service, "tcp", dns)
		if err != nil {
			return err
		}
		for _, srv := range addrs {
			port := fmt.Sprintf("%d", srv.Port)
			host := net.JoinHostPort(srv.Target, port)
			tcpAddr, terr := resolveTCPAddr("tcp", host)
			if terr != nil {
				err = terr
				continue
			}
			n := ""
			url, ok := tcp2ap[tcpAddr.String()]
			if ok {
				n = name
			}
			if n == "" {
				n = fmt.Sprintf("%d", tempName)
				tempName++
			}
			// SRV records have a trailing dot but URL shouldn't.
			shortHost := strings.TrimSuffix(srv.Target, ".")
			urlHost := net.JoinHostPort(shortHost, port)
			stringParts = append(stringParts, fmt.Sprintf("%s=%s://%s", n, scheme, urlHost))
			if ok && url.Scheme != scheme {
				err = fmt.Errorf("bootstrap at %s from DNS for %s has scheme mismatch with expected peer %s", scheme+"://"+urlHost, service, url.String())
			}
		}
		if len(stringParts) == 0 {
			return err
		}
		return nil
	}

	failCount := 0
	err := updateNodeMap(service+"-ssl", "https")
	srvErr := make([]string, 2)
	if err != nil {
		srvErr[0] = fmt.Sprintf("error querying DNS SRV records for _%s-ssl %s", service, err)
		failCount++
	}
	err = updateNodeMap(service, "http")
	if err != nil {
		srvErr[1] = fmt.Sprintf("error querying DNS SRV records for _%s %s", service, err)
		failCount++
	}
	if failCount == 2 {
		return nil, fmt.Errorf("srv: too many errors querying DNS SRV records (%q, %q)", srvErr[0], srvErr[1])
	}
	return stringParts, nil
}

type SRVClients struct {
	Endpoints []string
	SRVs      []*net.SRV
}

// GetClient looks up the client endpoints for a service and domain.
func GetClient(service, domain string) (*SRVClients, error) {
	var urls []*url.URL
	var srvs []*net.SRV

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
		srvs = append(srvs, addrs...)
		return nil
	}

	errHTTPS := updateURLs(service+"-ssl", "https")
	errHTTP := updateURLs(service, "http")

	if errHTTPS != nil && errHTTP != nil {
		return nil, fmt.Errorf("dns lookup errors: %s and %s", errHTTPS, errHTTP)
	}

	endpoints := make([]string, len(urls))
	for i := range urls {
		endpoints[i] = urls[i].String()
	}
	return &SRVClients{Endpoints: endpoints, SRVs: srvs}, nil
}
