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

package discovery

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/coreos/etcd/pkg/types"
)

var (
	// indirection for testing
	lookupSRV      = net.LookupSRV
	resolveTCPAddr = net.ResolveTCPAddr
)

// SRVGetCluster gets the cluster information via DNS discovery.
// TODO(barakmich): Currently ignores priority and weight (as they don't make as much sense for a bootstrap)
// Also doesn't do any lookups for the token (though it could)
// Also sees each entry as a separate instance.
func SRVGetCluster(name, dns string, defaultToken string, apurls types.URLs) (string, string, error) {
	tempName := int(0)
	tcp2ap := make(map[string]url.URL)

	// First, resolve the apurls
	for _, url := range apurls {
		tcpAddr, err := resolveTCPAddr("tcp", url.Host)
		if err != nil {
			plog.Errorf("couldn't resolve host %s during SRV discovery", url.Host)
			return "", "", err
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
			tcpAddr, err := resolveTCPAddr("tcp", host)
			if err != nil {
				plog.Warningf("couldn't resolve host %s during SRV discovery", host)
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
			plog.Noticef("got bootstrap from DNS for %s at %s://%s", service, scheme, urlHost)
			if ok && url.Scheme != scheme {
				plog.Errorf("bootstrap at %s from DNS for %s has scheme mismatch with expected peer %s", scheme+"://"+urlHost, service, url.String())
			}
		}
		return nil
	}

	failCount := 0
	err := updateNodeMap("etcd-server-ssl", "https")
	srvErr := make([]string, 2)
	if err != nil {
		srvErr[0] = fmt.Sprintf("error querying DNS SRV records for _etcd-server-ssl %s", err)
		failCount++
	}
	err = updateNodeMap("etcd-server", "http")
	if err != nil {
		srvErr[1] = fmt.Sprintf("error querying DNS SRV records for _etcd-server %s", err)
		failCount++
	}
	if failCount == 2 {
		plog.Warningf(srvErr[0])
		plog.Warningf(srvErr[1])
		plog.Errorf("SRV discovery failed: too many errors querying DNS SRV records")
		return "", "", err
	}
	return strings.Join(stringParts, ","), defaultToken, nil
}
