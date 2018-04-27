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

// Package netutil implements network-related utility functions.
package netutil

import (
	"context"
	"net"
	"net/url"
	"reflect"
	"sort"
	"time"

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "pkg/netutil")

	// indirection for testing
	resolveTCPAddr = resolveTCPAddrDefault
)

const retryInterval = time.Second

// taken from go's ResolveTCP code but uses configurable ctx
func resolveTCPAddrDefault(ctx context.Context, addr string) (*net.TCPAddr, error) {
	host, port, serr := net.SplitHostPort(addr)
	if serr != nil {
		return nil, serr
	}
	portnum, perr := net.DefaultResolver.LookupPort(ctx, "tcp", port)
	if perr != nil {
		return nil, perr
	}

	var ips []net.IPAddr
	if ip := net.ParseIP(host); ip != nil {
		ips = []net.IPAddr{{IP: ip}}
	} else {
		// Try as a DNS name.
		ipss, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		ips = ipss
	}
	// randomize?
	ip := ips[0]
	return &net.TCPAddr{IP: ip.IP, Port: portnum, Zone: ip.Zone}, nil
}

// resolveTCPAddrs is a convenience wrapper for net.ResolveTCPAddr.
// resolveTCPAddrs return a new set of url.URLs, in which all DNS hostnames
// are resolved.
func resolveTCPAddrs(ctx context.Context, urls [][]url.URL) ([][]url.URL, error) {
	newurls := make([][]url.URL, 0)
	for _, us := range urls {
		nus := make([]url.URL, len(us))
		for i, u := range us {
			nu, err := url.Parse(u.String())
			if err != nil {
				return nil, err
			}
			nus[i] = *nu
		}
		for i, u := range nus {
			h, err := resolveURL(ctx, u)
			if err != nil {
				return nil, err
			}
			if h != "" {
				nus[i].Host = h
			}
		}
		newurls = append(newurls, nus)
	}
	return newurls, nil
}

func resolveURL(ctx context.Context, u url.URL) (string, error) {
	for ctx.Err() == nil {
		host, _, err := net.SplitHostPort(u.Host)
		if err != nil {
			plog.Errorf("could not parse url %s during tcp resolving", u.Host)
			return "", err
		}
		if host == "localhost" || net.ParseIP(host) != nil {
			return "", nil
		}
		tcpAddr, err := resolveTCPAddr(ctx, u.Host)
		if err == nil {
			plog.Infof("resolving %s to %s", u.Host, tcpAddr.String())
			return tcpAddr.String(), nil
		}
		plog.Warningf("failed resolving host %s (%v); retrying in %v", u.Host, err, retryInterval)
		select {
		case <-ctx.Done():
			plog.Errorf("could not resolve host %s", u.Host)
			return "", err
		case <-time.After(retryInterval):
		}
	}
	return "", ctx.Err()
}

// urlsEqual checks equality of url.URLS between two arrays.
// This check pass even if an URL is in hostname and opposite is in IP address.
func urlsEqual(ctx context.Context, a []url.URL, b []url.URL) bool {
	if len(a) != len(b) {
		return false
	}
	urls, err := resolveTCPAddrs(ctx, [][]url.URL{a, b})
	if err != nil {
		return false
	}
	a, b = urls[0], urls[1]
	sort.Sort(types.URLs(a))
	sort.Sort(types.URLs(b))
	for i := range a {
		if !reflect.DeepEqual(a[i], b[i]) {
			return false
		}
	}

	return true
}

func URLStringsEqual(ctx context.Context, a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	urlsA := make([]url.URL, 0)
	for _, str := range a {
		u, err := url.Parse(str)
		if err != nil {
			return false
		}
		urlsA = append(urlsA, *u)
	}
	urlsB := make([]url.URL, 0)
	for _, str := range b {
		u, err := url.Parse(str)
		if err != nil {
			return false
		}
		urlsB = append(urlsB, *u)
	}

	return urlsEqual(ctx, urlsA, urlsB)
}

func IsNetworkTimeoutError(err error) bool {
	nerr, ok := err.(net.Error)
	return ok && nerr.Timeout()
}
