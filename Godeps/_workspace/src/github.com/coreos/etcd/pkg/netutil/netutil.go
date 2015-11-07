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

package netutil

import (
	"encoding/base64"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strings"

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd/pkg", "netutil")

	// indirection for testing
	resolveTCPAddr = net.ResolveTCPAddr
)

// resolveTCPAddrs is a convenience wrapper for net.ResolveTCPAddr.
// resolveTCPAddrs return a new set of url.URLs, in which all DNS hostnames
// are resolved.
func resolveTCPAddrs(urls [][]url.URL) ([][]url.URL, error) {
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
			host, _, err := net.SplitHostPort(u.Host)
			if err != nil {
				plog.Errorf("could not parse url %s during tcp resolving", u.Host)
				return nil, err
			}
			if host == "localhost" {
				continue
			}
			if net.ParseIP(host) != nil {
				continue
			}
			tcpAddr, err := resolveTCPAddr("tcp", u.Host)
			if err != nil {
				plog.Errorf("could not resolve host %s", u.Host)
				return nil, err
			}
			plog.Infof("resolving %s to %s", u.Host, tcpAddr.String())
			nus[i].Host = tcpAddr.String()
		}
		newurls = append(newurls, nus)
	}
	return newurls, nil
}

// urlsEqual checks equality of url.URLS between two arrays.
// This check pass even if an URL is in hostname and opposite is in IP address.
func urlsEqual(a []url.URL, b []url.URL) bool {
	if len(a) != len(b) {
		return false
	}
	urls, err := resolveTCPAddrs([][]url.URL{a, b})
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

func URLStringsEqual(a []string, b []string) bool {
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

	return urlsEqual(urlsA, urlsB)
}

// BasicAuth returns the username and password provided in the request's
// Authorization header, if the request uses HTTP Basic Authentication.
// See RFC 2617, Section 2.
// Based on the BasicAuth method from the Golang standard lib.
// TODO: use the standard lib BasicAuth method when we move to Go 1.4.
func BasicAuth(r *http.Request) (username, password string, ok bool) {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return
	}
	return parseBasicAuth(auth)
}

// parseBasicAuth parses an HTTP Basic Authentication string.
// "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==" returns ("Aladdin", "open sesame", true).
// Taken from the Golang standard lib.
// TODO: use the standard lib BasicAuth method when we move to Go 1.4.
func parseBasicAuth(auth string) (username, password string, ok bool) {
	if !strings.HasPrefix(auth, "Basic ") {
		return
	}
	c, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(auth, "Basic "))
	if err != nil {
		return
	}
	cs := string(c)
	s := strings.IndexByte(cs, ':')
	if s < 0 {
		return
	}
	return cs[:s], cs[s+1:], true
}
