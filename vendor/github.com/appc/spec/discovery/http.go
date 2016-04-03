// Copyright 2015 The appc Authors
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
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"
)

type InsecureOption int

const (
	defaultDialTimeout = 5 * time.Second
)

const (
	InsecureNone InsecureOption = 0

	InsecureTls InsecureOption = 1 << iota
	InsecureHttp
)

var (
	// Client is the default http.Client used for discovery requests.
	Client            *http.Client
	ClientInsecureTls *http.Client

	// httpDo is the internal object used by discovery to retrieve URLs; it is
	// defined here so it can be overridden for testing
	httpDo            httpDoer
	httpDoInsecureTls httpDoer
)

// httpDoer is an interface used to wrap http.Client for real requests and
// allow easy mocking in local tests.
type httpDoer interface {
	Do(req *http.Request) (resp *http.Response, err error)
}

func init() {
	t := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: func(n, a string) (net.Conn, error) {
			return net.DialTimeout(n, a, defaultDialTimeout)
		},
	}
	Client = &http.Client{
		Transport: t,
	}
	httpDo = Client

	// copy for InsecureTls
	tInsecureTls := *t
	tInsecureTls.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	ClientInsecureTls = &http.Client{
		Transport: &tInsecureTls,
	}
	httpDoInsecureTls = ClientInsecureTls
}

func httpsOrHTTP(name string, hostHeaders map[string]http.Header, insecure InsecureOption) (urlStr string, body io.ReadCloser, err error) {
	fetch := func(scheme string) (urlStr string, res *http.Response, err error) {
		u, err := url.Parse(scheme + "://" + name)
		if err != nil {
			return "", nil, err
		}
		u.RawQuery = "ac-discovery=1"
		urlStr = u.String()
		req, err := http.NewRequest("GET", urlStr, nil)
		if err != nil {
			return "", nil, err
		}
		if hostHeader, ok := hostHeaders[u.Host]; ok {
			req.Header = hostHeader
		}
		if insecure&InsecureTls != 0 {
			res, err = httpDoInsecureTls.Do(req)
			return
		}
		res, err = httpDo.Do(req)
		return
	}
	closeBody := func(res *http.Response) {
		if res != nil {
			res.Body.Close()
		}
	}
	urlStr, res, err := fetch("https")
	if err != nil || res.StatusCode != http.StatusOK {
		if insecure&InsecureHttp != 0 {
			closeBody(res)
			urlStr, res, err = fetch("http")
		}
	}

	if res != nil && res.StatusCode != http.StatusOK {
		err = fmt.Errorf("expected a 200 OK got %d", res.StatusCode)
	}

	if err != nil {
		closeBody(res)
		return "", nil, err
	}
	return urlStr, res.Body, nil
}
