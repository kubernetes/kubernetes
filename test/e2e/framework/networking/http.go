/*
Copyright 2019 The Kubernetes Authors.

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

package networking

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// HTTPPokeParams is a struct for HTTP poke parameters.
type HTTPPokeParams struct {
	Timeout        time.Duration
	ExpectCode     int // default = 200
	BodyContains   string
	RetriableCodes []int
}

// HTTPPokeResult is a struct for HTTP poke result.
type HTTPPokeResult struct {
	Status HTTPPokeStatus
	Code   int    // HTTP code: 0 if the connection was not made
	Error  error  // if there was any error
	Body   []byte // if code != 0
}

// HTTPPokeStatus is string for representing HTTP poke status.
type HTTPPokeStatus string

const (
	// HTTPSuccess is HTTP poke status which is success.
	HTTPSuccess HTTPPokeStatus = "Success"
	// HTTPError is HTTP poke status which is error.
	HTTPError HTTPPokeStatus = "UnknownError"
	// HTTPTimeout is HTTP poke status which is timeout.
	HTTPTimeout HTTPPokeStatus = "TimedOut"
	// HTTPRefused is HTTP poke status which is connection refused.
	HTTPRefused HTTPPokeStatus = "ConnectionRefused"
	// HTTPRetryCode is HTTP poke status which is retry code.
	HTTPRetryCode HTTPPokeStatus = "RetryCode"
	// HTTPWrongCode is HTTP poke status which is wrong code.
	HTTPWrongCode HTTPPokeStatus = "WrongCode"
	// HTTPBadResponse is HTTP poke status which is bad response.
	HTTPBadResponse HTTPPokeStatus = "BadResponse"
	// Any time we add new errors, we should audit all callers of this.
)

// PokeHTTP tries to connect to a host on a port for a given URL path.  Callers
// can specify additional success parameters, if desired.
//
// The result status will be characterized as precisely as possible, given the
// known users of this.
//
// The result code will be zero in case of any failure to connect, or non-zero
// if the HTTP transaction completed (even if the other test params make this a
// failure).
//
// The result error will be populated for any status other than Success.
//
// The result body will be populated if the HTTP transaction was completed, even
// if the other test params make this a failure).
func PokeHTTP(host string, port int, path string, params *HTTPPokeParams) HTTPPokeResult {
	hostPort := net.JoinHostPort(host, strconv.Itoa(port))
	url := fmt.Sprintf("http://%s%s", hostPort, path)

	ret := HTTPPokeResult{}

	// Sanity check inputs, because it has happened.  These are the only things
	// that should hard fail the test - they are basically ASSERT()s.
	if host == "" {
		Failf("Got empty host for HTTP poke (%s)", url)
		return ret
	}
	if port == 0 {
		Failf("Got port==0 for HTTP poke (%s)", url)
		return ret
	}

	// Set default params.
	if params == nil {
		params = &HTTPPokeParams{}
	}
	if params.ExpectCode == 0 {
		params.ExpectCode = http.StatusOK
	}

	e2elog.Logf("Poking %q", url)

	resp, err := httpGetNoConnectionPoolTimeout(url, params.Timeout)
	if err != nil {
		ret.Error = err
		neterr, ok := err.(net.Error)
		if ok && neterr.Timeout() {
			ret.Status = HTTPTimeout
		} else if strings.Contains(err.Error(), "connection refused") {
			ret.Status = HTTPRefused
		} else {
			ret.Status = HTTPError
		}
		e2elog.Logf("Poke(%q): %v", url, err)
		return ret
	}

	ret.Code = resp.StatusCode

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		ret.Status = HTTPError
		ret.Error = fmt.Errorf("error reading HTTP body: %v", err)
		e2elog.Logf("Poke(%q): %v", url, ret.Error)
		return ret
	}
	ret.Body = make([]byte, len(body))
	copy(ret.Body, body)

	if resp.StatusCode != params.ExpectCode {
		for _, code := range params.RetriableCodes {
			if resp.StatusCode == code {
				ret.Error = fmt.Errorf("retriable status code: %d", resp.StatusCode)
				ret.Status = HTTPRetryCode
				e2elog.Logf("Poke(%q): %v", url, ret.Error)
				return ret
			}
		}
		ret.Status = HTTPWrongCode
		ret.Error = fmt.Errorf("bad status code: %d", resp.StatusCode)
		e2elog.Logf("Poke(%q): %v", url, ret.Error)
		return ret
	}

	if params.BodyContains != "" && !strings.Contains(string(body), params.BodyContains) {
		ret.Status = HTTPBadResponse
		ret.Error = fmt.Errorf("response does not contain expected substring: %q", string(body))
		e2elog.Logf("Poke(%q): %v", url, ret.Error)
		return ret
	}

	ret.Status = HTTPSuccess
	e2elog.Logf("Poke(%q): success", url)
	return ret
}

// Does an HTTP GET, but does not reuse TCP connections
// This masks problems where the iptables rule has changed, but we don't see it
func httpGetNoConnectionPoolTimeout(url string, timeout time.Duration) (*http.Response, error) {
	tr := utilnet.SetTransportDefaults(&http.Transport{
		DisableKeepAlives: true,
	})
	client := &http.Client{
		Transport: tr,
		Timeout:   timeout,
	}

	return client.Get(url)
}
