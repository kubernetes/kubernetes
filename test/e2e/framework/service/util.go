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

package service

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
)

// TestReachableHTTP tests that the given host serves HTTP on the given port.
func TestReachableHTTP(host string, port int, timeout time.Duration) {
	TestReachableHTTPWithRetriableErrorCodes(host, port, []int{}, timeout)
}

// TestReachableHTTPWithRetriableErrorCodes tests that the given host serves HTTP on the given port with the given retriableErrCodes.
func TestReachableHTTPWithRetriableErrorCodes(host string, port int, retriableErrCodes []int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, "/echo?msg=hello",
			&e2enetwork.HTTPPokeParams{
				BodyContains:   "hello",
				RetriableCodes: retriableErrCodes,
			})
		if result.Status == e2enetwork.HTTPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		if err == wait.ErrWaitTimeout {
			framework.Failf("Could not reach HTTP service through %v:%v after %v", host, port, timeout)
		} else {
			framework.Failf("Failed to reach HTTP service through %v:%v: %v", host, port, err)
		}
	}
}

// TestNotReachableHTTP tests that a HTTP request doesn't connect to the given host and port.
func TestNotReachableHTTP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, "/", nil)
		if result.Code == 0 {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		framework.Failf("HTTP service %v:%v reachable after %v: %v", host, port, timeout, err)
	}
}

// TestRejectedHTTP tests that the given host rejects a HTTP request on the given port.
func TestRejectedHTTP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, "/", nil)
		if result.Status == e2enetwork.HTTPRefused {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		framework.Failf("HTTP service %v:%v not rejected: %v", host, port, err)
	}
}

// TestReachableUDP tests that the given host serves UDP on the given port.
func TestReachableUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := pokeUDP(host, port, "echo hello", &UDPPokeParams{
			Timeout:  3 * time.Second,
			Response: "hello",
		})
		if result.Status == UDPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		framework.Failf("Could not reach UDP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

// TestNotReachableUDP tests that the given host doesn't serve UDP on the given port.
func TestNotReachableUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := pokeUDP(host, port, "echo hello", &UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status != UDPSuccess && result.Status != UDPError {
			return true, nil
		}
		return false, nil // caller can retry
	}
	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		framework.Failf("UDP service %v:%v reachable after %v: %v", host, port, timeout, err)
	}
}

// TestRejectedUDP tests that the given host rejects a UDP request on the given port.
func TestRejectedUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := pokeUDP(host, port, "echo hello", &UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status == UDPRefused {
			return true, nil
		}
		return false, nil // caller can retry
	}
	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		framework.Failf("UDP service %v:%v not rejected: %v", host, port, err)
	}
}

// TestHTTPHealthCheckNodePort tests a HTTP connection by the given request to the given host and port.
func TestHTTPHealthCheckNodePort(host string, port int, request string, timeout time.Duration, expectSucceed bool, threshold int) error {
	count := 0
	condition := func() (bool, error) {
		success, _ := testHTTPHealthCheckNodePort(host, port, request)
		if success && expectSucceed ||
			!success && !expectSucceed {
			count++
		}
		if count >= threshold {
			return true, nil
		}
		return false, nil
	}

	if err := wait.PollImmediate(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error waiting for healthCheckNodePort: expected at least %d succeed=%v on %v%v, got %d", threshold, expectSucceed, host, port, count)
	}
	return nil
}

func testHTTPHealthCheckNodePort(ip string, port int, request string) (bool, error) {
	ipPort := net.JoinHostPort(ip, strconv.Itoa(port))
	url := fmt.Sprintf("http://%s%s", ipPort, request)
	if ip == "" || port == 0 {
		framework.Failf("Got empty IP for reachability check (%s)", url)
		return false, fmt.Errorf("invalid input ip or port")
	}
	framework.Logf("Testing HTTP health check on %v", url)
	resp, err := httpGetNoConnectionPoolTimeout(url, 5*time.Second)
	if err != nil {
		framework.Logf("Got error testing for reachability of %s: %v", url, err)
		return false, err
	}
	defer resp.Body.Close()
	if err != nil {
		framework.Logf("Got error reading response from %s: %v", url, err)
		return false, err
	}
	// HealthCheck responder returns 503 for no local endpoints
	if resp.StatusCode == 503 {
		return false, nil
	}
	// HealthCheck responder returns 200 for non-zero local endpoints
	if resp.StatusCode == 200 {
		return true, nil
	}
	return false, fmt.Errorf("unexpected HTTP response code %s from health check responder at %s", resp.Status, url)
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

// GetHTTPContent returns the content of the given url by HTTP.
func GetHTTPContent(host string, port int, timeout time.Duration, url string) bytes.Buffer {
	var body bytes.Buffer
	if pollErr := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, url, nil)
		if result.Status == e2enetwork.HTTPSuccess {
			body.Write(result.Body)
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		framework.Failf("Could not reach HTTP service through %v:%v%v after %v: %v", host, port, url, timeout, pollErr)
	}
	return body
}

// UDPPokeParams is a struct for UDP poke parameters.
type UDPPokeParams struct {
	Timeout  time.Duration
	Response string
}

// UDPPokeResult is a struct for UDP poke result.
type UDPPokeResult struct {
	Status   UDPPokeStatus
	Error    error  // if there was any error
	Response []byte // if code != 0
}

// UDPPokeStatus is string for representing UDP poke status.
type UDPPokeStatus string

const (
	// UDPSuccess is UDP poke status which is success.
	UDPSuccess UDPPokeStatus = "Success"
	// UDPError is UDP poke status which is error.
	UDPError UDPPokeStatus = "UnknownError"
	// UDPTimeout is UDP poke status which is timeout.
	UDPTimeout UDPPokeStatus = "TimedOut"
	// UDPRefused is UDP poke status which is connection refused.
	UDPRefused UDPPokeStatus = "ConnectionRefused"
	// UDPBadResponse is UDP poke status which is bad response.
	UDPBadResponse UDPPokeStatus = "BadResponse"
	// Any time we add new errors, we should audit all callers of this.
)

// pokeUDP tries to connect to a host on a port and send the given request. Callers
// can specify additional success parameters, if desired.
//
// The result status will be characterized as precisely as possible, given the
// known users of this.
//
// The result error will be populated for any status other than Success.
//
// The result response will be populated if the UDP transaction was completed, even
// if the other test params make this a failure).
func pokeUDP(host string, port int, request string, params *UDPPokeParams) UDPPokeResult {
	hostPort := net.JoinHostPort(host, strconv.Itoa(port))
	url := fmt.Sprintf("udp://%s", hostPort)

	ret := UDPPokeResult{}

	// Sanity check inputs, because it has happened.  These are the only things
	// that should hard fail the test - they are basically ASSERT()s.
	if host == "" {
		framework.Failf("Got empty host for UDP poke (%s)", url)
		return ret
	}
	if port == 0 {
		framework.Failf("Got port==0 for UDP poke (%s)", url)
		return ret
	}

	// Set default params.
	if params == nil {
		params = &UDPPokeParams{}
	}

	framework.Logf("Poking %v", url)

	con, err := net.Dial("udp", hostPort)
	if err != nil {
		ret.Status = UDPError
		ret.Error = err
		framework.Logf("Poke(%q): %v", url, err)
		return ret
	}

	_, err = con.Write([]byte(fmt.Sprintf("%s\n", request)))
	if err != nil {
		ret.Error = err
		neterr, ok := err.(net.Error)
		if ok && neterr.Timeout() {
			ret.Status = UDPTimeout
		} else if strings.Contains(err.Error(), "connection refused") {
			ret.Status = UDPRefused
		} else {
			ret.Status = UDPError
		}
		framework.Logf("Poke(%q): %v", url, err)
		return ret
	}

	if params.Timeout != 0 {
		err = con.SetDeadline(time.Now().Add(params.Timeout))
		if err != nil {
			ret.Status = UDPError
			ret.Error = err
			framework.Logf("Poke(%q): %v", url, err)
			return ret
		}
	}

	bufsize := len(params.Response) + 1
	if bufsize == 0 {
		bufsize = 4096
	}
	var buf = make([]byte, bufsize)
	n, err := con.Read(buf)
	if err != nil {
		ret.Error = err
		neterr, ok := err.(net.Error)
		if ok && neterr.Timeout() {
			ret.Status = UDPTimeout
		} else if strings.Contains(err.Error(), "connection refused") {
			ret.Status = UDPRefused
		} else {
			ret.Status = UDPError
		}
		framework.Logf("Poke(%q): %v", url, err)
		return ret
	}
	ret.Response = buf[0:n]

	if params.Response != "" && string(ret.Response) != params.Response {
		ret.Status = UDPBadResponse
		ret.Error = fmt.Errorf("response does not match expected string: %q", string(ret.Response))
		framework.Logf("Poke(%q): %v", url, ret.Error)
		return ret
	}

	ret.Status = UDPSuccess
	framework.Logf("Poke(%q): success", url)
	return ret
}
