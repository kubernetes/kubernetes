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
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// TestReachableHTTP tests that the given host serves HTTP on the given port.
func TestReachableHTTP(host string, port int, timeout time.Duration) {
	TestReachableHTTPWithRetriableErrorCodes(host, port, []int{}, timeout)
}

// TestReachableHTTPWithRetriableErrorCodes tests that the given host serves HTTP on the given port with the given retriableErrCodes.
func TestReachableHTTPWithRetriableErrorCodes(host string, port int, retriableErrCodes []int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeHTTP(host, port, "/echo?msg=hello",
			&framework.HTTPPokeParams{
				BodyContains:   "hello",
				RetriableCodes: retriableErrCodes,
			})
		if result.Status == framework.HTTPSuccess {
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
		result := framework.PokeHTTP(host, port, "/", nil)
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
		result := framework.PokeHTTP(host, port, "/", nil)
		if result.Status == framework.HTTPRefused {
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
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{
			Timeout:  3 * time.Second,
			Response: "hello",
		})
		if result.Status == framework.UDPSuccess {
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
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status != framework.UDPSuccess && result.Status != framework.UDPError {
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
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status == framework.UDPRefused {
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
		result := framework.PokeHTTP(host, port, url, nil)
		if result.Status == framework.HTTPSuccess {
			body.Write(result.Body)
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		framework.Failf("Could not reach HTTP service through %v:%v%v after %v: %v", host, port, url, timeout, pollErr)
	}
	return body
}
