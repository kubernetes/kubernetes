/*
Copyright 2016 The Kubernetes Authors.

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

package network

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"

	compute "google.golang.org/api/compute/v1"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2eendpoints "k8s.io/kubernetes/test/e2e/framework/endpoints"
	e2ekubesystem "k8s.io/kubernetes/test/e2e/framework/kubesystem"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	gcecloud "k8s.io/legacy-cloud-providers/gce"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	defaultServeHostnameServicePort = 80
	defaultServeHostnameServiceName = "svc-hostname"

	// AffinityTimeout is the maximum time that CheckAffinity is allowed to take; this
	// needs to be more than long enough for AffinityConfirmCount HTTP requests to
	// complete in a busy CI cluster, but shouldn't be too long since we will end up
	// waiting the entire time in the tests where affinity is not expected.
	AffinityTimeout = 2 * time.Minute

	// AffinityConfirmCount is the number of needed continuous requests to confirm that
	// affinity is enabled.
	AffinityConfirmCount = 15
)

var (
	defaultServeHostnameService = v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultServeHostnameServiceName,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port:       int32(defaultServeHostnameServicePort),
				TargetPort: intstr.FromInt(9376),
				Protocol:   v1.ProtocolTCP,
			}},
			Selector: map[string]string{
				"name": defaultServeHostnameServiceName,
			},
		},
	}
)

// portsByPodName is a map that maps pod name to container ports.
type portsByPodName map[string][]int

// CheckAffinity function tests whether the service affinity works as expected.
// If affinity is expected, the test will return true once affinityConfirmCount
// number of same response observed in a row. If affinity is not expected, the
// test will keep observe until different responses observed. The function will
// return false only in case of unexpected errors.
func checkAffinity(cs clientset.Interface, execPod *v1.Pod, serviceIP string, servicePort int, shouldHold bool) bool {
	serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	curl := fmt.Sprintf(`curl -q -s --connect-timeout 2 http://%s/`, serviceIPPort)
	cmd := fmt.Sprintf("for i in $(seq 0 %d); do echo; %s ; done", AffinityConfirmCount, curl)
	timeout := AffinityTimeout
	if execPod == nil {
		timeout = e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
	}
	var tracker affinityTracker
	// interval considering a maximum of 2 seconds per connection
	interval := 2 * AffinityConfirmCount * time.Second
	if pollErr := wait.PollImmediate(interval, timeout, func() (bool, error) {
		if execPod != nil {
			stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				framework.Logf("Failed to get response from %s. Retry until timeout", serviceIPPort)
				return false, nil
			}
			hosts := strings.Split(stdout, "\n")
			for _, host := range hosts {
				tracker.recordHost(strings.TrimSpace(host))
			}
		} else {
			rawResponse := GetHTTPContent(serviceIP, servicePort, timeout, "")
			tracker.recordHost(rawResponse.String())
		}
		trackerFulfilled, affinityHolds := tracker.checkHostTrace(AffinityConfirmCount)
		if !shouldHold && !affinityHolds {
			return true, nil
		}
		if shouldHold && trackerFulfilled && affinityHolds {
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		trackerFulfilled, _ := tracker.checkHostTrace(AffinityConfirmCount)
		if pollErr != wait.ErrWaitTimeout {
			checkAffinityFailed(tracker, pollErr.Error())
			return false
		}
		if !trackerFulfilled {
			checkAffinityFailed(tracker, fmt.Sprintf("Connection to %s timed out or not enough responses.", serviceIPPort))
		}
		if shouldHold {
			checkAffinityFailed(tracker, "Affinity should hold but didn't.")
		} else {
			checkAffinityFailed(tracker, "Affinity shouldn't hold but did.")
		}
		return true
	}
	return true
}

// affinityTracker tracks the destination of a request for the affinity tests.
type affinityTracker struct {
	hostTrace []string
}

// Record the response going to a given host.
func (at *affinityTracker) recordHost(host string) {
	at.hostTrace = append(at.hostTrace, host)
	framework.Logf("Received response from host: %s", host)
}

// Check that we got a constant count requests going to the same host.
func (at *affinityTracker) checkHostTrace(count int) (fulfilled, affinityHolds bool) {
	fulfilled = (len(at.hostTrace) >= count)
	if len(at.hostTrace) == 0 {
		return fulfilled, true
	}
	last := at.hostTrace[0:]
	if len(at.hostTrace)-count >= 0 {
		last = at.hostTrace[len(at.hostTrace)-count:]
	}
	host := at.hostTrace[len(at.hostTrace)-1]
	for _, h := range last {
		if h != host {
			return fulfilled, false
		}
	}
	return fulfilled, true
}

func checkAffinityFailed(tracker affinityTracker, err string) {
	framework.Logf("%v", tracker.hostTrace)
	framework.Failf(err)
}

// StartServeHostnameService creates a replication controller that serves its
// hostname and a service on top of it.
func StartServeHostnameService(c clientset.Interface, svc *v1.Service, ns string, replicas int) ([]string, string, error) {
	podNames := make([]string, replicas)
	name := svc.ObjectMeta.Name
	ginkgo.By("creating service " + name + " in namespace " + ns)
	_, err := c.CoreV1().Services(ns).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		return podNames, "", err
	}

	var createdPods []*v1.Pod
	maxContainerFailures := 0
	config := testutils.RCConfig{
		Client:               c,
		Image:                framework.ServeHostnameImage,
		Command:              []string{"/agnhost", "serve-hostname"},
		Name:                 name,
		Namespace:            ns,
		PollInterval:         3 * time.Second,
		Timeout:              framework.PodReadyBeforeTimeout,
		Replicas:             replicas,
		CreatedPods:          &createdPods,
		MaxContainerFailures: &maxContainerFailures,
	}
	err = e2erc.RunRC(config)
	if err != nil {
		return podNames, "", err
	}

	if len(createdPods) != replicas {
		return podNames, "", fmt.Errorf("incorrect number of running pods: %v", len(createdPods))
	}

	for i := range createdPods {
		podNames[i] = createdPods[i].ObjectMeta.Name
	}
	sort.StringSlice(podNames).Sort()

	service, err := c.CoreV1().Services(ns).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return podNames, "", err
	}
	if service.Spec.ClusterIP == "" {
		return podNames, "", fmt.Errorf("service IP is blank for %v", name)
	}
	serviceIP := service.Spec.ClusterIP
	return podNames, serviceIP, nil
}

// StopServeHostnameService stops the given service.
func StopServeHostnameService(clientset clientset.Interface, ns, name string) error {
	if err := e2erc.DeleteRCAndWaitForGC(clientset, ns, name); err != nil {
		return err
	}
	if err := clientset.CoreV1().Services(ns).Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
		return err
	}
	return nil
}

// verifyServeHostnameServiceUp wgets the given serviceIP:servicePort from the
// given host and from within a pod. The host is expected to be an SSH-able node
// in the cluster. Each pod in the service is expected to echo its name. These
// names are compared with the given expectedPods list after a sort | uniq.
func verifyServeHostnameServiceUp(c clientset.Interface, ns, host string, expectedPods []string, serviceIP string, servicePort int) error {
	execPod := e2epod.CreateExecPodOrFail(c, ns, "execpod-", nil)
	defer func() {
		e2epod.DeletePodOrFail(c, ns, execPod.Name)
	}()

	// Loop a bunch of times - the proxy is randomized, so we want a good
	// chance of hitting each backend at least once.
	buildCommand := func(wget string) string {
		serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
		return fmt.Sprintf("for i in $(seq 1 %d); do %s http://%s 2>&1 || true; echo; done",
			50*len(expectedPods), wget, serviceIPPort)
	}
	commands := []func() string{
		// verify service from node
		func() string {
			cmd := "set -e; " + buildCommand("wget -q --timeout=0.2 --tries=1 -O -")
			framework.Logf("Executing cmd %q on host %v", cmd, host)
			result, err := e2essh.SSH(cmd, host, framework.TestContext.Provider)
			if err != nil || result.Code != 0 {
				e2essh.LogResult(result)
				framework.Logf("error while SSH-ing to node: %v", err)
			}
			return result.Stdout
		},
		// verify service from pod
		func() string {
			cmd := buildCommand("wget -q -T 1 -O -")
			framework.Logf("Executing cmd %q in pod %v/%v", cmd, ns, execPod.Name)
			// TODO: Use exec-over-http via the netexec pod instead of kubectl exec.
			output, err := framework.RunHostCmd(ns, execPod.Name, cmd)
			if err != nil {
				framework.Logf("error while kubectl execing %q in pod %v/%v: %v\nOutput: %v", cmd, ns, execPod.Name, err, output)
			}
			return output
		},
	}

	expectedEndpoints := sets.NewString(expectedPods...)
	ginkgo.By(fmt.Sprintf("verifying service has %d reachable backends", len(expectedPods)))
	for _, cmdFunc := range commands {
		passed := false
		gotEndpoints := sets.NewString()

		// Retry cmdFunc for a while
		for start := time.Now(); time.Since(start) < e2eservice.KubeProxyLagTimeout; time.Sleep(5 * time.Second) {
			for _, endpoint := range strings.Split(cmdFunc(), "\n") {
				trimmedEp := strings.TrimSpace(endpoint)
				if trimmedEp != "" {
					gotEndpoints.Insert(trimmedEp)
				}
			}
			// TODO: simply checking that the retrieved endpoints is a superset
			// of the expected allows us to ignore intermitten network flakes that
			// result in output like "wget timed out", but these should be rare
			// and we need a better way to track how often it occurs.
			if gotEndpoints.IsSuperset(expectedEndpoints) {
				if !gotEndpoints.Equal(expectedEndpoints) {
					framework.Logf("Ignoring unexpected output wgetting endpoints of service %s: %v", serviceIP, gotEndpoints.Difference(expectedEndpoints))
				}
				passed = true
				break
			}
			framework.Logf("Unable to reach the following endpoints of service %s: %v", serviceIP, expectedEndpoints.Difference(gotEndpoints))
		}
		if !passed {
			// Sort the lists so they're easier to visually diff.
			exp := expectedEndpoints.List()
			got := gotEndpoints.List()
			sort.StringSlice(exp).Sort()
			sort.StringSlice(got).Sort()
			return fmt.Errorf("service verification failed for: %s\nexpected %v\nreceived %v", serviceIP, exp, got)
		}
	}
	return nil
}

// verifyServeHostnameServiceDown verifies that the given service isn't served.
func verifyServeHostnameServiceDown(c clientset.Interface, host string, serviceIP string, servicePort int) error {
	ipPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	// The current versions of curl included in CentOS and RHEL distros
	// misinterpret square brackets around IPv6 as globbing, so use the -g
	// argument to disable globbing to handle the IPv6 case.
	command := fmt.Sprintf(
		"curl -g -s --connect-timeout 2 http://%s && exit 99", ipPort)

	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(command, host, framework.TestContext.Provider)
		if err != nil {
			e2essh.LogResult(result)
			framework.Logf("error while SSH-ing to node: %v", err)
		}
		if result.Code != 99 {
			return nil
		}
		framework.Logf("service still alive - still waiting")
	}
	return fmt.Errorf("waiting for service to be down timed out")
}

// testNotReachableHTTP tests that a HTTP request doesn't connect to the given host and port.
func testNotReachableHTTP(host string, port int, timeout time.Duration) {
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

// testRejectedHTTP tests that the given host rejects a HTTP request on the given port.
func testRejectedHTTP(host string, port int, timeout time.Duration) {
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

// continuousEcho() uses the same connection for multiple requests, made to run as a goroutine so that
// manipulations can be made to the service and backend pods while a connection is ongoing
// it starts by sending a series of packets to establish conntrack entries and waits for a signal to keep
// sending packts. It returns an error if the number of failed attempts is >= 5
func continuousEcho(host string, port int, timeout time.Duration, maxAttempts int, signal chan struct{}, errorChannel chan error) {
	defer ginkgo.GinkgoRecover()
	const threshold = 10

	// Sanity check inputs, because it has happened.  These are the only things
	// that should hard fail the test - they are basically ASSERT()s.
	if host == "" {
		errorChannel <- fmt.Errorf("Got empty host for continuous echo (%s)", host)
		return
	}
	if port == 0 {
		errorChannel <- fmt.Errorf("Got port ==0 for continuous echo (%d)", port)
		return
	}

	hostPort := net.JoinHostPort(host, strconv.Itoa(port))
	url := fmt.Sprintf("udp://%s", hostPort)

	ret := UDPPokeResult{}

	con, err := net.Dial("udp", hostPort)
	if err != nil {
		ret.Status = UDPError
		ret.Error = err
		errorChannel <- fmt.Errorf("Connection to %q failed: %v", url, err)
		return
	}

	numErrors := 0
	bufsize := len(strconv.Itoa(maxAttempts)) + 1
	var buf = make([]byte, bufsize)

	for i := 0; i < maxAttempts; i++ {
		if i == threshold {
			framework.Logf("Continuous echo waiting for signal to continue")
			<-signal
			if numErrors == threshold {
				errorChannel <- fmt.Errorf("continuous echo was not able to communicate with initial server pod")
				return
			}
		}
		time.Sleep(1 * time.Second)
		err = con.SetDeadline(time.Now().Add(timeout))
		if err != nil {
			ret.Status = UDPError
			ret.Error = err
			framework.Logf("Continuous echo (%q): %v", url, err)
			numErrors++
			continue
		}
		myRequest := fmt.Sprintf("echo %d", i)
		_, err = con.Write([]byte(fmt.Sprintf("%s\n", myRequest)))
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
			numErrors++
			framework.Logf("Continuous echo (%q): %v - %d errors seen so far", url, err, numErrors)
			continue
		}

		err = con.SetDeadline(time.Now().Add(timeout))
		if err != nil {
			ret.Status = UDPError
			ret.Error = err
			numErrors++
			framework.Logf("Continuous echo (%q): %v - %d errors seen so far", url, err, numErrors)
			continue
		}

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
			numErrors++
			framework.Logf("Continuous echo (%q): %v - %d errors seen so far", url, err, numErrors)
			continue
		}
		ret.Response = buf[0:n]

		if string(ret.Response) != fmt.Sprintf("%d", i) {
			ret.Status = UDPBadResponse
			ret.Error = fmt.Errorf("response does not match expected string: %q", string(ret.Response))
			framework.Logf("Continuous echo (%q): %v", url, ret.Error)
			numErrors++
			continue

		}
		ret.Status = UDPSuccess
		framework.Logf("Continuous echo(%q): success", url)
	}

	err = nil
	if numErrors >= threshold {
		err = fmt.Errorf("Too many Errors in continuous echo")
	}

	errorChannel <- err
}

// testReachableUDP tests that the given host serves UDP on the given port.
func testReachableUDP(host string, port int, timeout time.Duration) {
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

// testNotReachableUDP tests that the given host doesn't serve UDP on the given port.
func testNotReachableUDP(host string, port int, timeout time.Duration) {
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

// testRejectedUDP tests that the given host rejects a UDP request on the given port.
func testRejectedUDP(host string, port int, timeout time.Duration) {
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

func getServeHostnameService(name string) *v1.Service {
	svc := defaultServeHostnameService.DeepCopy()
	svc.ObjectMeta.Name = name
	svc.Spec.Selector["name"] = name
	return svc
}

// restartKubeProxy restarts kube-proxy on the given host.
func restartKubeProxy(host string) error {
	// TODO: Make it work for all providers.
	if !framework.ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider for restartKubeProxy: %s", framework.TestContext.Provider)
	}
	// kubelet will restart the kube-proxy since it's running in a static pod
	framework.Logf("Killing kube-proxy on node %v", host)
	result, err := e2essh.SSH("sudo pkill kube-proxy", host, framework.TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart kube-proxy: %v", err)
	}
	// wait for kube-proxy to come back up
	sshCmd := "sudo /bin/sh -c 'pgrep kube-proxy | wc -l'"
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		framework.Logf("Waiting for kubeproxy to come back up with %v on %v", sshCmd, host)
		result, err := e2essh.SSH(sshCmd, host, framework.TestContext.Provider)
		if err != nil {
			return false, err
		}
		if result.Code != 0 {
			e2essh.LogResult(result)
			return false, fmt.Errorf("failed to run command, exited %d", result.Code)
		}
		if result.Stdout == "0\n" {
			return false, nil
		}
		framework.Logf("kube-proxy is back up.")
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("kube-proxy didn't recover: %v", err)
	}
	return nil
}

// waitForApiserverUp waits for the kube-apiserver to be up.
func waitForApiserverUp(c clientset.Interface) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		body, err := c.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.TODO()).Raw()
		if err == nil && string(body) == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

var _ = SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface
	serviceLBNames := []string{}

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentGinkgoTestDescription().Failed {
			DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning load balancer resource for %s", lb)
			e2eservice.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	// TODO: We get coverage of TCP/UDP and multi-port services through the DNS test. We should have a simpler test for multi-port TCP here.

	/*
		Release : v1.9
		Testname: Kubernetes Service
		Description: By default when a kubernetes cluster is running there MUST be a 'kubernetes' service running in the cluster.
	*/
	framework.ConformanceIt("should provide secure master service ", func() {
		_, err := cs.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the service object for the service named kubernetes")
	})

	/*
		Release : v1.9
		Testname: Service, endpoints
		Description: Create a service with a endpoint without any Pods, the service MUST run and show empty endpoints. Add a pod to the service and the service MUST validate to show all the endpoints for the ports exposed by the Pod. Add another Pod then the list of all Ports exposed by both the Pods MUST be valid and have corresponding service endpoint. Once the second Pod is deleted then set of endpoint MUST be validated to show only ports from the first container that are exposed. Once both pods are deleted the endpoints from the service MUST be empty.
	*/
	framework.ConformanceIt("should serve a basic endpoint from pods ", func() {
		serviceName := "endpoint-test2"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		defer func() {
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		_, err := jig.CreateTCPServiceWithPort(nil, 80)
		framework.ExpectNoError(err)

		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		name1 := "pod1"
		name2 := "pod2"

		createPodOrFail(cs, ns, name1, jig.Labels, []v1.ContainerPort{{ContainerPort: 80}})
		names[name1] = true
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{name1: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		createPodOrFail(cs, ns, name2, jig.Labels, []v1.ContainerPort{{ContainerPort: 80}})
		names[name2] = true
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{name1: {80}, name2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, name1)
		delete(names, name1)
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{name2: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, name2)
		delete(names, name2)
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)
	})

	/*
		Release : v1.9
		Testname: Service, endpoints with multiple ports
		Description: Create a service with two ports but no Pods are added to the service yet.  The service MUST run and show empty set of endpoints. Add a Pod to the first port, service MUST list one endpoint for the Pod on that port. Add another Pod to the second port, service MUST list both the endpoints. Delete the first Pod and the service MUST list only the endpoint to the second Pod. Delete the second Pod and the service must now have empty set of endpoints.
	*/
	framework.ConformanceIt("should serve multiport endpoints from pods ", func() {
		// repacking functionality is intentionally not tested here - it's better to test it in an integration test.
		serviceName := "multi-endpoint-test"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		defer func() {
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()

		svc1port := "svc1"
		svc2port := "svc2"

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		_, err := jig.CreateTCPService(func(service *v1.Service) {
			service.Spec.Ports = []v1.ServicePort{
				{
					Name:       "portname1",
					Port:       80,
					TargetPort: intstr.FromString(svc1port),
				},
				{
					Name:       "portname2",
					Port:       81,
					TargetPort: intstr.FromString(svc2port),
				},
			}
		})
		framework.ExpectNoError(err)

		port1 := 100
		port2 := 101
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		containerPorts1 := []v1.ContainerPort{
			{
				Name:          svc1port,
				ContainerPort: int32(port1),
			},
		}
		containerPorts2 := []v1.ContainerPort{
			{
				Name:          svc2port,
				ContainerPort: int32(port2),
			},
		}

		podname1 := "pod1"
		podname2 := "pod2"

		createPodOrFail(cs, ns, podname1, jig.Labels, containerPorts1)
		names[podname1] = true
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podname1: {port1}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		createPodOrFail(cs, ns, podname2, jig.Labels, containerPorts2)
		names[podname2] = true
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podname1: {port1}, podname2: {port2}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, podname1)
		delete(names, podname1)
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{podname2: {port2}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		e2epod.DeletePodOrFail(cs, ns, podname2)
		delete(names, podname2)
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)
	})

	ginkgo.It("should preserve source pod IP for traffic thru service cluster IP [LinuxOnly]", func() {
		// this test is creating a pod with HostNetwork=true, which is not supported on Windows.
		e2eskipper.SkipIfNodeOSDistroIs("windows")

		// This behavior is not supported if Kube-proxy is in "userspace" mode.
		// So we check the kube-proxy mode and skip this test if that's the case.
		if proxyMode, err := proxyMode(f); err == nil {
			if proxyMode == "userspace" {
				e2eskipper.Skipf("The test doesn't work with kube-proxy in userspace mode")
			}
		} else {
			framework.Logf("Couldn't detect KubeProxy mode - test failure may be expected: %v", err)
		}

		serviceName := "sourceip-test"
		ns := f.Namespace.Name

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		servicePort := 8080
		tcpService, err := jig.CreateTCPServiceWithPort(nil, int32(servicePort))
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the sourceip test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		serviceIP := tcpService.Spec.ClusterIP
		framework.Logf("sourceip-test cluster ip: %s", serviceIP)

		ginkgo.By("Picking 2 Nodes to test whether source IP is preserved or not")
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}

		ginkgo.By("Creating a webserver pod to be part of the TCP service which echoes back source ip")
		serverPodName := "echo-sourceip"
		pod := newAgnhostPod(serverPodName, "netexec", "--http-port", strconv.Itoa(servicePort))
		pod.Labels = jig.Labels
		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
		defer func() {
			framework.Logf("Cleaning up the echo server pod")
			err := cs.CoreV1().Pods(ns).Delete(context.TODO(), serverPodName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete pod: %s on node", serverPodName)
		}()

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{serverPodName: {servicePort}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		ginkgo.By("Creating pause pod deployment")
		deployment := createPausePodDeployment(cs, "pause-pod", ns, nodeCounts)

		defer func() {
			framework.Logf("Deleting deployment")
			err = cs.AppsV1().Deployments(ns).Delete(context.TODO(), deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete deployment %s", deployment.Name)
		}()

		framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(cs, deployment), "Failed to complete pause pod deployment")

		deployment, err = cs.AppsV1().Deployments(ns).Get(context.TODO(), deployment.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error in retrieving pause pod deployment")
		labelSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)

		pausePods, err := cs.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector.String()})
		framework.ExpectNoError(err, "Error in listing pods associated with pause pod deployments")

		framework.ExpectNotEqual(pausePods.Items[0].Spec.NodeName, pausePods.Items[1].Spec.NodeName)

		serviceAddress := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))

		for _, pausePod := range pausePods.Items {
			sourceIP, execPodIP := execSourceipTest(pausePod, serviceAddress)
			ginkgo.By("Verifying the preserved source ip")
			framework.ExpectEqual(sourceIP, execPodIP)
		}
	})

	ginkgo.It("should allow pods to hairpin back to themselves through services", func() {
		serviceName := "hairpin-test"
		ns := f.Namespace.Name

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		servicePort := 8080
		svc, err := jig.CreateTCPServiceWithPort(nil, int32(servicePort))
		framework.ExpectNoError(err)
		serviceIP := svc.Spec.ClusterIP
		framework.Logf("hairpin-test cluster ip: %s", serviceIP)

		ginkgo.By("creating a client/server pod")
		serverPodName := "hairpin"
		podTemplate := newAgnhostPod(serverPodName, "netexec", "--http-port", strconv.Itoa(servicePort))
		podTemplate.Labels = jig.Labels
		pod, err := cs.CoreV1().Pods(ns).Create(context.TODO(), podTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))

		ginkgo.By("waiting for the service to expose an endpoint")
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{serverPodName: {servicePort}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		ginkgo.By("Checking if the pod can reach itself")
		err = jig.CheckServiceReachability(svc, pod)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should be able to up and down services", func() {
		// TODO: use the ServiceTestJig here
		// this test uses e2essh.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		// this test does not work if the Node does not support SSH Key
		e2eskipper.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort

		svc1 := "up-down-1"
		svc2 := "up-down-2"
		svc3 := "up-down-3"

		ginkgo.By("creating " + svc1 + " in namespace " + ns)
		podNames1, svc1IP, err := StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)
		ginkgo.By("creating " + svc2 + " in namespace " + ns)
		podNames2, svc2IP, err := StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ginkgo.By("verifying service " + svc1 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		ginkgo.By("verifying service " + svc2 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		// Stop service 1 and make sure it is gone.
		ginkgo.By("stopping service " + svc1)
		framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc1))

		ginkgo.By("verifying service " + svc1 + " is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svc1IP, servicePort))
		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		// Start another service and verify both are up.
		ginkgo.By("creating service " + svc3 + " in namespace " + ns)
		podNames3, svc3IP, err := StartServeHostnameService(cs, getServeHostnameService(svc3), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc3, ns)

		if svc2IP == svc3IP {
			framework.Failf("service IPs conflict: %v", svc2IP)
		}

		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		ginkgo.By("verifying service " + svc3 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames3, svc3IP, servicePort))
	})

	ginkgo.It("should work after restarting kube-proxy [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		e2eskipper.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort

		svc1 := "restart-proxy-1"
		svc2 := "restart-proxy-2"

		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc1))
		}()
		podNames1, svc1IP, err := StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)

		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc2))
		}()
		podNames2, svc2IP, err := StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))

		ginkgo.By(fmt.Sprintf("Restarting kube-proxy on %v", host))
		if err := restartKubeProxy(host); err != nil {
			framework.Failf("error restarting kube-proxy: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))
	})

	ginkgo.It("should work after restarting apiserver [Disruptive]", func() {
		// TODO: use the ServiceTestJig here
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		e2eskipper.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, 80

		svc1 := "restart-apiserver-1"
		svc2 := "restart-apiserver-2"

		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc1))
		}()
		podNames1, svc1IP, err := StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		// Restart apiserver
		ginkgo.By("Restarting apiserver")
		if err := restartApiserver(ns, cs); err != nil {
			framework.Failf("error restarting apiserver: %v", err)
		}
		ginkgo.By("Waiting for apiserver to come up by polling /healthz")
		if err := waitForApiserverUp(cs); err != nil {
			framework.Failf("error while waiting for apiserver up: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))

		// Create a new service and check if it's not reusing IP.
		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc2))
		}()
		podNames2, svc2IP, err := StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podNames2, svc2IP, servicePort))
	})

	ginkgo.It("should be able to preserve UDP traffic when server pod cycles for a NodePort service", func() {
		serviceName := "clusterip-test"
		serverPod1Name := "server-1"
		serverPod2Name := "server-2"

		ns := f.Namespace.Name

		nodeIP, err := e2enode.PickIP(cs) // for later
		framework.ExpectNoError(err)

		// Create a NodePort service
		udpJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a UDP service " + serviceName + " with type=NodePort in " + ns)
		udpService, err := udpJig.CreateUDPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolUDP, TargetPort: intstr.FromInt(80)},
			}
		})
		framework.ExpectNoError(err)

		// Add a backend pod to the service
		ginkgo.By("creating a backend pod for the service " + serviceName)
		serverPod1 := newAgnhostPod(serverPod1Name, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod1.Labels = udpJig.Labels
		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), serverPod1, metav1.CreateOptions{})
		ginkgo.By(fmt.Sprintf("checking NodePort service %s on node with public IP %s", serviceName, nodeIP))
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, serverPod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// Waiting for service to expose endpoint.
		err = validateEndpointsPorts(cs, ns, serviceName, portsByPodName{serverPod1Name: {80}})
		framework.ExpectNoError(err, "failed to validate endpoints for service %s in namespace: %s", serviceName, ns)

		// Check that the pod reveives the traffic
		ginkgo.By("Sending UDP traffic to NodePort service " + serviceName + " on node with publicIP " + nodeIP)
		errorChannel := make(chan error)
		signal := make(chan struct{}, 1)
		go continuousEcho(nodeIP, int(udpService.Spec.Ports[0].NodePort), 3*time.Second, 20, signal, errorChannel)

		// Create a second pod
		ginkgo.By("creating a second pod for the service " + serviceName)
		serverPod2 := newAgnhostPod(serverPod2Name, "netexec", fmt.Sprintf("--udp-port=%d", 80))
		serverPod2.Labels = udpJig.Labels
		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), serverPod2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, serverPod2.Name, f.Namespace.Name, framework.PodStartTimeout))

		// and delete the first pod
		framework.Logf("Cleaning up %s pod", serverPod1Name)
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), serverPod1Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod: %s on node", serverPod1Name)

		// Check that the second pod keeps receiving traffic
		ginkgo.By("Sending UDP traffic to NodePort service " + serviceName + " on node with publicIP " + nodeIP)
		signal <- struct{}{}

		// Check that there are no errors
		err = <-errorChannel
		framework.ExpectNoError(err, "pod communication failed")

	})

	/*
		Release : v1.16
		Testname: Service, NodePort Service
		Description: Create a TCP NodePort service, and test reachability from a client Pod.
		The client Pod MUST be able to access the NodePort service by service name and cluster
		IP on the service port, and on nodes' internal and external IPs on the NodePort.
	*/
	framework.ConformanceIt("should be able to create a functioning NodePort service", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name

		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		nodePortService, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
			}
		})
		framework.ExpectNoError(err)
		err = jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(nodePortService, execPod)
		framework.ExpectNoError(err)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	ginkgo.It("should be able to change the type and ports of a service [Slow] [DisabledForLargeClusters]", func() {
		// requires cloud load-balancer support
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerSupportsUDP := !framework.ProviderIs("aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		// This test is more monolithic than we'd like because LB turnup can be
		// very slow, so we lumped all the tests into one LB lifecycle.

		serviceName := "mutability-test"
		ns1 := f.Namespace.Name // LB1 in ns1 on TCP
		framework.Logf("namespace for TCP test: %s", ns1)

		ginkgo.By("creating a second namespace")
		namespacePtr, err := f.CreateNamespace("services", nil)
		framework.ExpectNoError(err, "failed to create namespace")
		ns2 := namespacePtr.Name // LB2 in ns2 on UDP
		framework.Logf("namespace for UDP test: %s", ns2)

		nodeIP, err := e2enode.PickIP(cs) // for later
		framework.ExpectNoError(err)

		// Test TCP and UDP Services.  Services with the same name in different
		// namespaces should get different node ports and load balancers.

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns1)
		tcpJig := e2eservice.NewTestJig(cs, ns1, serviceName)
		tcpService, err := tcpJig.CreateTCPService(nil)
		framework.ExpectNoError(err)

		ginkgo.By("creating a UDP service " + serviceName + " with type=ClusterIP in namespace " + ns2)
		udpJig := e2eservice.NewTestJig(cs, ns2, serviceName)
		udpService, err := udpJig.CreateUDPService(nil)
		framework.ExpectNoError(err)

		ginkgo.By("verifying that TCP and UDP use the same port")
		if tcpService.Spec.Ports[0].Port != udpService.Spec.Ports[0].Port {
			framework.Failf("expected to use the same port for TCP and UDP")
		}
		svcPort := int(tcpService.Spec.Ports[0].Port)
		framework.Logf("service port (TCP and UDP): %d", svcPort)

		ginkgo.By("creating a pod to be part of the TCP service " + serviceName)
		_, err = tcpJig.Run(nil)
		framework.ExpectNoError(err)

		ginkgo.By("creating a pod to be part of the UDP service " + serviceName)
		_, err = udpJig.Run(nil)
		framework.ExpectNoError(err)

		// Change the services to NodePort.

		ginkgo.By("changing the TCP service to type=NodePort")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		tcpNodePort := int(tcpService.Spec.Ports[0].NodePort)
		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("changing the UDP service to type=NodePort")
		udpService, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		udpNodePort := int(udpService.Spec.Ports[0].NodePort)
		framework.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		// Change the services to LoadBalancer.

		// Here we test that LoadBalancers can receive static IP addresses.  This isn't
		// necessary, but is an additional feature this monolithic test checks.
		requestedIP := ""
		staticIPName := ""
		if framework.ProviderIs("gce", "gke") {
			ginkgo.By("creating a static load balancer IP")
			staticIPName = fmt.Sprintf("e2e-external-lb-test-%s", framework.RunID)
			gceCloud, err := gce.GetGCECloud()
			framework.ExpectNoError(err, "failed to get GCE cloud provider")

			err = gceCloud.ReserveRegionAddress(&compute.Address{Name: staticIPName}, gceCloud.Region())
			defer func() {
				if staticIPName != "" {
					// Release GCE static IP - this is not kube-managed and will not be automatically released.
					if err := gceCloud.DeleteRegionAddress(staticIPName, gceCloud.Region()); err != nil {
						framework.Logf("failed to release static IP %s: %v", staticIPName, err)
					}
				}
			}()
			framework.ExpectNoError(err, "failed to create region address: %s", staticIPName)
			reservedAddr, err := gceCloud.GetRegionAddress(staticIPName, gceCloud.Region())
			framework.ExpectNoError(err, "failed to get region address: %s", staticIPName)

			requestedIP = reservedAddr.Address
			framework.Logf("Allocated static load balancer IP: %s", requestedIP)
		}

		ginkgo.By("changing the TCP service to type=LoadBalancer")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.LoadBalancerIP = requestedIP // will be "" if not applicable
			s.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		if loadBalancerSupportsUDP {
			ginkgo.By("changing the UDP service to type=LoadBalancer")
			udpService, err = udpJig.UpdateService(func(s *v1.Service) {
				s.Spec.Type = v1.ServiceTypeLoadBalancer
			})
			framework.ExpectNoError(err)
		}
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(tcpService))
		if loadBalancerSupportsUDP {
			serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(udpService))
		}

		ginkgo.By("waiting for the TCP service to have a load balancer")
		// Wait for the load balancer to be created asynchronously
		tcpService, err = tcpJig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", tcpNodePort, tcpService.Spec.Ports[0].NodePort)
		}
		if requestedIP != "" && e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != requestedIP {
			framework.Failf("unexpected TCP Status.LoadBalancer.Ingress (expected %s, got %s)", requestedIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		tcpIngressIP := e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0])
		framework.Logf("TCP load balancer: %s", tcpIngressIP)

		if framework.ProviderIs("gce", "gke") {
			// Do this as early as possible, which overrides the `defer` above.
			// This is mostly out of fear of leaking the IP in a timeout case
			// (as of this writing we're not 100% sure where the leaks are
			// coming from, so this is first-aid rather than surgery).
			ginkgo.By("demoting the static IP to ephemeral")
			if staticIPName != "" {
				gceCloud, err := gce.GetGCECloud()
				framework.ExpectNoError(err, "failed to get GCE cloud provider")
				// Deleting it after it is attached "demotes" it to an
				// ephemeral IP, which can be auto-released.
				if err := gceCloud.DeleteRegionAddress(staticIPName, gceCloud.Region()); err != nil {
					framework.Failf("failed to release static IP %s: %v", staticIPName, err)
				}
				staticIPName = ""
			}
		}

		var udpIngressIP string
		if loadBalancerSupportsUDP {
			ginkgo.By("waiting for the UDP service to have a load balancer")
			// 2nd one should be faster since they ran in parallel.
			udpService, err = udpJig.WaitForLoadBalancer(loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
				framework.Failf("UDP Spec.Ports[0].NodePort changed (%d -> %d) when not expected", udpNodePort, udpService.Spec.Ports[0].NodePort)
			}
			udpIngressIP = e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0])
			framework.Logf("UDP load balancer: %s", udpIngressIP)

			ginkgo.By("verifying that TCP and UDP use different load balancers")
			if tcpIngressIP == udpIngressIP {
				framework.Failf("Load balancers are not different: %s", e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
			}
		}

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' node ports.

		ginkgo.By("changing the TCP service's NodePort")
		tcpService, err = tcpJig.ChangeServiceNodePort(tcpNodePort)
		framework.ExpectNoError(err)
		tcpNodePortOld := tcpNodePort
		tcpNodePort = int(tcpService.Spec.Ports[0].NodePort)
		if tcpNodePort == tcpNodePortOld {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) did not change", tcpNodePort)
		}
		if e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("TCP node port: %d", tcpNodePort)

		ginkgo.By("changing the UDP service's NodePort")
		udpService, err = udpJig.ChangeServiceNodePort(udpNodePort)
		framework.ExpectNoError(err)
		udpNodePortOld := udpNodePort
		udpNodePort = int(udpService.Spec.Ports[0].NodePort)
		if udpNodePort == udpNodePortOld {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) did not change", udpNodePort)
		}
		if loadBalancerSupportsUDP && e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}
		framework.Logf("UDP node port: %d", udpNodePort)

		ginkgo.By("hitting the TCP service's new NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's new NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the old TCP NodePort is closed")
		testNotReachableHTTP(nodeIP, tcpNodePortOld, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the old UDP NodePort is closed")
		testNotReachableUDP(nodeIP, udpNodePortOld, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			testReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}

		// Change the services' main ports.

		ginkgo.By("changing the TCP service's port")
		tcpService, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		framework.ExpectNoError(err)
		svcPortOld := svcPort
		svcPort = int(tcpService.Spec.Ports[0].Port)
		if svcPort == svcPortOld {
			framework.Failf("TCP Spec.Ports[0].Port (%d) did not change", svcPort)
		}
		if int(tcpService.Spec.Ports[0].NodePort) != tcpNodePort {
			framework.Failf("TCP Spec.Ports[0].NodePort (%d) changed", tcpService.Spec.Ports[0].NodePort)
		}
		if e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]) != tcpIngressIP {
			framework.Failf("TCP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", tcpIngressIP, e2eservice.GetIngressPoint(&tcpService.Status.LoadBalancer.Ingress[0]))
		}

		ginkgo.By("changing the UDP service's port")
		udpService, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Ports[0].Port++
		})
		framework.ExpectNoError(err)
		if int(udpService.Spec.Ports[0].Port) != svcPort {
			framework.Failf("UDP Spec.Ports[0].Port (%d) did not change", udpService.Spec.Ports[0].Port)
		}
		if int(udpService.Spec.Ports[0].NodePort) != udpNodePort {
			framework.Failf("UDP Spec.Ports[0].NodePort (%d) changed", udpService.Spec.Ports[0].NodePort)
		}
		if loadBalancerSupportsUDP && e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]) != udpIngressIP {
			framework.Failf("UDP Status.LoadBalancer.Ingress changed (%s -> %s) when not expected", udpIngressIP, e2eservice.GetIngressPoint(&udpService.Status.LoadBalancer.Ingress[0]))
		}

		framework.Logf("service port (TCP and UDP): %d", svcPort)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		ginkgo.By("Scaling the pods to 0")
		err = tcpJig.Scale(0)
		framework.ExpectNoError(err)
		err = udpJig.Scale(0)
		framework.ExpectNoError(err)

		ginkgo.By("looking for ICMP REJECT on the TCP service's NodePort")
		testRejectedHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the UDP service's NodePort")
		testRejectedUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("looking for ICMP REJECT on the TCP service's LoadBalancer")
		testRejectedHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("looking for ICMP REJECT on the UDP service's LoadBalancer")
			testRejectedUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		ginkgo.By("Scaling the pods to 1")
		err = tcpJig.Scale(1)
		framework.ExpectNoError(err)
		err = udpJig.Scale(1)
		framework.ExpectNoError(err)

		ginkgo.By("hitting the TCP service's NodePort")
		e2eservice.TestReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the UDP service's NodePort")
		testReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("hitting the TCP service's LoadBalancer")
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, loadBalancerCreateTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("hitting the UDP service's LoadBalancer")
			testReachableUDP(udpIngressIP, svcPort, loadBalancerCreateTimeout)
		}

		// Change the services back to ClusterIP.

		ginkgo.By("changing TCP service back to type=ClusterIP")
		_, err = tcpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		framework.ExpectNoError(err)
		// Wait for the load balancer to be destroyed asynchronously
		_, err = tcpJig.WaitForLoadBalancerDestroy(tcpIngressIP, svcPort, loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("changing UDP service back to type=ClusterIP")
		_, err = udpJig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.Ports[0].NodePort = 0
		})
		framework.ExpectNoError(err)
		if loadBalancerSupportsUDP {
			// Wait for the load balancer to be destroyed asynchronously
			_, err = udpJig.WaitForLoadBalancerDestroy(udpIngressIP, svcPort, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
		}

		ginkgo.By("checking the TCP NodePort is closed")
		testNotReachableHTTP(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the UDP NodePort is closed")
		testNotReachableUDP(nodeIP, udpNodePort, e2eservice.KubeProxyLagTimeout)

		ginkgo.By("checking the TCP LoadBalancer is closed")
		testNotReachableHTTP(tcpIngressIP, svcPort, loadBalancerLagTimeout)

		if loadBalancerSupportsUDP {
			ginkgo.By("checking the UDP LoadBalancer is closed")
			testNotReachableUDP(udpIngressIP, svcPort, loadBalancerLagTimeout)
		}
	})

	/*
		Testname: Service, update NodePort, same port different protocol
		Description: Create a service to accept TCP requests. By default, created service MUST be of type ClusterIP and an ClusterIP MUST be assigned to the service.
		When service type is updated to NodePort supporting TCP protocol, it MUST be reachable on nodeIP over allocated NodePort to serve TCP requests.
		When this NodePort service is updated to use two protocols i.e. TCP and UDP for same assigned service port 80, service update MUST be successful by allocating two NodePorts to the service and
		service MUST be able to serve both TCP and UDP requests over same service port 80.
	*/
	ginkgo.It("should be able to update service type to NodePort listening on same port number but different protocols", func() {
		serviceName := "nodeport-update-service"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		tcpService, err := jig.CreateTCPService(nil)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the updating NodePorts test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()
		framework.Logf("Service Port TCP: %v", tcpService.Spec.Ports[0].Port)

		ginkgo.By("changing the TCP service to type=NodePort")
		nodePortService, err := jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
			s.Spec.Ports = []v1.ServicePort{
				{
					Name:       "tcp-port",
					Port:       80,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt(9376),
				},
			}
		})
		framework.ExpectNoError(err)

		err = jig.CreateTCPUDPServicePods(2)
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(nodePortService, execPod)
		framework.ExpectNoError(err)

		ginkgo.By("Updating NodePort service to listen TCP and UDP based requests over same Port")
		nodePortService, err = jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
			s.Spec.Ports = []v1.ServicePort{
				{
					Name:       "tcp-port",
					Port:       80,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt(9376),
				},
				{
					Name:       "udp-port",
					Port:       80,
					Protocol:   v1.ProtocolUDP,
					TargetPort: intstr.FromInt(9376),
				},
			}
		})
		framework.ExpectNoError(err)
		err = jig.CheckServiceReachability(nodePortService, execPod)
		framework.ExpectNoError(err)
		nodePortCounts := len(nodePortService.Spec.Ports)
		framework.ExpectEqual(nodePortCounts, 2, "updated service should have two Ports but found %d Ports", nodePortCounts)

		for _, port := range nodePortService.Spec.Ports {
			framework.ExpectNotEqual(port.NodePort, 0, "NodePort service failed to allocate NodePort for Port %s", port.Name)
			framework.Logf("NodePort service allocates NodePort: %d for Port: %s over Protocol: %s", port.NodePort, port.Name, port.Protocol)
		}
	})

	/*
		Release: v1.16
		Testname: Service, change type, ExternalName to ClusterIP
		Description: Create a service of type ExternalName, pointing to external DNS. ClusterIP MUST not be assigned to the service.
		Update the service from ExternalName to ClusterIP by removing ExternalName entry, assigning port 80 as service port and TCP as protocol.
		Service update MUST be successful by assigning ClusterIP to the service and it MUST be reachable over serviceName and ClusterIP on provided service port.
	*/
	framework.ConformanceIt("should be able to change the type from ExternalName to ClusterIP", func() {
		serviceName := "externalname-service"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ExternalName in namespace " + ns)
		_, err := jig.CreateExternalNameService(nil)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the ExternalName to ClusterIP test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()

		ginkgo.By("changing the ExternalName service to type=ClusterIP")
		clusterIPService, err := jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeClusterIP
			s.Spec.ExternalName = ""
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
			}
		})
		framework.ExpectNoError(err)

		err = jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(clusterIPService, execPod)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: Service, change type, ExternalName to NodePort
		Description: Create a service of type ExternalName, pointing to external DNS. ClusterIP MUST not be assigned to the service.
		Update the service from ExternalName to NodePort, assigning port 80 as service port and, TCP as protocol.
		service update MUST be successful by exposing service on every node's IP on dynamically assigned NodePort and, ClusterIP MUST be assigned to route service requests.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort. Service MUST also be reachable over node's IP on NodePort.
	*/
	framework.ConformanceIt("should be able to change the type from ExternalName to NodePort", func() {
		serviceName := "externalname-service"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ExternalName in namespace " + ns)
		_, err := jig.CreateExternalNameService(nil)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the ExternalName to NodePort test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()

		ginkgo.By("changing the ExternalName service to type=NodePort")
		nodePortService, err := jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeNodePort
			s.Spec.ExternalName = ""
			s.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
			}
		})
		framework.ExpectNoError(err)
		err = jig.CreateServicePods(2)
		framework.ExpectNoError(err)

		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(nodePortService, execPod)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: Service, change type, ClusterIP to ExternalName
		Description: Create a service of type ClusterIP. Service creation MUST be successful by assigning ClusterIP to the service.
		Update service type from ClusterIP to ExternalName by setting CNAME entry as externalName. Service update MUST be successful and service MUST not has associated ClusterIP.
		Service MUST be able to resolve to IP address by returning A records ensuring service is pointing to provided externalName.
	*/
	framework.ConformanceIt("should be able to change the type from ClusterIP to ExternalName", func() {
		serviceName := "clusterip-service"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=ClusterIP in namespace " + ns)
		_, err := jig.CreateTCPService(nil)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the ClusterIP to ExternalName test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()

		ginkgo.By("Creating active service to test reachability when its FQDN is referred as externalName for another service")
		externalServiceName := "externalsvc"
		externalServiceFQDN := createAndGetExternalServiceFQDN(cs, ns, externalServiceName)
		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, externalServiceName))
		}()

		ginkgo.By("changing the ClusterIP service to type=ExternalName")
		externalNameService, err := jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeExternalName
			s.Spec.ExternalName = externalServiceFQDN
			s.Spec.ClusterIP = ""
		})
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(externalNameService, execPod)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: Service, change type, NodePort to ExternalName
		Description: Create a service of type NodePort. Service creation MUST be successful by exposing service on every node's IP on dynamically assigned NodePort and, ClusterIP MUST be assigned to route service requests.
		Update the service type from NodePort to ExternalName by setting CNAME entry as externalName. Service update MUST be successful and, MUST not has ClusterIP associated with the service and, allocated NodePort MUST be released.
		Service MUST be able to resolve to IP address by returning A records ensuring service is pointing to provided externalName.
	*/
	framework.ConformanceIt("should be able to change the type from NodePort to ExternalName", func() {
		serviceName := "nodeport-service"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating a service " + serviceName + " with the type=NodePort in namespace " + ns)
		_, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("Cleaning up the NodePort to ExternalName test service")
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
		}()

		ginkgo.By("Creating active service to test reachability when its FQDN is referred as externalName for another service")
		externalServiceName := "externalsvc"
		externalServiceFQDN := createAndGetExternalServiceFQDN(cs, ns, externalServiceName)
		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, externalServiceName))
		}()

		ginkgo.By("changing the NodePort service to type=ExternalName")
		externalNameService, err := jig.UpdateService(func(s *v1.Service) {
			s.Spec.Type = v1.ServiceTypeExternalName
			s.Spec.ExternalName = externalServiceFQDN
			s.Spec.ClusterIP = ""
			s.Spec.Ports[0].NodePort = 0
		})
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(externalNameService, execPod)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should prevent NodePort collisions", func() {
		// TODO: use the ServiceTestJig here
		baseName := "nodeport-collision-"
		serviceName1 := baseName + "1"
		serviceName2 := baseName + "2"
		ns := f.Namespace.Name

		t := NewServerTest(cs, ns, serviceName1)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		ginkgo.By("creating service " + serviceName1 + " with type NodePort in namespace " + ns)
		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		result, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName1, ns)

		if result.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", result)
		}
		if len(result.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", result)
		}
		port := result.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].NodePort for new service: %v", result)
		}

		ginkgo.By("creating service " + serviceName2 + " with conflicting NodePort")
		service2 := t.BuildServiceSpec()
		service2.Name = serviceName2
		service2.Spec.Type = v1.ServiceTypeNodePort
		service2.Spec.Ports[0].NodePort = port.NodePort
		result2, err := t.CreateService(service2)
		if err == nil {
			framework.Failf("Created service with conflicting NodePort: %v", result2)
		}
		expectedErr := fmt.Sprintf("%d.*port is already allocated", port.NodePort)
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))

		ginkgo.By("deleting service " + serviceName1 + " to release NodePort")
		err = t.DeleteService(serviceName1)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName1, ns)

		ginkgo.By("creating service " + serviceName2 + " with no-longer-conflicting NodePort")
		_, err = t.CreateService(service2)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName1, ns)
	})

	ginkgo.It("should check NodePort out-of-range", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-range-test"
		ns := f.Namespace.Name

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort

		ginkgo.By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		if service.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !e2eservice.NodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}

		outOfRangeNodePort := 0
		for {
			outOfRangeNodePort = 1 + rand.Intn(65535)
			if !e2eservice.NodePortRange.Contains(outOfRangeNodePort) {
				break
			}
		}
		ginkgo.By(fmt.Sprintf("changing service "+serviceName+" to out-of-range NodePort %d", outOfRangeNodePort))
		result, err := e2eservice.UpdateService(cs, ns, serviceName, func(s *v1.Service) {
			s.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		})
		if err == nil {
			framework.Failf("failed to prevent update of service with out-of-range NodePort: %v", result)
		}
		expectedErr := fmt.Sprintf("%d.*port is not in the valid range", outOfRangeNodePort)
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))

		ginkgo.By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)

		ginkgo.By(fmt.Sprintf("creating service "+serviceName+" with out-of-range NodePort %d", outOfRangeNodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = int32(outOfRangeNodePort)
		service, err = t.CreateService(service)
		if err == nil {
			framework.Failf("failed to prevent create of service with out-of-range NodePort (%d): %v", outOfRangeNodePort, service)
		}
		gomega.Expect(fmt.Sprintf("%v", err)).To(gomega.MatchRegexp(expectedErr))
	})

	ginkgo.It("should release NodePorts on delete", func() {
		// TODO: use the ServiceTestJig here
		serviceName := "nodeport-reuse"
		ns := f.Namespace.Name

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		service := t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort

		ginkgo.By("creating service " + serviceName + " with type NodePort in namespace " + ns)
		service, err := t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)

		if service.Spec.Type != v1.ServiceTypeNodePort {
			framework.Failf("got unexpected Spec.Type for new service: %v", service)
		}
		if len(service.Spec.Ports) != 1 {
			framework.Failf("got unexpected len(Spec.Ports) for new service: %v", service)
		}
		port := service.Spec.Ports[0]
		if port.NodePort == 0 {
			framework.Failf("got unexpected Spec.Ports[0].nodePort for new service: %v", service)
		}
		if !e2eservice.NodePortRange.Contains(int(port.NodePort)) {
			framework.Failf("got unexpected (out-of-range) port for new service: %v", service)
		}
		nodePort := port.NodePort

		ginkgo.By("deleting original service " + serviceName)
		err = t.DeleteService(serviceName)
		framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)

		hostExec := launchHostExecPod(f.ClientSet, f.Namespace.Name, "hostexec")
		cmd := fmt.Sprintf(`! ss -ant46 'sport = :%d' | tail -n +2 | grep LISTEN`, nodePort)
		var stdout string
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				framework.Logf("expected node port (%d) to not be in use, stdout: %v", nodePort, stdout)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected node port (%d) to not be in use in %v, stdout: %v", nodePort, e2eservice.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By(fmt.Sprintf("creating service "+serviceName+" with same NodePort %d", nodePort))
		service = t.BuildServiceSpec()
		service.Spec.Type = v1.ServiceTypeNodePort
		service.Spec.Ports[0].NodePort = nodePort
		_, err = t.CreateService(service)
		framework.ExpectNoError(err, "failed to create service: %s in namespace: %s", serviceName, ns)
	})

	ginkgo.It("should create endpoints for unready pods", func() {
		serviceName := "tolerate-unready"
		ns := f.Namespace.Name

		t := NewServerTest(cs, ns, serviceName)
		defer func() {
			defer ginkgo.GinkgoRecover()
			errs := t.Cleanup()
			if len(errs) != 0 {
				framework.Failf("errors in cleanup: %v", errs)
			}
		}()

		t.Name = "slow-terminating-unready-pod"
		t.Image = imageutils.GetE2EImage(imageutils.Agnhost)
		port := 80
		terminateSeconds := int64(600)

		service := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      t.ServiceName,
				Namespace: t.Namespace,
			},
			Spec: v1.ServiceSpec{
				Selector: t.Labels,
				Ports: []v1.ServicePort{{
					Name:       "http",
					Port:       int32(port),
					TargetPort: intstr.FromInt(port),
				}},
				PublishNotReadyAddresses: true,
			},
		}
		rcSpec := e2erc.ByNameContainer(t.Name, 1, t.Labels, v1.Container{
			Args:  []string{"netexec", fmt.Sprintf("--http-port=%d", port)},
			Name:  t.Name,
			Image: t.Image,
			Ports: []v1.ContainerPort{{ContainerPort: int32(port), Protocol: v1.ProtocolTCP}},
			ReadinessProbe: &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"/bin/false"},
					},
				},
			},
			Lifecycle: &v1.Lifecycle{
				PreStop: &v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"/bin/sleep", fmt.Sprintf("%d", terminateSeconds)},
					},
				},
			},
		}, nil)
		rcSpec.Spec.Template.Spec.TerminationGracePeriodSeconds = &terminateSeconds

		ginkgo.By(fmt.Sprintf("creating RC %v with selectors %v", rcSpec.Name, rcSpec.Spec.Selector))
		_, err := t.CreateRC(rcSpec)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("creating Service %v with selectors %v", service.Name, service.Spec.Selector))
		_, err = t.CreateService(service)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying pods for RC " + t.Name)
		framework.ExpectNoError(e2epod.VerifyPods(t.Client, t.Namespace, t.Name, false, 1))

		svcName := fmt.Sprintf("%v.%v.svc.%v", serviceName, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		ginkgo.By("Waiting for endpoints of Service with DNS name " + svcName)

		execPod := e2epod.CreateExecPodOrFail(f.ClientSet, f.Namespace.Name, "execpod-", nil)
		execPodName := execPod.Name
		cmd := fmt.Sprintf("curl -q -s --connect-timeout 2 http://%s:%d/", svcName, port)
		var stdout string
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				framework.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, e2eservice.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Scaling down replication controller to zero")
		e2erc.ScaleRC(f.ClientSet, f.ScalesGetter, t.Namespace, rcSpec.Name, 0, false)

		ginkgo.By("Update service to not tolerate unready services")
		_, err = e2eservice.UpdateService(f.ClientSet, t.Namespace, t.ServiceName, func(s *v1.Service) {
			s.Spec.PublishNotReadyAddresses = false
		})
		framework.ExpectNoError(err)

		ginkgo.By("Check if pod is unreachable")
		cmd = fmt.Sprintf("curl -q -s --connect-timeout 2 http://%s:%d/; test \"$?\" -ne \"0\"", svcName, port)
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				framework.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, e2eservice.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Update service to tolerate unready services again")
		_, err = e2eservice.UpdateService(f.ClientSet, t.Namespace, t.ServiceName, func(s *v1.Service) {
			s.Spec.PublishNotReadyAddresses = true
		})
		framework.ExpectNoError(err)

		ginkgo.By("Check if terminating pod is available through service")
		cmd = fmt.Sprintf("curl -q -s --connect-timeout 2 http://%s:%d/", svcName, port)
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			var err error
			stdout, err = framework.RunHostCmd(f.Namespace.Name, execPodName, cmd)
			if err != nil {
				framework.Logf("expected un-ready endpoint for Service %v, stdout: %v, err %v", t.Name, stdout, err)
				return false, nil
			}
			return true, nil
		}); pollErr != nil {
			framework.Failf("expected un-ready endpoint for Service %v within %v, stdout: %v", t.Name, e2eservice.KubeProxyLagTimeout, stdout)
		}

		ginkgo.By("Remove pods immediately")
		label := labels.SelectorFromSet(labels.Set(t.Labels))
		options := metav1.ListOptions{LabelSelector: label.String()}
		podClient := t.Client.CoreV1().Pods(f.Namespace.Name)
		pods, err := podClient.List(context.TODO(), options)
		if err != nil {
			framework.Logf("warning: error retrieving pods: %s", err)
		} else {
			for _, pod := range pods.Items {
				var gracePeriodSeconds int64 = 0
				err := podClient.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriodSeconds})
				if err != nil {
					framework.Logf("warning: error force deleting pod '%s': %s", pod.Name, err)
				}
			}
		}
	})

	ginkgo.It("should only allow access from service loadbalancer source ranges [Slow]", func() {
		// this feature currently supported only on GCE/GKE/AWS
		e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")

		loadBalancerLagTimeout := e2eservice.LoadBalancerLagTimeoutDefault
		if framework.ProviderIs("aws") {
			loadBalancerLagTimeout = e2eservice.LoadBalancerLagTimeoutAWS
		}
		loadBalancerCreateTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)

		namespace := f.Namespace.Name
		serviceName := "lb-sourcerange"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("Prepare allow source ips")
		// prepare the exec pods
		// acceptPod are allowed to access the loadbalancer
		acceptPod := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-accept", nil)
		dropPod := e2epod.CreateExecPodOrFail(cs, namespace, "execpod-drop", nil)

		ginkgo.By("creating a pod to be part of the service " + serviceName)
		// This container is an nginx container listening on port 80
		// See kubernetes/contrib/ingress/echoheaders/nginx.conf for content of response
		_, err := jig.Run(nil)
		framework.ExpectNoError(err)
		// Make sure acceptPod is running. There are certain chances that pod might be teminated due to unexpected reasons.
		acceptPod, err = cs.CoreV1().Pods(namespace).Get(context.TODO(), acceptPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", acceptPod.Name)
		framework.ExpectEqual(acceptPod.Status.Phase, v1.PodRunning)
		framework.ExpectNotEqual(acceptPod.Status.PodIP, "")

		// Create loadbalancer service with source range from node[0] and podAccept
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.LoadBalancerSourceRanges = []string{acceptPod.Status.PodIP + "/32"}
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(loadBalancerCreateTimeout)
		framework.ExpectNoError(err)

		// timeout when we haven't just created the load balancer
		normalReachabilityTimeout := 2 * time.Minute

		ginkgo.By("check reachability from different sources")
		svcIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		// Wait longer as this is our first request after creation.  We can't check using a separate method,
		// because the LB should only be reachable from the "accept" pod
		checkReachabilityFromPod(true, loadBalancerLagTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(false, normalReachabilityTimeout, namespace, dropPod.Name, svcIP)

		// Make sure dropPod is running. There are certain chances that the pod might be teminated due to unexpected reasons.		dropPod, err = cs.CoreV1().Pods(namespace).Get(dropPod.Name, metav1.GetOptions{})
		dropPod, err = cs.CoreV1().Pods(namespace).Get(context.TODO(), dropPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Unable to get pod %s", dropPod.Name)
		framework.ExpectEqual(acceptPod.Status.Phase, v1.PodRunning)
		framework.ExpectNotEqual(acceptPod.Status.PodIP, "")

		ginkgo.By("Update service LoadBalancerSourceRange and check reachability")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			// only allow access from dropPod
			svc.Spec.LoadBalancerSourceRanges = []string{dropPod.Status.PodIP + "/32"}
		})
		framework.ExpectNoError(err)
		checkReachabilityFromPod(false, normalReachabilityTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(true, normalReachabilityTimeout, namespace, dropPod.Name, svcIP)

		ginkgo.By("Delete LoadBalancerSourceRange field and check reachability")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.LoadBalancerSourceRanges = nil
		})
		framework.ExpectNoError(err)
		checkReachabilityFromPod(true, normalReachabilityTimeout, namespace, acceptPod.Name, svcIP)
		checkReachabilityFromPod(true, normalReachabilityTimeout, namespace, dropPod.Name, svcIP)
	})

	ginkgo.It("should be able to create an internal type load balancer [Slow]", func() {
		e2eskipper.SkipUnlessProviderIs("azure", "gke", "gce")

		createTimeout := e2eservice.GetServiceLoadBalancerCreationTimeout(cs)
		pollInterval := framework.Poll * 10

		namespace := f.Namespace.Name
		serviceName := "lb-internal"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("creating pod to be part of service " + serviceName)
		_, err := jig.Run(nil)
		framework.ExpectNoError(err)

		enableILB, disableILB := enableAndDisableInternalLB()

		isInternalEndpoint := func(lbIngress *v1.LoadBalancerIngress) bool {
			ingressEndpoint := e2eservice.GetIngressPoint(lbIngress)
			// Needs update for providers using hostname as endpoint.
			return strings.HasPrefix(ingressEndpoint, "10.")
		}

		ginkgo.By("creating a service with type LoadBalancer and cloud specific Internal-LB annotation enabled")
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			enableILB(svc)
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(createTimeout)
		framework.ExpectNoError(err)
		lbIngress := &svc.Status.LoadBalancer.Ingress[0]
		svcPort := int(svc.Spec.Ports[0].Port)
		// should have an internal IP.
		framework.ExpectEqual(isInternalEndpoint(lbIngress), true)

		// ILBs are not accessible from the test orchestrator, so it's necessary to use
		//  a pod to test the service.
		ginkgo.By("hitting the internal load balancer from pod")
		framework.Logf("creating pod with host network")
		hostExec := launchHostExecPod(f.ClientSet, f.Namespace.Name, "ilb-host-exec")

		framework.Logf("Waiting up to %v for service %q's internal LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP := e2eservice.GetIngressPoint(lbIngress)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			cmd := fmt.Sprintf(`curl -m 5 'http://%v:%v/echo?msg=hello'`, tcpIngressIP, svcPort)
			stdout, err := framework.RunHostCmd(hostExec.Namespace, hostExec.Name, cmd)
			if err != nil {
				framework.Logf("error curling; stdout: %v. err: %v", stdout, err)
				return false, nil
			}

			if !strings.Contains(stdout, "hello") {
				framework.Logf("Expected output to contain 'hello', got %q; retrying...", stdout)
				return false, nil
			}

			framework.Logf("Successful curl; stdout: %v", stdout)
			return true, nil
		}); pollErr != nil {
			framework.Failf("ginkgo.Failed to hit ILB IP, err: %v", pollErr)
		}

		ginkgo.By("switching to external type LoadBalancer")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			disableILB(svc)
		})
		framework.ExpectNoError(err)
		framework.Logf("Waiting up to %v for service %q to have an external LoadBalancer", createTimeout, serviceName)
		if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
			svc, err := cs.CoreV1().Services(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			lbIngress = &svc.Status.LoadBalancer.Ingress[0]
			return !isInternalEndpoint(lbIngress), nil
		}); pollErr != nil {
			framework.Failf("Loadbalancer IP not changed to external.")
		}
		// should have an external IP.
		gomega.Expect(isInternalEndpoint(lbIngress)).To(gomega.BeFalse())

		ginkgo.By("hitting the external load balancer")
		framework.Logf("Waiting up to %v for service %q's external LB to respond to requests", createTimeout, serviceName)
		tcpIngressIP = e2eservice.GetIngressPoint(lbIngress)
		e2eservice.TestReachableHTTP(tcpIngressIP, svcPort, e2eservice.LoadBalancerLagTimeoutDefault)

		// GCE cannot test a specific IP because the test may not own it. This cloud specific condition
		// will be removed when GCP supports similar functionality.
		if framework.ProviderIs("azure") {
			ginkgo.By("switching back to interal type LoadBalancer, with static IP specified.")
			internalStaticIP := "10.240.11.11"
			svc, err = jig.UpdateService(func(svc *v1.Service) {
				svc.Spec.LoadBalancerIP = internalStaticIP
				enableILB(svc)
			})
			framework.ExpectNoError(err)
			framework.Logf("Waiting up to %v for service %q to have an internal LoadBalancer", createTimeout, serviceName)
			if pollErr := wait.PollImmediate(pollInterval, createTimeout, func() (bool, error) {
				svc, err := cs.CoreV1().Services(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				lbIngress = &svc.Status.LoadBalancer.Ingress[0]
				return isInternalEndpoint(lbIngress), nil
			}); pollErr != nil {
				framework.Failf("Loadbalancer IP not changed to internal.")
			}
			// should have the given static internal IP.
			framework.ExpectEqual(e2eservice.GetIngressPoint(lbIngress), internalStaticIP)
		}
	})

	// This test creates a load balancer, make sure its health check interval
	// equals to gceHcCheckIntervalSeconds. Then the interval is manipulated
	// to be something else, see if the interval will be reconciled.
	ginkgo.It("should reconcile LB health check interval [Slow][Serial]", func() {
		const gceHcCheckIntervalSeconds = int64(8)
		// This test is for clusters on GCE.
		// (It restarts kube-controller-manager, which we don't support on GKE)
		e2eskipper.SkipUnlessProviderIs("gce")
		e2eskipper.SkipUnlessSSHKeyPresent()

		clusterID, err := gce.GetClusterID(cs)
		if err != nil {
			framework.Failf("framework.GetClusterID(cs) = _, %v; want nil", err)
		}
		gceCloud, err := gce.GetGCECloud()
		if err != nil {
			framework.Failf("framework.GetGCECloud() = _, %v; want nil", err)
		}

		namespace := f.Namespace.Name
		serviceName := "lb-hc-int"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		ginkgo.By("create load balancer service")
		// Create loadbalancer service with source range from node[0] and podAccept
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Clean up loadbalancer service")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		svc, err = jig.WaitForLoadBalancer(e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)

		hcName := gcecloud.MakeNodesHealthCheckName(clusterID)
		hc, err := gceCloud.GetHTTPHealthCheck(hcName)
		if err != nil {
			framework.Failf("gceCloud.GetHttpHealthCheck(%q) = _, %v; want nil", hcName, err)
		}
		framework.ExpectEqual(hc.CheckIntervalSec, gceHcCheckIntervalSeconds)

		ginkgo.By("modify the health check interval")
		hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1
		if err = gceCloud.UpdateHTTPHealthCheck(hc); err != nil {
			framework.Failf("gcecloud.UpdateHttpHealthCheck(%#v) = %v; want nil", hc, err)
		}

		ginkgo.By("restart kube-controller-manager")
		if err := e2ekubesystem.RestartControllerManager(); err != nil {
			framework.Failf("e2ekubesystem.RestartControllerManager() = %v; want nil", err)
		}
		if err := e2ekubesystem.WaitForControllerManagerUp(); err != nil {
			framework.Failf("e2ekubesystem.WaitForControllerManagerUp() = %v; want nil", err)
		}

		ginkgo.By("health check should be reconciled")
		pollInterval := framework.Poll * 10
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
		if pollErr := wait.PollImmediate(pollInterval, loadBalancerPropagationTimeout, func() (bool, error) {
			hc, err := gceCloud.GetHTTPHealthCheck(hcName)
			if err != nil {
				framework.Logf("ginkgo.Failed to get HttpHealthCheck(%q): %v", hcName, err)
				return false, err
			}
			framework.Logf("hc.CheckIntervalSec = %v", hc.CheckIntervalSec)
			return hc.CheckIntervalSec == gceHcCheckIntervalSeconds, nil
		}); pollErr != nil {
			framework.Failf("Health check %q does not reconcile its check interval to %d.", hcName, gceHcCheckIntervalSeconds)
		}
	})

	/*
		Release: v1.19
		Testname: Service, ClusterIP type, session affinity to ClientIP
		Description: Create a service of type "ClusterIP". Service's sessionAffinity is set to "ClientIP". Service creation MUST be successful by assigning "ClusterIP" to the service.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when requests are sent to the service.
		Create another pod to make requests to the service. Service MUST serve the hostname from the same pod of the replica for all consecutive requests.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should have session affinity work for service with type clusterIP [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-clusterip")
		svc.Spec.Type = v1.ServiceTypeClusterIP
		execAffinityTestForNonLBService(f, cs, svc)
	})

	/*
		Release: v1.19
		Testname: Service, ClusterIP type, session affinity to ClientIP with timeout
		Description: Create a service of type "ClusterIP". Service's sessionAffinity is set to "ClientIP" and session affinity timeout is set. Service creation MUST be successful by assigning "ClusterIP" to the service.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when requests are sent to the service.
		Create another pod to make requests to the service. Service MUST serve the hostname from the same pod of the replica for all consecutive requests until timeout expires.
		After timeout, requests MUST be served from different pods of the replica.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should have session affinity timeout work for service with type clusterIP [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-clusterip-timeout")
		svc.Spec.Type = v1.ServiceTypeClusterIP
		execAffinityTestForSessionAffinityTimeout(f, cs, svc)
	})

	/*
		Release: v1.19
		Testname: Service, ClusterIP type, session affinity to None
		Description: Create a service of type "ClusterIP". Service's sessionAffinity is set to "ClientIP". Service creation MUST be successful by assigning "ClusterIP" to the service.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when requests are sent to the service.
		Create another pod to make requests to the service. Update the service's sessionAffinity to "None". Service update MUST be successful. When a requests are made to the service, it MUST be able serve the hostname from any pod of the replica.
		When service's sessionAffinily is updated back to "ClientIP", service MUST serve the hostname from the same pod of the replica for all consecutive requests.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should be able to switch session affinity for service with type clusterIP [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-clusterip-transition")
		svc.Spec.Type = v1.ServiceTypeClusterIP
		execAffinityTestForNonLBServiceWithTransition(f, cs, svc)
	})

	/*
		Release: v1.19
		Testname: Service, NodePort type, session affinity to ClientIP
		Description: Create a service of type "NodePort" and provide service port and protocol. Service's sessionAffinity is set to "ClientIP". Service creation MUST be successful by assigning a "ClusterIP" to service and allocating NodePort on all nodes.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when a requests are sent to the service.
		Create another pod to make requests to the service on node's IP and NodePort. Service MUST serve the hostname from the same pod of the replica for all consecutive requests.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort. Service MUST also be reachable over node's IP on NodePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should have session affinity work for NodePort service [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-nodeport")
		svc.Spec.Type = v1.ServiceTypeNodePort
		execAffinityTestForNonLBService(f, cs, svc)
	})

	/*
		Release: v1.19
		Testname: Service, NodePort type, session affinity to ClientIP with timeout
		Description: Create a service of type "NodePort" and provide service port and protocol. Service's sessionAffinity is set to "ClientIP" and session affinity timeout is set.
		Service creation MUST be successful by assigning a "ClusterIP" to service and allocating NodePort on all nodes.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when requests are sent to the service.
		Create another pod to make requests to the service on node's IP and NodePort. Service MUST serve the hostname from the same pod of the replica for all consecutive requests until timeout.
		After timeout, requests MUST be served from different pods of the replica.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort. Service MUST also be reachable over node's IP on NodePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should have session affinity timeout work for NodePort service [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-nodeport-timeout")
		svc.Spec.Type = v1.ServiceTypeNodePort
		execAffinityTestForSessionAffinityTimeout(f, cs, svc)
	})

	/*
		Release: v1.19
		Testname: Service, NodePort type, session affinity to None
		Description: Create a service of type "NodePort" and provide service port and protocol. Service's sessionAffinity is set to "ClientIP". Service creation MUST be successful by assigning a "ClusterIP" to the service and allocating NodePort on all the nodes.
		Create a Replication Controller to ensure that 3 pods are running and are targeted by the service to serve hostname of the pod when requests are sent to the service.
		Create another pod to make requests to the service. Update the service's sessionAffinity to "None". Service update MUST be successful. When a requests are made to the service on node's IP and NodePort, service MUST be able serve the hostname from any pod of the replica.
		When service's sessionAffinily is updated back to "ClientIP", service MUST serve the hostname from the same pod of the replica for all consecutive requests.
		Service MUST be reachable over serviceName and the ClusterIP on servicePort. Service MUST also be reachable over node's IP on NodePort.
		[LinuxOnly]: Windows does not support session affinity.
	*/
	framework.ConformanceIt("should be able to switch session affinity for NodePort service [LinuxOnly]", func() {
		svc := getServeHostnameService("affinity-nodeport-transition")
		svc.Spec.Type = v1.ServiceTypeNodePort
		execAffinityTestForNonLBServiceWithTransition(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP on [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-esipp-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should have session affinity work for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBService(f, cs, svc)
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #56138 is fixed.
	// [LinuxOnly]: Windows does not support session affinity.
	ginkgo.It("should be able to switch session affinity for LoadBalancer service with ESIPP off [Slow] [DisabledForLargeClusters] [LinuxOnly]", func() {
		// L4 load balancer affinity `ClientIP` is not supported on AWS ELB.
		e2eskipper.SkipIfProviderIs("aws")

		svc := getServeHostnameService("affinity-lb-transition")
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		execAffinityTestForLBServiceWithTransition(f, cs, svc)
	})

	ginkgo.It("should implement service.kubernetes.io/service-proxy-name", func() {
		// this test uses e2essh.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		// this test does not work if the Node does not support SSH Key
		e2eskipper.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort
		serviceProxyNameLabels := map[string]string{"service.kubernetes.io/service-proxy-name": "foo-bar"}

		// We will create 2 services to test creating services in both states and also dynamic updates
		// svcDisabled: Created with the label, will always be disabled. We create this early and
		//              test again late to make sure it never becomes available.
		// svcToggled: Created without the label then the label is toggled verifying reachability at each step.

		ginkgo.By("creating service-disabled in namespace " + ns)
		svcDisabled := getServeHostnameService("service-proxy-disabled")
		svcDisabled.ObjectMeta.Labels = serviceProxyNameLabels
		_, svcDisabledIP, err := StartServeHostnameService(cs, svcDisabled, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svcDisabledIP, ns)

		ginkgo.By("creating service in namespace " + ns)
		svcToggled := getServeHostnameService("service-proxy-toggled")
		podToggledNames, svcToggledIP, err := StartServeHostnameService(cs, svcToggled, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svcToggledIP, ns)

		jig := e2eservice.NewTestJig(cs, ns, svcToggled.ObjectMeta.Name)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcDisabledIP, servicePort))

		ginkgo.By("adding service-proxy-name label")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = serviceProxyNameLabels
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcToggledIP, servicePort))

		ginkgo.By("removing service-proxy-name annotation")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is still not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcDisabledIP, servicePort))
	})

	ginkgo.It("should implement service.kubernetes.io/headless", func() {
		// this test uses e2essh.NodeSSHHosts that does not work if a Node only reports LegacyHostIP
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		// this test does not work if the Node does not support SSH Key
		e2eskipper.SkipUnlessSSHKeyPresent()

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort
		serviceHeadlessLabels := map[string]string{v1.IsHeadlessService: ""}

		// We will create 2 services to test creating services in both states and also dynamic updates
		// svcHeadless: Created with the label, will always be disabled. We create this early and
		//              test again late to make sure it never becomes available.
		// svcHeadlessToggled: Created without the label then the label is toggled verifying reachability at each step.

		ginkgo.By("creating service-headless in namespace " + ns)
		svcHeadless := getServeHostnameService("service-headless")
		svcHeadless.ObjectMeta.Labels = serviceHeadlessLabels
		// This should be improved, as we do not want a Headlesss Service to contain an IP...
		_, svcHeadlessIP, err := StartServeHostnameService(cs, svcHeadless, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with headless service: %s in the namespace: %s", svcHeadlessIP, ns)

		ginkgo.By("creating service in namespace " + ns)
		svcHeadlessToggled := getServeHostnameService("service-headless-toggled")
		podHeadlessToggledNames, svcHeadlessToggledIP, err := StartServeHostnameService(cs, svcHeadlessToggled, ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svcHeadlessToggledIP, ns)

		jig := e2eservice.NewTestJig(cs, ns, svcHeadlessToggled.ObjectMeta.Name)

		hosts, err := e2essh.NodeSSHHosts(cs)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podHeadlessToggledNames, svcHeadlessToggledIP, servicePort))

		ginkgo.By("verifying service-headless is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcHeadlessIP, servicePort))

		ginkgo.By("adding service.kubernetes.io/headless label")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = serviceHeadlessLabels
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcHeadlessToggledIP, servicePort))

		ginkgo.By("removing service.kubernetes.io/headless annotation")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, host, podHeadlessToggledNames, svcHeadlessToggledIP, servicePort))

		ginkgo.By("verifying service-headless is still not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, host, svcHeadlessIP, servicePort))
	})

	ginkgo.It("should be rejected when no endpoints exist", func() {
		namespace := f.Namespace.Name
		serviceName := "no-pods"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		port := 80

		ginkgo.By("creating a service with no endpoints")
		_, err = jig.CreateTCPServiceWithPort(nil, int32(port))
		framework.ExpectNoError(err)

		nodeName := nodes.Items[0].Name
		podName := "execpod-noendpoints"

		ginkgo.By(fmt.Sprintf("creating %v on node %v", podName, nodeName))
		execPod := e2epod.CreateExecPodOrFail(f.ClientSet, namespace, podName, func(pod *v1.Pod) {
			pod.Spec.NodeName = nodeName
		})

		serviceAddress := net.JoinHostPort(serviceName, strconv.Itoa(port))
		framework.Logf("waiting up to %v to connect to %v", e2eservice.KubeProxyEndpointLagTimeout, serviceAddress)
		cmd := fmt.Sprintf("/agnhost connect --timeout=3s %s", serviceAddress)

		ginkgo.By(fmt.Sprintf("hitting service %v from pod %v on node %v", serviceAddress, podName, nodeName))
		expectedErr := "REFUSED"
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyEndpointLagTimeout, func() (bool, error) {
			_, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)

			if err != nil {
				if strings.Contains(err.Error(), expectedErr) {
					framework.Logf("error contained '%s', as expected: %s", expectedErr, err.Error())
					return true, nil
				}
				framework.Logf("error didn't contain '%s', keep trying: %s", expectedErr, err.Error())
				return false, nil
			}
			return true, errors.New("expected connect call to fail")
		}); pollErr != nil {
			framework.ExpectNoError(pollErr)
		}
	})

	// This test verifies if service load balancer cleanup finalizer is properly
	// handled during service lifecycle.
	// 1. Create service with type=LoadBalancer. Finalizer should be added.
	// 2. Update service to type=ClusterIP. Finalizer should be removed.
	// 3. Update service to type=LoadBalancer. Finalizer should be added.
	// 4. Delete service with type=LoadBalancer. Finalizer should be removed.
	ginkgo.It("should handle load balancer cleanup finalizer for service [Slow]", func() {
		jig := e2eservice.NewTestJig(cs, f.Namespace.Name, "lb-finalizer")

		ginkgo.By("Create load balancer service")
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
		})
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Check that service can be deleted with finalizer")
			e2eservice.WaitForServiceDeletedWithFinalizer(cs, svc.Namespace, svc.Name)
		}()

		ginkgo.By("Wait for load balancer to serve traffic")
		svc, err = jig.WaitForLoadBalancer(e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)

		ginkgo.By("Check if finalizer presents on service with type=LoadBalancer")
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)

		ginkgo.By("Check if finalizer is removed on service after changed to type=ClusterIP")
		err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, false)

		ginkgo.By("Check if finalizer is added back to service after changed to type=LoadBalancer")
		err = jig.ChangeServiceType(v1.ServiceTypeLoadBalancer, e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
		framework.ExpectNoError(err)
		e2eservice.WaitForServiceUpdatedWithFinalizer(cs, svc.Namespace, svc.Name, true)
	})

	/*
	   Release : v1.18
	   Testname: Find Kubernetes Service in default Namespace
	   Description: List all Services in all Namespaces, response MUST include a Service named Kubernetes with the Namespace of default.
	*/
	framework.ConformanceIt("should find a service from listing all namespaces", func() {
		ginkgo.By("fetching services")
		svcs, _ := f.ClientSet.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{})

		foundSvc := false
		for _, svc := range svcs.Items {
			if svc.ObjectMeta.Name == "kubernetes" && svc.ObjectMeta.Namespace == "default" {
				foundSvc = true
				break
			}
		}

		framework.ExpectEqual(foundSvc, true, "could not find service 'kubernetes' in service list in all namespaces")
	})

	ginkgo.It("should test the lifecycle of an Endpoint", func() {
		ns := f.Namespace.Name
		testEndpointName := "testservice"

		ginkgo.By("creating an Endpoint")
		_, err := f.ClientSet.CoreV1().Endpoints(ns).Create(context.TODO(), &v1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      testEndpointName,
				Namespace: ns,
				Labels: map[string]string{
					"testendpoint-static": "true",
				},
			},
			Subsets: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: "10.0.0.24",
				}},
				Ports: []v1.EndpointPort{{
					Name:     "http",
					Port:     80,
					Protocol: v1.ProtocolTCP,
				}},
			}},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Endpoint")

		// set up a watch for the Endpoint
		// this timeout was chosen as there was timeout failure from the CI
		endpointWatchTimeoutSeconds := int64(180)
		endpointWatch, err := f.ClientSet.CoreV1().Endpoints(ns).Watch(context.TODO(), metav1.ListOptions{LabelSelector: "testendpoint-static=true", TimeoutSeconds: &endpointWatchTimeoutSeconds})
		framework.ExpectNoError(err, "failed to setup watch on newly created Endpoint")
		endpointWatchChan := endpointWatch.ResultChan()
		ginkgo.By("waiting for available Endpoint")
		for watchEvent := range endpointWatchChan {
			if watchEvent.Type == "ADDED" {
				break
			}
		}

		ginkgo.By("listing all Endpoints")
		endpointsList, err := f.ClientSet.CoreV1().Endpoints("").List(context.TODO(), metav1.ListOptions{LabelSelector: "testendpoint-static=true"})
		framework.ExpectNoError(err, "failed to list Endpoints")
		foundEndpointService := false
		var foundEndpoint v1.Endpoints
		for _, endpoint := range endpointsList.Items {
			if endpoint.ObjectMeta.Name == testEndpointName && endpoint.ObjectMeta.Namespace == ns {
				foundEndpointService = true
				foundEndpoint = endpoint
				break
			}
		}
		framework.ExpectEqual(foundEndpointService, true, "unable to find Endpoint Service in list of Endpoints")

		ginkgo.By("updating the Endpoint")
		foundEndpoint.ObjectMeta.Labels["testservice"] = "first-modification"
		_, err = f.ClientSet.CoreV1().Endpoints(ns).Update(context.TODO(), &foundEndpoint, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update Endpoint with new label")

		ginkgo.By("fetching the Endpoint")
		_, err = f.ClientSet.CoreV1().Endpoints(ns).Get(context.TODO(), testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		framework.ExpectEqual(foundEndpoint.ObjectMeta.Labels["testservice"], "first-modification", "label not patched")

		endpointPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{
					"testservice": "second-modification",
				},
			},
			"subsets": []map[string]interface{}{
				{
					"addresses": []map[string]string{
						{
							"ip": "10.0.0.25",
						},
					},
					"ports": []map[string]interface{}{
						{
							"name": "http-test",
							"port": int32(8080),
						},
					},
				},
			},
		})
		framework.ExpectNoError(err, "failed to marshal JSON for WatchEvent patch")
		ginkgo.By("patching the Endpoint")
		_, err = f.ClientSet.CoreV1().Endpoints(ns).Patch(context.TODO(), testEndpointName, types.StrategicMergePatchType, []byte(endpointPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Endpoint")

		ginkgo.By("fetching the Endpoint")
		endpoint, err := f.ClientSet.CoreV1().Endpoints(ns).Get(context.TODO(), testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		framework.ExpectEqual(endpoint.ObjectMeta.Labels["testservice"], "second-modification", "failed to patch Endpoint with Label")
		endpointSubsetOne := endpoint.Subsets[0]
		endpointSubsetOneAddresses := endpointSubsetOne.Addresses[0]
		endpointSubsetOnePorts := endpointSubsetOne.Ports[0]
		framework.ExpectEqual(endpointSubsetOneAddresses.IP, "10.0.0.25", "failed to patch Endpoint")
		framework.ExpectEqual(endpointSubsetOnePorts.Name, "http-test", "failed to patch Endpoint")
		framework.ExpectEqual(endpointSubsetOnePorts.Port, int32(8080), "failed to patch Endpoint")

		ginkgo.By("deleting the Endpoint by Collection")
		err = f.ClientSet.CoreV1().Endpoints(ns).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "testendpoint-static=true"})
		framework.ExpectNoError(err, "failed to delete Endpoint by Collection")

		ginkgo.By("waiting for Endpoint deletion")
		for watchEvent := range endpointWatchChan {
			if watchEvent.Type == "DELETED" {
				break
			}
		}
	})
})

var _ = SIGDescribe("ESIPP [Slow]", func() {
	f := framework.NewDefaultFramework("esipp")
	var loadBalancerCreateTimeout time.Duration

	var cs clientset.Interface
	serviceLBNames := []string{}

	ginkgo.BeforeEach(func() {
		// requires cloud load-balancer support - this feature currently supported only on GCE/GKE
		e2eskipper.SkipUnlessProviderIs("gce", "gke")

		cs = f.ClientSet
		loadBalancerCreateTimeout = e2eservice.GetServiceLoadBalancerCreationTimeout(cs)
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentGinkgoTestDescription().Failed {
			DescribeSvc(f.Namespace.Name)
		}
		for _, lb := range serviceLBNames {
			framework.Logf("cleaning load balancer resource for %s", lb)
			e2eservice.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
		}
		//reset serviceLBNames
		serviceLBNames = []string{}
	})

	ginkgo.It("should work for type=LoadBalancer", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-lb"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)

			// Make sure we didn't leak the health check node port.
			threshold := 2
			nodes, err := jig.GetEndpointNodes()
			framework.ExpectNoError(err)
			for _, ips := range nodes {
				err := TestHTTPHealthCheckNodePort(ips[0], healthCheckNodePort, "/healthz", e2eservice.KubeProxyEndpointLagTimeout, false, threshold)
				framework.ExpectNoError(err)
			}
			err = cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])

		ginkgo.By("reading clientIP using the TCP service's service port via its external VIP")
		content := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, "/clientip")
		clientIP := content.String()
		framework.Logf("ClientIP detected by target pod using VIP:SvcPort is %s", clientIP)

		ginkgo.By("checking if Source IP is preserved")
		if strings.HasPrefix(clientIP, "10.") {
			framework.Failf("Source IP was NOT preserved")
		}
	})

	ginkgo.It("should work for type=NodePort", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodeport"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalNodePortService(true)
		framework.ExpectNoError(err)
		defer func() {
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		tcpNodePort := int(svc.Spec.Ports[0].NodePort)
		endpointsNodeMap, err := jig.GetEndpointNodes()
		framework.ExpectNoError(err)
		path := "/clientip"

		for nodeName, nodeIPs := range endpointsNodeMap {
			nodeIP := nodeIPs[0]
			ginkgo.By(fmt.Sprintf("reading clientIP using the TCP service's NodePort, on node %v: %v%v%v", nodeName, nodeIP, tcpNodePort, path))
			content := GetHTTPContent(nodeIP, tcpNodePort, e2eservice.KubeProxyLagTimeout, path)
			clientIP := content.String()
			framework.Logf("ClientIP detected by target pod using NodePort is %s", clientIP)
			if strings.HasPrefix(clientIP, "10.") {
				framework.Failf("Source IP was NOT preserved")
			}
		}
	})

	ginkgo.It("should only target nodes with endpoints", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-nodes"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, false,
			func(svc *v1.Service) {
				// Change service port to avoid collision with opened hostPorts
				// in other tests that run in parallel.
				if len(svc.Spec.Ports) != 0 {
					svc.Spec.Ports[0].TargetPort = intstr.FromInt(int(svc.Spec.Ports[0].Port))
					svc.Spec.Ports[0].Port = 8081
				}

			})
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)
		if healthCheckNodePort == 0 {
			framework.Failf("Service HealthCheck NodePort was not allocated")
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeExternalIP)

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		svcTCPPort := int(svc.Spec.Ports[0].Port)

		threshold := 2
		path := "/healthz"
		for i := 0; i < len(nodes.Items); i++ {
			endpointNodeName := nodes.Items[i].Name

			ginkgo.By("creating a pod to be part of the service " + serviceName + " on node " + endpointNodeName)
			_, err = jig.Run(func(rc *v1.ReplicationController) {
				rc.Name = serviceName
				if endpointNodeName != "" {
					rc.Spec.Template.Spec.NodeName = endpointNodeName
				}
			})
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("waiting for service endpoint on node %v", endpointNodeName))
			err = jig.WaitForEndpointOnNode(endpointNodeName)
			framework.ExpectNoError(err)

			// HealthCheck should pass only on the node where num(endpoints) > 0
			// All other nodes should fail the healthcheck on the service healthCheckNodePort
			for n, publicIP := range ips {
				// Make sure the loadbalancer picked up the health check change.
				// Confirm traffic can reach backend through LB before checking healthcheck nodeport.
				e2eservice.TestReachableHTTP(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout)
				expectedSuccess := nodes.Items[n].Name == endpointNodeName
				port := strconv.Itoa(healthCheckNodePort)
				ipPort := net.JoinHostPort(publicIP, port)
				framework.Logf("Health checking %s, http://%s%s, expectedSuccess %v", nodes.Items[n].Name, ipPort, path, expectedSuccess)
				err := TestHTTPHealthCheckNodePort(publicIP, healthCheckNodePort, path, e2eservice.KubeProxyEndpointLagTimeout, expectedSuccess, threshold)
				framework.ExpectNoError(err)
			}
			framework.ExpectNoError(e2erc.DeleteRCAndWaitForGC(f.ClientSet, namespace, serviceName))
		}
	})

	ginkgo.It("should work from pods", func() {
		var err error
		namespace := f.Namespace.Name
		serviceName := "external-local-pods"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		port := strconv.Itoa(int(svc.Spec.Ports[0].Port))
		ipPort := net.JoinHostPort(ingressIP, port)
		path := fmt.Sprintf("%s/clientip", ipPort)

		ginkgo.By("Creating pause pod deployment to make sure, pausePods are in desired state")
		deployment := createPausePodDeployment(cs, "pause-pod-deployment", namespace, 1)
		framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(cs, deployment), "Failed to complete pause pod deployment")

		defer func() {
			framework.Logf("Deleting deployment")
			err = cs.AppsV1().Deployments(namespace).Delete(context.TODO(), deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete deployment %s", deployment.Name)
		}()

		deployment, err = cs.AppsV1().Deployments(namespace).Get(context.TODO(), deployment.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error in retrieving pause pod deployment")
		labelSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
		framework.ExpectNoError(err, "Error in setting LabelSelector as selector from deployment")

		pausePods, err := cs.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector.String()})
		framework.ExpectNoError(err, "Error in listing pods associated with pause pod deployments")

		pausePod := pausePods.Items[0]
		framework.Logf("Waiting up to %v curl %v", e2eservice.KubeProxyLagTimeout, path)
		cmd := fmt.Sprintf(`curl -q -s --connect-timeout 30 %v`, path)

		var srcIP string
		loadBalancerPropagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)
		ginkgo.By(fmt.Sprintf("Hitting external lb %v from pod %v on node %v", ingressIP, pausePod.Name, pausePod.Spec.NodeName))
		if pollErr := wait.PollImmediate(framework.Poll, loadBalancerPropagationTimeout, func() (bool, error) {
			stdout, err := framework.RunHostCmd(pausePod.Namespace, pausePod.Name, cmd)
			if err != nil {
				framework.Logf("got err: %v, retry until timeout", err)
				return false, nil
			}
			srcIP = strings.TrimSpace(strings.Split(stdout, ":")[0])
			return srcIP == pausePod.Status.PodIP, nil
		}); pollErr != nil {
			framework.Failf("Source IP not preserved from %v, expected '%v' got '%v'", pausePod.Name, pausePod.Status.PodIP, srcIP)
		}
	})

	// TODO: Get rid of [DisabledForLargeClusters] tag when issue #90047 is fixed.
	ginkgo.It("should handle updates to ExternalTrafficPolicy field [DisabledForLargeClusters]", func() {
		namespace := f.Namespace.Name
		serviceName := "external-local-update"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			framework.Failf("Need at least 2 nodes to verify source ip from a node without endpoint")
		}

		svc, err := jig.CreateOnlyLocalLoadBalancerService(loadBalancerCreateTimeout, true, nil)
		framework.ExpectNoError(err)
		serviceLBNames = append(serviceLBNames, cloudprovider.DefaultLoadBalancerName(svc))
		defer func() {
			err = jig.ChangeServiceType(v1.ServiceTypeClusterIP, loadBalancerCreateTimeout)
			framework.ExpectNoError(err)
			err := cs.CoreV1().Services(svc.Namespace).Delete(context.TODO(), svc.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		// save the health check node port because it disappears when ESIPP is turned off.
		healthCheckNodePort := int(svc.Spec.HealthCheckNodePort)

		ginkgo.By("turning ESIPP off")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeCluster
		})
		framework.ExpectNoError(err)
		if svc.Spec.HealthCheckNodePort > 0 {
			framework.Failf("Service HealthCheck NodePort still present")
		}

		endpointNodeMap, err := jig.GetEndpointNodes()
		framework.ExpectNoError(err)
		noEndpointNodeMap := map[string][]string{}
		for _, n := range nodes.Items {
			if _, ok := endpointNodeMap[n.Name]; ok {
				continue
			}
			noEndpointNodeMap[n.Name] = e2enode.GetAddresses(&n, v1.NodeExternalIP)
		}

		svcTCPPort := int(svc.Spec.Ports[0].Port)
		svcNodePort := int(svc.Spec.Ports[0].NodePort)
		ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
		path := "/clientip"

		ginkgo.By(fmt.Sprintf("endpoints present on nodes %v, absent on nodes %v", endpointNodeMap, noEndpointNodeMap))
		for nodeName, nodeIPs := range noEndpointNodeMap {
			ginkgo.By(fmt.Sprintf("Checking %v (%v:%v%v) proxies to endpoints on another node", nodeName, nodeIPs[0], svcNodePort, path))
			GetHTTPContent(nodeIPs[0], svcNodePort, e2eservice.KubeProxyLagTimeout, path)
		}

		for nodeName, nodeIPs := range endpointNodeMap {
			ginkgo.By(fmt.Sprintf("checking kube-proxy health check fails on node with endpoint (%s), public IP %s", nodeName, nodeIPs[0]))
			var body bytes.Buffer
			pollfn := func() (bool, error) {
				result := e2enetwork.PokeHTTP(nodeIPs[0], healthCheckNodePort, "/healthz", nil)
				if result.Code == 0 {
					return true, nil
				}
				body.Reset()
				body.Write(result.Body)
				return false, nil
			}
			if pollErr := wait.PollImmediate(framework.Poll, e2eservice.TestTimeout, pollfn); pollErr != nil {
				framework.Failf("Kube-proxy still exposing health check on node %v:%v, after ESIPP was turned off. body %s",
					nodeName, healthCheckNodePort, body.String())
			}
		}

		// Poll till kube-proxy re-adds the MASQUERADE rule on the node.
		ginkgo.By(fmt.Sprintf("checking source ip is NOT preserved through loadbalancer %v", ingressIP))
		var clientIP string
		pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			content := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, "/clientip")
			clientIP = content.String()
			if strings.HasPrefix(clientIP, "10.") {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP WAS preserved even after ESIPP turned off. Got %v, expected a ten-dot cluster ip.", clientIP)
		}

		// TODO: We need to attempt to create another service with the previously
		// allocated healthcheck nodePort. If the health check nodePort has been
		// freed, the new service creation will succeed, upon which we cleanup.
		// If the health check nodePort has NOT been freed, the new service
		// creation will fail.

		ginkgo.By("setting ExternalTraffic field back to OnlyLocal")
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			// Request the same healthCheckNodePort as before, to test the user-requested allocation path
			svc.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		})
		framework.ExpectNoError(err)
		pollErr = wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			content := GetHTTPContent(ingressIP, svcTCPPort, e2eservice.KubeProxyLagTimeout, path)
			clientIP = content.String()
			ginkgo.By(fmt.Sprintf("Endpoint %v:%v%v returned client ip %v", ingressIP, svcTCPPort, path, clientIP))
			if !strings.HasPrefix(clientIP, "10.") {
				return true, nil
			}
			return false, nil
		})
		if pollErr != nil {
			framework.Failf("Source IP (%v) is not the client IP even after ESIPP turned on, expected a public IP.", clientIP)
		}
	})
})

func execSourceipTest(pausePod v1.Pod, serviceAddress string) (string, string) {
	var err error
	var stdout string
	timeout := 2 * time.Minute

	framework.Logf("Waiting up to %v to get response from %s", timeout, serviceAddress)
	cmd := fmt.Sprintf(`curl -q -s --connect-timeout 30 %s/clientip`, serviceAddress)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		stdout, err = framework.RunHostCmd(pausePod.Namespace, pausePod.Name, cmd)
		if err != nil {
			framework.Logf("got err: %v, retry until timeout", err)
			continue
		}
		// Need to check output because it might omit in case of error.
		if strings.TrimSpace(stdout) == "" {
			framework.Logf("got empty stdout, retry until timeout")
			continue
		}
		break
	}

	framework.ExpectNoError(err)

	// The stdout return from RunHostCmd is in this format: x.x.x.x:port or [xx:xx:xx::x]:port
	host, _, err := net.SplitHostPort(stdout)
	if err != nil {
		// ginkgo.Fail the test if output format is unexpected.
		framework.Failf("exec pod returned unexpected stdout: [%v]\n", stdout)
	}
	return pausePod.Status.PodIP, host
}

// execAffinityTestForSessionAffinityTimeout is a helper function that wrap the logic of
// affinity test for non-load-balancer services. Session afinity will be
// enabled when the service is created and a short timeout will be configured so
// session affinity must change after the timeout expirese.
func execAffinityTestForSessionAffinityTimeout(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	ns := f.Namespace.Name
	numPods, servicePort, serviceName := 3, defaultServeHostnameServicePort, svc.ObjectMeta.Name
	ginkgo.By("creating service in namespace " + ns)
	serviceType := svc.Spec.Type
	// set an affinity timeout equal to the number of connection requests
	svcSessionAffinityTimeout := int32(AffinityConfirmCount)
	if proxyMode, err := proxyMode(f); err == nil {
		if proxyMode == "ipvs" {
			// session affinity timeout must be greater than 120 in ipvs mode,
			// because IPVS module has a hardcoded TIME_WAIT timeout of 120s,
			// and that value can't be sysctl'ed now.
			// Ref: https://github.com/torvalds/linux/blob/master/net/netfilter/ipvs/ip_vs_proto_tcp.c
			// TODO: remove this to speed up testing when IPVS does really respect session affinity timeout
			svcSessionAffinityTimeout = int32(125)
		}
	} else {
		framework.Logf("Couldn't detect KubeProxy mode - test failure may be expected: %v", err)
	}
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
		ClientIP: &v1.ClientIPConfig{TimeoutSeconds: &svcSessionAffinityTimeout},
	}
	_, _, err := StartServeHostnameService(cs, svc, ns, numPods)
	framework.ExpectNoError(err, "failed to create replication controller with service in the namespace: %s", ns)
	defer func() {
		StopServeHostnameService(cs, ns, serviceName)
	}()
	jig := e2eservice.NewTestJig(cs, ns, serviceName)
	svc, err = jig.Client.CoreV1().Services(ns).Get(context.TODO(), serviceName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to fetch service: %s in namespace: %s", serviceName, ns)
	var svcIP string
	if serviceType == v1.ServiceTypeNodePort {
		nodes, err := e2enode.GetReadySchedulableNodes(cs)
		framework.ExpectNoError(err)
		addrs := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)
		gomega.Expect(len(addrs)).To(gomega.BeNumerically(">", 0), "ginkgo.Failed to get Node internal IP")
		svcIP = addrs[0]
		servicePort = int(svc.Spec.Ports[0].NodePort)
	} else {
		svcIP = svc.Spec.ClusterIP
	}

	execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod-affinity", nil)
	defer func() {
		framework.Logf("Cleaning up the exec pod")
		err := cs.CoreV1().Pods(ns).Delete(context.TODO(), execPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", execPod.Name, ns)
	}()
	err = jig.CheckServiceReachability(svc, execPod)
	framework.ExpectNoError(err)

	// the service should be sticky until the timeout expires
	framework.ExpectEqual(checkAffinity(cs, execPod, svcIP, servicePort, true), true)
	// but it should return different hostnames after the timeout expires
	// try several times to avoid the probability that we hit the same pod twice
	hosts := sets.NewString()
	cmd := fmt.Sprintf(`curl -q -s --connect-timeout 2 http://%s/`, net.JoinHostPort(svcIP, strconv.Itoa(servicePort)))
	for i := 0; i < 10; i++ {
		hostname, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
		if err == nil {
			hosts.Insert(hostname)
			if hosts.Len() > 1 {
				return
			}
			time.Sleep(time.Duration(svcSessionAffinityTimeout) * time.Second)
		}
	}
	framework.Fail("Session is sticky after reaching the timeout")
}

func execAffinityTestForNonLBServiceWithTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForNonLBServiceWithOptionalTransition(f, cs, svc, true)
}

func execAffinityTestForNonLBService(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForNonLBServiceWithOptionalTransition(f, cs, svc, false)
}

// execAffinityTestForNonLBServiceWithOptionalTransition is a helper function that wrap the logic of
// affinity test for non-load-balancer services. Session afinity will be
// enabled when the service is created. If parameter isTransitionTest is true,
// session affinity will be switched off/on and test if the service converges
// to a stable affinity state.
func execAffinityTestForNonLBServiceWithOptionalTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service, isTransitionTest bool) {
	ns := f.Namespace.Name
	numPods, servicePort, serviceName := 3, defaultServeHostnameServicePort, svc.ObjectMeta.Name
	ginkgo.By("creating service in namespace " + ns)
	serviceType := svc.Spec.Type
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	_, _, err := StartServeHostnameService(cs, svc, ns, numPods)
	framework.ExpectNoError(err, "failed to create replication controller with service in the namespace: %s", ns)
	defer func() {
		StopServeHostnameService(cs, ns, serviceName)
	}()
	jig := e2eservice.NewTestJig(cs, ns, serviceName)
	svc, err = jig.Client.CoreV1().Services(ns).Get(context.TODO(), serviceName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to fetch service: %s in namespace: %s", serviceName, ns)
	var svcIP string
	if serviceType == v1.ServiceTypeNodePort {
		nodes, err := e2enode.GetReadySchedulableNodes(cs)
		framework.ExpectNoError(err)
		addrs := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)
		gomega.Expect(len(addrs)).To(gomega.BeNumerically(">", 0), "ginkgo.Failed to get Node internal IP")
		svcIP = addrs[0]
		servicePort = int(svc.Spec.Ports[0].NodePort)
	} else {
		svcIP = svc.Spec.ClusterIP
	}

	execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod-affinity", nil)
	defer func() {
		framework.Logf("Cleaning up the exec pod")
		err := cs.CoreV1().Pods(ns).Delete(context.TODO(), execPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", execPod.Name, ns)
	}()
	err = jig.CheckServiceReachability(svc, execPod)
	framework.ExpectNoError(err)

	if !isTransitionTest {
		framework.ExpectEqual(checkAffinity(cs, execPod, svcIP, servicePort, true), true)
	}
	if isTransitionTest {
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(checkAffinity(cs, execPod, svcIP, servicePort, false), true)
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(checkAffinity(cs, execPod, svcIP, servicePort, true), true)
	}
}

func execAffinityTestForLBServiceWithTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForLBServiceWithOptionalTransition(f, cs, svc, true)
}

func execAffinityTestForLBService(f *framework.Framework, cs clientset.Interface, svc *v1.Service) {
	execAffinityTestForLBServiceWithOptionalTransition(f, cs, svc, false)
}

// execAffinityTestForLBServiceWithOptionalTransition is a helper function that wrap the logic of
// affinity test for load balancer services, similar to
// execAffinityTestForNonLBServiceWithOptionalTransition.
func execAffinityTestForLBServiceWithOptionalTransition(f *framework.Framework, cs clientset.Interface, svc *v1.Service, isTransitionTest bool) {
	numPods, ns, serviceName := 3, f.Namespace.Name, svc.ObjectMeta.Name

	ginkgo.By("creating service in namespace " + ns)
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	_, _, err := StartServeHostnameService(cs, svc, ns, numPods)
	framework.ExpectNoError(err, "failed to create replication controller with service in the namespace: %s", ns)
	jig := e2eservice.NewTestJig(cs, ns, serviceName)
	ginkgo.By("waiting for loadbalancer for service " + ns + "/" + serviceName)
	svc, err = jig.WaitForLoadBalancer(e2eservice.GetServiceLoadBalancerCreationTimeout(cs))
	framework.ExpectNoError(err)
	defer func() {
		podNodePairs, err := e2enode.PodNodePairs(cs, ns)
		framework.Logf("[pod,node] pairs: %+v; err: %v", podNodePairs, err)
		StopServeHostnameService(cs, ns, serviceName)
		lb := cloudprovider.DefaultLoadBalancerName(svc)
		framework.Logf("cleaning load balancer resource for %s", lb)
		e2eservice.CleanupServiceResources(cs, lb, framework.TestContext.CloudConfig.Region, framework.TestContext.CloudConfig.Zone)
	}()
	ingressIP := e2eservice.GetIngressPoint(&svc.Status.LoadBalancer.Ingress[0])
	port := int(svc.Spec.Ports[0].Port)

	if !isTransitionTest {
		framework.ExpectEqual(checkAffinity(cs, nil, ingressIP, port, true), true)
	}
	if isTransitionTest {
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(checkAffinity(cs, nil, ingressIP, port, false), true)
		svc, err = jig.UpdateService(func(svc *v1.Service) {
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(checkAffinity(cs, nil, ingressIP, port, true), true)
	}
}

func createAndGetExternalServiceFQDN(cs clientset.Interface, ns, serviceName string) string {
	_, _, err := StartServeHostnameService(cs, getServeHostnameService(serviceName), ns, 2)
	framework.ExpectNoError(err, "Expected Service %s to be running", serviceName)
	return fmt.Sprintf("%s.%s.svc.%s", serviceName, ns, framework.TestContext.ClusterDNSDomain)
}

func createPausePodDeployment(cs clientset.Interface, name, ns string, replicas int) *appsv1.Deployment {
	labels := map[string]string{"deployment": "agnhost-pause"}
	pauseDeployment := e2edeployment.NewDeployment(name, int32(replicas), labels, "", "", appsv1.RollingUpdateDeploymentStrategyType)

	pauseDeployment.Spec.Template.Spec.Containers[0] = v1.Container{
		Name:  "agnhost-pause",
		Image: imageutils.GetE2EImage(imageutils.Agnhost),
		Args:  []string{"pause"},
	}
	pauseDeployment.Spec.Template.Spec.Affinity = &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{MatchLabels: labels},
					TopologyKey:   "kubernetes.io/hostname",
					Namespaces:    []string{ns},
				},
			},
		},
	}

	deployment, err := cs.AppsV1().Deployments(ns).Create(context.TODO(), pauseDeployment, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error in creating deployment for pause pod")
	return deployment
}

// createPodOrFail creates a pod with the specified containerPorts.
func createPodOrFail(c clientset.Interface, ns, name string, labels map[string]string, containerPorts []v1.ContainerPort) {
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", name, ns))
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"pause"},
					Ports: containerPorts,
					// Add a dummy environment variable to work around a docker issue.
					// https://github.com/docker/docker/issues/14203
					Env: []v1.EnvVar{{Name: "FOO", Value: " "}},
				},
			},
		},
	}
	_, err := c.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod %s in namespace %s", name, ns)
}

// launchHostExecPod launches a hostexec pod in the given namespace and waits
// until it's Running
func launchHostExecPod(client clientset.Interface, ns, name string) *v1.Pod {
	hostExecPod := e2epod.NewExecPodSpec(ns, name, true)
	pod, err := client.CoreV1().Pods(ns).Create(context.TODO(), hostExecPod, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2epod.WaitForPodRunningInNamespace(client, pod)
	framework.ExpectNoError(err)
	return pod
}

// checkReachabilityFromPod checks reachability from the specified pod.
func checkReachabilityFromPod(expectToBeReachable bool, timeout time.Duration, namespace, pod, target string) {
	cmd := fmt.Sprintf("wget -T 5 -qO- %q", target)
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		_, err := framework.RunHostCmd(namespace, pod, cmd)
		if expectToBeReachable && err != nil {
			framework.Logf("Expect target to be reachable. But got err: %v. Retry until timeout", err)
			return false, nil
		}

		if !expectToBeReachable && err == nil {
			framework.Logf("Expect target NOT to be reachable. But it is reachable. Retry until timeout")
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)
}

// proxyMode returns a proxyMode of a kube-proxy.
func proxyMode(f *framework.Framework) (string, error) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-proxy-mode-detector",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			HostNetwork: true,
			Containers: []v1.Container{
				{
					Name:  "detector",
					Image: agnHostImage,
					Args:  []string{"pause"},
				},
			},
		},
	}
	f.PodClient().CreateSync(pod)
	defer f.PodClient().DeleteSync(pod.Name, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

	cmd := "curl -q -s --connect-timeout 1 http://localhost:10249/proxyMode"
	stdout, err := framework.RunHostCmd(pod.Namespace, pod.Name, cmd)
	if err != nil {
		return "", err
	}
	framework.Logf("proxyMode: %s", stdout)
	return stdout, nil
}

// enableAndDisableInternalLB returns two functions for enabling and disabling the internal load balancer
// setting for the supported cloud providers (currently GCE/GKE and Azure) and empty functions for others.
func enableAndDisableInternalLB() (enable func(svc *v1.Service), disable func(svc *v1.Service)) {
	return framework.TestContext.CloudConfig.Provider.EnableAndDisableInternalLB()
}

func validatePorts(ep e2eendpoints.PortsByPodUID, expectedEndpoints e2eendpoints.PortsByPodUID) error {
	if len(ep) != len(expectedEndpoints) {
		// should not happen because we check this condition before
		return fmt.Errorf("invalid number of endpoints got %v, expected %v", ep, expectedEndpoints)
	}
	for podUID := range expectedEndpoints {
		if _, ok := ep[podUID]; !ok {
			return fmt.Errorf("endpoint %v not found", podUID)
		}
		if len(ep[podUID]) != len(expectedEndpoints[podUID]) {
			return fmt.Errorf("invalid list of ports for uid %v. Got %v, expected %v", podUID, ep[podUID], expectedEndpoints[podUID])
		}
		sort.Ints(ep[podUID])
		sort.Ints(expectedEndpoints[podUID])
		for index := range ep[podUID] {
			if ep[podUID][index] != expectedEndpoints[podUID][index] {
				return fmt.Errorf("invalid list of ports for uid %v. Got %v, expected %v", podUID, ep[podUID], expectedEndpoints[podUID])
			}
		}
	}
	return nil
}

func translatePodNameToUID(c clientset.Interface, ns string, expectedEndpoints portsByPodName) (e2eendpoints.PortsByPodUID, error) {
	portsByUID := make(e2eendpoints.PortsByPodUID)
	for name, portList := range expectedEndpoints {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get pod %s, that's pretty weird. validation failed: %s", name, err)
		}
		portsByUID[pod.ObjectMeta.UID] = portList
	}
	return portsByUID, nil
}

// validateEndpointsPorts validates that the given service exists and is served by the given expectedEndpoints.
func validateEndpointsPorts(c clientset.Interface, namespace, serviceName string, expectedEndpoints portsByPodName) error {
	ginkgo.By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to expose endpoints %v", framework.ServiceStartTimeout, serviceName, namespace, expectedEndpoints))
	i := 1
	for start := time.Now(); time.Since(start) < framework.ServiceStartTimeout; time.Sleep(1 * time.Second) {
		ep, err := c.CoreV1().Endpoints(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Get endpoints failed (%v elapsed, ignoring for 5s): %v", time.Since(start), err)
			continue
		}
		portsByPodUID := e2eendpoints.GetContainerPortsByPodUID(ep)
		expectedPortsByPodUID, err := translatePodNameToUID(c, namespace, expectedEndpoints)
		if err != nil {
			return err
		}
		if len(portsByPodUID) == len(expectedEndpoints) {
			err := validatePorts(portsByPodUID, expectedPortsByPodUID)
			if err != nil {
				return err
			}
			framework.Logf("successfully validated that service %s in namespace %s exposes endpoints %v (%v elapsed)",
				serviceName, namespace, expectedEndpoints, time.Since(start))
			return nil
		}
		if i%5 == 0 {
			framework.Logf("Unexpected endpoints: found %v, expected %v (%v elapsed, will retry)", portsByPodUID, expectedEndpoints, time.Since(start))
		}
		i++
	}
	if pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{}); err == nil {
		for _, pod := range pods.Items {
			framework.Logf("Pod %s\t%s\t%s\t%s", pod.Namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
		}
	} else {
		framework.Logf("Can't list pod debug info: %v", err)
	}
	return fmt.Errorf("Timed out waiting for service %s in namespace %s to expose endpoints %v (%v elapsed)", serviceName, namespace, expectedEndpoints, framework.ServiceStartTimeout)
}

// restartApiserver restarts the kube-apiserver.
func restartApiserver(namespace string, cs clientset.Interface) error {
	// TODO: Make it work for all providers.
	if !framework.ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider for RestartApiserver: %s", framework.TestContext.Provider)
	}
	if framework.ProviderIs("gce", "aws") {
		initialRestartCount, err := getApiserverRestartCount(cs)
		if err != nil {
			return fmt.Errorf("failed to get apiserver's restart count: %v", err)
		}
		if err := sshRestartMaster(); err != nil {
			return fmt.Errorf("failed to restart apiserver: %v", err)
		}
		return waitForApiserverRestarted(cs, initialRestartCount)
	}
	// GKE doesn't allow ssh access, so use a same-version master
	// upgrade to teardown/recreate master.
	v, err := cs.Discovery().ServerVersion()
	if err != nil {
		return err
	}
	return framework.MasterUpgradeGKE(namespace, v.GitVersion[1:]) // strip leading 'v'
}

func sshRestartMaster() error {
	if !framework.ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider for sshRestartMaster: %s", framework.TestContext.Provider)
	}
	var command string
	if framework.ProviderIs("gce") {
		command = "pidof kube-apiserver | xargs sudo kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	framework.Logf("Restarting master via ssh, running: %v", command)
	result, err := e2essh.SSH(command, net.JoinHostPort(framework.GetMasterHost(), e2essh.SSHPort), framework.TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
	}
	return nil
}

// waitForApiserverRestarted waits until apiserver's restart count increased.
func waitForApiserverRestarted(c clientset.Interface, initialRestartCount int32) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		restartCount, err := getApiserverRestartCount(c)
		if err != nil {
			framework.Logf("Failed to get apiserver's restart count: %v", err)
			continue
		}
		if restartCount > initialRestartCount {
			framework.Logf("Apiserver has restarted.")
			return nil
		}
		framework.Logf("Waiting for apiserver restart count to increase")
	}
	return fmt.Errorf("timed out waiting for apiserver to be restarted")
}

func getApiserverRestartCount(c clientset.Interface) (int32, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"component": "kube-apiserver"}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), listOpts)
	if err != nil {
		return -1, err
	}
	if len(pods.Items) != 1 {
		return -1, fmt.Errorf("unexpected number of apiserver pod: %d", len(pods.Items))
	}
	for _, s := range pods.Items[0].Status.ContainerStatuses {
		if s.Name != "kube-apiserver" {
			continue
		}
		return s.RestartCount, nil
	}
	return -1, fmt.Errorf("Failed to find kube-apiserver container in pod")
}
