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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	watch "k8s.io/apimachinery/pkg/watch"
	admissionapi "k8s.io/pod-security-admission/api"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/util/retry"

	cloudprovider "k8s.io/cloud-provider"
	netutils "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"

	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2eendpoints "k8s.io/kubernetes/test/e2e/framework/endpoints"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
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

	// label define which is used to find kube-proxy and kube-apiserver pod
	kubeProxyLabelName     = "kube-proxy"
	clusterAddonLabelKey   = "k8s-app"
	kubeAPIServerLabelName = "kube-apiserver"
	clusterComponentKey    = "component"

	svcReadyTimeout = 1 * time.Minute
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

// portsByPodUID is a map that maps pod name to container ports.
type portsByPodUID map[types.UID][]int

// affinityCheckFromPod returns interval, timeout and function pinging the service and
// returning pinged hosts for pinging the service from execPod.
func affinityCheckFromPod(execPod *v1.Pod, serviceIP string, servicePort int) (time.Duration, time.Duration, func() []string) {
	timeout := AffinityTimeout
	// interval considering a maximum of 2 seconds per connection
	interval := 2 * AffinityConfirmCount * time.Second

	serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	curl := fmt.Sprintf(`curl -q -s --connect-timeout 2 http://%s/`, serviceIPPort)
	cmd := fmt.Sprintf("for i in $(seq 0 %d); do echo; %s ; done", AffinityConfirmCount, curl)
	getHosts := func() []string {
		stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
		if err != nil {
			framework.Logf("Failed to get response from %s. Retry until timeout", serviceIPPort)
			return nil
		}
		return strings.Split(stdout, "\n")
	}

	return interval, timeout, getHosts
}

// affinityCheckFromTest returns interval, timeout and function pinging the service and
// returning pinged hosts for pinging the service from the test itself.
func affinityCheckFromTest(cs clientset.Interface, serviceIP string, servicePort int) (time.Duration, time.Duration, func() []string) {
	interval := 2 * time.Second
	timeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(cs)

	params := &e2enetwork.HTTPPokeParams{Timeout: 2 * time.Second}
	getHosts := func() []string {
		var hosts []string
		for i := 0; i < AffinityConfirmCount; i++ {
			result := e2enetwork.PokeHTTP(serviceIP, servicePort, "", params)
			if result.Status == e2enetwork.HTTPSuccess {
				hosts = append(hosts, string(result.Body))
			}
		}
		return hosts
	}

	return interval, timeout, getHosts
}

// CheckAffinity function tests whether the service affinity works as expected.
// If affinity is expected, the test will return true once affinityConfirmCount
// number of same response observed in a row. If affinity is not expected, the
// test will keep observe until different responses observed. The function will
// return false only in case of unexpected errors.
func checkAffinity(cs clientset.Interface, execPod *v1.Pod, serviceIP string, servicePort int, shouldHold bool) bool {
	var interval, timeout time.Duration
	var getHosts func() []string
	if execPod != nil {
		interval, timeout, getHosts = affinityCheckFromPod(execPod, serviceIP, servicePort)
	} else {
		interval, timeout, getHosts = affinityCheckFromTest(cs, serviceIP, servicePort)
	}

	var tracker affinityTracker
	if pollErr := wait.PollImmediate(interval, timeout, func() (bool, error) {
		hosts := getHosts()
		for _, host := range hosts {
			if len(host) > 0 {
				tracker.recordHost(strings.TrimSpace(host))
			}
		}

		trackerFulfilled, affinityHolds := tracker.checkHostTrace(AffinityConfirmCount)
		if !trackerFulfilled {
			return false, nil
		}

		if !shouldHold && !affinityHolds {
			return true, nil
		}
		if shouldHold && affinityHolds {
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
			checkAffinityFailed(tracker, fmt.Sprintf("Connection timed out or not enough responses."))
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
// the host exec pod of host network type and from the exec pod of container network type.
// Each pod in the service is expected to echo its name. These names are compared with the
// given expectedPods list after a sort | uniq.
func verifyServeHostnameServiceUp(c clientset.Interface, ns string, expectedPods []string, serviceIP string, servicePort int) error {
	// to verify from host network
	hostExecPod := launchHostExecPod(c, ns, "verify-service-up-host-exec-pod")

	// to verify from container's network
	execPod := e2epod.CreateExecPodOrFail(c, ns, "verify-service-up-exec-pod-", nil)
	defer func() {
		e2epod.DeletePodOrFail(c, ns, hostExecPod.Name)
		e2epod.DeletePodOrFail(c, ns, execPod.Name)
	}()

	// verify service from pod
	cmdFunc := func(podName string) string {
		wgetCmd := "wget -q -O -"
		// Command 'wget' in Windows image may not support option 'T'
		if !framework.NodeOSDistroIs("windows") {
			wgetCmd += " -T 1"
		}
		serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
		cmd := fmt.Sprintf("for i in $(seq 1 %d); do %s http://%s 2>&1 || true; echo; done",
			50*len(expectedPods), wgetCmd, serviceIPPort)
		framework.Logf("Executing cmd %q in pod %v/%v", cmd, ns, podName)
		// TODO: Use exec-over-http via the netexec pod instead of kubectl exec.
		output, err := framework.RunHostCmd(ns, podName, cmd)
		if err != nil {
			framework.Logf("error while kubectl execing %q in pod %v/%v: %v\nOutput: %v", cmd, ns, podName, err, output)
		}
		return output
	}

	expectedEndpoints := sets.NewString(expectedPods...)
	ginkgo.By(fmt.Sprintf("verifying service has %d reachable backends", len(expectedPods)))
	for _, podName := range []string{hostExecPod.Name, execPod.Name} {
		passed := false
		gotEndpoints := sets.NewString()

		// Retry cmdFunc for a while
		for start := time.Now(); time.Since(start) < e2eservice.KubeProxyLagTimeout; time.Sleep(5 * time.Second) {
			for _, endpoint := range strings.Split(cmdFunc(podName), "\n") {
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
func verifyServeHostnameServiceDown(c clientset.Interface, ns string, serviceIP string, servicePort int) error {
	// verify from host network
	hostExecPod := launchHostExecPod(c, ns, "verify-service-down-host-exec-pod")

	defer func() {
		e2epod.DeletePodOrFail(c, ns, hostExecPod.Name)
	}()

	ipPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	// The current versions of curl included in CentOS and RHEL distros
	// misinterpret square brackets around IPv6 as globbing, so use the -g
	// argument to disable globbing to handle the IPv6 case.
	command := fmt.Sprintf(
		"curl -g -s --connect-timeout 2 http://%s && echo service-down-failed", ipPort)

	for start := time.Now(); time.Since(start) < e2eservice.KubeProxyLagTimeout; time.Sleep(5 * time.Second) {
		output, err := framework.RunHostCmd(ns, hostExecPod.Name, command)
		if err != nil {
			framework.Logf("error while kubectl execing %q in pod %v/%v: %v\nOutput: %v", command, ns, hostExecPod.Name, err, output)
		}
		if !strings.Contains(output, "service-down-failed") {
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

func testHTTPHealthCheckNodePortFromTestContainer(config *e2enetwork.NetworkingTestConfig, host string, port int, timeout time.Duration, expectSucceed bool, threshold int) error {
	count := 0
	pollFn := func() (bool, error) {
		statusCode, err := config.GetHTTPCodeFromTestContainer(
			"/healthz",
			host,
			port)
		if err != nil {
			framework.Logf("Got error reading status code from http://%s:%d/healthz via test container: %v", host, port, err)
			return false, nil
		}
		framework.Logf("Got status code from http://%s:%d/healthz via test container: %d", host, port, statusCode)
		success := statusCode == 200
		if (success && expectSucceed) ||
			(!success && !expectSucceed) {
			count++
		}
		return count >= threshold, nil
	}
	err := wait.PollImmediate(time.Second, timeout, pollFn)
	if err != nil {
		return fmt.Errorf("error waiting for healthCheckNodePort: expected at least %d succeed=%v on %v:%v/healthz, got %d", threshold, expectSucceed, host, port, count)
	}
	return nil
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

// waitForAPIServerUp waits for the kube-apiserver to be up.
func waitForAPIServerUp(c clientset.Interface) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		body, err := c.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.TODO()).Raw()
		if err == nil && string(body) == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

// getEndpointNodesWithInternalIP returns a map of nodenames:internal-ip on which the
// endpoints of the Service are running.
func getEndpointNodesWithInternalIP(jig *e2eservice.TestJig) (map[string]string, error) {
	nodesWithIPs, err := jig.GetEndpointNodesWithIP(v1.NodeInternalIP)
	if err != nil {
		return nil, err
	}
	endpointsNodeMap := make(map[string]string, len(nodesWithIPs))
	for nodeName, internalIPs := range nodesWithIPs {
		if len(internalIPs) < 1 {
			return nil, fmt.Errorf("no internal ip found for node %s", nodeName)
		}
		endpointsNodeMap[nodeName] = internalIPs[0]
	}
	return endpointsNodeMap, nil
}

var _ = common.SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface
	serviceLBNames := []string{}

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.AfterEach(func() {
		if ginkgo.CurrentSpecReport().Failed() {
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
		Release: v1.9
		Testname: Kubernetes Service
		Description: By default when a kubernetes cluster is running there MUST be a 'kubernetes' service running in the cluster.
	*/
	framework.ConformanceIt("should provide secure master service ", func() {
		_, err := cs.CoreV1().Services(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the service object for the service named kubernetes")
	})

	/*
		Release: v1.9
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
		svc, err := jig.CreateTCPServiceWithPort(nil, 80)
		framework.ExpectNoError(err)

		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})

		names := map[string]bool{}
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		name1 := "pod1"
		name2 := "pod2"

		createPodOrFail(f, ns, name1, jig.Labels, []v1.ContainerPort{{ContainerPort: 80}}, "netexec", "--http-port", "80")
		names[name1] = true
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{name1: {80}})

		ginkgo.By("Checking if the Service forwards traffic to pod1")
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)

		createPodOrFail(f, ns, name2, jig.Labels, []v1.ContainerPort{{ContainerPort: 80}}, "netexec", "--http-port", "80")
		names[name2] = true
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{name1: {80}, name2: {80}})

		ginkgo.By("Checking if the Service forwards traffic to pod1 and pod2")
		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)

		e2epod.DeletePodOrFail(cs, ns, name1)
		delete(names, name1)
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{name2: {80}})

		ginkgo.By("Checking if the Service forwards traffic to pod2")
		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)

		e2epod.DeletePodOrFail(cs, ns, name2)
		delete(names, name2)
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})
	})

	/*
		Release: v1.9
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
		svc, err := jig.CreateTCPService(func(service *v1.Service) {
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
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})

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

		createPodOrFail(f, ns, podname1, jig.Labels, containerPorts1, "netexec", "--http-port", strconv.Itoa(port1))
		names[podname1] = true
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{podname1: {port1}})

		createPodOrFail(f, ns, podname2, jig.Labels, containerPorts2, "netexec", "--http-port", strconv.Itoa(port2))
		names[podname2] = true
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{podname1: {port1}, podname2: {port2}})

		ginkgo.By("Checking if the Service forwards traffic to pods")
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(svc, execPod)
		framework.ExpectNoError(err)

		e2epod.DeletePodOrFail(cs, ns, podname1)
		delete(names, podname1)
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{podname2: {port2}})

		e2epod.DeletePodOrFail(cs, ns, podname2)
		delete(names, podname2)
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})
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
		jig.ExternalIPs = true
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
		pod := e2epod.NewAgnhostPod(ns, serverPodName, nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort))
		pod.Labels = jig.Labels
		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
		defer func() {
			framework.Logf("Cleaning up the echo server pod")
			err := cs.CoreV1().Pods(ns).Delete(context.TODO(), serverPodName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete pod: %s on node", serverPodName)
		}()

		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{serverPodName: {servicePort}})

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
			sourceIP, execPodIP := execSourceIPTest(pausePod, serviceAddress)
			ginkgo.By("Verifying the preserved source ip")
			framework.ExpectEqual(sourceIP, execPodIP)
		}
	})

	ginkgo.It("should allow pods to hairpin back to themselves through services", func() {
		serviceName := "hairpin-test"
		ns := f.Namespace.Name

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP in namespace " + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		jig.ExternalIPs = true
		servicePort := 8080
		svc, err := jig.CreateTCPServiceWithPort(nil, int32(servicePort))
		framework.ExpectNoError(err)
		serviceIP := svc.Spec.ClusterIP
		framework.Logf("hairpin-test cluster ip: %s", serviceIP)

		ginkgo.By("creating a client/server pod")
		serverPodName := "hairpin"
		podTemplate := e2epod.NewAgnhostPod(ns, serverPodName, nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort))
		podTemplate.Labels = jig.Labels
		pod, err := cs.CoreV1().Pods(ns).Create(context.TODO(), podTemplate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))

		ginkgo.By("waiting for the service to expose an endpoint")
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{serverPodName: {servicePort}})

		ginkgo.By("Checking if the pod can reach itself")
		err = jig.CheckServiceReachability(svc, pod)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should be able to up and down services", func() {
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

		ginkgo.By("verifying service " + svc1 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))

		ginkgo.By("verifying service " + svc2 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))

		// Stop service 1 and make sure it is gone.
		ginkgo.By("stopping service " + svc1)
		framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc1))

		ginkgo.By("verifying service " + svc1 + " is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svc1IP, servicePort))
		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))

		// Start another service and verify both are up.
		ginkgo.By("creating service " + svc3 + " in namespace " + ns)
		podNames3, svc3IP, err := StartServeHostnameService(cs, getServeHostnameService(svc3), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc3, ns)

		if svc2IP == svc3IP {
			framework.Failf("service IPs conflict: %v", svc2IP)
		}

		ginkgo.By("verifying service " + svc2 + " is still up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))

		ginkgo.By("verifying service " + svc3 + " is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames3, svc3IP, servicePort))
	})

	ginkgo.It("should work after restarting kube-proxy [Disruptive]", func() {
		kubeProxyLabelSet := map[string]string{clusterAddonLabelKey: kubeProxyLabelName}
		e2eskipper.SkipUnlessComponentRunsAsPodsAndClientCanDeleteThem(kubeProxyLabelName, cs, metav1.NamespaceSystem, kubeProxyLabelSet)

		// TODO: use the ServiceTestJig here
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

		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))

		if err := restartComponent(cs, kubeProxyLabelName, metav1.NamespaceSystem, kubeProxyLabelSet); err != nil {
			framework.Failf("error restarting kube-proxy: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))
	})

	ginkgo.It("should work after restarting apiserver [Disruptive]", func() {

		if !framework.ProviderIs("gke") {
			e2eskipper.SkipUnlessComponentRunsAsPodsAndClientCanDeleteThem(kubeAPIServerLabelName, cs, metav1.NamespaceSystem, map[string]string{clusterComponentKey: kubeAPIServerLabelName})
		}

		// TODO: use the ServiceTestJig here
		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort

		svc1 := "restart-apiserver-1"
		svc2 := "restart-apiserver-2"

		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc1))
		}()
		podNames1, svc1IP, err := StartServeHostnameService(cs, getServeHostnameService(svc1), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc1, ns)

		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))

		// Restart apiserver
		ginkgo.By("Restarting apiserver")
		if err := restartApiserver(ns, cs); err != nil {
			framework.Failf("error restarting apiserver: %v", err)
		}
		ginkgo.By("Waiting for apiserver to come up by polling /healthz")
		if err := waitForAPIServerUp(cs); err != nil {
			framework.Failf("error while waiting for apiserver up: %v", err)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))

		// Create a new service and check if it's not reusing IP.
		defer func() {
			framework.ExpectNoError(StopServeHostnameService(f.ClientSet, ns, svc2))
		}()
		podNames2, svc2IP, err := StartServeHostnameService(cs, getServeHostnameService(svc2), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc2, ns)

		if svc1IP == svc2IP {
			framework.Failf("VIPs conflict: %v", svc1IP)
		}
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames1, svc1IP, servicePort))
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podNames2, svc2IP, servicePort))
	})

	/*
		Release: v1.16
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

	/*
		Create a ClusterIP service with an External IP that is not assigned to an interface.
		The IP ranges here are reserved for documentation according to
		[RFC 5737](https://tools.ietf.org/html/rfc5737) Section 3 and should not be used by any host.
	*/
	ginkgo.It("should be possible to connect to a service via ExternalIP when the external IP is not assigned to a node", func() {
		serviceName := "externalip-test"
		ns := f.Namespace.Name
		externalIP := "203.0.113.250"
		if framework.TestContext.ClusterIsIPv6() {
			externalIP = "2001:DB8::cb00:71fa"
		}

		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		jig.ExternalIPs = true

		ginkgo.By("creating service " + serviceName + " with type=clusterIP in namespace " + ns)
		clusterIPService, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ExternalIPs = []string{externalIP}
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
			}
		})
		if err != nil && strings.Contains(err.Error(), "Use of external IPs is denied by admission control") {
			e2eskipper.Skipf("Admission controller to deny services with external IPs is enabled - skip.")
		}
		framework.ExpectNoError(err)
		err = jig.CreateServicePods(2)
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(clusterIPService, execPod)
		framework.ExpectNoError(err)
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
		jig.ExternalIPs = true

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

		err = jig.CreateServicePods(2)
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
		})
		framework.ExpectNoError(err)
		if externalNameService.Spec.ClusterIP != "" {
			framework.Failf("Spec.ClusterIP was not cleared")
		}
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
		})
		framework.ExpectNoError(err)
		if externalNameService.Spec.ClusterIP != "" {
			framework.Failf("Spec.ClusterIP was not cleared")
		}
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
				ProbeHandler: v1.ProbeHandler{
					Exec: &v1.ExecAction{
						Command: []string{"/bin/false"},
					},
				},
			},
			Lifecycle: &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
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

	ginkgo.It("should be able to connect to terminating and unready endpoints if PublishNotReadyAddresses is true", func() {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-tolerate-unready"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a NodePort TCP service " + serviceName + " that PublishNotReadyAddresses on" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.PublishNotReadyAddresses = true
		})
		framework.ExpectNoError(err, "failed to create Service")

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		gracePeriod := int64(300)
		webserverPod0 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "webserver-pod",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "agnhost",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod)},
						Ports: []v1.ContainerPort{
							{
								ContainerPort: 80,
							},
						},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/readyz",
									Port: intstr.IntOrString{
										IntVal: int32(80),
									},
									Scheme: v1.URISchemeHTTP,
								},
							},
						},
						LivenessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/healthz",
									Port: intstr.IntOrString{
										IntVal: int32(80),
									},
									Scheme: v1.URISchemeHTTP,
								},
							},
						},
					},
				},
			},
		}
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(gracePeriod)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod")
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout)
		if err != nil {
			framework.Failf("error waiting for pod %s to be ready %v", webserverPod0.Name, err)
		}
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 1 pause pods that will try to connect to the webservers")
		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod")
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout)
		if err != nil {
			framework.Failf("error waiting for pod %s to be ready %v", pausePod1.Name, err)
		}

		// webserver should continue to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is unready but PublishNotReadyAddresses is true
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// Wait until the pod becomes unready
		err = e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, webserverPod0.Name, "pod not ready", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
			return !podutil.IsPodReady(pod), nil
		})
		if err != nil {
			framework.Failf("error waiting for pod %s to be unready %v", webserverPod0.Name, err)
		}
		// assert 5 times that the pause pod can connect to the Service
		nodeIPs0 := e2enode.GetAddresses(&node0, v1.NodeInternalIP)
		nodeIPs1 := e2enode.GetAddresses(&node1, v1.NodeInternalIP)
		clusterIPAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		nodePortAddress0 := net.JoinHostPort(nodeIPs0[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		nodePortAddress1 := net.JoinHostPort(nodeIPs1[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		// connect 3 times every 5 seconds to the Service with the unready and terminating endpoint
		for i := 0; i < 5; i++ {
			execHostnameTest(*pausePod1, clusterIPAddress, webserverPod0.Name)
			execHostnameTest(*pausePod1, nodePortAddress0, webserverPod0.Name)
			execHostnameTest(*pausePod1, nodePortAddress1, webserverPod0.Name)
			time.Sleep(5 * time.Second)
		}
	})

	ginkgo.It("should not be able to connect to terminating and unready endpoints if PublishNotReadyAddresses is false", func() {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-not-tolerate-unready"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a NodePort TCP service " + serviceName + " that PublishNotReadyAddresses on" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.PublishNotReadyAddresses = false
		})
		framework.ExpectNoError(err, "failed to create Service")

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		gracePeriod := int64(300)
		webserverPod0 := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "webserver-pod",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "agnhost",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod)},
						Ports: []v1.ContainerPort{
							{
								ContainerPort: 80,
							},
						},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/readyz",
									Port: intstr.IntOrString{
										IntVal: int32(80),
									},
									Scheme: v1.URISchemeHTTP,
								},
							},
						},
						LivenessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								HTTPGet: &v1.HTTPGetAction{
									Path: "/healthz",
									Port: intstr.IntOrString{
										IntVal: int32(80),
									},
									Scheme: v1.URISchemeHTTP,
								},
							},
						},
					},
				},
			},
		}
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(gracePeriod)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod")
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout)
		if err != nil {
			framework.Failf("error waiting for pod %s to be ready %v", webserverPod0.Name, err)
		}
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 1 pause pods that will try to connect to the webservers")
		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod")
		err = e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout)
		if err != nil {
			framework.Failf("error waiting for pod %s to be ready %v", pausePod1.Name, err)
		}

		// webserver should stop to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is unready but PublishNotReadyAddresses is false
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// Wait until the pod becomes unready
		err = e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, webserverPod0.Name, "pod not ready", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
			return !podutil.IsPodReady(pod), nil
		})
		if err != nil {
			framework.Failf("error waiting for pod %s to be unready %v", webserverPod0.Name, err)
		}
		// Wait the change has been propagated and the service start to fail
		clusterIPAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, clusterIPAddress)
		if pollErr := wait.PollImmediate(framework.Poll, e2eservice.KubeProxyEndpointLagTimeout, func() (bool, error) {
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			if err != nil {
				return true, nil
			}
			return false, nil
		}); pollErr != nil {
			framework.ExpectNoError(pollErr, "service still serves traffic")
		}

		nodeIPs0 := e2enode.GetAddresses(&node0, v1.NodeInternalIP)
		nodeIPs1 := e2enode.GetAddresses(&node1, v1.NodeInternalIP)
		nodePortAddress0 := net.JoinHostPort(nodeIPs0[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		nodePortAddress1 := net.JoinHostPort(nodeIPs1[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		// connect 3 times every 5 seconds to the Service and expect a failure
		for i := 0; i < 5; i++ {
			cmd = fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, clusterIPAddress)
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")

			cmd = fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, nodePortAddress0)
			_, err = framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to NodePort address")

			cmd = fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, nodePortAddress1)
			_, err = framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to NodePort address")

			time.Sleep(5 * time.Second)
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

	ginkgo.It("should implement service.kubernetes.io/service-proxy-name", func() {
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

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcDisabledIP, servicePort))

		ginkgo.By("adding service-proxy-name label")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = serviceProxyNameLabels
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcToggledIP, servicePort))

		ginkgo.By("removing service-proxy-name annotation")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podToggledNames, svcToggledIP, servicePort))

		ginkgo.By("verifying service-disabled is still not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcDisabledIP, servicePort))
	})

	ginkgo.It("should implement service.kubernetes.io/headless", func() {
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

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podHeadlessToggledNames, svcHeadlessToggledIP, servicePort))

		ginkgo.By("verifying service-headless is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcHeadlessIP, servicePort))

		ginkgo.By("adding service.kubernetes.io/headless label")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = serviceHeadlessLabels
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcHeadlessToggledIP, servicePort))

		ginkgo.By("removing service.kubernetes.io/headless annotation")
		_, err = jig.UpdateService(func(svc *v1.Service) {
			svc.ObjectMeta.Labels = nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("verifying service is up")
		framework.ExpectNoError(verifyServeHostnameServiceUp(cs, ns, podHeadlessToggledNames, svcHeadlessToggledIP, servicePort))

		ginkgo.By("verifying service-headless is still not up")
		framework.ExpectNoError(verifyServeHostnameServiceDown(cs, ns, svcHeadlessIP, servicePort))
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
			nodeSelection := e2epod.NodeSelection{Name: nodeName}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
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

	// regression test for https://issues.k8s.io/109414 and https://issues.k8s.io/109718
	ginkgo.It("should be rejected for evicted pods (no endpoints exist)", func() {
		namespace := f.Namespace.Name
		serviceName := "evicted-pods"
		jig := e2eservice.NewTestJig(cs, namespace, serviceName)
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, e2eservice.MaxNodesForEndpointsTests)
		framework.ExpectNoError(err)
		nodeName := nodes.Items[0].Name

		port := 80

		ginkgo.By("creating a service with no endpoints")
		_, err = jig.CreateTCPServiceWithPort(func(s *v1.Service) {
			// set publish not ready addresses to cover edge cases too
			s.Spec.PublishNotReadyAddresses = true
		}, int32(port))
		framework.ExpectNoError(err)

		// Create a pod in one node to get evicted
		ginkgo.By("creating a client pod that is going to be evicted for the service " + serviceName)
		evictedPod := e2epod.NewAgnhostPod(namespace, "evicted-pod", nil, nil, nil)
		evictedPod.Spec.Containers[0].Command = []string{"/bin/sh", "-c", "sleep 10; fallocate -l 10M file; sleep 10000"}
		evictedPod.Spec.Containers[0].Name = "evicted-pod"
		evictedPod.Spec.Containers[0].Resources = v1.ResourceRequirements{
			Limits: v1.ResourceList{"ephemeral-storage": resource.MustParse("5Mi")},
		}
		f.PodClient().Create(evictedPod)
		err = e2epod.WaitForPodTerminatedInNamespace(f.ClientSet, evictedPod.Name, "Evicted", f.Namespace.Name)
		if err != nil {
			framework.Failf("error waiting for pod to be evicted: %v", err)
		}

		podName := "execpod-evictedpods"
		ginkgo.By(fmt.Sprintf("creating %v on node %v", podName, nodeName))
		execPod := e2epod.CreateExecPodOrFail(f.ClientSet, namespace, podName, func(pod *v1.Pod) {
			nodeSelection := e2epod.NodeSelection{Name: nodeName}
			e2epod.SetNodeSelection(&pod.Spec, nodeSelection)
		})

		if epErr := wait.PollImmediate(framework.Poll, e2eservice.ServiceEndpointsTimeout, func() (bool, error) {
			endpoints, err := cs.CoreV1().Endpoints(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
			if err != nil {
				framework.Logf("error fetching '%s/%s' Endpoints: %s", namespace, serviceName, err.Error())
				return false, err
			}
			if len(endpoints.Subsets) > 0 {
				framework.Logf("expected '%s/%s' Endpoints to be empty, got: %v", namespace, serviceName, endpoints.Subsets)
				return false, nil
			}
			epsList, err := cs.DiscoveryV1().EndpointSlices(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, serviceName)})
			if err != nil {
				framework.Logf("error fetching '%s/%s' EndpointSlices: %s", namespace, serviceName, err.Error())
				return false, err
			}
			if len(epsList.Items) != 1 {
				framework.Logf("expected exactly 1 EndpointSlice, got: %d", len(epsList.Items))
				return false, nil
			}
			endpointSlice := epsList.Items[0]
			if len(endpointSlice.Endpoints) > 0 {
				framework.Logf("expected EndpointSlice to be empty, got %d endpoints", len(endpointSlice.Endpoints))
				return false, nil
			}
			return true, nil
		}); epErr != nil {
			framework.ExpectNoError(epErr)
		}

		serviceAddress := net.JoinHostPort(serviceName, strconv.Itoa(port))
		framework.Logf("waiting up to %v to connect to %v", e2eservice.KubeProxyEndpointLagTimeout, serviceAddress)
		cmd := fmt.Sprintf("/agnhost connect --timeout=3s %s", serviceAddress)

		ginkgo.By(fmt.Sprintf("hitting service %v from pod %v on node %v expected to be refused", serviceAddress, podName, nodeName))
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

	ginkgo.It("should respect internalTrafficPolicy=Local Pod to Pod [Feature:ServiceInternalTrafficPolicy]", func() {
		// windows kube-proxy does not support this feature yet
		// TODO: remove this skip when windows-based proxies implement internalTrafficPolicy
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-itp"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP and internalTrafficPolicy=Local in namespace " + ns)
		local := v1.ServiceInternalTrafficPolicyLocal
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.InternalTrafficPolicy = &local
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort))
		webserverPod0.Labels = jig.Labels
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		serviceAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		for i := 0; i < 5; i++ {
			// the first pause pod should be on the same node as the webserver, so it can connect to the local pod using clusterIP
			execHostnameTest(*pausePod0, serviceAddress, webserverPod0.Name)

			// the second pause pod is on a different node, so it should see a connection error every time
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, serviceAddress)
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")
		}
	})

	ginkgo.It("should respect internalTrafficPolicy=Local Pod (hostNetwork: true) to Pod [Feature:ServiceInternalTrafficPolicy]", func() {
		// windows kube-proxy does not support this feature yet
		// TODO: remove this skip when windows-based proxies implement internalTrafficPolicy
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-itp"
		ns := f.Namespace.Name
		servicePort := 8000

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP and internalTrafficPolicy=Local in namespace " + ns)
		local := v1.ServiceInternalTrafficPolicyLocal
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 8000, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(8000)},
			}
			svc.Spec.InternalTrafficPolicy = &local
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort))
		webserverPod0.Labels = jig.Labels
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		pausePod0.Spec.HostNetwork = true
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		pausePod1.Spec.HostNetwork = true
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		serviceAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		for i := 0; i < 5; i++ {
			// the first pause pod should be on the same node as the webserver, so it can connect to the local pod using clusterIP
			execHostnameTest(*pausePod0, serviceAddress, webserverPod0.Name)

			// the second pause pod is on a different node, so it should see a connection error every time
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, serviceAddress)
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")
		}
	})

	ginkgo.It("should respect internalTrafficPolicy=Local Pod and Node, to Pod (hostNetwork: true) [Feature:ServiceInternalTrafficPolicy]", func() {
		// windows kube-proxy does not support this feature yet
		// TODO: remove this skip when windows-based proxies implement internalTrafficPolicy
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-itp"
		ns := f.Namespace.Name
		servicePort := 80
		// If the pod can't bind to this port, it will fail to start, and it will fail the test,
		// because is using hostNetwork. Using a not common port will reduce this possibility.
		endpointPort := 10180

		ginkgo.By("creating a TCP service " + serviceName + " with type=ClusterIP and internalTrafficPolicy=Local in namespace " + ns)
		local := v1.ServiceInternalTrafficPolicyLocal
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(endpointPort)},
			}
			svc.Spec.InternalTrafficPolicy = &local
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(endpointPort), "--udp-port", strconv.Itoa(endpointPort))
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.HostNetwork = true
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {endpointPort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webserver")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		serviceAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		for i := 0; i < 5; i++ {
			// the first pause pod should be on the same node as the webserver, so it can connect to the local pod using clusterIP
			// note that the expected hostname is the node name because the backend pod is on host network
			execHostnameTest(*pausePod0, serviceAddress, node0.Name)

			// the second pause pod is on a different node, so it should see a connection error every time
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, serviceAddress)
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")
		}

		ginkgo.By("Creating 2 pause hostNetwork pods that will try to connect to the webserver")
		pausePod2 := e2epod.NewAgnhostPod(ns, "pause-pod-2", nil, nil, nil)
		pausePod2.Spec.HostNetwork = true
		e2epod.SetNodeSelection(&pausePod2.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod2, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod2.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod3 := e2epod.NewAgnhostPod(ns, "pause-pod-3", nil, nil, nil)
		pausePod3.Spec.HostNetwork = true
		e2epod.SetNodeSelection(&pausePod3.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod3, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod3, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod3.Name, f.Namespace.Name, framework.PodStartTimeout))

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		for i := 0; i < 5; i++ {
			// the first pause pod should be on the same node as the webserver, so it can connect to the local pod using clusterIP
			// note that the expected hostname is the node name because the backend pod is on host network
			execHostnameTest(*pausePod2, serviceAddress, node0.Name)

			// the second pause pod is on a different node, so it should see a connection error every time
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, serviceAddress)
			_, err := framework.RunHostCmd(pausePod3.Namespace, pausePod3.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")
		}
	})

	ginkgo.It("should fail health check node port if there are only terminating endpoints [Feature:ProxyTerminatingEndpoints]", func() {
		// windows kube-proxy does not support this feature yet
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]

		serviceName := "svc-proxy-terminating"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " where all pods are terminating" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort), "--delay-shutdown", "600")
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(600)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		nodeIPs := e2enode.GetAddresses(&node0, v1.NodeInternalIP)
		healthCheckNodePortAddr := net.JoinHostPort(nodeIPs[0], strconv.Itoa(int(svc.Spec.HealthCheckNodePort)))
		// validate that the health check node port from kube-proxy returns 200 when there are ready endpoints
		err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
			cmd := fmt.Sprintf(`curl -s -o /dev/null -w "%%{http_code}" --connect-timeout 5 http://%s/healthz`, healthCheckNodePortAddr)
			out, err := framework.RunHostCmd(pausePod0.Namespace, pausePod0.Name, cmd)
			if err != nil {
				return false, err
			}

			expectedOut := "200"
			if out != expectedOut {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err)

		// webserver should continue to serve traffic through the Service after deletion, even though the health check node port should return 503
		ginkgo.By("Terminating the webserver pod")
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// validate that the health check node port from kube-proxy returns 503 when there are no ready endpoints
		err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
			cmd := fmt.Sprintf(`curl -s -o /dev/null -w "%%{http_code}" --connect-timeout 5 http://%s/healthz`, healthCheckNodePortAddr)
			out, err := framework.RunHostCmd(pausePod0.Namespace, pausePod0.Name, cmd)
			if err != nil {
				return false, err
			}

			expectedOut := "503"
			if out != expectedOut {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err)

		// also verify that while health check node port indicates 0 endpoints and returns 503, the endpoint still serves traffic.
		nodePortAddress := net.JoinHostPort(nodeIPs[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		execHostnameTest(*pausePod0, nodePortAddress, webserverPod0.Name)
	})

	ginkgo.It("should fallback to terminating endpoints when there are no ready endpoints with internalTrafficPolicy=Cluster [Feature:ProxyTerminatingEndpoints]", func() {
		// windows kube-proxy does not support this feature yet
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-proxy-terminating"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " where all pods are terminating" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort), "--delay-shutdown", "600")
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(600)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// webserver should continue to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is the only ready endpoint
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// assert 5 times that both the local and remote pod can connect to the Service while all endpoints are terminating
		serviceAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		for i := 0; i < 5; i++ {
			// There's a Service with internalTrafficPolicy=Cluster,
			// with a single endpoint (which is terminating) called webserver0 running on node0.
			// pausePod0 and pausePod1 are on node0 and node1 respectively.
			// pausePod0 -> Service clusterIP succeeds because traffic policy is "Cluster"
			// pausePod1 -> Service clusterIP succeeds because traffic policy is "Cluster"
			execHostnameTest(*pausePod0, serviceAddress, webserverPod0.Name)
			execHostnameTest(*pausePod1, serviceAddress, webserverPod0.Name)

			time.Sleep(5 * time.Second)
		}
	})

	ginkgo.It("should fallback to local terminating endpoints when there are no ready endpoints with internalTrafficPolicy=Local [Feature:ProxyTerminatingEndpoints]", func() {
		// windows kube-proxy does not support this feature yet
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-proxy-terminating"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " where all pods are terminating" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		local := v1.ServiceInternalTrafficPolicyLocal
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.InternalTrafficPolicy = &local
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort), "--delay-shutdown", "600")
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(600)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// webserver should continue to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is the only ready endpoint
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		serviceAddress := net.JoinHostPort(svc.Spec.ClusterIP, strconv.Itoa(servicePort))
		for i := 0; i < 5; i++ {
			// There's a Service with internalTrafficPolicy=Local,
			// with a single endpoint (which is terminating) called webserver0 running on node0.
			// pausePod0 and pausePod1 are on node0 and node1 respectively.
			// pausePod0 -> Service clusterIP succeeds because webserver0 is running on node0 and traffic policy is "Local"
			// pausePod1 -> Service clusterIP fails because webserver0 is on a different node and traffic policy is "Local"
			execHostnameTest(*pausePod0, serviceAddress, webserverPod0.Name)

			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, serviceAddress)
			_, err := framework.RunHostCmd(pausePod1.Namespace, pausePod1.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to cluster IP")

			time.Sleep(5 * time.Second)
		}
	})

	ginkgo.It("should fallback to terminating endpoints when there are no ready endpoints with externallTrafficPolicy=Cluster [Feature:ProxyTerminatingEndpoints]", func() {
		// windows kube-proxy does not support this feature yet
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-proxy-terminating"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " where all pods are terminating" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort), "--delay-shutdown", "600")
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(600)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// webserver should continue to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is the only ready endpoint
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// assert 5 times that both the local and remote pod can connect to the Service NodePort while all endpoints are terminating
		nodeIPs := e2enode.GetAddresses(&node0, v1.NodeInternalIP)
		nodePortAddress := net.JoinHostPort(nodeIPs[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		for i := 0; i < 5; i++ {
			// There's a Service Type=NodePort with externalTrafficPolicy=Cluster,
			// with a single endpoint (which is terminating) called webserver0 running on node0.
			// pausePod0 and pausePod1 are on node0 and node1 respectively.
			// pausePod0 -> node0 node port succeeds because webserver0 is running on node0 and traffic policy is "Cluster"
			// pausePod1 -> node0 node port succeeds because webserver0 is running on node0 and traffic policy is "Cluster"
			execHostnameTest(*pausePod0, nodePortAddress, webserverPod0.Name)
			execHostnameTest(*pausePod1, nodePortAddress, webserverPod0.Name)

			time.Sleep(5 * time.Second)
		}
	})

	ginkgo.It("should fallback to local terminating endpoints when there are no ready endpoints with externalTrafficPolicy=Local [Feature:ProxyTerminatingEndpoints]", func() {
		// windows kube-proxy does not support this feature yet
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

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		nodeCounts := len(nodes.Items)
		if nodeCounts < 2 {
			e2eskipper.Skipf("The test requires at least two ready nodes on %s, but found %v", framework.TestContext.Provider, nodeCounts)
		}
		node0 := nodes.Items[0]
		node1 := nodes.Items[1]

		serviceName := "svc-proxy-terminating"
		ns := f.Namespace.Name
		servicePort := 80

		ginkgo.By("creating a TCP service " + serviceName + " where all pods are terminating" + ns)
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(80)},
			}
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 1 webserver pod to be part of the TCP service")
		webserverPod0 := e2epod.NewAgnhostPod(ns, "echo-hostname-0", nil, nil, nil, "netexec", "--http-port", strconv.Itoa(servicePort), "--delay-shutdown", "600")
		webserverPod0.Labels = jig.Labels
		webserverPod0.Spec.TerminationGracePeriodSeconds = utilpointer.Int64(600)
		e2epod.SetNodeSelection(&webserverPod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		_, err = cs.CoreV1().Pods(ns).Create(context.TODO(), webserverPod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, webserverPod0.Name, f.Namespace.Name, framework.PodStartTimeout))
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{webserverPod0.Name: {servicePort}})

		ginkgo.By("Creating 2 pause pods that will try to connect to the webservers")
		pausePod0 := e2epod.NewAgnhostPod(ns, "pause-pod-0", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod0.Spec, e2epod.NodeSelection{Name: node0.Name})

		pausePod0, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod0, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod0.Name, f.Namespace.Name, framework.PodStartTimeout))

		pausePod1 := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		e2epod.SetNodeSelection(&pausePod1.Spec, e2epod.NodeSelection{Name: node1.Name})

		pausePod1, err = cs.CoreV1().Pods(ns).Create(context.TODO(), pausePod1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(f.ClientSet, pausePod1.Name, f.Namespace.Name, framework.PodStartTimeout))

		// webserver should continue to serve traffic through the Service after delete since:
		//  - it has a 600s termination grace period
		//  - it is the only ready endpoint
		err = cs.CoreV1().Pods(ns).Delete(context.TODO(), webserverPod0.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		// assert 5 times that the first pause pod can connect to the Service locally and the second one errors with a timeout
		nodeIPs0 := e2enode.GetAddresses(&node0, v1.NodeInternalIP)
		nodeIPs1 := e2enode.GetAddresses(&node1, v1.NodeInternalIP)
		nodePortAddress0 := net.JoinHostPort(nodeIPs0[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		nodePortAddress1 := net.JoinHostPort(nodeIPs1[0], strconv.Itoa(int(svc.Spec.Ports[0].NodePort)))
		for i := 0; i < 5; i++ {
			// There's a Service Type=NodePort with externalTrafficPolicy=Local,
			// with a single endpoint (which is terminating) called webserver0 running on node0.
			// pausePod0 and pausePod1 are on node0 and node1 respectively.
			// pausePod0 -> node1 node port fails because it's "external" and there are no local endpoints
			// pausePod1 -> node0 node port succeeds because webserver0 is running on node0
			// pausePod0 -> node0 and pausePod1 -> node1 both succeed because pod-to-same-node-NodePort
			// connections are neither internal nor external and always get Cluster traffic policy.
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, nodePortAddress1)
			_, err := framework.RunHostCmd(pausePod0.Namespace, pausePod0.Name, cmd)
			framework.ExpectError(err, "expected error when trying to connect to node port for pausePod0")

			execHostnameTest(*pausePod0, nodePortAddress0, webserverPod0.Name)
			execHostnameTest(*pausePod1, nodePortAddress0, webserverPod0.Name)
			execHostnameTest(*pausePod1, nodePortAddress1, webserverPod0.Name)

			time.Sleep(5 * time.Second)
		}
	})

	/*
	   Release: v1.18
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

		if !foundSvc {
			framework.Fail("could not find service 'kubernetes' in service list in all namespaces")
		}
	})

	/*
	   Release: v1.19
	   Testname: Endpoint resource lifecycle
	   Description: Create an endpoint, the endpoint MUST exist.
	   The endpoint is updated with a new label, a check after the update MUST find the changes.
	   The endpoint is then patched with a new IPv4 address and port, a check after the patch MUST the changes.
	   The endpoint is deleted by it's label, a watch listens for the deleted watch event.
	*/
	framework.ConformanceIt("should test the lifecycle of an Endpoint", func() {
		testNamespaceName := f.Namespace.Name
		testEndpointName := "testservice"
		testEndpoints := v1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name: testEndpointName,
				Labels: map[string]string{
					"test-endpoint-static": "true",
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
		}
		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = "test-endpoint-static=true"
				return f.ClientSet.CoreV1().Endpoints(testNamespaceName).Watch(context.TODO(), options)
			},
		}
		endpointsList, err := f.ClientSet.CoreV1().Endpoints("").List(context.TODO(), metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to list Endpoints")

		ginkgo.By("creating an Endpoint")
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Create(context.TODO(), &testEndpoints, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Endpoint")
		ginkgo.By("waiting for available Endpoint")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, endpointsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Added:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Added)

		ginkgo.By("listing all Endpoints")
		endpointsList, err = f.ClientSet.CoreV1().Endpoints("").List(context.TODO(), metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to list Endpoints")
		eventFound := false
		var foundEndpoint v1.Endpoints
		for _, endpoint := range endpointsList.Items {
			if endpoint.ObjectMeta.Name == testEndpointName && endpoint.ObjectMeta.Namespace == testNamespaceName {
				eventFound = true
				foundEndpoint = endpoint
				break
			}
		}
		if !eventFound {
			framework.Fail("unable to find Endpoint Service in list of Endpoints")
		}

		ginkgo.By("updating the Endpoint")
		foundEndpoint.ObjectMeta.Labels["test-service"] = "updated"
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Update(context.TODO(), &foundEndpoint, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update Endpoint with new label")

		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, endpointsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the Endpoint")
		endpoints, err := f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(context.TODO(), testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		framework.ExpectEqual(foundEndpoint.ObjectMeta.Labels["test-service"], "updated", "failed to update Endpoint %v in namespace %v label not updated", testEndpointName, testNamespaceName)

		endpointPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{
					"test-service": "patched",
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
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Patch(context.TODO(), testEndpointName, types.StrategicMergePatchType, []byte(endpointPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Endpoint")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, endpoints.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Modified)

		ginkgo.By("fetching the Endpoint")
		endpoints, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(context.TODO(), testEndpointName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch Endpoint")
		framework.ExpectEqual(endpoints.ObjectMeta.Labels["test-service"], "patched", "failed to patch Endpoint with Label")
		endpointSubsetOne := endpoints.Subsets[0]
		endpointSubsetOneAddresses := endpointSubsetOne.Addresses[0]
		endpointSubsetOnePorts := endpointSubsetOne.Ports[0]
		framework.ExpectEqual(endpointSubsetOneAddresses.IP, "10.0.0.25", "failed to patch Endpoint")
		framework.ExpectEqual(endpointSubsetOnePorts.Name, "http-test", "failed to patch Endpoint")
		framework.ExpectEqual(endpointSubsetOnePorts.Port, int32(8080), "failed to patch Endpoint")

		ginkgo.By("deleting the Endpoint by Collection")
		err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test-endpoint-static=true"})
		framework.ExpectNoError(err, "failed to delete Endpoint by Collection")

		ginkgo.By("waiting for Endpoint deletion")
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, err = watchtools.Until(ctx, endpoints.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				if endpoints, ok := event.Object.(*v1.Endpoints); ok {
					found := endpoints.ObjectMeta.Name == endpoints.Name &&
						endpoints.Labels["test-endpoint-static"] == "true"
					return found, nil
				}
			default:
				framework.Logf("observed event type %v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see %v event", watch.Deleted)

		ginkgo.By("fetching the Endpoint")
		_, err = f.ClientSet.CoreV1().Endpoints(testNamespaceName).Get(context.TODO(), testEndpointName, metav1.GetOptions{})
		framework.ExpectError(err, "should not be able to fetch Endpoint")
	})

	/*
		Release: v1.21
		Testname: Service, complete ServiceStatus lifecycle
		Description: Create a service, the service MUST exist.
		When retrieving /status the action MUST be validated.
		When patching /status the action MUST be validated.
		When updating /status the action MUST be validated.
		When patching a service the action MUST be validated.
	*/
	framework.ConformanceIt("should complete a service status lifecycle", func() {

		ns := f.Namespace.Name
		svcResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "services"}
		svcClient := f.ClientSet.CoreV1().Services(ns)

		testSvcName := "test-service-" + utilrand.String(5)
		testSvcLabels := map[string]string{"test-service-static": "true"}
		testSvcLabelsFlat := "test-service-static=true"

		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = testSvcLabelsFlat
				return cs.CoreV1().Services(ns).Watch(context.TODO(), options)
			},
		}

		svcList, err := cs.CoreV1().Services("").List(context.TODO(), metav1.ListOptions{LabelSelector: testSvcLabelsFlat})
		framework.ExpectNoError(err, "failed to list Services")

		ginkgo.By("creating a Service")
		testService := v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:   testSvcName,
				Labels: testSvcLabels,
			},
			Spec: v1.ServiceSpec{
				Type: "ClusterIP",
				Ports: []v1.ServicePort{{
					Name:       "http",
					Protocol:   v1.ProtocolTCP,
					Port:       int32(80),
					TargetPort: intstr.FromInt(80),
				}},
			},
		}
		_, err = cs.CoreV1().Services(ns).Create(context.TODO(), &testService, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Service")

		ginkgo.By("watching for the Service to be added")
		ctx, cancel := context.WithTimeout(context.Background(), svcReadyTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if svc, ok := event.Object.(*v1.Service); ok {
				found := svc.ObjectMeta.Name == testService.ObjectMeta.Name &&
					svc.ObjectMeta.Namespace == ns &&
					svc.Labels["test-service-static"] == "true"
				if !found {
					framework.Logf("observed Service %v in namespace %v with labels: %v & ports %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Spec.Ports)
					return false, nil
				}
				framework.Logf("Found Service %v in namespace %v with labels: %v & ports %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Spec.Ports)
				return found, nil
			}
			framework.Logf("Observed event: %+v", event.Object)
			return false, nil
		})
		framework.ExpectNoError(err, "Failed to locate Service %v in namespace %v", testService.ObjectMeta.Name, ns)
		framework.Logf("Service %s created", testSvcName)

		ginkgo.By("Getting /status")
		svcStatusUnstructured, err := f.DynamicClient.Resource(svcResource).Namespace(ns).Get(context.TODO(), testSvcName, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "Failed to fetch ServiceStatus of Service %s in namespace %s", testSvcName, ns)
		svcStatusBytes, err := json.Marshal(svcStatusUnstructured)
		framework.ExpectNoError(err, "Failed to marshal unstructured response. %v", err)

		var svcStatus v1.Service
		err = json.Unmarshal(svcStatusBytes, &svcStatus)
		framework.ExpectNoError(err, "Failed to unmarshal JSON bytes to a Service object type")
		framework.Logf("Service %s has LoadBalancer: %v", testSvcName, svcStatus.Status.LoadBalancer)

		ginkgo.By("patching the ServiceStatus")
		lbStatus := v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{{IP: "203.0.113.1"}},
		}
		lbStatusJSON, err := json.Marshal(lbStatus)
		framework.ExpectNoError(err, "Failed to marshal JSON. %v", err)
		_, err = svcClient.Patch(context.TODO(), testSvcName, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"loadBalancer":`+string(lbStatusJSON)+`}}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err, "Could not patch service status", err)

		ginkgo.By("watching for the Service to be patched")
		ctx, cancel = context.WithTimeout(context.Background(), svcReadyTimeout)
		defer cancel()

		_, err = watchtools.Until(ctx, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if svc, ok := event.Object.(*v1.Service); ok {
				found := svc.ObjectMeta.Name == testService.ObjectMeta.Name &&
					svc.ObjectMeta.Namespace == ns &&
					svc.Annotations["patchedstatus"] == "true"
				if !found {
					framework.Logf("observed Service %v in namespace %v with annotations: %v & LoadBalancer: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
					return false, nil
				}
				framework.Logf("Found Service %v in namespace %v with annotations: %v & LoadBalancer: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
				return found, nil
			}
			framework.Logf("Observed event: %+v", event.Object)
			return false, nil
		})
		framework.ExpectNoError(err, "failed to locate Service %v in namespace %v", testService.ObjectMeta.Name, ns)
		framework.Logf("Service %s has service status patched", testSvcName)

		ginkgo.By("updating the ServiceStatus")

		var statusToUpdate, updatedStatus *v1.Service
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = svcClient.Get(context.TODO(), testSvcName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to retrieve service %s", testSvcName)

			statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, metav1.Condition{
				Type:    "StatusUpdate",
				Status:  metav1.ConditionTrue,
				Reason:  "E2E",
				Message: "Set from e2e test",
			})

			updatedStatus, err = svcClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "\n\n Failed to UpdateStatus. %v\n\n", err)
		framework.Logf("updatedStatus.Conditions: %#v", updatedStatus.Status.Conditions)

		ginkgo.By("watching for the Service to be updated")
		ctx, cancel = context.WithTimeout(context.Background(), svcReadyTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if svc, ok := event.Object.(*v1.Service); ok {
				found := svc.ObjectMeta.Name == testService.ObjectMeta.Name &&
					svc.ObjectMeta.Namespace == ns &&
					svc.Annotations["patchedstatus"] == "true"
				if !found {
					framework.Logf("Observed Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
					return false, nil
				}
				for _, cond := range svc.Status.Conditions {
					if cond.Type == "StatusUpdate" &&
						cond.Reason == "E2E" &&
						cond.Message == "Set from e2e test" {
						framework.Logf("Found Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.Conditions)
						return found, nil
					} else {
						framework.Logf("Observed Service %v in namespace %v with annotations: %v & Conditions: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Annotations, svc.Status.LoadBalancer)
						return false, nil
					}
				}
			}
			framework.Logf("Observed event: %+v", event.Object)
			return false, nil
		})
		framework.ExpectNoError(err, "failed to locate Service %v in namespace %v", testService.ObjectMeta.Name, ns)
		framework.Logf("Service %s has service status updated", testSvcName)

		ginkgo.By("patching the service")
		servicePatchPayload, err := json.Marshal(v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"test-service": "patched",
				},
			},
		})

		_, err = svcClient.Patch(context.TODO(), testSvcName, types.StrategicMergePatchType, []byte(servicePatchPayload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch service. %v", err)

		ginkgo.By("watching for the Service to be patched")
		ctx, cancel = context.WithTimeout(context.Background(), svcReadyTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if svc, ok := event.Object.(*v1.Service); ok {
				found := svc.ObjectMeta.Name == testService.ObjectMeta.Name &&
					svc.ObjectMeta.Namespace == ns &&
					svc.Labels["test-service"] == "patched"
				if !found {
					framework.Logf("observed Service %v in namespace %v with labels: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels)
					return false, nil
				}
				framework.Logf("Found Service %v in namespace %v with labels: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels)
				return found, nil
			}
			framework.Logf("Observed event: %+v", event.Object)
			return false, nil
		})
		framework.ExpectNoError(err, "failed to locate Service %v in namespace %v", testService.ObjectMeta.Name, ns)
		framework.Logf("Service %s patched", testSvcName)

		ginkgo.By("deleting the service")
		err = cs.CoreV1().Services(ns).Delete(context.TODO(), testSvcName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the Service. %v", err)

		ginkgo.By("watching for the Service to be deleted")
		ctx, cancel = context.WithTimeout(context.Background(), svcReadyTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, svcList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				if svc, ok := event.Object.(*v1.Service); ok {
					found := svc.ObjectMeta.Name == testService.ObjectMeta.Name &&
						svc.ObjectMeta.Namespace == ns &&
						svc.Labels["test-service-static"] == "true"
					if !found {
						framework.Logf("observed Service %v in namespace %v with labels: %v & annotations: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Annotations)
						return false, nil
					}
					framework.Logf("Found Service %v in namespace %v with labels: %v & annotations: %v", svc.ObjectMeta.Name, svc.ObjectMeta.Namespace, svc.Labels, svc.Annotations)
					return found, nil
				}
			default:
				framework.Logf("Observed event: %+v", event.Type)
			}
			return false, nil
		})
		framework.ExpectNoError(err, "failed to delete Service %v in namespace %v", testService.ObjectMeta.Name, ns)
		framework.Logf("Service %s deleted", testSvcName)
	})

	/*
		Release: v1.23
		Testname: Service, deletes a collection of services
		Description: Create three services with the required
		labels and ports. It MUST locate three services in the
		test namespace. It MUST succeed at deleting a collection
		of services via a label selector. It MUST locate only
		one service after deleting the service collection.
	*/
	framework.ConformanceIt("should delete a collection of services", func() {

		ns := f.Namespace.Name
		svcClient := f.ClientSet.CoreV1().Services(ns)
		svcResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "services"}
		svcDynamicClient := f.DynamicClient.Resource(svcResource).Namespace(ns)

		svcLabel := map[string]string{"e2e-test-service": "delete"}
		deleteLabel := labels.SelectorFromSet(svcLabel).String()

		ginkgo.By("creating a collection of services")

		testServices := []struct {
			name  string
			label map[string]string
			port  int
		}{
			{
				name:  "e2e-svc-a-" + utilrand.String(5),
				label: map[string]string{"e2e-test-service": "delete"},
				port:  8001,
			},
			{
				name:  "e2e-svc-b-" + utilrand.String(5),
				label: map[string]string{"e2e-test-service": "delete"},
				port:  8002,
			},
			{
				name:  "e2e-svc-c-" + utilrand.String(5),
				label: map[string]string{"e2e-test-service": "keep"},
				port:  8003,
			},
		}

		for _, testService := range testServices {
			func() {
				framework.Logf("Creating %s", testService.name)

				svc := v1.Service{
					ObjectMeta: metav1.ObjectMeta{
						Name:   testService.name,
						Labels: testService.label,
					},
					Spec: v1.ServiceSpec{
						Type: "ClusterIP",
						Ports: []v1.ServicePort{{
							Name:       "http",
							Protocol:   v1.ProtocolTCP,
							Port:       int32(testService.port),
							TargetPort: intstr.FromInt(testService.port),
						}},
					},
				}
				_, err := svcClient.Create(context.TODO(), &svc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "failed to create Service")

			}()
		}

		svcList, err := cs.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list Services")
		framework.ExpectEqual(len(svcList.Items), 3, "Required count of services out of sync")

		ginkgo.By("deleting service collection")
		err = svcDynamicClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: deleteLabel})
		framework.ExpectNoError(err, "failed to delete service collection. %v", err)

		svcList, err = cs.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list Services")
		framework.ExpectEqual(len(svcList.Items), 1, "Required count of services out of sync")

		framework.Logf("Collection of services has been deleted")
	})

})

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
		// The node addresses must have the same IP family as the ClusterIP
		family := v1.IPv4Protocol
		if netutils.IsIPv6String(svc.Spec.ClusterIP) {
			family = v1.IPv6Protocol
		}
		svcIP = e2enode.FirstAddressByTypeAndFamily(nodes, v1.NodeInternalIP, family)
		framework.ExpectNotEqual(svcIP, "", "failed to get Node internal IP for family: %s", family)
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
			// In some case, ipvs didn't deleted the persistent connection after timeout expired,
			// use 'ipvsadm -lnc' command can found the expire time become '13171233:02' after '00:00'
			//
			// pro expire state       source             virtual            destination
			// TCP 00:00  NONE        10.105.253.160:0   10.105.253.160:80  10.244.1.25:9376
			//
			// pro expire state       source             virtual            destination
			// TCP 13171233:02 NONE        10.105.253.160:0   10.105.253.160:80  10.244.1.25:9376
			//
			// And 2 seconds later, the connection will be ensure deleted,
			// so we sleep 'svcSessionAffinityTimeout+5' seconds to avoid this issue.
			// TODO: figure out why the expired connection didn't be deleted and fix this issue in ipvs side.
			time.Sleep(time.Duration(svcSessionAffinityTimeout+5) * time.Second)
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
		// The node addresses must have the same IP family as the ClusterIP
		family := v1.IPv4Protocol
		if netutils.IsIPv6String(svc.Spec.ClusterIP) {
			family = v1.IPv6Protocol
		}
		svcIP = e2enode.FirstAddressByTypeAndFamily(nodes, v1.NodeInternalIP, family)
		framework.ExpectNotEqual(svcIP, "", "failed to get Node internal IP for family: %s", family)
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

	pauseDeployment.Spec.Template.Spec.Containers[0] = e2epod.NewAgnhostContainer("agnhost-pause", nil, nil, "pause")
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
func createPodOrFail(f *framework.Framework, ns, name string, labels map[string]string, containerPorts []v1.ContainerPort, args ...string) {
	ginkgo.By(fmt.Sprintf("Creating pod %s in namespace %s", name, ns))
	pod := e2epod.NewAgnhostPod(ns, name, nil, nil, containerPorts, args...)
	pod.ObjectMeta.Labels = labels
	// Add a dummy environment variable to work around a docker issue.
	// https://github.com/docker/docker/issues/14203
	pod.Spec.Containers[0].Env = []v1.EnvVar{{Name: "FOO", Value: " "}}
	f.PodClient().CreateSync(pod)
}

// launchHostExecPod launches a hostexec pod in the given namespace and waits
// until it's Running
func launchHostExecPod(client clientset.Interface, ns, name string) *v1.Pod {
	framework.Logf("Creating new host exec pod")
	hostExecPod := e2epod.NewExecPodSpec(ns, name, true)
	pod, err := client.CoreV1().Pods(ns).Create(context.TODO(), hostExecPod, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2epod.WaitTimeoutForPodReadyInNamespace(client, name, ns, framework.PodStartTimeout)
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
	pod := e2epod.NewAgnhostPod(f.Namespace.Name, "kube-proxy-mode-detector", nil, nil, nil)
	pod.Spec.HostNetwork = true
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

func validatePorts(ep, expectedEndpoints portsByPodUID) error {
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

func translatePodNameToUID(c clientset.Interface, ns string, expectedEndpoints portsByPodName) (portsByPodUID, error) {
	portsByUID := make(portsByPodUID)
	for name, portList := range expectedEndpoints {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get pod %s, that's pretty weird. validation failed: %s", name, err)
		}
		portsByUID[pod.ObjectMeta.UID] = portList
	}
	return portsByUID, nil
}

// validateEndpointsPortsOrFail validates that the given service exists and is served by the given expectedEndpoints.
func validateEndpointsPortsOrFail(c clientset.Interface, namespace, serviceName string, expectedEndpoints portsByPodName) {
	ginkgo.By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to expose endpoints %v", framework.ServiceStartTimeout, serviceName, namespace, expectedEndpoints))
	expectedPortsByPodUID, err := translatePodNameToUID(c, namespace, expectedEndpoints)
	framework.ExpectNoError(err, "failed to translate pod name to UID, ns:%s, expectedEndpoints:%v", namespace, expectedEndpoints)

	var (
		pollErr error
		i       = 0
	)
	if pollErr = wait.PollImmediate(time.Second, framework.ServiceStartTimeout, func() (bool, error) {
		i++

		ep, err := c.CoreV1().Endpoints(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed go get Endpoints object: %v", err)
			// Retry the error
			return false, nil
		}
		portsByUID := portsByPodUID(e2eendpoints.GetContainerPortsByPodUID(ep))
		if err := validatePorts(portsByUID, expectedPortsByPodUID); err != nil {
			if i%5 == 0 {
				framework.Logf("Unexpected endpoints: found %v, expected %v, will retry", portsByUID, expectedEndpoints)
			}
			return false, nil
		}

		// If EndpointSlice API is enabled, then validate if appropriate EndpointSlice objects
		// were also create/updated/deleted.
		if _, err := c.Discovery().ServerResourcesForGroupVersion(discoveryv1.SchemeGroupVersion.String()); err == nil {
			opts := metav1.ListOptions{
				LabelSelector: "kubernetes.io/service-name=" + serviceName,
			}
			es, err := c.DiscoveryV1().EndpointSlices(namespace).List(context.TODO(), opts)
			if err != nil {
				framework.Logf("Failed go list EndpointSlice objects: %v", err)
				// Retry the error
				return false, nil
			}
			portsByUID = portsByPodUID(e2eendpointslice.GetContainerPortsByPodUID(es.Items))
			if err := validatePorts(portsByUID, expectedPortsByPodUID); err != nil {
				if i%5 == 0 {
					framework.Logf("Unexpected endpoint slices: found %v, expected %v, will retry", portsByUID, expectedEndpoints)
				}
				return false, nil
			}
		}
		framework.Logf("successfully validated that service %s in namespace %s exposes endpoints %v",
			serviceName, namespace, expectedEndpoints)
		return true, nil
	}); pollErr != nil {
		if pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{}); err == nil {
			for _, pod := range pods.Items {
				framework.Logf("Pod %s\t%s\t%s\t%s", pod.Namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
			}
		} else {
			framework.Logf("Can't list pod debug info: %v", err)
		}
	}
	framework.ExpectNoError(pollErr, "error waithing for service %s in namespace %s to expose endpoints %v: %v", serviceName, namespace, expectedEndpoints)
}

func restartApiserver(namespace string, cs clientset.Interface) error {
	if framework.ProviderIs("gke") {
		// GKE use a same-version master upgrade to teardown/recreate master.
		v, err := cs.Discovery().ServerVersion()
		if err != nil {
			return err
		}
		return framework.MasterUpgradeGKE(namespace, v.GitVersion[1:]) // strip leading 'v'
	}

	return restartComponent(cs, kubeAPIServerLabelName, metav1.NamespaceSystem, map[string]string{clusterComponentKey: kubeAPIServerLabelName})
}

// restartComponent restarts component static pod
func restartComponent(cs clientset.Interface, cName, ns string, matchLabels map[string]string) error {
	pods, err := e2epod.GetPods(cs, ns, matchLabels)
	if err != nil {
		return fmt.Errorf("failed to get %s's pods, err: %v", cName, err)
	}
	if len(pods) == 0 {
		return fmt.Errorf("%s pod count is 0", cName)
	}

	if err := e2epod.DeletePodsWithGracePeriod(cs, pods, 0); err != nil {
		return fmt.Errorf("failed to restart component: %s, err: %v", cName, err)
	}

	_, err = e2epod.PodsCreatedByLabel(cs, ns, cName, int32(len(pods)), labels.SelectorFromSet(matchLabels))
	return err
}

var _ = common.SIGDescribe("SCTP [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("sctp")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.It("should allow creating a basic SCTP service with pod and endpoints", func() {
		serviceName := "sctp-endpoint-test"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("getting the state of the sctp module on nodes")
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		sctpLoadedAtStart := CheckSCTPModuleLoadedOnNodes(f, nodes)

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		_, err = jig.CreateSCTPServiceWithPort(nil, 5060)
		framework.ExpectNoError(err)
		defer func() {
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()

		err = e2enetwork.WaitForService(f.ClientSet, ns, serviceName, true, 5*time.Second, e2eservice.TestTimeout)
		framework.ExpectNoError(err, fmt.Sprintf("error while waiting for service:%s err: %v", serviceName, err))

		ginkgo.By("validating endpoints do not exist yet")
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})

		ginkgo.By("creating a pod for the service")
		names := map[string]bool{}

		name1 := "pod1"

		createPodOrFail(f, ns, name1, jig.Labels, []v1.ContainerPort{{ContainerPort: 5060, Protocol: v1.ProtocolSCTP}})
		names[name1] = true
		defer func() {
			for name := range names {
				err := cs.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", name, ns)
			}
		}()

		ginkgo.By("validating endpoints exists")
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{name1: {5060}})

		ginkgo.By("deleting the pod")
		e2epod.DeletePodOrFail(cs, ns, name1)
		delete(names, name1)
		ginkgo.By("validating endpoints do not exist anymore")
		validateEndpointsPortsOrFail(cs, ns, serviceName, portsByPodName{})

		ginkgo.By("validating sctp module is still not loaded")
		sctpLoadedAtEnd := CheckSCTPModuleLoadedOnNodes(f, nodes)
		if !sctpLoadedAtStart && sctpLoadedAtEnd {
			framework.Failf("The state of the sctp module has changed due to the test case")
		}
	})

	ginkgo.It("should create a Pod with SCTP HostPort", func() {
		node, err := e2enode.GetRandomReadySchedulableNode(cs)
		framework.ExpectNoError(err)
		hostExec := utils.NewHostExec(f)
		defer hostExec.Cleanup()

		ginkgo.By("getting the state of the sctp module on the selected node")
		nodes := &v1.NodeList{}
		nodes.Items = append(nodes.Items, *node)
		sctpLoadedAtStart := CheckSCTPModuleLoadedOnNodes(f, nodes)

		ginkgo.By("creating a pod with hostport on the selected node")
		podName := "hostport"
		ports := []v1.ContainerPort{{Protocol: v1.ProtocolSCTP, ContainerPort: 5060, HostPort: 5060}}
		podSpec := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, ports)
		nodeSelection := e2epod.NodeSelection{Name: node.Name}
		e2epod.SetNodeSelection(&podSpec.Spec, nodeSelection)

		ginkgo.By(fmt.Sprintf("Launching the pod on node %v", node.Name))
		f.PodClient().CreateSync(podSpec)
		defer func() {
			err := cs.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete pod: %s in namespace: %s", podName, f.Namespace.Name)
		}()
		// wait until host port manager syncs rules
		cmd := "iptables-save"
		if framework.TestContext.ClusterIsIPv6() {
			cmd = "ip6tables-save"
		}
		err = wait.PollImmediate(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
			framework.Logf("Executing cmd %q on node %v", cmd, node.Name)
			result, err := hostExec.IssueCommandWithResult(cmd, node)
			if err != nil {
				framework.Logf("Interrogation of iptables rules failed on node %v", node.Name)
				return false, nil
			}

			for _, line := range strings.Split(result, "\n") {
				if strings.Contains(line, "-p sctp") && strings.Contains(line, "--dport 5060") {
					return true, nil
				}
			}
			framework.Logf("retrying ... not hostport sctp iptables rules found on node %v", node.Name)
			return false, nil
		})
		if err != nil {
			framework.Failf("iptables rules are not set for a pod with sctp hostport")
		}
		ginkgo.By("validating sctp module is still not loaded")
		sctpLoadedAtEnd := CheckSCTPModuleLoadedOnNodes(f, nodes)
		if !sctpLoadedAtStart && sctpLoadedAtEnd {
			framework.Failf("The state of the sctp module has changed due to the test case")
		}
	})
	ginkgo.It("should create a ClusterIP Service with SCTP ports", func() {
		ginkgo.By("checking that kube-proxy is in iptables mode")
		if proxyMode, err := proxyMode(f); err != nil {
			e2eskipper.Skipf("Couldn't detect KubeProxy mode - skip, %v", err)
		} else if proxyMode != "iptables" {
			e2eskipper.Skipf("The test doesn't work if kube-proxy is not in iptables mode")
		}

		serviceName := "sctp-clusterip"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("getting the state of the sctp module on nodes")
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		sctpLoadedAtStart := CheckSCTPModuleLoadedOnNodes(f, nodes)

		ginkgo.By("creating service " + serviceName + " in namespace " + ns)
		_, err = jig.CreateSCTPServiceWithPort(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolSCTP, Port: 5060}}
		}, 5060)
		framework.ExpectNoError(err)
		defer func() {
			err := cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		}()

		err = e2enetwork.WaitForService(f.ClientSet, ns, serviceName, true, 5*time.Second, e2eservice.TestTimeout)
		framework.ExpectNoError(err, fmt.Sprintf("error while waiting for service:%s err: %v", serviceName, err))
		hostExec := utils.NewHostExec(f)
		defer hostExec.Cleanup()
		node := &nodes.Items[0]
		cmd := "iptables-save"
		if framework.TestContext.ClusterIsIPv6() {
			cmd = "ip6tables-save"
		}
		err = wait.PollImmediate(framework.Poll, e2eservice.KubeProxyLagTimeout, func() (bool, error) {
			framework.Logf("Executing cmd %q on node %v", cmd, node.Name)
			result, err := hostExec.IssueCommandWithResult(cmd, node)
			if err != nil {
				framework.Logf("Interrogation of iptables rules failed on node %v", node.Name)
				return false, nil
			}

			for _, line := range strings.Split(result, "\n") {
				if strings.Contains(line, "-A KUBE-SERVICES") && strings.Contains(line, "-p sctp") {
					return true, nil
				}
			}
			framework.Logf("retrying ... no iptables rules found for service with sctp ports on node %v", node.Name)
			return false, nil
		})
		if err != nil {
			framework.Failf("iptables rules are not set for a clusterip service with sctp ports")
		}
		ginkgo.By("validating sctp module is still not loaded")
		sctpLoadedAtEnd := CheckSCTPModuleLoadedOnNodes(f, nodes)
		if !sctpLoadedAtStart && sctpLoadedAtEnd {
			framework.Failf("The state of the sctp module has changed due to the test case")
		}
	})
})
