/*
Copyright 2014 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Proxy", func() {
	version := testapi.Default.GroupVersion().Version
	Context("version "+version, func() { proxyContext(version) })
})

const (
	// Try all the proxy tests this many times (to catch even rare flakes).
	proxyAttempts = 20
	// Only print this many characters of the response (to keep the logs
	// legible).
	maxDisplayBodyLen = 100

	// We have seen one of these calls take just over 15 seconds, so putting this at 30.
	proxyHTTPCallTimeout = 30 * time.Second
)

func proxyContext(version string) {
	options := framework.FrameworkOptions{
		ClientQPS: -1.0,
	}
	f := framework.NewFramework("proxy", options, nil)
	prefix := "/api/" + version

	// Port here has to be kept in sync with default kubelet port.
	It("should proxy logs on node with explicit kubelet port [Conformance]", func() { nodeProxyTest(f, prefix+"/proxy/nodes/", ":10250/logs/") })
	It("should proxy logs on node [Conformance]", func() { nodeProxyTest(f, prefix+"/proxy/nodes/", "/logs/") })
	It("should proxy to cadvisor [Conformance]", func() { nodeProxyTest(f, prefix+"/proxy/nodes/", ":4194/containers/") })

	It("should proxy logs on node with explicit kubelet port using proxy subresource [Conformance]", func() { nodeProxyTest(f, prefix+"/nodes/", ":10250/proxy/logs/") })
	It("should proxy logs on node using proxy subresource [Conformance]", func() { nodeProxyTest(f, prefix+"/nodes/", "/proxy/logs/") })
	It("should proxy to cadvisor using proxy subresource [Conformance]", func() { nodeProxyTest(f, prefix+"/nodes/", ":4194/proxy/containers/") })

	// using the porter image to serve content, access the content
	// (of multiple pods?) from multiple (endpoints/services?)
	It("should proxy through a service and a pod [Conformance]", func() {
		start := time.Now()
		labels := map[string]string{"proxy-service-target": "true"}
		service, err := f.Client.Services(f.Namespace.Name).Create(&api.Service{
			ObjectMeta: api.ObjectMeta{
				GenerateName: "proxy-service-",
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{
					{
						Name:       "portname1",
						Port:       80,
						TargetPort: intstr.FromString("dest1"),
					},
					{
						Name:       "portname2",
						Port:       81,
						TargetPort: intstr.FromInt(162),
					},
					{
						Name:       "tlsportname1",
						Port:       443,
						TargetPort: intstr.FromString("tlsdest1"),
					},
					{
						Name:       "tlsportname2",
						Port:       444,
						TargetPort: intstr.FromInt(462),
					},
				},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func(name string) {
			err := f.Client.Services(f.Namespace.Name).Delete(name)
			if err != nil {
				framework.Logf("Failed deleting service %v: %v", name, err)
			}
		}(service.Name)

		// Make an RC with a single pod. The 'porter' image is
		// a simple server which serves the values of the
		// environmental variables below.
		By("starting an echo server on multiple ports")
		pods := []*api.Pod{}
		cfg := framework.RCConfig{
			Client:       f.Client,
			Image:        "gcr.io/google_containers/porter:cd5cb5791ebaa8641955f0e8c2a9bed669b1eaab",
			Name:         service.Name,
			Namespace:    f.Namespace.Name,
			Replicas:     1,
			PollInterval: time.Second,
			Env: map[string]string{
				"SERVE_PORT_80":   `<a href="/rewriteme">test</a>`,
				"SERVE_PORT_1080": `<a href="/rewriteme">test</a>`,
				"SERVE_PORT_160":  "foo",
				"SERVE_PORT_162":  "bar",

				"SERVE_TLS_PORT_443": `<a href="/tlsrewriteme">test</a>`,
				"SERVE_TLS_PORT_460": `tls baz`,
				"SERVE_TLS_PORT_462": `tls qux`,
			},
			Ports: map[string]int{
				"dest1": 160,
				"dest2": 162,

				"tlsdest1": 460,
				"tlsdest2": 462,
			},
			ReadinessProbe: &api.Probe{
				Handler: api.Handler{
					HTTPGet: &api.HTTPGetAction{
						Port: intstr.FromInt(80),
					},
				},
				InitialDelaySeconds: 1,
				TimeoutSeconds:      5,
				PeriodSeconds:       10,
			},
			Labels:      labels,
			CreatedPods: &pods,
		}
		Expect(framework.RunRC(cfg)).NotTo(HaveOccurred())
		defer framework.DeleteRCAndPods(f.Client, f.Namespace.Name, cfg.Name)

		Expect(f.WaitForAnEndpoint(service.Name)).NotTo(HaveOccurred())

		// table constructors
		// Try proxying through the service and directly to through the pod.
		svcProxyURL := func(scheme, port string) string {
			return prefix + "/proxy/namespaces/" + f.Namespace.Name + "/services/" + net.JoinSchemeNamePort(scheme, service.Name, port)
		}
		subresourceServiceProxyURL := func(scheme, port string) string {
			return prefix + "/namespaces/" + f.Namespace.Name + "/services/" + net.JoinSchemeNamePort(scheme, service.Name, port) + "/proxy"
		}
		podProxyURL := func(scheme, port string) string {
			return prefix + "/proxy/namespaces/" + f.Namespace.Name + "/pods/" + net.JoinSchemeNamePort(scheme, pods[0].Name, port)
		}
		subresourcePodProxyURL := func(scheme, port string) string {
			return prefix + "/namespaces/" + f.Namespace.Name + "/pods/" + net.JoinSchemeNamePort(scheme, pods[0].Name, port) + "/proxy"
		}

		// construct the table
		expectations := map[string]string{
			svcProxyURL("", "portname1") + "/": "foo",
			svcProxyURL("", "80") + "/":        "foo",
			svcProxyURL("", "portname2") + "/": "bar",
			svcProxyURL("", "81") + "/":        "bar",

			svcProxyURL("http", "portname1") + "/": "foo",
			svcProxyURL("http", "80") + "/":        "foo",
			svcProxyURL("http", "portname2") + "/": "bar",
			svcProxyURL("http", "81") + "/":        "bar",

			svcProxyURL("https", "tlsportname1") + "/": "tls baz",
			svcProxyURL("https", "443") + "/":          "tls baz",
			svcProxyURL("https", "tlsportname2") + "/": "tls qux",
			svcProxyURL("https", "444") + "/":          "tls qux",

			subresourceServiceProxyURL("", "portname1") + "/":         "foo",
			subresourceServiceProxyURL("http", "portname1") + "/":     "foo",
			subresourceServiceProxyURL("", "portname2") + "/":         "bar",
			subresourceServiceProxyURL("http", "portname2") + "/":     "bar",
			subresourceServiceProxyURL("https", "tlsportname1") + "/": "tls baz",
			subresourceServiceProxyURL("https", "tlsportname2") + "/": "tls qux",

			podProxyURL("", "1080") + "/": `<a href="` + podProxyURL("", "1080") + `/rewriteme">test</a>`,
			podProxyURL("", "160") + "/":  "foo",
			podProxyURL("", "162") + "/":  "bar",

			podProxyURL("http", "1080") + "/": `<a href="` + podProxyURL("http", "1080") + `/rewriteme">test</a>`,
			podProxyURL("http", "160") + "/":  "foo",
			podProxyURL("http", "162") + "/":  "bar",

			subresourcePodProxyURL("", "") + "/":         `<a href="` + subresourcePodProxyURL("", "") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("", "1080") + "/":     `<a href="` + subresourcePodProxyURL("", "1080") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("http", "1080") + "/": `<a href="` + subresourcePodProxyURL("http", "1080") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("", "160") + "/":      "foo",
			subresourcePodProxyURL("http", "160") + "/":  "foo",
			subresourcePodProxyURL("", "162") + "/":      "bar",
			subresourcePodProxyURL("http", "162") + "/":  "bar",

			subresourcePodProxyURL("https", "443") + "/": `<a href="` + subresourcePodProxyURL("https", "443") + `/tlsrewriteme">test</a>`,
			subresourcePodProxyURL("https", "460") + "/": "tls baz",
			subresourcePodProxyURL("https", "462") + "/": "tls qux",

			// TODO: below entries don't work, but I believe we should make them work.
			// podPrefix + ":dest1": "foo",
			// podPrefix + ":dest2": "bar",
		}

		wg := sync.WaitGroup{}
		errs := []string{}
		errLock := sync.Mutex{}
		recordError := func(s string) {
			errLock.Lock()
			defer errLock.Unlock()
			errs = append(errs, s)
		}
		d := time.Since(start)
		framework.Logf("setup took %v, starting test cases", d)
		numberTestCases := len(expectations)
		totalAttempts := numberTestCases * proxyAttempts
		By(fmt.Sprintf("running %v cases, %v attempts per case, %v total attempts", numberTestCases, proxyAttempts, totalAttempts))

		for i := 0; i < proxyAttempts; i++ {
			wg.Add(numberTestCases)
			for path, val := range expectations {
				go func(i int, path, val string) {
					defer wg.Done()
					// this runs the test case
					body, status, d, err := doProxy(f, path, i)

					if err != nil {
						if serr, ok := err.(*errors.StatusError); ok {
							recordError(fmt.Sprintf("%v (%v; %v): path %v gave status error: %+v",
								i, status, d, path, serr.Status()))
						} else {
							recordError(fmt.Sprintf("%v: path %v gave error: %v", i, path, err))
						}
						return
					}
					if status != http.StatusOK {
						recordError(fmt.Sprintf("%v: path %v gave status: %v", i, path, status))
					}
					if e, a := val, string(body); e != a {
						recordError(fmt.Sprintf("%v: path %v: wanted %v, got %v", i, path, e, a))
					}
					if d > proxyHTTPCallTimeout {
						recordError(fmt.Sprintf("%v: path %v took %v > %v", i, path, d, proxyHTTPCallTimeout))
					}
				}(i, path, val)
			}
			wg.Wait()
		}

		if len(errs) != 0 {
			body, err := f.Client.Pods(f.Namespace.Name).GetLogs(pods[0].Name, &api.PodLogOptions{}).Do().Raw()
			if err != nil {
				framework.Logf("Error getting logs for pod %s: %v", pods[0].Name, err)
			} else {
				framework.Logf("Pod %s has the following error logs: %s", pods[0].Name, body)
			}

			Fail(strings.Join(errs, "\n"))
		}
	})
}

func doProxy(f *framework.Framework, path string, i int) (body []byte, statusCode int, d time.Duration, err error) {
	// About all of the proxy accesses in this file:
	// * AbsPath is used because it preserves the trailing '/'.
	// * Do().Raw() is used (instead of DoRaw()) because it will turn an
	//   error from apiserver proxy into an actual error, and there is no
	//   chance of the things we are talking to being confused for an error
	//   that apiserver would have emitted.
	start := time.Now()
	body, err = f.Client.Get().AbsPath(path).Do().StatusCode(&statusCode).Raw()
	d = time.Since(start)
	if len(body) > 0 {
		framework.Logf("(%v) %v: %s (%v; %v)", i, path, truncate(body, maxDisplayBodyLen), statusCode, d)
	} else {
		framework.Logf("%v: %s (%v; %v)", path, "no body", statusCode, d)
	}
	return
}

func truncate(b []byte, maxLen int) []byte {
	if len(b) <= maxLen-3 {
		return b
	}
	b2 := append([]byte(nil), b[:maxLen-3]...)
	b2 = append(b2, '.', '.', '.')
	return b2
}

func pickNode(c *client.Client) (string, error) {
	// TODO: investigate why it doesn't work on master Node.
	nodes := framework.GetReadySchedulableNodesOrDie(c)
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("no nodes exist, can't test node proxy")
	}
	return nodes.Items[0].Name, nil
}

func nodeProxyTest(f *framework.Framework, prefix, nodeDest string) {
	node, err := pickNode(f.Client)
	Expect(err).NotTo(HaveOccurred())
	// TODO: Change it to test whether all requests succeeded when requests
	// not reaching Kubelet issue is debugged.
	serviceUnavailableErrors := 0
	for i := 0; i < proxyAttempts; i++ {
		_, status, d, err := doProxy(f, prefix+node+nodeDest, i)
		if status == http.StatusServiceUnavailable {
			framework.Logf("Failed proxying node logs due to service unavailable: %v", err)
			time.Sleep(time.Second)
			serviceUnavailableErrors++
		} else {
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(http.StatusOK))
			Expect(d).To(BeNumerically("<", proxyHTTPCallTimeout))
		}
	}
	if serviceUnavailableErrors > 0 {
		framework.Logf("error: %d requests to proxy node logs failed", serviceUnavailableErrors)
	}
	maxFailures := int(math.Floor(0.1 * float64(proxyAttempts)))
	Expect(serviceUnavailableErrors).To(BeNumerically("<", maxFailures))
}
