/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Proxy", func() {
	version := testapi.Default.Version()
	Context("version "+version, func() { proxyContext(version) })
})

const (
	// Try all the proxy tests this many times (to catch even rare flakes).
	proxyAttempts = 20
	// Only print this many characters of the response (to keep the logs
	// legible).
	maxDisplayBodyLen = 100
)

func proxyContext(version string) {
	f := NewFramework("proxy")
	prefix := "/api/" + version

	// Port here has to be kept in sync with default kubelet port.
	It("should proxy logs on node with explicit kubelet port", func() { nodeProxyTest(f, version, ":10250/logs/") })

	It("should proxy logs on node", func() { nodeProxyTest(f, version, "/logs/") })

	It("should proxy to cadvisor", func() { nodeProxyTest(f, version, ":4194/containers/") })

	It("should proxy through a service and a pod", func() {
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
						TargetPort: util.NewIntOrStringFromString("dest1"),
					},
					{
						Name:       "portname2",
						Port:       81,
						TargetPort: util.NewIntOrStringFromInt(162),
					},
					{
						Name:       "tlsportname1",
						Port:       443,
						TargetPort: util.NewIntOrStringFromString("tlsdest1"),
					},
					{
						Name:       "tlsportname2",
						Port:       444,
						TargetPort: util.NewIntOrStringFromInt(462),
					},
				},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func(name string) {
			err := f.Client.Services(f.Namespace.Name).Delete(name)
			if err != nil {
				Logf("Failed deleting service %v: %v", name, err)
			}
		}(service.Name)

		// Make an RC with a single pod.
		pods := []*api.Pod{}
		cfg := RCConfig{
			Client:       f.Client,
			Image:        "gcr.io/google_containers/porter:cd5cb5791ebaa8641955f0e8c2a9bed669b1eaab",
			Name:         service.Name,
			Namespace:    f.Namespace.Name,
			Replicas:     1,
			PollInterval: time.Second,
			Env: map[string]string{
				"SERVE_PORT_80":  `<a href="/rewriteme">test</a>`,
				"SERVE_PORT_160": "foo",
				"SERVE_PORT_162": "bar",

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
			Labels:      labels,
			CreatedPods: &pods,
		}
		Expect(RunRC(cfg)).NotTo(HaveOccurred())
		defer DeleteRC(f.Client, f.Namespace.Name, cfg.Name)

		Expect(f.WaitForAnEndpoint(service.Name)).NotTo(HaveOccurred())

		// Try proxying through the service and directly to through the pod.
		svcProxyURL := func(scheme, port string) string {
			return prefix + "/proxy/namespaces/" + f.Namespace.Name + "/services/" + util.JoinSchemeNamePort(scheme, service.Name, port)
		}
		podProxyURL := func(scheme, port string) string {
			return prefix + "/proxy/namespaces/" + f.Namespace.Name + "/pods/" + util.JoinSchemeNamePort(scheme, pods[0].Name, port)
		}
		subresourcePodProxyURL := func(scheme, port string) string {
			return prefix + "/namespaces/" + f.Namespace.Name + "/pods/" + util.JoinSchemeNamePort(scheme, pods[0].Name, port) + "/proxy"
		}
		expectations := map[string]string{
			svcProxyURL("", "portname1") + "/": "foo",
			svcProxyURL("", "portname2") + "/": "bar",

			svcProxyURL("http", "portname1") + "/": "foo",
			svcProxyURL("http", "portname2") + "/": "bar",

			svcProxyURL("https", "tlsportname1") + "/": "tls baz",
			svcProxyURL("https", "tlsportname2") + "/": "tls qux",

			podProxyURL("", "80") + "/":  `<a href="` + podProxyURL("", "80") + `/rewriteme">test</a>`,
			podProxyURL("", "160") + "/": "foo",
			podProxyURL("", "162") + "/": "bar",

			podProxyURL("http", "80") + "/":  `<a href="` + podProxyURL("http", "80") + `/rewriteme">test</a>`,
			podProxyURL("http", "160") + "/": "foo",
			podProxyURL("http", "162") + "/": "bar",

			subresourcePodProxyURL("", "") + "/":        `<a href="` + subresourcePodProxyURL("", "") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("", "80") + "/":      `<a href="` + subresourcePodProxyURL("", "80") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("http", "80") + "/":  `<a href="` + subresourcePodProxyURL("http", "80") + `/rewriteme">test</a>`,
			subresourcePodProxyURL("", "160") + "/":     "foo",
			subresourcePodProxyURL("http", "160") + "/": "foo",
			subresourcePodProxyURL("", "162") + "/":     "bar",
			subresourcePodProxyURL("http", "162") + "/": "bar",

			subresourcePodProxyURL("https", "443") + "/": `<a href="` + subresourcePodProxyURL("https", "443") + `/tlsrewriteme">test</a>`,
			subresourcePodProxyURL("https", "460") + "/": "tls baz",
			subresourcePodProxyURL("https", "462") + "/": "tls qux",
			// TODO: below entries don't work, but I believe we should make them work.
			// svcPrefix + ":80": "foo",
			// svcPrefix + ":81": "bar",
			// podPrefix + ":dest1": "foo",
			// podPrefix + ":dest2": "bar",
		}

		wg := sync.WaitGroup{}
		errors := []string{}
		errLock := sync.Mutex{}
		recordError := func(s string) {
			errLock.Lock()
			defer errLock.Unlock()
			errors = append(errors, s)
		}
		for i := 0; i < proxyAttempts; i++ {
			for path, val := range expectations {
				wg.Add(1)
				go func(i int, path, val string) {
					defer wg.Done()
					body, status, d, err := doProxy(f, path)
					if err != nil {
						recordError(fmt.Sprintf("%v: path %v gave error: %v", i, path, err))
						return
					}
					if status != http.StatusOK {
						recordError(fmt.Sprintf("%v: path %v gave status: %v", i, path, status))
					}
					if e, a := val, string(body); e != a {
						recordError(fmt.Sprintf("%v: path %v: wanted %v, got %v", i, path, e, a))
					}
					if d > 15*time.Second {
						recordError(fmt.Sprintf("%v: path %v took %v > 15s", i, path, d))
					}
				}(i, path, val)
				// default QPS is 5
				time.Sleep(200 * time.Millisecond)
			}
		}
		wg.Wait()

		if len(errors) != 0 {
			Fail(strings.Join(errors, "\n"))
		}
	})
}

func doProxy(f *Framework, path string) (body []byte, statusCode int, d time.Duration, err error) {
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
		Logf("%v: %s (%v; %v)", path, truncate(body, maxDisplayBodyLen), statusCode, d)
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
	nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("no nodes exist, can't test node proxy")
	}
	return nodes.Items[0].Name, nil
}

func nodeProxyTest(f *Framework, version, nodeDest string) {
	prefix := "/api/" + version
	node, err := pickNode(f.Client)
	Expect(err).NotTo(HaveOccurred())
	// TODO: Change it to test whether all requests succeeded when requests
	// not reaching Kubelet issue is debugged.
	serviceUnavailableErrors := 0
	for i := 0; i < proxyAttempts; i++ {
		_, status, d, err := doProxy(f, prefix+"/proxy/nodes/"+node+nodeDest)
		if status == http.StatusServiceUnavailable {
			Logf("Failed proxying node logs due to service unavailable: %v", err)
			time.Sleep(time.Second)
			serviceUnavailableErrors++
		} else {
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(http.StatusOK))
			Expect(d).To(BeNumerically("<", 15*time.Second))
		}
	}
	if serviceUnavailableErrors > 0 {
		Logf("error: %d requests to proxy node logs failed", serviceUnavailableErrors)
	}
	maxFailures := int(math.Floor(0.1 * float64(proxyAttempts)))
	Expect(serviceUnavailableErrors).To(BeNumerically("<", maxFailures))
}
