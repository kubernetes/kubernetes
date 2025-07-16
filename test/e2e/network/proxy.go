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

// OWNER = sig/network

package network

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/transport"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	// Try all the proxy tests this many times (to catch even rare flakes).
	proxyAttempts = 20
	// Only print this many characters of the response (to keep the logs
	// legible).
	maxDisplayBodyLen = 100

	// We have seen one of these calls take just over 15 seconds, so putting this at 30.
	proxyHTTPCallTimeout = 30 * time.Second

	requestRetryPeriod  = 10 * time.Millisecond
	requestRetryTimeout = 1 * time.Minute
)

type jsonResponse struct {
	Method string
	Body   string
}

var _ = common.SIGDescribe("Proxy", func() {
	version := "v1"
	ginkgo.Context("version "+version, func() {
		options := framework.Options{
			ClientQPS: -1.0,
		}
		f := framework.NewFramework("proxy", options, nil)
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
		prefix := "/api/" + version

		/*
			Test for Proxy, logs port endpoint
			Select any node in the cluster to invoke /proxy/nodes/<nodeip>:10250/logs endpoint. This endpoint MUST be reachable.
		*/
		ginkgo.It("should proxy logs on node with explicit kubelet port using proxy subresource ", func(ctx context.Context) { nodeProxyTest(ctx, f, prefix+"/nodes/", ":10250/proxy/logs/") })

		/*
			Test for Proxy, logs endpoint
			Select any node in the cluster to invoke /proxy/nodes/<nodeip>//logs endpoint. This endpoint MUST be reachable.
		*/
		ginkgo.It("should proxy logs on node using proxy subresource ", func(ctx context.Context) { nodeProxyTest(ctx, f, prefix+"/nodes/", "/proxy/logs/") })

		// using the porter image to serve content, access the content
		// (of multiple pods?) from multiple (endpoints/services?)
		/*
			Release: v1.9
			Testname: Proxy, logs service endpoint
			Description: Select any node in the cluster to invoke  /logs endpoint  using the /nodes/proxy subresource from the kubelet port. This endpoint MUST be reachable.
		*/
		framework.ConformanceIt("should proxy through a service and a pod", func(ctx context.Context) {
			start := time.Now()
			labels := map[string]string{"proxy-service-target": "true"}
			service, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "proxy-service-",
				},
				Spec: v1.ServiceSpec{
					Selector: labels,
					Ports: []v1.ServicePort{
						{
							Name:       "portname1",
							Port:       80,
							TargetPort: intstr.FromString("dest1"),
						},
						{
							Name:       "portname2",
							Port:       81,
							TargetPort: intstr.FromInt32(162),
						},
						{
							Name:       "tlsportname1",
							Port:       443,
							TargetPort: intstr.FromString("tlsdest1"),
						},
						{
							Name:       "tlsportname2",
							Port:       444,
							TargetPort: intstr.FromInt32(462),
						},
					},
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			// Make a deployment with a single pod. The 'porter' image is
			// a simple server which serves the values of the
			// environmental variables below.
			ginkgo.By("starting an echo server on multiple ports")

			deploymentConfig := e2edeployment.NewDeployment(service.Name,
				1,
				labels,
				service.Name,
				imageutils.GetE2EImage(imageutils.Agnhost),
				appsv1.RecreateDeploymentStrategyType)
			deploymentConfig.Spec.Template.Spec.Containers[0].Command = []string{"/agnhost", "porter"}
			deploymentConfig.Spec.Template.Spec.Containers[0].Env = []v1.EnvVar{
				{
					Name:  "SERVE_PORT_80",
					Value: `<a href="/rewriteme">test</a>`,
				},
				{
					Name:  "SERVE_PORT_1080",
					Value: `<a href="/rewriteme">test</a>`,
				},
				{
					Name:  "SERVE_PORT_160",
					Value: "foo",
				},
				{
					Name:  "SERVE_PORT_162",
					Value: "bar",
				},
				{
					Name:  "SERVE_TLS_PORT_443",
					Value: `<a href="/tlsrewriteme">test</a>`,
				},
				{
					Name:  "SERVE_TLS_PORT_460",
					Value: "tls baz",
				},
				{
					Name:  "SERVE_TLS_PORT_462",
					Value: "tls qux",
				},
			}
			deploymentConfig.Spec.Template.Spec.Containers[0].Ports = []v1.ContainerPort{
				{
					ContainerPort: 80,
				},
				{
					Name:          "dest1",
					ContainerPort: 160,
				},
				{
					Name:          "dest2",
					ContainerPort: 162,
				},
				{
					Name:          "tlsdest1",
					ContainerPort: 460,
				},
				{
					Name:          "tlsdest2",
					ContainerPort: 462,
				},
			}
			deploymentConfig.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					HTTPGet: &v1.HTTPGetAction{
						Port: intstr.FromInt32(80),
					},
				},
				InitialDelaySeconds: 1,
				TimeoutSeconds:      5,
				PeriodSeconds:       10,
			}

			deployment, err := f.ClientSet.AppsV1().Deployments(f.Namespace.Name).Create(ctx,
				deploymentConfig,
				metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return f.ClientSet.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, name, metav1.DeleteOptions{})
			}, deployment.Name)

			err = e2edeployment.WaitForDeploymentComplete(f.ClientSet, deployment)
			framework.ExpectNoError(err)

			podList, err := e2edeployment.GetPodsForDeployment(ctx, f.ClientSet, deployment)
			framework.ExpectNoError(err)
			pods := podList.Items

			err = waitForEndpoint(ctx, f.ClientSet, f.Namespace.Name, service.Name)
			framework.ExpectNoError(err)

			// table constructors
			// Try proxying through the service and directly to through the pod.
			subresourceServiceProxyURL := func(scheme, port string) string {
				return prefix + "/namespaces/" + f.Namespace.Name + "/services/" + net.JoinSchemeNamePort(scheme, service.Name, port) + "/proxy"
			}
			subresourcePodProxyURL := func(scheme, port string) string {
				return prefix + "/namespaces/" + f.Namespace.Name + "/pods/" + net.JoinSchemeNamePort(scheme, pods[0].Name, port) + "/proxy"
			}

			// construct the table
			expectations := map[string]string{
				subresourceServiceProxyURL("", "portname1") + "/":         "foo",
				subresourceServiceProxyURL("http", "portname1") + "/":     "foo",
				subresourceServiceProxyURL("", "portname2") + "/":         "bar",
				subresourceServiceProxyURL("http", "portname2") + "/":     "bar",
				subresourceServiceProxyURL("https", "tlsportname1") + "/": "tls baz",
				subresourceServiceProxyURL("https", "tlsportname2") + "/": "tls qux",

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
			ginkgo.By(fmt.Sprintf("running %v cases, %v attempts per case, %v total attempts", numberTestCases, proxyAttempts, totalAttempts))

			for i := 0; i < proxyAttempts; i++ {
				wg.Add(numberTestCases)
				for path, val := range expectations {
					go func(i int, path, val string) {
						defer wg.Done()
						// this runs the test case
						body, status, d, err := doProxy(ctx, f, path, i)

						if err != nil {
							if serr, ok := err.(*apierrors.StatusError); ok {
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
				body, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).GetLogs(pods[0].Name, &v1.PodLogOptions{}).Do(ctx).Raw()
				if err != nil {
					framework.Logf("Error getting logs for pod %s: %v", pods[0].Name, err)
				} else {
					framework.Logf("Pod %s has the following error logs: %s", pods[0].Name, body)
				}

				framework.Fail(strings.Join(errs, "\n"))
			}
		})

		/*
			Release: v1.21
			Testname: Proxy, validate ProxyWithPath responses
			Description: Attempt to create a pod and a service. A
			set of pod and service endpoints MUST be accessed via
			ProxyWithPath using a list of http methods. A valid
			response MUST be returned for each endpoint.
		*/
		framework.ConformanceIt("A set of valid responses are returned for both pod and service ProxyWithPath", func(ctx context.Context) {

			ns := f.Namespace.Name
			msg := "foo"
			testSvcName := "test-service"
			testSvcLabels := map[string]string{"test": "response"}

			framework.Logf("Creating pod...")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "agnhost",
					Namespace: ns,
					Labels: map[string]string{
						"test": "response"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Image:   imageutils.GetE2EImage(imageutils.Agnhost),
						Name:    "agnhost",
						Command: []string{"/agnhost", "porter", "--json-response"},
						Env: []v1.EnvVar{{
							Name:  "SERVE_PORT_80",
							Value: msg,
						}},
					}},
					RestartPolicy: v1.RestartPolicyNever,
				}}
			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod")
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "Pod didn't start within time out period")

			framework.Logf("Creating service...")
			_, err = f.ClientSet.CoreV1().Services(ns).Create(ctx, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testSvcName,
					Namespace: ns,
					Labels:    testSvcLabels,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:       80,
						TargetPort: intstr.FromInt32(80),
						Protocol:   v1.ProtocolTCP,
					}},
					Selector: map[string]string{
						"test": "response",
					},
				}}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Failed to create the service")

			transportCfg, err := f.ClientConfig().TransportConfig()
			framework.ExpectNoError(err, "Error creating transportCfg")
			restTransport, err := transport.New(transportCfg)
			framework.ExpectNoError(err, "Error creating restTransport")

			client := &http.Client{
				CheckRedirect: func(req *http.Request, via []*http.Request) error {
					return http.ErrUseLastResponse
				},
				Transport: restTransport,
			}

			// All methods for Pod ProxyWithPath return 200
			// For all methods other than HEAD the response body returns 'foo' with the received http method
			httpVerbs := []string{"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"}
			for _, httpVerb := range httpVerbs {

				urlString := strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/pods/agnhost/proxy/some/path/with/" + httpVerb
				framework.Logf("Starting http.Client for %s", urlString)

				pollErr := wait.PollImmediate(requestRetryPeriod, requestRetryTimeout, validateProxyVerbRequest(client, urlString, httpVerb, msg))
				framework.ExpectNoError(err, "Service didn't start within time out period. %v", pollErr)
			}

			// All methods for Service ProxyWithPath return 200
			// For all methods other than HEAD the response body returns 'foo' with the received http method
			for _, httpVerb := range httpVerbs {

				urlString := strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/services/test-service/proxy/some/path/with/" + httpVerb
				framework.Logf("Starting http.Client for %s", urlString)

				pollErr := wait.PollImmediate(requestRetryPeriod, requestRetryTimeout, validateProxyVerbRequest(client, urlString, httpVerb, msg))
				framework.ExpectNoError(err, "Service didn't start within time out period. %v", pollErr)
			}
		})

		/*
			Release: v1.24
			Testname: Proxy, validate Proxy responses
			Description: Attempt to create a pod and a service. A
			set of pod and service endpoints MUST be accessed via
			Proxy using a list of http methods. A valid response
			MUST be returned for each endpoint.
		*/
		framework.ConformanceIt("A set of valid responses are returned for both pod and service Proxy", func(ctx context.Context) {

			ns := f.Namespace.Name
			msg := "foo"
			testSvcName := "e2e-proxy-test-service"
			testSvcLabels := map[string]string{"e2e-test": "proxy-endpoints"}

			framework.Logf("Creating pod...")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "agnhost",
					Namespace: ns,
					Labels: map[string]string{
						"e2e-test": "proxy-endpoints"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Image:   imageutils.GetE2EImage(imageutils.Agnhost),
						Name:    "agnhost",
						Command: []string{"/agnhost", "porter", "--json-response"},
						Env: []v1.EnvVar{{
							Name:  "SERVE_PORT_80",
							Value: msg,
						}},
					}},
					RestartPolicy: v1.RestartPolicyNever,
				}}
			_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod")
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod), "Pod didn't start within time out period")

			framework.Logf("Creating service...")
			_, err = f.ClientSet.CoreV1().Services(ns).Create(ctx, &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testSvcName,
					Namespace: ns,
					Labels:    testSvcLabels,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Port:       80,
						TargetPort: intstr.FromInt32(80),
						Protocol:   v1.ProtocolTCP,
					}},
					Selector: map[string]string{
						"e2e-test": "proxy-endpoints",
					},
				}}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Failed to create the service")

			transportCfg, err := f.ClientConfig().TransportConfig()
			framework.ExpectNoError(err, "Error creating transportCfg")
			restTransport, err := transport.New(transportCfg)
			framework.ExpectNoError(err, "Error creating restTransport")

			client := &http.Client{
				CheckRedirect: func(req *http.Request, via []*http.Request) error {
					return http.ErrUseLastResponse
				},
				Transport: restTransport,
			}

			// All methods for Pod Proxy return 200
			// The response body returns 'foo' with the received http method
			httpVerbs := []string{"DELETE", "OPTIONS", "PATCH", "POST", "PUT"}
			for _, httpVerb := range httpVerbs {

				urlString := strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/pods/agnhost/proxy?method=" + httpVerb
				framework.Logf("Starting http.Client for %s", urlString)

				pollErr := wait.PollImmediate(requestRetryPeriod, requestRetryTimeout, validateProxyVerbRequest(client, urlString, httpVerb, msg))
				framework.ExpectNoError(pollErr, "Pod didn't start within time out period. %v", pollErr)
			}

			// All methods for Service Proxy return 200
			// The response body returns 'foo' with the received http method
			for _, httpVerb := range httpVerbs {

				urlString := strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/services/" + testSvcName + "/proxy?method=" + httpVerb
				framework.Logf("Starting http.Client for %s", urlString)

				pollErr := wait.PollImmediate(requestRetryPeriod, requestRetryTimeout, validateProxyVerbRequest(client, urlString, httpVerb, msg))
				framework.ExpectNoError(pollErr, "Service didn't start within time out period. %v", pollErr)
			}

			// Test that each method returns 301 for both pod and service endpoints
			redirectVerbs := []string{"GET", "HEAD"}
			for _, redirectVerb := range redirectVerbs {
				urlString := strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/pods/agnhost/proxy?method=" + redirectVerb
				validateRedirectRequest(client, redirectVerb, urlString)

				urlString = strings.TrimRight(f.ClientConfig().Host, "/") + "/api/v1/namespaces/" + ns + "/services/" + testSvcName + "/proxy?method=" + redirectVerb
				validateRedirectRequest(client, redirectVerb, urlString)
			}
		})
	})
})

func validateRedirectRequest(client *http.Client, redirectVerb string, urlString string) {
	framework.Logf("Starting http.Client for %s", urlString)
	request, err := http.NewRequest(redirectVerb, urlString, nil)
	framework.ExpectNoError(err, "processing request")

	resp, err := client.Do(request)
	framework.ExpectNoError(err, "processing response")
	defer resp.Body.Close()

	framework.Logf("http.Client request:%s StatusCode:%d", redirectVerb, resp.StatusCode)
	gomega.Expect(resp.StatusCode).To(gomega.Equal(301), "The resp.StatusCode returned: %d", resp.StatusCode)
}

// validateProxyVerbRequest checks that a http request to a pod
// or service was valid for any http verb. Requires agnhost image
// with porter --json-response
func validateProxyVerbRequest(client *http.Client, urlString string, httpVerb string, msg string) func() (bool, error) {
	return func() (bool, error) {
		var err error

		request, err := http.NewRequest(httpVerb, urlString, nil)
		if err != nil {
			framework.Logf("Failed to get a new request. %v", err)
			return false, nil
		}

		resp, err := client.Do(request)
		if err != nil {
			framework.Logf("Failed to get a response. %v", err)
			return false, nil
		}
		defer resp.Body.Close()

		buf := new(bytes.Buffer)
		buf.ReadFrom(resp.Body)
		response := buf.String()

		switch httpVerb {
		case "HEAD":
			framework.Logf("http.Client request:%s | StatusCode:%d", httpVerb, resp.StatusCode)
			if resp.StatusCode != 200 {
				return false, nil
			}
			return true, nil
		default:
			var jr *jsonResponse
			err = json.Unmarshal([]byte(response), &jr)
			if err != nil {
				framework.Logf("Failed to process jsonResponse. %v", err)
				return false, nil
			}

			framework.Logf("http.Client request:%s | StatusCode:%d | Response:%s | Method:%s", httpVerb, resp.StatusCode, jr.Body, jr.Method)
			if resp.StatusCode != 200 {
				return false, nil
			}

			if msg != jr.Body {
				return false, nil
			}

			if httpVerb != jr.Method {
				return false, nil
			}
			return true, nil
		}
	}
}

func doProxy(ctx context.Context, f *framework.Framework, path string, i int) (body []byte, statusCode int, d time.Duration, err error) {
	// About all of the proxy accesses in this file:
	// * AbsPath is used because it preserves the trailing '/'.
	// * Do().Raw() is used (instead of DoRaw()) because it will turn an
	//   error from apiserver proxy into an actual error, and there is no
	//   chance of the things we are talking to being confused for an error
	//   that apiserver would have emitted.
	start := time.Now()
	body, err = f.ClientSet.CoreV1().RESTClient().Get().AbsPath(path).Do(ctx).StatusCode(&statusCode).Raw()
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

func nodeProxyTest(ctx context.Context, f *framework.Framework, prefix, nodeDest string) {
	// TODO: investigate why it doesn't work on master Node.
	node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
	framework.ExpectNoError(err)

	// TODO: Change it to test whether all requests succeeded when requests
	// not reaching Kubelet issue is debugged.
	serviceUnavailableErrors := 0
	for i := 0; i < proxyAttempts; i++ {
		_, status, d, err := doProxy(ctx, f, prefix+node.Name+nodeDest, i)
		if status == http.StatusServiceUnavailable {
			framework.Logf("ginkgo.Failed proxying node logs due to service unavailable: %v", err)
			time.Sleep(time.Second)
			serviceUnavailableErrors++
		} else {
			framework.ExpectNoError(err)
			gomega.Expect(status).To(gomega.Equal(http.StatusOK))
			gomega.Expect(d).To(gomega.BeNumerically("<", proxyHTTPCallTimeout))
		}
	}
	if serviceUnavailableErrors > 0 {
		framework.Logf("error: %d requests to proxy node logs failed", serviceUnavailableErrors)
	}
	maxFailures := int(math.Floor(0.1 * float64(proxyAttempts)))
	gomega.Expect(serviceUnavailableErrors).To(gomega.BeNumerically("<", maxFailures))
}

// waitForEndpoint waits for the specified endpoint to be ready.
func waitForEndpoint(ctx context.Context, c clientset.Interface, ns, name string) error {
	// registerTimeout is how long to wait for an endpoint to be registered.
	registerTimeout := time.Minute
	for t := time.Now(); time.Since(t) < registerTimeout; time.Sleep(framework.Poll) {
		endpoint, err := c.CoreV1().Endpoints(ns).Get(ctx, name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			framework.Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		}
		framework.ExpectNoError(err, "Failed to get endpoints for %s/%s", ns, name)
		if len(endpoint.Subsets) == 0 || len(endpoint.Subsets[0].Addresses) == 0 {
			framework.Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		}
		return nil
	}
	return fmt.Errorf("failed to get endpoints for %s/%s", ns, name)
}
