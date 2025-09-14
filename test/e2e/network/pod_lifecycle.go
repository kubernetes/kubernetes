/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Connectivity Pod Lifecycle", func() {

	fr := framework.NewDefaultFramework("podlifecycle")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		cs        clientset.Interface
		ns        string
		podClient *e2epod.PodClient
	)
	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = fr.ClientSet
		ns = fr.Namespace.Name
		podClient = e2epod.NewPodClient(fr)
	})

	ginkgo.It("should be able to connect from a Pod to a terminating Pod", func(ctx context.Context) {
		ginkgo.By("Creating 1 webserver pod able to serve traffic during the grace period of 300 seconds")
		gracePeriod := int64(100)
		webserverPod := e2epod.NewAgnhostPod(ns, "webserver-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		webserverPod.Spec.TerminationGracePeriodSeconds = &gracePeriod
		webserverPod = podClient.CreateSync(ctx, webserverPod)

		ginkgo.By("Creating 1 client pod that will try to connect to the webserver")
		pausePod := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil)
		pausePod = podClient.CreateSync(ctx, pausePod)

		ginkgo.By("Try to connect to the webserver")
		// Wait until we are able to connect to the Pod
		podIPAddress := net.JoinHostPort(webserverPod.Status.PodIP, strconv.Itoa(80))
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)

		// webserver should continue to serve traffic after delete
		// since will be gracefully terminating
		err := cs.CoreV1().Pods(ns).Delete(ctx, webserverPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting webserver pod")
		// wait some time to ensure the test hit the pod during the terminating phase
		time.Sleep(15 * time.Second)
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)
	})

	ginkgo.It("should be able to connect to other Pod from a terminating Pod", func(ctx context.Context) {
		ginkgo.By("Creating 1 webserver pod able to serve traffic during the grace period of 300 seconds")
		gracePeriod := int64(100)
		webserverPod := e2epod.NewAgnhostPod(ns, "webserver-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		webserverPod = podClient.CreateSync(ctx, webserverPod)

		ginkgo.By("Creating 1 client pod that will try to connect to the webservers")
		pausePod := e2epod.NewAgnhostPod(ns, "pause-pod-1", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		pausePod.Spec.TerminationGracePeriodSeconds = &gracePeriod
		pausePod = podClient.CreateSync(ctx, pausePod)

		ginkgo.By("Try to connect to the webserver")
		// Wait until we are able to connect to the Pod
		podIPAddress := net.JoinHostPort(webserverPod.Status.PodIP, strconv.Itoa(80))
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)

		// pod client should continue to connect to the webserver after delete
		// since will be gracefully terminating
		err := cs.CoreV1().Pods(ns).Delete(ctx, pausePod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "error deleting client pod")
		// wait some time to ensure the test hit the pod during the terminating
		time.Sleep(15 * time.Second)
		execHostnameTest(*pausePod, podIPAddress, webserverPod.Name)
	})

	ginkgo.It("should be able to have zero downtime on a Blue Green deployment using Services and Readiness Gates", func(ctx context.Context) {
		readinessGate := "k8s.io/blue-green"
		patchStatusFmt := `{"status":{"conditions":[{"type":%q, "status":%q}]}}`

		serviceName := "blue-green-svc"
		blueGreenJig := e2eservice.NewTestJig(cs, ns, serviceName)
		ginkgo.By("creating a service " + serviceName + " with type=ClusterIP in " + ns)
		blueGreenService, err := blueGreenJig.CreateTCPService(ctx, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.Ports = []v1.ServicePort{
				{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt32(80)},
			}
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating 2 webserver pod (green and blue) able to serve traffic during the grace period of 30 seconds")
		gracePeriod := int64(30)
		bluePod := e2epod.NewAgnhostPod(ns, "blue-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		bluePod.Labels = blueGreenService.Labels
		bluePod.Spec.ReadinessGates = []v1.PodReadinessGate{
			{ConditionType: v1.PodConditionType(readinessGate)},
		}
		podClient.Create(ctx, bluePod)
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, cs, bluePod.Name, ns)
		if err != nil {
			framework.Failf("waiting for pod %s : %v", bluePod.Name, err)
		}
		if podClient.PodIsReady(ctx, bluePod.Name) {
			framework.Failf("Expect pod(%s/%s)'s Ready condition to be false initially.", ns, bluePod.Name)
		}

		greenPod := e2epod.NewAgnhostPod(ns, "green-pod", nil, nil, nil, "netexec", "--http-port=80", fmt.Sprintf("--delay-shutdown=%d", gracePeriod))
		greenPod.Labels = blueGreenService.Labels
		greenPod.Spec.ReadinessGates = []v1.PodReadinessGate{
			{ConditionType: v1.PodConditionType(readinessGate)},
		}
		podClient.Create(ctx, greenPod)
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, cs, greenPod.Name, ns)
		if err != nil {
			framework.Failf("waiting for pod %s : %v", greenPod.Name, err)
		}
		if podClient.PodIsReady(ctx, greenPod.Name) {
			framework.Failf("Expect pod(%s/%s)'s Ready condition to be false initially.", ns, greenPod.Name)
		}

		ginkgo.By("Creating 1 client pod that will try to connect to the blue-green-svc")
		clientPod := e2epod.NewAgnhostPod(ns, "client-pod-1", nil, nil, nil)
		clientPod.Spec.TerminationGracePeriodSeconds = &gracePeriod
		clientPod = podClient.CreateSync(ctx, clientPod)

		ginkgo.By(fmt.Sprintf("patching blue pod status with condition %q to true", readinessGate))
		_, err = podClient.Patch(ctx, bluePod.Name, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate, "True")), metav1.PatchOptions{}, "status")
		if err != nil {
			framework.Failf("failed to patch %s pod condition: %v", bluePod.Name, err)
		}

		// Expect EndpointSlice resource to have the blue pod ready to serve traffic
		if err := e2eendpointslice.WaitForEndpointSlices(ctx, cs, blueGreenJig.Namespace, blueGreenJig.Name, 2*time.Second, wait.ForeverTestTimeout, func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (bool, error) {
			for _, slice := range endpointSlices {
				for _, ep := range slice.Endpoints {
					if ep.TargetRef != nil &&
						ep.TargetRef.Name == bluePod.Name &&
						ep.TargetRef.Namespace == bluePod.Namespace &&
						ep.Conditions.Ready != nil && *ep.Conditions.Ready {
						return true, nil
					}
				}
			}
			return false, nil
		}); err != nil {
			framework.Failf("No EndpointSlice found for Service %s/%s: %s", blueGreenJig.Namespace, blueGreenJig.Name, err)
		}

		ginkgo.By("Try to connect to the blue pod through the service")
		scvAddress := net.JoinHostPort(blueGreenService.Spec.ClusterIP, strconv.Itoa(80))
		// assert 5 times that we can connect only to the blue pod
		for i := 0; i < 5; i++ {
			err := wait.PollUntilContextTimeout(ctx, 3*time.Second, 30*time.Second, true, func(ctx context.Context) (done bool, err error) {
				cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, scvAddress)
				stdout, err := e2eoutput.RunHostCmd(clientPod.Namespace, clientPod.Name, cmd)
				if err != nil {
					framework.Logf("expected error when trying to connect to cluster IP : %v", err)
					return false, nil
				}
				if strings.TrimSpace(stdout) == "" {
					framework.Logf("got empty stdout, retry until timeout")
					return false, nil
				}
				// Ensure we're comparing hostnames and not FQDNs
				targetHostname := strings.Split(bluePod.Name, ".")[0]
				hostname := strings.TrimSpace(strings.Split(stdout, ".")[0])
				if hostname != targetHostname {
					return false, fmt.Errorf("expecting hostname %s got %s", targetHostname, hostname)
				}
				return true, nil
			})
			if err != nil {
				framework.Failf("can not connect to pod %s on address %s : %v", bluePod.Name, scvAddress, err)
			}
		}

		// Switch from blue to green
		ginkgo.By(fmt.Sprintf("patching green pod status with condition %q to true", readinessGate))
		_, err = podClient.Patch(ctx, greenPod.Name, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate, "True")), metav1.PatchOptions{}, "status")
		if err != nil {
			framework.Failf("failed to patch %s pod condition: %v", greenPod.Name, err)
		}

		// Expect EndpointSlice resource to have the green pod ready to serve traffic
		if err := e2eendpointslice.WaitForEndpointSlices(ctx, cs, blueGreenJig.Namespace, blueGreenJig.Name, 2*time.Second, wait.ForeverTestTimeout, func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (bool, error) {
			for _, slice := range endpointSlices {
				for _, ep := range slice.Endpoints {
					if ep.TargetRef != nil &&
						ep.TargetRef.Name == greenPod.Name &&
						ep.TargetRef.Namespace == greenPod.Namespace &&
						ep.Conditions.Ready != nil && *ep.Conditions.Ready {
						return true, nil
					}
				}
			}
			return false, nil
		}); err != nil {
			framework.Failf("No EndpointSlice found for Service %s/%s: %s", blueGreenJig.Namespace, blueGreenJig.Name, err)
		}

		ginkgo.By(fmt.Sprintf("patching blue pod status with condition %q to false", readinessGate))
		_, err = podClient.Patch(ctx, bluePod.Name, types.StrategicMergePatchType, []byte(fmt.Sprintf(patchStatusFmt, readinessGate, "False")), metav1.PatchOptions{}, "status")
		if err != nil {
			framework.Failf("failed to patch %s pod condition: %v", bluePod.Name, err)
		}

		// Expect EndpointSlice resource to have the blue pod NOT ready to serve traffic
		if err := e2eendpointslice.WaitForEndpointSlices(ctx, cs, blueGreenJig.Namespace, blueGreenJig.Name, 2*time.Second, wait.ForeverTestTimeout, func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (bool, error) {
			for _, slice := range endpointSlices {
				for _, ep := range slice.Endpoints {
					if ep.TargetRef != nil &&
						ep.TargetRef.Name == bluePod.Name &&
						ep.TargetRef.Namespace == bluePod.Namespace &&
						ep.Conditions.Ready != nil && !*ep.Conditions.Ready {
						return true, nil
					}
				}
			}
			return false, nil
		}); err != nil {
			framework.Failf("No EndpointSlice found for Service %s/%s: %s", blueGreenJig.Namespace, blueGreenJig.Name, err)
		}

		// We have checked the endpoint slices reflect the desired state:
		// bluePod not ready and greenPod ready, but we need to remember kubernetes
		// is a distributed system eventually consistent, so there is a propagation
		// delay until this information is present on the nodes and a programming delay
		// until the corresponding node components program the information on the dataplane.
		// Since there are only two backends, we need to ensure that only one is active,
		// the chance of hitting any backend is 50% and each request is independent.
		// We can use binomial probability to calculate the probability of hitting only
		// the new pod despite there are two active pods, using 6 requests gives us
		// a 0.5^6=0.015625,  1.5625% of chances of not able to detect the transition.
		consecutiveHits := 0
		expectedHits := 6
		err = wait.PollUntilContextTimeout(ctx, 3*time.Second, e2eservice.ServiceReachabilityShortPollTimeout, true, func(ctx context.Context) (done bool, err error) {
			cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, scvAddress)
			stdout, err := e2eoutput.RunHostCmd(clientPod.Namespace, clientPod.Name, cmd)
			if err != nil {
				framework.Logf("expected error when trying to connect to cluster IP : %v", err)
				consecutiveHits = 0
				return false, nil
			}
			if strings.TrimSpace(stdout) == "" {
				framework.Logf("got empty stdout, retry until timeout")
				consecutiveHits = 0
				return false, nil
			}
			// Ensure we're comparing hostnames and not FQDNs
			targetHostname := strings.Split(greenPod.Name, ".")[0]
			hostname := strings.TrimSpace(strings.Split(stdout, ".")[0])
			if hostname != targetHostname {
				framework.Logf("expecting hostname %s got %s", targetHostname, hostname)
				consecutiveHits = 0
				return false, nil
			}
			consecutiveHits++
			if consecutiveHits < expectedHits {
				framework.Logf("got %s %d times, needs %d hits to ensure the dataplane is programmed with more 98.5 percent accuracy", targetHostname, consecutiveHits, expectedHits)
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			framework.Failf("can not connect to pod %s on address %s : %v", greenPod.Name, scvAddress, err)
		}

		ginkgo.By("Try to connect to the green pod through the service")
		// assert 5 times that we can connect ONLY to the green pod
		for i := 0; i < 5; i++ {
			err := wait.PollUntilContextTimeout(ctx, 3*time.Second, e2eservice.KubeProxyEndpointLagTimeout, true, func(ctx context.Context) (done bool, err error) {
				cmd := fmt.Sprintf(`curl -q -s --connect-timeout 5 %s/hostname`, scvAddress)
				stdout, err := e2eoutput.RunHostCmd(clientPod.Namespace, clientPod.Name, cmd)
				if err != nil {
					framework.Logf("expected error when trying to connect to cluster IP : %v", err)
					return false, nil
				}
				if strings.TrimSpace(stdout) == "" {
					framework.Logf("got empty stdout, retry until timeout")
					return false, nil
				}
				// Ensure we're comparing hostnames and not FQDNs
				targetHostname := strings.Split(greenPod.Name, ".")[0]
				hostname := strings.TrimSpace(strings.Split(stdout, ".")[0])
				// At this point we should only receive traffic from the green Pod.
				if hostname != targetHostname {
					return false, fmt.Errorf("expecting hostname %s got %s", targetHostname, hostname)
				}
				return true, nil
			})
			if err != nil {
				framework.Failf("can not connect to pod %s on address %s : %v", greenPod.Name, scvAddress, err)
			}
		}

		// TODO there can be multiple combinations like:
		// test zero downtime deleting the blue pod instead setting the readiness to false
		// test roll back setting back the readiness to true on the blue pod
		// ...

	})

})
