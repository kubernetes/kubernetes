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

package e2e

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/test/e2e/framework"

	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

/*
The following Network Policy tests verify that policy object definitions
are correctly enforced by a networking plugin. It accomplishes this by launching
a simple netcat server, and two clients with different
attributes. Each test case creates a network policy which should only allow
connections from one of the clients. The test then asserts that the clients
failed or succesfully connected as expected.
*/

var _ = framework.KubeDescribe("NetworkPolicy", func() {
	var service *v1.Service
	var podServer *v1.Pod
	f := framework.NewDefaultFramework("network-policy")

	Context("Single-port servers", func() {
		BeforeEach(func() {
			framework.Logf("Creating a simple server.")
			podServer, service = createServerPodAndService(f, f.Namespace, "server", []int{80})
			framework.Logf("Waiting for Server to come up.")
			if err := framework.WaitForPodRunningInNamespace(f.ClientSet, podServer); err != nil {
				framework.Failf("Base server pod didn't come up: %v", err)
			}

			// Create a pod with name 'client-can-connect', which should be able to communicate with server.
			framework.Logf("Ensuring base connectivity by creating a client which will be able to contact the server (since no policies are present.)")
			if err := ensureCanConnect(f, f.Namespace, "base-client-can-connect", service, 80); err != nil {
				framework.Failf("Base connectivity connection failed: %v", err)
			}
		})

		AfterEach(func() {
			cleanupServerPodAndService(f, podServer, service)
		})

		It("should support a 'default-deny' policy [Feature:NetworkPolicy]", func() {
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "deny-all",
				},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []networking.NetworkPolicyIngressRule{},
				},
			}

			policy, err := f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred(), "Test Panic: could not create NetworkPolicy.")
			defer cleanupNetworkPolicy(f, policy)

			// Create a pod with name 'client-cannot-connect', which will attempt to comunicate with the server,
			// but should not be able to now that isolation is on.
			err = ensureCanConnect(f, f.Namespace, "client-cannot-connect", service, 80)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should enforce policy based on PodSelector [Feature:NetworkPolicy]", func() {
			By("Creating a network policy for the server which allows traffic from the pod 'client-a'.")
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-a-via-pod-selector",
				},
				Spec: networking.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServer.Name,
						},
					},
					// Allow traffic only from client-a
					Ingress: []networking.NetworkPolicyIngressRule{{
						From: []networking.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-a",
								},
							},
						}},
					}},
				},
			}

			policy, err := f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy)

			By("Creating client-a which should be able to contact the server.")
			err = ensureCanConnect(f, f.Namespace, "client-a", service, 80)
			Expect(err).NotTo(HaveOccurred(), "client-a unexpectedly unable to connect to server")
			err = ensureCanConnect(f, f.Namespace, "client-b", service, 80)
			Expect(err).To(HaveOccurred(), "client-b unexpectedly able to connect to server")
		})

		It("should enforce policy based on NamespaceSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			// The CreateNamespace helper uses the input name as a Name Generator, so the namespace itself
			// will have a different name than what we are setting as the value of ns-name.
			// This is fine as long as we don't try to match the label as nsB.Name in our policy.
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			Expect(err).NotTo(HaveOccurred(), "Test panic: couldn't create additional namespace")

			// Create Server with Service in NS-B
			framework.Logf("Waiting for server-b to come up.")
			err = framework.WaitForPodRunningInNamespace(f.ClientSet, podServer)
			Expect(err).NotTo(HaveOccurred(), "Test panic: server-b did not come up.")

			// Create Policy for that service that allows traffic only via namespace B
			By("Creating a network policy for the server which allows traffic from namespace-b.")
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ns-b-via-namespace-selector",
				},
				Spec: networking.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServer.Name,
						},
					},
					// Allow traffic only from NS-B
					Ingress: []networking.NetworkPolicyIngressRule{{
						From: []networking.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
						}},
					}},
				},
			}
			policy, err = f.InternalClientset.Networking().NetworkPolicies(nsA.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy)

			err = ensureCanConnect(f, nsA, "client-a", service, 80)
			Expect(err).To(HaveOccurred(), "client-a unexpectedly unable to connect to server-b")

			err = ensureCanConnect(f, nsB, "client-b", service, 80)
			Expect(err).To(HaveOccurred(), "client-b unexpectedly unable to connect to server-b")
		})
	})

	Context("Multi-port servers", func() {
		BeforeEach(func() {
			// Create Server with Service
			By("Creating a simple server.")
			podServer, service = createServerPodAndService(f, f.Namespace, "server", []int{80, 81})
			framework.Logf("Waiting for Server to come up.")
			err := framework.WaitForPodRunningInNamespace(f.ClientSet, podServer)
			Expect(err).NotTo(HaveOccurred())

			By("Testing pods can connect to both ports when no policy is present.")
			err = ensureCanConnect(f, f.Namespace, "basecase-reachable-80", service, 80)
			testCanConnect(f, f.Namespace, "basecase-reachable-81", service, 81)
		})
		AfterEach(func() {
			cleanupServerPodAndService(f, podServer, service)
		})
		It("should enforce policy based on Ports [Feature:NetworkPolicy]", func() {
			By("Creating a network policy for the Service which allows traffic only to one port.")
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-81",
				},
				Spec: networking.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServer.Name,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networking.NetworkPolicyIngressRule{{
						Ports: []networking.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 81},
						}},
					}},
				},
			}
			policy, err := f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy)

			By("Testing pods can connect only to the port allowed by the policy.")
			testCannotConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})

		It("should enforce multiple, stacked policies with overlapping podSelectors [Feature:NetworkPolicy]", func() {
			By("Creating a network policy for the Service which allows traffic only to one port.")
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-80",
				},
				Spec: networking.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServer.Name,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networking.NetworkPolicyIngressRule{{
						Ports: []networking.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 80},
						}},
					}},
				},
			}
			policy, err := f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy)

			By("Creating a network policy for the Service which allows traffic only to another port.")
			policy2 := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-81",
				},
				Spec: networking.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServer.Name,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networking.NetworkPolicyIngressRule{{
						Ports: []networking.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 81},
						}},
					}},
				},
			}
			policy2, err = f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy2)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy2)

			By("Testing pods can connect to both ports when both policies are present.")
			testCanConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})

		It("should support allow-all policy [Feature:NetworkPolicy]", func() {
			By("Creating a network policy which allows all traffic.")
			policy := &networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-all",
				},
				Spec: networking.NetworkPolicySpec{
					// Allow all traffic
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{},
					},
					Ingress: []networking.NetworkPolicyIngressRule{{}},
				},
			}
			policy, err := f.InternalClientset.Networking().NetworkPolicies(f.Namespace.Name).Create(policy)
			Expect(err).NotTo(HaveOccurred())
			defer cleanupNetworkPolicy(f, policy)

			By("Testing pods can connect to both ports when an 'allow-all' policy is present.")
			testCanConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})
	})
})

func ensureCanConnect(f *framework.Framework, ns *v1.Namespace, podName string, service *v1.Service, targetPort int) error {
	By(fmt.Sprintf("Creating client pod %s that should successfully connect to %s.", podName, service.Name))
	podClient := createNetworkClientPod(f, ns, podName, service, targetPort)
	defer func() {
		By(fmt.Sprintf("Cleaning up the pod %s", podName))
		if err := f.ClientSet.Core().Pods(ns.Name).Delete(podClient.Name, nil); err != nil {
			framework.Failf("unable to cleanup pod %v: %v", podClient.Name, err)
		}
	}()

	framework.Logf("Waiting for %s to complete.", podClient.Name)
	err := framework.WaitForPodNoLongerRunningInNamespace(f.ClientSet, podClient.Name, ns.Name)
	Expect(err).NotTo(HaveOccurred(), "Test panic: Client Pod did not complete as expected.")

	framework.Logf("Waiting for %s to complete.", podClient.Name)
	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, podClient.Name, ns.Name)
	if err != nil {
		logs, logErr := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, fmt.Sprintf("%s-container", podName))
		if logErr != nil {
			return fmt.Errorf("Client connection failed: %v. Unable to gather pod logs: %v", err, logErr)
		}
		return fmt.Errorf("Client connection failed: %v. Client logs: %s", err, logs)
	}
}

// Create a server pod with a listening container for each port in ports[].
// Will also assign a pod label with key: "pod-name" and label set to the given podname for later use by the network
// policy.
func createServerPodAndService(f *framework.Framework, namespace *v1.Namespace, podName string, ports []int) (*v1.Pod, *v1.Service) {
	// Because we have a variable amount of ports, we'll first loop through and generate our Containers for our pod,
	// and ServicePorts.for our Service.
	containers := []v1.Container{}
	servicePorts := []v1.ServicePort{}
	for _, port := range ports {
		// Build the containers for the server pod.
		containers = append(containers, v1.Container{
			Name:  fmt.Sprintf("%s-container-%d", podName, port),
			Image: "gcr.io/google_containers/porter:4524579c0eb935c056c8e75563b4e1eda31587e0",
			Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port)}},
			Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
		})

		// Build the Service Ports for the service.
		servicePorts = append(servicePorts, v1.ServicePort{
			Name:       fmt.Sprintf("%s-%d", podName, port),
			Port:       int32(port),
			TargetPort: intstr.FromInt(port),
		})
	}

	By(fmt.Sprintf("Creating a server pod %s in namespace %s", podName, namespace.Name))
	pod, err := f.ClientSet.Core().Pods(namespace.Name).Create(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers:    containers,
			RestartPolicy: v1.RestartPolicyNever,
		},
	})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created pod %v", pod.ObjectMeta.Name)

	svcName := fmt.Sprintf("svc-%s", podName)
	By(fmt.Sprintf("Creating a service %s for pod %s in namespace %s", svcName, podName, namespace.Name))
	svc, err := f.ClientSet.Core().Services(namespace.Name).Create(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: svcName,
		},
		Spec: v1.ServiceSpec{
			Ports: servicePorts,
			Selector: map[string]string{
				"pod-name": podName,
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created service %s", svc.Name)

	return pod, svc
}

func cleanupServerPodAndService(f *framework.Framework, pod *v1.Pod, service *v1.Service) {
	By("Cleaning up the server.")
	if err := f.ClientSet.Core().Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
		framework.Failf("unable to cleanup pod %v: %v", pod.Name, err)
	}
	By("Cleaning up the server's service.")
	if err := f.ClientSet.Core().Services(service.Namespace).Delete(service.Name, nil); err != nil {
		framework.Failf("unable to cleanup svc %v: %v", service.Name, err)
	}
}

// Create a client pod which will attempt a netcat to the provided service, on the specified port.
// This client will attempt a oneshot connection, then die, without restarting the pod.
// Test can then be asserted based on whether the pod quit with an error or not.
func createNetworkClientPod(f *framework.Framework, namespace *v1.Namespace, podName string, targetService *v1.Service, targetPort int) *v1.Pod {
	pod, err := f.ClientSet.Core().Pods(namespace.Name).Create(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  fmt.Sprintf("%s-container", podName),
					Image: "gcr.io/google_containers/busybox:1.24",
					Args: []string{
						"/bin/wget",
						"-T", "8",
						fmt.Sprintf("%s.%s:%d", targetService.Name, targetService.Namespace, targetPort),
						"-O",
						"-",
					},
				},
			},
		},
	})

	Expect(err).NotTo(HaveOccurred())
	return pod
}

func cleanupNetworkPolicy(f *framework.Framework, policy *networking.NetworkPolicy) {
	By("Cleaning up the policy.")
	if err := f.InternalClientset.Networking().NetworkPolicies(policy.Namespace).Delete(policy.Name, nil); err != nil {
		framework.Failf("unable to cleanup policy %v: %v", policy.Name, err)
	}
}
