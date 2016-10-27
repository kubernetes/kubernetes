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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	"fmt"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

/*
The following Network Policy tests verify that policy object definitions
are correctly enforced by a networking plugin. It accomplishes this by launching
a simple netcat server, and (in these tests) two clients with different
attributes. Each test case creates a network policy which should only allow
connections from one of the clients. The test then asserts that the clients
failed or succesfully connected as expected.
*/

const (
	listeningPort1 = 80
	listeningPort2 = 81
	serverName     = "server"
	clientAName    = "client-a"
	clientBName    = "client-b"
)

var _ = framework.KubeDescribe("NetworkPolicy", func() {
	f := framework.NewDefaultFramework("network-policy")

	It("should support setting DefaultDeny namespace policy [Feature:NetworkPolicy]", func() {
		ns := f.Namespace

		By("Create a simple server.")
		podServer, service := createServerPodAndService(f, ns, serverName, []int{listeningPort1})
		defer func() {
			By("Cleaning up the server.")
			if err := f.Client.Pods(ns.Name).Delete(podServer.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podServer.Name, err)
			}
		}()
		defer func() {
			By("Cleaning up the server's service.")
			if err := f.Client.Services(ns.Name).Delete(service.Name); err != nil {
				framework.Failf("unable to delete svc %v: %v", service.Name, err)
			}
		}()
		framework.Logf("Waiting for Server to come up.")
		err := framework.WaitForPodRunningInNamespace(f.Client, podServer)
		Expect(err).NotTo(HaveOccurred())

		// Create a pod with name 'client-a', which should be able to communicate with server.
		By("Creating client-a which will be able to contact the server since isolation is off.")
		podClientA := createClientPod(f, ns, clientAName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod client-a")
			if err := f.Client.Pods(ns.Name).Delete(podClientA.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientA.Name, err)
			}
		}()

		framework.Logf("Waiting for client-a to to come up.")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientA)
		Expect(err).NotTo(HaveOccurred(), "waiting for client-a to run")
		framework.Logf("Waiting for client-a to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientA.Name, ns.Name)
		Expect(err).NotTo(HaveOccurred(), "checking client-a could communicate with server.")

		framework.Logf("Enabling network isolation.")
		setNamespaceIsolation(f, ns, "DefaultDeny")

		// Create a pod with name 'client-b', which will attempt to comunicate with the server,
		// but should not be able to now that isolation is on.
		By("Creating client-b which should *not* be able to contact the server.")
		podClientB := createClientPod(f, ns, clientBName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod client-b.")
			if err := f.Client.Pods(ns.Name).Delete(podClientB.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientB.Name, err)
			}
		}()

		framework.Logf("Waiting for client-b to come up.")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientB)
		Expect(err).NotTo(HaveOccurred(), "waiting for client-b to come up.")

		framework.Logf("Waiting for client-b to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientB.Name, ns.Name)
		Expect(err).To(HaveOccurred(), "checking client-b could not communicate with server")
	})

	It("should enforce policy based on PodSelector [Feature:NetworkPolicy]", func() {
		ns := f.Namespace
		setNamespaceIsolation(f, ns, "DefaultDeny")

		By("Creating a simple server.")
		serverPod, service := createServerPodAndService(f, ns, serverName, []int{listeningPort1})
		defer func() {
			By("Cleaning up the server.")
			if err := f.Client.Pods(ns.Name).Delete(serverPod.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", serverPod.Name, err)
			}
		}()
		defer func() {
			By("Cleaning up the server's service.")
			if err := f.Client.Services(ns.Name).Delete(service.Name); err != nil {
				framework.Failf("unable to delete svc %v: %v", service.Name, err)
			}
		}()
		framework.Logf("Waiting for Server to come up.")
		err := framework.WaitForPodRunningInNamespace(f.Client, serverPod)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a network policy for the server which allows traffic from the pod 'client-a'.")
		policy, err := f.Client.NetworkPolicies(ns.Name).Create(&extensions.NetworkPolicy{
			ObjectMeta: api.ObjectMeta{
				Name: "allow-client-a-via-pod-selector",
			},
			Spec: extensions.NetworkPolicySpec{
				// Apply this policy to the Server
				PodSelector: unversioned.LabelSelector{
					MatchLabels: map[string]string{
						"pod-name": serverPod.Name,
					},
				},
				// Allow traffic only from client-a
				Ingress: []extensions.NetworkPolicyIngressRule{{
					From: []extensions.NetworkPolicyPeer{{
						PodSelector: &unversioned.LabelSelector{
							MatchLabels: map[string]string{
								"pod-name": clientAName,
							},
						},
					}},
				}},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("Cleaning up the policy.")
			if err := f.Client.NetworkPolicies(ns.Name).Delete(policy.Name, nil); err != nil {
				framework.Failf("unable to delete policy %v: %v", policy.Name, err)
			}
		}()

		By("Creating client-a which should be able to contact the server.")
		podClientA := createClientPod(f, ns, clientAName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod client-a")
			if err := f.Client.Pods(ns.Name).Delete(podClientA.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientA.Name, err)
			}
		}()

		// Create a pod with name 'client-b', which will attempt to comunicate with the server,
		// but should not be able to due to the label-based policy.
		By("Creating client-b which should *not* be able to contact the server.")
		podClientB := createClientPod(f, ns, clientBName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod Server A")
			if err := f.Client.Pods(ns.Name).Delete(podClientB.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientB.Name, err)
			}
		}()

		framework.Logf("Waiting for both clients to run.")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientA)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-A to run")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientB)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-B to run")

		framework.Logf("Waiting for client-a to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientA.Name, ns.Name)
		Expect(err).NotTo(HaveOccurred(), "checking client-a could communicate with server.")

		framework.Logf("Waiting for client-b to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientB.Name, ns.Name)
		Expect(err).To(HaveOccurred(), "checking client-b could not communicate with server")
	})

	It("should enforce policy based on Ports [Feature:NetworkPolicy]", func() {
		ns := f.Namespace
		setNamespaceIsolation(f, ns, "DefaultDeny")

		// Create Server with Service
		By("Creating a simple server.")
		serverPod, service := createServerPodAndService(f, ns, serverName, []int{listeningPort1, listeningPort2})
		defer func() {
			By("Cleaning up the server.")
			if err := f.Client.Pods(ns.Name).Delete(serverPod.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", serverPod.Name, err)
			}
		}()
		defer func() {
			By("Cleaning up the server's service.")
			if err := f.Client.Services(ns.Name).Delete(service.Name); err != nil {
				framework.Failf("unable to delete svc %v: %v", service.Name, err)
			}
		}()
		framework.Logf("Waiting for Server to come up.")
		err := framework.WaitForPodRunningInNamespace(f.Client, serverPod)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a network policy for the Service which allows traffic only to one port.")
		policy, err := f.Client.NetworkPolicies(ns.Name).Create(&extensions.NetworkPolicy{
			ObjectMeta: api.ObjectMeta{
				Name: fmt.Sprintf("allow-ingress-on-port-%d", listeningPort1),
			},
			Spec: extensions.NetworkPolicySpec{
				// Apply to server
				PodSelector: unversioned.LabelSelector{
					MatchLabels: map[string]string{
						"pod-name": serverPod.Name,
					},
				},
				// Allow traffic only to one port.
				Ingress: []extensions.NetworkPolicyIngressRule{{
					Ports: []extensions.NetworkPolicyPort{{
						Port: &intstr.IntOrString{IntVal: listeningPort1},
					}},
				}},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("Cleaning up the policy")
			if err := f.Client.NetworkPolicies(ns.Name).Delete(policy.Name, nil); err != nil {
				framework.Failf("unable to delete policy %v: %v", policy.Name, err)
			}
		}()

		By("Creating client-a that should succesfully connect to the opened port.")
		podClientA := createClientPod(f, ns, clientAName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod Server A")
			if err := f.Client.Pods(ns.Name).Delete(podClientA.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientA.Name, err)
			}
		}()

		By("Creating a client-b that should not succesfully connect to the closed port.")
		podClientB := createClientPod(f, ns, clientBName, service, listeningPort2)
		defer func() {
			By("Cleaning up client-b.")
			if err := f.Client.Pods(ns.Name).Delete(podClientB.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientB.Name, err)
			}
		}()

		framework.Logf("Waiting for both clients to run.")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientA)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-A to run")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientB)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-B to run")

		framework.Logf("Waiting for client-a to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientA.Name, ns.Name)
		Expect(err).NotTo(HaveOccurred(), "checking Client-A could communicate with server.")

		framework.Logf("Waiting for client-b to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientB.Name, ns.Name)
		Expect(err).To(HaveOccurred(), "checking Client-B could not communicate with server")
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
		Expect(err).NotTo(HaveOccurred())
		setNamespaceIsolation(f, nsA, "DefaultDeny")

		// Create Server with Service in NS-B
		By("Creating a webserver tied to a service.")
		serverPod, service := createServerPodAndService(f, nsA, serverName, []int{listeningPort1})
		defer func() {
			By("Cleaning up the server.")
			if err := f.Client.Pods(nsA.Name).Delete(serverPod.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", serverPod.Name, err)
			}
		}()
		defer func() {
			By("Cleaning up the server's service.")
			if err := f.Client.Services(nsA.Name).Delete(service.Name); err != nil {
				framework.Failf("unable to delete svc %v: %v", service.Name, err)
			}
		}()
		framework.Logf("Waiting for server to come up.")
		err = framework.WaitForPodRunningInNamespace(f.Client, serverPod)
		Expect(err).NotTo(HaveOccurred())

		// Create Policy for that service that allows traffic only via namespace B
		By("Creating a network policy for the server which allows traffic from namespace-b.")
		policy, err := f.Client.NetworkPolicies(nsA.Name).Create(&extensions.NetworkPolicy{
			ObjectMeta: api.ObjectMeta{
				Name: "allow-ns-b-via-namespace-selector",
			},
			Spec: extensions.NetworkPolicySpec{
				// Apply to server
				PodSelector: unversioned.LabelSelector{
					MatchLabels: map[string]string{
						"pod-name": serverPod.Name,
					},
				},
				// Allow traffic only from NS-B
				Ingress: []extensions.NetworkPolicyIngressRule{{
					From: []extensions.NetworkPolicyPeer{{
						NamespaceSelector: &unversioned.LabelSelector{
							MatchLabels: map[string]string{
								"ns-name": nsBName,
							},
						},
					}},
				}},
			},
		})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			By("Cleaning up the policy")
			if err := f.Client.NetworkPolicies(nsA.Name).Delete(policy.Name, nil); err != nil {
				framework.Failf("unable to delete policy %v: %v", policy.Name, err)
			}
		}()

		// Create a pod with name 'client-a', which should be able to communicate with server.
		By("Creating client-a in ns-a which should *not* be able to contact the server.")
		podClientA := createClientPod(f, nsA, clientAName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod client-a")
			if err := f.Client.Pods(nsA.Name).Delete(podClientA.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientA.Name, err)
			}
		}()

		// Create a pod with name 'client-b', which will attempt to comunicate with the server,
		// but should not be able to do to the label-based policy.
		By("Creating client-b in ns-b which should be able to contact the server.")
		podClientB := createClientPod(f, nsB, clientBName, service, listeningPort1)
		defer func() {
			By("Cleaning up the pod client B")
			if err := f.Client.Pods(nsB.Name).Delete(podClientB.Name, nil); err != nil {
				framework.Failf("unable to delete pod %v: %v", podClientB.Name, err)
			}
		}()

		framework.Logf("Waiting for both clients to run.")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientA)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-A to run")
		err = framework.WaitForPodRunningInNamespace(f.Client, podClientB)
		Expect(err).NotTo(HaveOccurred(), "waiting for Client-B to run")

		framework.Logf("Waiting for client-a to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientA.Name, nsA.Name)
		Expect(err).To(HaveOccurred(), "checking Client-A could not communicate with server.")

		framework.Logf("Waiting for client-b to complete.")
		err = framework.WaitForPodSuccessInNamespace(f.Client, podClientB.Name, nsB.Name)
		Expect(err).NotTo(HaveOccurred(), "checking Client-B could communicate with server")
	})
})

// Create a server pod with a listening container for each port in ports[].
// Will also assign a pod label with key: "pod-name" and label set to the given podname for later use by the network
// policy.
func createServerPodAndService(f *framework.Framework, namespace *api.Namespace, podName string, ports []int) (*api.Pod, *api.Service) {
	// Because we have a variable amount of ports, we'll first loop through and generate our Containers for our pod,
	// and ServicePorts.for our Service.
	containers := []api.Container{}
	servicePorts := []api.ServicePort{}
	for _, port := range ports {
		// Build the containers for the server pod.
		containers = append(containers, api.Container{
			Name:  fmt.Sprintf("%s-container-%d", podName, port),
			Image: "gcr.io/google_containers/redis:e2e",
			Args: []string{
				"/bin/sh",
				"-c",
				fmt.Sprintf("/bin/nc -kl %d", port),
			},
			Ports: []api.ContainerPort{{ContainerPort: int32(port)}},
		})

		// Build the Service Ports for the service.
		servicePorts = append(servicePorts, api.ServicePort{
			Name:       fmt.Sprintf("%s-%d", podName, port),
			Port:       int32(port),
			TargetPort: intstr.FromInt(port),
		})
	}

	By(fmt.Sprintf("Creating a server pod %s in namespace %s", podName, namespace.Name))
	pod, err := f.Client.Pods(namespace.Name).Create(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: api.PodSpec{
			Containers:    containers,
			RestartPolicy: api.RestartPolicyNever,
		},
	})
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Created pod %v", pod.ObjectMeta.Name)

	svcName := fmt.Sprintf("svc-%s", podName)
	By(fmt.Sprintf("Creating a service %s for pod %s in namespace %s", svcName, podName, namespace.Name))
	svc, err := f.Client.Services(namespace.Name).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: svcName,
		},
		Spec: api.ServiceSpec{
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

// Create a client pod which will attempt a netcat to the provided service, on the specified port.
// This client will attempt a oneshot connection, then die, without restarting the pod.
// Test can then be asserted based on whether the pod quit with an error or not.
func createClientPod(f *framework.Framework, namespace *api.Namespace, podName string, targetService *api.Service, targetPort int) *api.Pod {
	pod, err := f.Client.Pods(namespace.Name).Create(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Name:  fmt.Sprintf("%s-container", podName),
					Image: "gcr.io/google_containers/redis:e2e",
					Args: []string{
						"/bin/sh",
						"-c",
						fmt.Sprintf("/usr/bin/printf dummy-data | /bin/nc -w 30 %s.%s %d", targetService.Name, targetService.Namespace, targetPort),
					},
				},
			},
		},
	})

	Expect(err).NotTo(HaveOccurred())
	return pod
}

// Configure namespace network isolation by setting the network-policy annotation
// on the namespace.
func setNamespaceIsolation(f *framework.Framework, namespace *api.Namespace, ingressIsolation string) {
	var annotations = map[string]string{}
	if ingressIsolation != "" {
		By(fmt.Sprintf("Enabling isolation through namespace annotations on namespace %v", namespace.Name))
		policy := fmt.Sprintf(`{"ingress":{"isolation":"%s"}}`, ingressIsolation)
		annotations["net.beta.kubernetes.io/network-policy"] = policy
	} else {
		By(fmt.Sprintf("Disabling isolation through namespace annotations on namespace %v", namespace.Name))
		delete(annotations, "net.beta.kubernetes.io/network-policy")
	}

	// Update the namespace.  We set the resource version to be an empty
	// string, this forces the update.  If we weren't to do this, we would
	// either need to re-query the namespace, or update the namespace
	// references with the one returned by the update.  This approach
	// requires less plumbing.
	namespace.ObjectMeta.Annotations = annotations
	namespace.ObjectMeta.ResourceVersion = ""
	_, err := f.Client.Namespaces().Update(namespace)
	Expect(err).NotTo(HaveOccurred())
}
