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
	"encoding/json"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"fmt"

	"github.com/onsi/ginkgo"
)

/*
The following Network Policy tests verify that policy object definitions
are correctly enforced by a networking plugin. It accomplishes this by launching
a simple netcat server, and two clients with different
attributes. Each test case creates a network policy which should only allow
connections from one of the clients. The test then asserts that the clients
failed or successfully connected as expected.
*/

var _ = SIGDescribe("NetworkPolicy [LinuxOnly]", func() {
	var service *v1.Service
	var podServer *v1.Pod
	var podServerLabelSelector string
	f := framework.NewDefaultFramework("network-policy")

	ginkgo.BeforeEach(func() {
		// Windows does not support network policies.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
	})

	ginkgo.Context("NetworkPolicy between server and client", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("Creating a simple server that serves on port 80 and 81.")
			podServer, service = createServerPodAndService(f, f.Namespace, "server", []int{80, 81})

			ginkgo.By("Waiting for pod ready", func() {
				err := f.WaitForPodReady(podServer.Name)
				framework.ExpectNoError(err)
			})

			// podServerLabelSelector holds the value for the podServer's label "pod-name".
			podServerLabelSelector = podServer.ObjectMeta.Labels["pod-name"]

			// Create pods, which should be able to communicate with the server on port 80 and 81.
			ginkgo.By("Testing pods can connect to both ports when no policy is present.")
			testCanConnect(f, f.Namespace, "client-can-connect-80", service, 80)
			testCanConnect(f, f.Namespace, "client-can-connect-81", service, 81)
		})

		ginkgo.AfterEach(func() {
			cleanupServerPodAndService(f, podServer, service)
		})

		ginkgo.It("should support a 'default-deny' policy [Feature:NetworkPolicy]", func() {
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "deny-all",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []networkingv1.NetworkPolicyIngressRule{},
				},
			}

			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			// Create a pod with name 'client-cannot-connect', which will attempt to communicate with the server,
			// but should not be able to now that isolation is on.
			testCannotConnect(f, f.Namespace, "client-cannot-connect", service, 80)
		})

		ginkgo.It("should enforce policy to allow traffic from pods within server namespace based on PodSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error occurred while creating namespace-b.")

			// All communication should be possible before applying the policy.
			ginkgo.By("Creating client-a, in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsA, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsA, "client-b", service, 80)
			})
			ginkgo.By("Creating client-a, not in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsB, "client-a", service, 80)
			})

			ginkgo.By("Creating a network policy for the server which allows traffic from the pod 'client-a' in same namespace.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-a-via-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only from client-a
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-a",
								},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating client-a, in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsA, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, in server's namespace, which should not be able to contact the server.", func() {
				testCannotConnect(f, nsA, "client-b", service, 80)
			})
			ginkgo.By("Creating client-a, not in server's namespace, which should not be able to contact the server.", func() {
				testCannotConnect(f, nsB, "client-a", service, 80)
			})
		})

		ginkgo.It("should enforce policy to allow traffic only from a different namespace, based on NamespaceSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err)

			// Create Server with Service in NS-B
			framework.Logf("Waiting for server to come up.")
			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, podServer)
			framework.ExpectNoError(err)

			// Create Policy for that service that allows traffic only via namespace B
			ginkgo.By("Creating a network policy for the server which allows traffic from namespace-b.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ns-b-via-namespace-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only from NS-B
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
						}},
					}},
				},
			}
			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(nsA.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, nsA, "client-a", service, 80)
			testCanConnect(f, nsB, "client-b", service, 80)
		})

		ginkgo.It("should enforce policy based on PodSelector with MatchExpressions[Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy for the server which allows traffic from the pod 'client-a'.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-a-via-pod-selector-with-match-expressions",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "pod-name",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"client-a"},
								}},
							},
						}},
					}},
				},
			}

			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b which should not be able to contact the server.", func() {
				testCannotConnect(f, f.Namespace, "client-b", service, 80)
			})
		})

		ginkgo.It("should enforce policy based on NamespaceSelector with MatchExpressions[Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsBName, err)

			nsCName := f.BaseName + "-c"
			nsC, err := f.CreateNamespace(nsCName, map[string]string{
				"ns-name": nsCName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsCName, err)

			// Create Policy for the server that allows traffic from namespace different than namespace-a
			ginkgo.By("Creating a network policy for the server which allows traffic from ns different than namespace-a.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-any-ns-different-than-ns-a-via-ns-selector-with-match-expressions",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "ns-name",
									Operator: metav1.LabelSelectorOpNotIn,
									Values:   []string{nsCName},
								}},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(nsA.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, nsC, "client-a", service, 80)
			testCanConnect(f, nsB, "client-a", service, 80)
		})

		ginkgo.It("should enforce policy based on PodSelector or NamespaceSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsBName, err)

			// Create Policy for the server that allows traffic only via client B or namespace B
			ginkgo.By("Creating a network policy for the server which allows traffic from client-b or namespace-b.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ns-b-via-namespace-selector-or-client-b-via-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-b",
								},
							},
						}, {
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(nsA.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCanConnect(f, nsB, "client-a", service, 80)
			testCanConnect(f, nsA, "client-b", service, 80)
			testCannotConnect(f, nsA, "client-c", service, 80)
		})

		ginkgo.It("should enforce policy based on PodSelector and NamespaceSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsBName, err)

			// Create Policy for the server that allows traffic only via client-b in namespace B
			ginkgo.By("Creating a network policy for the server which allows traffic from client-b in namespace-b.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-b-in-ns-b-via-ns-selector-and-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-b",
								},
							},
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(nsA.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, nsB, "client-a", service, 80)
			testCannotConnect(f, nsA, "client-b", service, 80)
			testCanConnect(f, nsB, "client-b", service, 80)
		})

		ginkgo.It("should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error occurred while creating namespace-b.")

			// Wait for Server in namespaces-a to be ready
			framework.Logf("Waiting for server to come up.")
			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, podServer)
			framework.ExpectNoError(err, "Error occurred while waiting for pod status in namespace: Running.")

			// Before application of the policy, all communication should be successful.
			ginkgo.By("Creating client-a, in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsA, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsA, "client-b", service, 80)
			})
			ginkgo.By("Creating client-a, not in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsB, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, not in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsB, "client-b", service, 80)
			})

			ginkgo.By("Creating a network policy for the server which allows traffic only from client-a in namespace-b.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: nsA.Name,
					Name:      "allow-ns-b-client-a-via-namespace-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only from client-a in namespace-b
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-a",
								},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err, "Error occurred while creating policy: policy.")
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating client-a, in server's namespace, which should not be able to contact the server.", func() {
				testCannotConnect(f, nsA, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, in server's namespace, which should not be able to contact the server.", func() {
				testCannotConnect(f, nsA, "client-b", service, 80)
			})
			ginkgo.By("Creating client-a, not in server's namespace, which should be able to contact the server.", func() {
				testCanConnect(f, nsB, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b, not in server's namespace, which should not be able to contact the server.", func() {
				testCannotConnect(f, nsB, "client-b", service, 80)
			})
		})

		ginkgo.It("should enforce policy based on Ports [Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy for the Service which allows traffic only to one port.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-81",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 81},
						}},
					}},
				},
			}
			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Testing pods can connect only to the port allowed by the policy.")
			testCannotConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})

		ginkgo.It("should enforce multiple, stacked policies with overlapping podSelectors [Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy for the Service which allows traffic only to one port.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-80",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 80},
						}},
					}},
				},
			}
			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating a network policy for the Service which allows traffic only to another port.")
			policy2 := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress-on-port-81",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: 81},
						}},
					}},
				},
			}
			policy2, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy2)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy2)

			ginkgo.By("Testing pods can connect to both ports when both policies are present.")
			testCanConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})

		ginkgo.It("should support allow-all policy [Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy which allows all traffic.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-all",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Allow all traffic
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{}},
				},
			}
			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Testing pods can connect to both ports when an 'allow-all' policy is present.")
			testCanConnect(f, f.Namespace, "client-a", service, 80)
			testCanConnect(f, f.Namespace, "client-b", service, 81)
		})

		ginkgo.It("should allow ingress access on one named port [Feature:NetworkPolicy]", func() {
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-a-via-named-port-ingress-rule",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic to only one named port: "serve-80".
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80"},
						}},
					}},
				},
			}

			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b which should not be able to contact the server on port 81.", func() {
				testCannotConnect(f, f.Namespace, "client-b", service, 81)
			})
		})

		ginkgo.It("should allow ingress access from namespace on one named port [Feature:NetworkPolicy]", func() {
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsBName, err)

			const allowedPort = 80
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-in-ns-b-via-named-port-ingress-rule",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic to only one named port: "serve-80" from namespace-b.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": nsBName,
								},
							},
						}},
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80"},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, f.Namespace, "client-a", service, allowedPort)
			testCanConnect(f, nsB, "client-b", service, allowedPort)
		})

		ginkgo.It("should allow egress access on one named port [Feature:NetworkPolicy]", func() {
			clientPodName := "client-a"
			protocolUDP := v1.ProtocolUDP
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-client-a-via-named-port-egress-rule",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to client-a
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": clientPodName,
						},
					},
					// Allow traffic to only one named port: "serve-80".
					Egress: []networkingv1.NetworkPolicyEgressRule{{
						Ports: []networkingv1.NetworkPolicyPort{
							{
								Port: &intstr.IntOrString{Type: intstr.String, StrVal: "serve-80"},
							},
							// Allow DNS look-ups
							{
								Protocol: &protocolUDP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 53},
							},
						},
					}},
				},
			}

			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, clientPodName, service, 80)
			})
			ginkgo.By("Creating client-a which should not be able to contact the server on port 81.", func() {
				testCannotConnect(f, f.Namespace, clientPodName, service, 81)
			})
		})

		ginkgo.It("should enforce updated policy [Feature:NetworkPolicy]", func() {
			const (
				clientAAllowedPort    = 80
				clientANotAllowedPort = 81
			)
			ginkgo.By("Creating a network policy for the Service which allows traffic from pod at a port")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-a",
								},
							},
						}},
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: clientAAllowedPort},
						}},
					}},
				},
			}
			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)

			testCanConnect(f, f.Namespace, "client-a", service, clientAAllowedPort)
			err = f.WaitForPodNotFound("client-a", framework.PodDeleteTimeout)
			framework.ExpectNoError(err, "Expected pod to be not found.")

			testCannotConnect(f, f.Namespace, "client-b", service, clientAAllowedPort)
			err = f.WaitForPodNotFound("client-b", framework.PodDeleteTimeout)
			framework.ExpectNoError(err, "Expected pod to be not found.")

			testCannotConnect(f, f.Namespace, "client-a", service, clientANotAllowedPort)
			err = f.WaitForPodNotFound("client-a", framework.PodDeleteTimeout)
			framework.ExpectNoError(err, "Expected pod to be not found.")

			const (
				clientBAllowedPort    = 81
				clientBNotAllowedPort = 80
			)
			ginkgo.By("Updating a network policy for the Service which allows traffic from another pod at another port.")
			policy = &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ingress",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply to server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					// Allow traffic only to one port.
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-b",
								},
							},
						}},
						Ports: []networkingv1.NetworkPolicyPort{{
							Port: &intstr.IntOrString{IntVal: clientBAllowedPort},
						}},
					}},
				},
			}
			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Update(policy)
			framework.ExpectNoError(err, "Error updating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, f.Namespace, "client-b", service, clientBNotAllowedPort)
			err = f.WaitForPodNotFound("client-b", framework.PodDeleteTimeout)
			framework.ExpectNoError(err, "Expected pod to be not found.")

			testCannotConnect(f, f.Namespace, "client-a", service, clientBNotAllowedPort)
			testCanConnect(f, f.Namespace, "client-b", service, clientBAllowedPort)
		})

		ginkgo.It("should allow ingress access from updated namespace [Feature:NetworkPolicy]", func() {
			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			newNsBName := nsBName + "-updated"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error creating namespace %v: %v", nsBName, err)

			const allowedPort = 80
			// Create Policy for that service that allows traffic only via namespace B
			ginkgo.By("Creating a network policy for the server which allows traffic from namespace-b.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-ns-b-via-namespace-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"ns-name": newNsBName,
								},
							},
						}},
					}},
				},
			}

			policy, err = f.ClientSet.NetworkingV1().NetworkPolicies(nsA.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			testCannotConnect(f, nsB, "client-a", service, allowedPort)

			nsB, err = f.ClientSet.CoreV1().Namespaces().Get(nsB.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Error getting Namespace %v: %v", nsB.ObjectMeta.Name, err)

			nsB.ObjectMeta.Labels["ns-name"] = newNsBName
			nsB, err = f.ClientSet.CoreV1().Namespaces().Update(nsB)
			framework.ExpectNoError(err, "Error updating Namespace %v: %v", nsB.ObjectMeta.Name, err)

			testCanConnect(f, nsB, "client-b", service, allowedPort)
		})

		ginkgo.It("should allow ingress access from updated pod [Feature:NetworkPolicy]", func() {
			const allowedPort = 80
			ginkgo.By("Creating a network policy for the server which allows traffic from client-a-updated.")
			policy := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-pod-b-via-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "pod-name",
									Operator: metav1.LabelSelectorOpDoesNotExist,
								}},
							},
						}},
					}},
				},
			}

			policy, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policy)
			framework.ExpectNoError(err, "Error creating Network Policy %v: %v", policy.ObjectMeta.Name, err)
			defer cleanupNetworkPolicy(f, policy)

			ginkgo.By(fmt.Sprintf("Creating client pod %s that should not be able to connect to %s.", "client-a", service.Name))
			podClient := createNetworkClientPod(f, f.Namespace, "client-a", service, allowedPort)
			defer func() {
				ginkgo.By(fmt.Sprintf("Cleaning up the pod %s", podClient.Name))
				if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(podClient.Name, nil); err != nil {
					framework.Failf("unable to cleanup pod %v: %v", podClient.Name, err)
				}
			}()
			checkNoConnectivity(f, f.Namespace, podClient, service)

			ginkgo.By(fmt.Sprintf("Updating client pod %s that should successfully connect to %s.", podClient.Name, service.Name))
			podClient = updateNetworkClientPodLabel(f, f.Namespace, podClient.Name, "replace", "/metadata/labels", map[string]string{})
			checkConnectivity(f, f.Namespace, podClient, service)
		})

		ginkgo.It("should enforce egress policy allowing traffic to a server in a different namespace based on PodSelector and NamespaceSelector [Feature:NetworkPolicy]", func() {
			var nsBserviceA, nsBserviceB *v1.Service
			var nsBpodServerA, nsBpodServerB *v1.Pod

			nsA := f.Namespace
			nsBName := f.BaseName + "-b"
			nsB, err := f.CreateNamespace(nsBName, map[string]string{
				"ns-name": nsBName,
			})
			framework.ExpectNoError(err, "Error occurred while creating namespace-b.")

			// Creating pods and services in namespace-b
			nsBpodServerA, nsBserviceA = createServerPodAndService(f, nsB, "ns-b-server-a", []int{80})
			nsBpodServerB, nsBserviceB = createServerPodAndService(f, nsB, "ns-b-server-b", []int{80})

			// Wait for Server with Service in NS-A to be ready
			framework.Logf("Waiting for servers to come up.")
			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, podServer)
			framework.ExpectNoError(err, "Error occurred while waiting for pod status in namespace: Running.")

			// Wait for Servers with Services in NS-B to be ready
			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, nsBpodServerA)
			framework.ExpectNoError(err, "Error occurred while waiting for pod status in namespace: Running.")

			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, nsBpodServerB)
			framework.ExpectNoError(err, "Error occurred while waiting for pod status in namespace: Running.")

			defer cleanupServerPodAndService(f, nsBpodServerA, nsBserviceA)
			defer cleanupServerPodAndService(f, nsBpodServerB, nsBserviceB)

			ginkgo.By("Creating a network policy for the server which allows traffic only to a server in different namespace.")
			protocolUDP := v1.ProtocolUDP
			policyAllowToServerInNSB := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: nsA.Name,
					Name:      "allow-to-ns-b-server-a-via-namespace-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the client
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": "client-a",
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
					// Allow traffic only to server-a in namespace-b
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							Ports: []networkingv1.NetworkPolicyPort{
								// Allow DNS look-ups
								{
									Protocol: &protocolUDP,
									Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 53},
								},
							},
						},
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{
											"ns-name": nsBName,
										},
									},
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{
											"pod-name": nsBpodServerA.ObjectMeta.Labels["pod-name"],
										},
									},
								},
							},
						},
					},
				},
			}

			policyAllowToServerInNSB, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowToServerInNSB)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowToServerInNSB.")
			defer cleanupNetworkPolicy(f, policyAllowToServerInNSB)

			ginkgo.By("Creating client-a, in 'namespace-a', which should be able to contact the server-a in namespace-b.", func() {
				testCanConnect(f, nsA, "client-a", nsBserviceA, 80)
			})
			ginkgo.By("Creating client-a, in 'namespace-a', which should not be able to contact the server-b in namespace-b.", func() {
				testCannotConnect(f, nsA, "client-a", nsBserviceB, 80)
			})
			ginkgo.By("Creating client-a, in 'namespace-a', which should not be able to contact the server in namespace-a.", func() {
				testCannotConnect(f, nsA, "client-a", service, 80)
			})
		})

		ginkgo.It("should enforce multiple ingress policies with ingress allow-all policy taking precedence [Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy for the server which allows traffic only from client-b.")
			policyAllowOnlyFromClientB := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "allow-from-client-b-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
					// Allow traffic only from "client-b"
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-b",
								},
							},
						}},
					}},
				},
			}

			policyAllowOnlyFromClientB, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowOnlyFromClientB)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowOnlyFromClientB.")
			defer cleanupNetworkPolicy(f, policyAllowOnlyFromClientB)

			ginkgo.By("Creating client-a which should not be able to contact the server.", func() {
				testCannotConnect(f, f.Namespace, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-b", service, 80)
			})

			ginkgo.By("Creating a network policy for the server which allows traffic from all clients.")
			policyIngressAllowAll := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					//Namespace: f.Namespace.Name,
					Name: "allow-all",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to all pods
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
					Ingress:     []networkingv1.NetworkPolicyIngressRule{{}},
				},
			}

			policyIngressAllowAll, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyIngressAllowAll)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyIngressAllowAll.")
			defer cleanupNetworkPolicy(f, policyIngressAllowAll)

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})
			ginkgo.By("Creating client-b which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-b", service, 80)
			})
		})

		ginkgo.It("should enforce multiple egress policies with egress allow-all policy taking precedence [Feature:NetworkPolicy]", func() {
			podServerB, serviceB := createServerPodAndService(f, f.Namespace, "server-b", []int{80})
			defer cleanupServerPodAndService(f, podServerB, serviceB)

			ginkgo.By("Waiting for pod ready", func() {
				err := f.WaitForPodReady(podServerB.Name)
				framework.ExpectNoError(err, "Error occurred while waiting for pod type: Ready.")
			})

			protocolUDP := v1.ProtocolUDP

			ginkgo.By("Creating client-a which should be able to contact the server before applying policy.", func() {
				testCanConnect(f, f.Namespace, "client-a", serviceB, 80)
			})

			ginkgo.By("Creating a network policy for the server which allows traffic only to server-a.")
			policyAllowOnlyToServerA := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "allow-to-server-a-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the "client-a"
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": "client-a",
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
					// Allow traffic only to "server-a"
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							Ports: []networkingv1.NetworkPolicyPort{
								// Allow DNS look-ups
								{
									Protocol: &protocolUDP,
									Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 53},
								},
							},
						},
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{
											"pod-name": podServerLabelSelector,
										},
									},
								},
							},
						},
					},
				},
			}
			policyAllowOnlyToServerA, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowOnlyToServerA)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowOnlyToServerA.")
			defer cleanupNetworkPolicy(f, policyAllowOnlyToServerA)

			ginkgo.By("Creating client-a which should not be able to contact the server-b.", func() {
				testCannotConnect(f, f.Namespace, "client-a", serviceB, 80)
			})
			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})

			ginkgo.By("Creating a network policy which allows traffic to all pods.")
			policyEgressAllowAll := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: "allow-all",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to all pods
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
					Egress:      []networkingv1.NetworkPolicyEgressRule{{}},
				},
			}

			policyEgressAllowAll, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyEgressAllowAll)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyEgressAllowAll.")
			defer cleanupNetworkPolicy(f, policyEgressAllowAll)

			ginkgo.By("Creating client-a which should be able to contact the server-b.", func() {
				testCanConnect(f, f.Namespace, "client-a", serviceB, 80)
			})
			ginkgo.By("Creating client-a which should be able to contact the server-a.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})
		})

		ginkgo.It("should stop enforcing policies after they are deleted [Feature:NetworkPolicy]", func() {
			ginkgo.By("Creating a network policy for the server which denies all traffic.")
			policyDenyAll := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "deny-all",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Deny all traffic
					PodSelector: metav1.LabelSelector{},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
					Ingress:     []networkingv1.NetworkPolicyIngressRule{},
				},
			}

			policyDenyAll, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyDenyAll)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyDenyAll.")

			ginkgo.By("Creating client-a which should not be able to contact the server.", func() {
				testCannotConnect(f, f.Namespace, "client-a", service, 80)
			})

			ginkgo.By("Creating a network policy for the server which allows traffic only from client-a.")
			policyAllowFromClientA := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "allow-from-client-a-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": podServerLabelSelector,
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
					// Allow traffic from "client-a"
					Ingress: []networkingv1.NetworkPolicyIngressRule{{
						From: []networkingv1.NetworkPolicyPeer{{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"pod-name": "client-a",
								},
							},
						}},
					}},
				},
			}

			policyAllowFromClientA, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowFromClientA)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowFromClientA.")

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})

			ginkgo.By("Deleting the network policy allowing traffic from client-a")
			cleanupNetworkPolicy(f, policyAllowFromClientA)

			ginkgo.By("Creating client-a which should not be able to contact the server.", func() {
				testCannotConnect(f, f.Namespace, "client-a", service, 80)
			})

			ginkgo.By("Deleting the network policy denying all traffic.")
			cleanupNetworkPolicy(f, policyDenyAll)

			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})

		})

		ginkgo.It("should allow egress access to server in CIDR block [Feature:NetworkPolicy]", func() {
			var serviceB *v1.Service
			var podServerB *v1.Pod

			protocolUDP := v1.ProtocolUDP

			// Getting podServer's status to get podServer's IP, to create the CIDR
			podServerStatus, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(podServer.Name, metav1.GetOptions{})
			if err != nil {
				framework.ExpectNoError(err, "Error occurred while getting pod status.")
			}

			podServerCIDR := fmt.Sprintf("%s/32", podServerStatus.Status.PodIP)

			// Creating pod-b and service-b
			podServerB, serviceB = createServerPodAndService(f, f.Namespace, "pod-b", []int{80})
			ginkgo.By("Waiting for pod-b to be ready", func() {
				err = f.WaitForPodReady(podServerB.Name)
				framework.ExpectNoError(err, "Error occurred while waiting for pod type: Ready.")
			})
			defer cleanupServerPodAndService(f, podServerB, serviceB)

			// Wait for podServerB with serviceB to be ready
			err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, podServerB)
			framework.ExpectNoError(err, "Error occurred while waiting for pod status in namespace: Running.")

			ginkgo.By("Creating client-a which should be able to contact the server-b.", func() {
				testCanConnect(f, f.Namespace, "client-a", serviceB, 80)
			})

			policyAllowCIDR := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "allow-client-a-via-cidr-egress-rule",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy to the Server
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": "client-a",
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
					// Allow traffic to only one CIDR block.
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							Ports: []networkingv1.NetworkPolicyPort{
								// Allow DNS look-ups
								{
									Protocol: &protocolUDP,
									Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 53},
								},
							},
						},
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									IPBlock: &networkingv1.IPBlock{
										CIDR: podServerCIDR,
									},
								},
							},
						},
					},
				},
			}

			policyAllowCIDR, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowCIDR)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowCIDR.")
			defer cleanupNetworkPolicy(f, policyAllowCIDR)

			ginkgo.By("Creating client-a which should not be able to contact the server-b.", func() {
				testCannotConnect(f, f.Namespace, "client-a", serviceB, 80)
			})
			ginkgo.By("Creating client-a which should be able to contact the server.", func() {
				testCanConnect(f, f.Namespace, "client-a", service, 80)
			})
		})

		ginkgo.It("should enforce policies to check ingress and egress policies can be controlled independently based on PodSelector [Feature:NetworkPolicy]", func() {
			var serviceA, serviceB *v1.Service
			var podA, podB *v1.Pod
			var err error

			protocolUDP := v1.ProtocolUDP

			// Before applying policy, communication should be successful between pod-a and pod-b
			podA, serviceA = createServerPodAndService(f, f.Namespace, "pod-a", []int{80})
			ginkgo.By("Waiting for pod-a to be ready", func() {
				err = f.WaitForPodReady(podA.Name)
				framework.ExpectNoError(err, "Error occurred while waiting for pod type: Ready.")
			})
			ginkgo.By("Creating client pod-b which should be able to contact the server pod-a.", func() {
				testCanConnect(f, f.Namespace, "pod-b", serviceA, 80)
			})
			cleanupServerPodAndService(f, podA, serviceA)

			podB, serviceB = createServerPodAndService(f, f.Namespace, "pod-b", []int{80})
			ginkgo.By("Waiting for pod-b to be ready", func() {
				err = f.WaitForPodReady(podB.Name)
				framework.ExpectNoError(err, "Error occurred while waiting for pod type: Ready.")
			})
			ginkgo.By("Creating client pod-a which should be able to contact the server pod-b.", func() {
				testCanConnect(f, f.Namespace, "pod-a", serviceB, 80)
			})

			ginkgo.By("Creating a network policy for pod-a which allows Egress traffic to pod-b.")
			policyAllowToPodB := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "allow-pod-a-to-pod-b-using-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy on pod-a
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": "pod-a",
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
					// Allow traffic to server on pod-b
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							Ports: []networkingv1.NetworkPolicyPort{
								// Allow DNS look-ups
								{
									Protocol: &protocolUDP,
									Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 53},
								},
							},
						},
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{
											"pod-name": "pod-b",
										},
									},
								},
							},
						},
					},
				},
			}

			policyAllowToPodB, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyAllowToPodB)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyAllowToPodB.")
			defer cleanupNetworkPolicy(f, policyAllowToPodB)

			ginkgo.By("Creating a network policy for pod-a that denies traffic from pod-b.")
			policyDenyFromPodB := &networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: f.Namespace.Name,
					Name:      "deny-pod-b-to-pod-a-pod-selector",
				},
				Spec: networkingv1.NetworkPolicySpec{
					// Apply this policy on the server on pod-a
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{
							"pod-name": "pod-a",
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
					// Deny traffic from all pods, including pod-b
					Ingress: []networkingv1.NetworkPolicyIngressRule{},
				},
			}

			policyDenyFromPodB, err = f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).Create(policyDenyFromPodB)
			framework.ExpectNoError(err, "Error occurred while creating policy: policyDenyFromPodB.")
			defer cleanupNetworkPolicy(f, policyDenyFromPodB)

			ginkgo.By("Creating client pod-a which should be able to contact the server pod-b.", func() {
				testCanConnect(f, f.Namespace, "pod-a", serviceB, 80)
			})
			cleanupServerPodAndService(f, podB, serviceB)

			// Creating server pod with label "pod-name": "pod-a" to deny traffic from client pod with label "pod-name": "pod-b"
			podA, serviceA = createServerPodAndService(f, f.Namespace, "pod-a", []int{80})
			ginkgo.By("Waiting for pod-a to be ready", func() {
				err = f.WaitForPodReady(podA.Name)
				framework.ExpectNoError(err, "Error occurred while waiting for pod type: Ready.")
			})

			ginkgo.By("Creating client pod-b which should be able to contact the server pod-a.", func() {
				testCannotConnect(f, f.Namespace, "pod-b", serviceA, 80)
			})
			cleanupServerPodAndService(f, podA, serviceA)
		})

	})

})

func testCanConnect(f *framework.Framework, ns *v1.Namespace, podName string, service *v1.Service, targetPort int) {
	ginkgo.By(fmt.Sprintf("Creating client pod %s that should successfully connect to %s.", podName, service.Name))
	podClient := createNetworkClientPod(f, ns, podName, service, targetPort)
	defer func() {
		ginkgo.By(fmt.Sprintf("Cleaning up the pod %s", podClient.Name))
		if err := f.ClientSet.CoreV1().Pods(ns.Name).Delete(podClient.Name, nil); err != nil {
			framework.Failf("unable to cleanup pod %v: %v", podClient.Name, err)
		}
	}()
	checkConnectivity(f, ns, podClient, service)
}

func testCannotConnect(f *framework.Framework, ns *v1.Namespace, podName string, service *v1.Service, targetPort int) {
	ginkgo.By(fmt.Sprintf("Creating client pod %s that should not be able to connect to %s.", podName, service.Name))
	podClient := createNetworkClientPod(f, ns, podName, service, targetPort)
	defer func() {
		ginkgo.By(fmt.Sprintf("Cleaning up the pod %s", podClient.Name))
		if err := f.ClientSet.CoreV1().Pods(ns.Name).Delete(podClient.Name, nil); err != nil {
			framework.Failf("unable to cleanup pod %v: %v", podClient.Name, err)
		}
	}()
	checkNoConnectivity(f, ns, podClient, service)
}

func checkConnectivity(f *framework.Framework, ns *v1.Namespace, podClient *v1.Pod, service *v1.Service) {
	framework.Logf("Waiting for %s to complete.", podClient.Name)
	err := e2epod.WaitForPodNoLongerRunningInNamespace(f.ClientSet, podClient.Name, ns.Name)
	framework.ExpectNoError(err, "Pod did not finish as expected.")

	framework.Logf("Waiting for %s to complete.", podClient.Name)
	err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, podClient.Name, ns.Name)
	if err != nil {
		// Collect pod logs when we see a failure.
		logs, logErr := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, podClient.Name, "client")
		if logErr != nil {
			framework.Failf("Error getting container logs: %s", logErr)
		}

		// Collect current NetworkPolicies applied in the test namespace.
		policies, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).List(metav1.ListOptions{})
		if err != nil {
			framework.Logf("error getting current NetworkPolicies for %s namespace: %s", f.Namespace.Name, err)
		}

		// Collect the list of pods running in the test namespace.
		podsInNS, err := e2epod.GetPodsInNamespace(f.ClientSet, f.Namespace.Name, map[string]string{})
		if err != nil {
			framework.Logf("error getting pods for %s namespace: %s", f.Namespace.Name, err)
		}

		pods := []string{}
		for _, p := range podsInNS {
			pods = append(pods, fmt.Sprintf("Pod: %s, Status: %s\n", p.Name, p.Status.String()))
		}

		framework.Failf("Pod %s should be able to connect to service %s, but was not able to connect.\nPod logs:\n%s\n\n Current NetworkPolicies:\n\t%v\n\n Pods:\n\t%v\n\n", podClient.Name, service.Name, logs, policies.Items, pods)

		// Dump debug information for the test namespace.
		framework.DumpDebugInfo(f.ClientSet, f.Namespace.Name)
	}
}

func checkNoConnectivity(f *framework.Framework, ns *v1.Namespace, podClient *v1.Pod, service *v1.Service) {
	framework.Logf("Waiting for %s to complete.", podClient.Name)
	err := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, podClient.Name, ns.Name)

	// We expect an error here since it's a cannot connect test.
	// Dump debug information if the error was nil.
	if err == nil {
		// Collect pod logs when we see a failure.
		logs, logErr := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, podClient.Name, "client")
		if logErr != nil {
			framework.Failf("Error getting container logs: %s", logErr)
		}

		// Collect current NetworkPolicies applied in the test namespace.
		policies, err := f.ClientSet.NetworkingV1().NetworkPolicies(f.Namespace.Name).List(metav1.ListOptions{})
		if err != nil {
			framework.Logf("error getting current NetworkPolicies for %s namespace: %s", f.Namespace.Name, err)
		}

		// Collect the list of pods running in the test namespace.
		podsInNS, err := e2epod.GetPodsInNamespace(f.ClientSet, f.Namespace.Name, map[string]string{})
		if err != nil {
			framework.Logf("error getting pods for %s namespace: %s", f.Namespace.Name, err)
		}

		pods := []string{}
		for _, p := range podsInNS {
			pods = append(pods, fmt.Sprintf("Pod: %s, Status: %s\n", p.Name, p.Status.String()))
		}

		framework.Failf("Pod %s should not be able to connect to service %s, but was able to connect.\nPod logs:\n%s\n\n Current NetworkPolicies:\n\t%v\n\n Pods:\n\t %v\n\n", podClient.Name, service.Name, logs, policies.Items, pods)

		// Dump debug information for the test namespace.
		framework.DumpDebugInfo(f.ClientSet, f.Namespace.Name)
	}
}

// Create a server pod with a listening container for each port in ports[].
// Will also assign a pod label with key: "pod-name" and label set to the given podName for later use by the network
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
			Image: imageutils.GetE2EImage(imageutils.Agnhost),
			Args:  []string{"porter"},
			Env: []v1.EnvVar{
				{
					Name:  fmt.Sprintf("SERVE_PORT_%d", port),
					Value: "foo",
				},
			},
			Ports: []v1.ContainerPort{
				{
					ContainerPort: int32(port),
					Name:          fmt.Sprintf("serve-%d", port),
				},
			},
			ReadinessProbe: &v1.Probe{
				Handler: v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Path: "/",
						Port: intstr.IntOrString{
							IntVal: int32(port),
						},
						Scheme: v1.URISchemeHTTP,
					},
				},
			},
		})

		// Build the Service Ports for the service.
		servicePorts = append(servicePorts, v1.ServicePort{
			Name:       fmt.Sprintf("%s-%d", podName, port),
			Port:       int32(port),
			TargetPort: intstr.FromInt(port),
		})
	}

	ginkgo.By(fmt.Sprintf("Creating a server pod %s in namespace %s", podName, namespace.Name))
	pod, err := f.ClientSet.CoreV1().Pods(namespace.Name).Create(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName + "-",
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers:    containers,
			RestartPolicy: v1.RestartPolicyNever,
		},
	})
	framework.ExpectNoError(err)
	framework.Logf("Created pod %v", pod.ObjectMeta.Name)

	svcName := fmt.Sprintf("svc-%s", podName)
	ginkgo.By(fmt.Sprintf("Creating a service %s for pod %s in namespace %s", svcName, podName, namespace.Name))
	svc, err := f.ClientSet.CoreV1().Services(namespace.Name).Create(&v1.Service{
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
	framework.ExpectNoError(err)
	framework.Logf("Created service %s", svc.Name)

	return pod, svc
}

func cleanupServerPodAndService(f *framework.Framework, pod *v1.Pod, service *v1.Service) {
	ginkgo.By("Cleaning up the server.")
	if err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
		framework.Failf("unable to cleanup pod %v: %v", pod.Name, err)
	}
	ginkgo.By("Cleaning up the server's service.")
	if err := f.ClientSet.CoreV1().Services(service.Namespace).Delete(service.Name, nil); err != nil {
		framework.Failf("unable to cleanup svc %v: %v", service.Name, err)
	}
}

// Create a client pod which will attempt a netcat to the provided service, on the specified port.
// This client will attempt a one-shot connection, then die, without restarting the pod.
// Test can then be asserted based on whether the pod quit with an error or not.
func createNetworkClientPod(f *framework.Framework, namespace *v1.Namespace, podName string, targetService *v1.Service, targetPort int) *v1.Pod {
	pod, err := f.ClientSet.CoreV1().Pods(namespace.Name).Create(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName + "-",
			Labels: map[string]string{
				"pod-name": podName,
			},
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "client",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
					Args: []string{
						"/bin/sh",
						"-c",
						fmt.Sprintf("for i in $(seq 1 5); do nc -vz -w 8 %s.%s %d && exit 0 || sleep 1; done; exit 1",
							targetService.Name, targetService.Namespace, targetPort),
					},
				},
			},
		},
	})
	framework.ExpectNoError(err)

	return pod
}

// Patch client pod with a map value
func updateNetworkClientPodLabel(f *framework.Framework, namespace *v1.Namespace, podName string, patchOperation string, patchPath string, patchValue map[string]string) *v1.Pod {
	type patchMapValue struct {
		Op    string            `json:"op"`
		Path  string            `json:"path"`
		Value map[string]string `json:"value,omitempty"`
	}
	payload := []patchMapValue{{
		Op:    patchOperation,
		Path:  patchPath,
		Value: patchValue,
	}}
	payloadBytes, err := json.Marshal(payload)
	framework.ExpectNoError(err)

	pod, err := f.ClientSet.CoreV1().Pods(namespace.Name).Patch(podName, types.JSONPatchType, payloadBytes)
	framework.ExpectNoError(err)

	return pod
}

func cleanupNetworkPolicy(f *framework.Framework, policy *networkingv1.NetworkPolicy) {
	ginkgo.By("Cleaning up the policy.")
	if err := f.ClientSet.NetworkingV1().NetworkPolicies(policy.Namespace).Delete(policy.Name, nil); err != nil {
		framework.Failf("unable to cleanup policy %v: %v", policy.Name, err)
	}
}
