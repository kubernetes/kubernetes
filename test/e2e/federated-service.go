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
	"fmt"
	"os"
	"reflect"
	"strconv"
	"time"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	UserAgentName = "federation-e2e-service-controller"
	// TODO(madhusudancs): Using the same values as defined in the federated
	// service controller. Replace it with the values from the e2e framework.
	KubeAPIQPS   = 20.0
	KubeAPIBurst = 30

	FederatedServiceTimeout = 60 * time.Second

	FederatedServiceName    = "federated-service"
	FederatedServicePodName = "federated-service-test-pod"

	DefaultFederationName = "federation"

	// We use this to decide how long to wait for our DNS probes to succeed.
	DNSTTL = 180 * time.Second // TODO: make k8s.io/kubernetes/federation/pkg/federation-controller/service.minDnsTtl exported, and import it here.
)

var FederatedServiceLabels = map[string]string{
	"foo": "bar",
}

/*
cluster keeps track of the assorted objects and state related to each cluster
in the federation
*/
type cluster struct {
	name string
	*release_1_3.Clientset
	namespaceCreated bool    // Did we need to create a new namespace in this cluster?  If so, we should delete it.
	backendPod       *v1.Pod // The backend pod, if one's been created.
}

var _ = framework.KubeDescribe("[Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federated-service")
	var clusters map[string]*cluster // All clusters, keyed by cluster name
	var federationName string
	var primaryClusterName string // The name of the "primary" cluster

	var _ = Describe("Federated Services", func() {
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)

			// TODO: Federation API server should be able to answer this.
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}

			contexts := f.GetUnderlyingFederatedContexts()

			for _, context := range contexts {
				createClusterObjectOrFail(f, &context)
			}

			var clusterList *federationapi.ClusterList
			By("Obtaining a list of all the clusters")
			if err := wait.PollImmediate(framework.Poll, FederatedServiceTimeout, func() (bool, error) {
				var err error
				clusterList, err = f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
				if err != nil {
					return false, err
				}
				framework.Logf("%d clusters registered, waiting for %d", len(clusterList.Items), len(contexts))
				if len(clusterList.Items) == len(contexts) {
					return true, nil
				}
				return false, nil
			}); err != nil {
				framework.Failf("Failed to list registered clusters: %+v", err)
			}

			framework.Logf("Checking that %d clusters are Ready", len(contexts))
			for _, context := range contexts {
				clusterIsReadyOrFail(f, &context)
			}
			framework.Logf("%d clusters are Ready", len(contexts))

			clusters = map[string]*cluster{}
			primaryClusterName = clusterList.Items[0].Name
			By(fmt.Sprintf("Labeling %q as the first cluster", primaryClusterName))
			for i, c := range clusterList.Items {
				framework.Logf("Creating a clientset for the cluster %s", c.Name)

				Expect(framework.TestContext.KubeConfig).ToNot(Equal(""), "KubeConfig must be specified to load clusters' client config")
				kubecfg, err := clientcmd.LoadFromFile(framework.TestContext.KubeConfig)
				framework.ExpectNoError(err, "error loading KubeConfig: %v", err)

				cfgOverride := &clientcmd.ConfigOverrides{
					ClusterInfo: clientcmdapi.Cluster{
						Server: c.Spec.ServerAddressByClientCIDRs[0].ServerAddress,
					},
				}
				ccfg := clientcmd.NewNonInteractiveClientConfig(*kubecfg, c.Name, cfgOverride, clientcmd.NewDefaultClientConfigLoadingRules())
				cfg, err := ccfg.ClientConfig()
				framework.ExpectNoError(err, "Error creating client config in cluster #%d (%q)", i, c.Name)

				cfg.QPS = KubeAPIQPS
				cfg.Burst = KubeAPIBurst
				clset := release_1_3.NewForConfigOrDie(restclient.AddUserAgent(cfg, UserAgentName))
				clusters[c.Name] = &cluster{c.Name, clset, false, nil}
			}

			for name, c := range clusters {
				// The e2e Framework created the required namespace in one of the clusters, but we need to create it in all the others, if it doesn't yet exist.
				if _, err := c.Clientset.Core().Namespaces().Get(f.Namespace.Name); errors.IsNotFound(err) {
					ns := &v1.Namespace{
						ObjectMeta: v1.ObjectMeta{
							Name: f.Namespace.Name,
						},
					}
					_, err := c.Clientset.Core().Namespaces().Create(ns)
					if err == nil {
						c.namespaceCreated = true
					}
					framework.ExpectNoError(err, "Couldn't create the namespace %s in cluster %q", f.Namespace.Name, name)
					framework.Logf("Namespace %s created in cluster %q", f.Namespace.Name, name)
				} else if err != nil {
					framework.Logf("Couldn't create the namespace %s in cluster %q: %v", f.Namespace.Name, name, err)
				}
			}
		})

		AfterEach(func() {
			for name, c := range clusters {
				if c.namespaceCreated {
					if _, err := c.Clientset.Core().Namespaces().Get(f.Namespace.Name); !errors.IsNotFound(err) {
						err := c.Clientset.Core().Namespaces().Delete(f.Namespace.Name, &api.DeleteOptions{})
						framework.ExpectNoError(err, "Couldn't delete the namespace %s in cluster %q: %v", f.Namespace.Name, name, err)
					}
					framework.Logf("Namespace %s deleted in cluster %q", f.Namespace.Name, name)
				}
			}

			// Delete the registered clusters in the federation API server.
			clusterList, err := f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
			framework.ExpectNoError(err, "Error listing clusters")
			for _, cluster := range clusterList.Items {
				err := f.FederationClientset_1_4.Federation().Clusters().Delete(cluster.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting cluster %q", cluster.Name)
			}
		})

		Describe("Service creation", func() {
			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)
				// Placeholder
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.Client)
				// Placeholder
			})

			It("should succeed", func() {
				framework.SkipUnlessFederated(f.Client)
				service := createServiceOrFail(f.FederationClientset_1_4, f.Namespace.Name)
				By(fmt.Sprintf("Creation of service %q in namespace %q succeeded.  Deleting service.", service.Name, f.Namespace.Name))
				// Cleanup
				err := f.FederationClientset_1_4.Services(f.Namespace.Name).Delete(service.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
				By(fmt.Sprintf("Deletion of service %q in namespace %q succeeded.", service.Name, f.Namespace.Name))
			})

			It("should create matching services in underlying clusters", func() {
				framework.SkipUnlessFederated(f.Client)
				service := createServiceOrFail(f.FederationClientset_1_4, f.Namespace.Name)
				defer func() { // Cleanup
					By(fmt.Sprintf("Deleting service %q in namespace %q", service.Name, f.Namespace.Name))
					err := f.FederationClientset_1_4.Services(f.Namespace.Name).Delete(service.Name, &api.DeleteOptions{})
					framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, f.Namespace.Name)
				}()
				waitForServiceShardsOrFail(f.Namespace.Name, service, clusters)
			})
		})

		var _ = Describe("DNS", func() {

			var (
				service *v1.Service
			)

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)
				createBackendPodsOrFail(clusters, f.Namespace.Name, FederatedServicePodName)
				service = createServiceOrFail(f.FederationClientset_1_4, f.Namespace.Name)
				waitForServiceShardsOrFail(f.Namespace.Name, service, clusters)
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.Client)
				deleteBackendPodsOrFail(clusters, f.Namespace.Name)

				if service != nil {
					deleteServiceOrFail(f.FederationClientset_1_4, f.Namespace.Name, service.Name)
					service = nil
				} else {
					By("No service to delete.  Service is nil")
				}
			})

			It("should be able to discover a federated service", func() {
				framework.SkipUnlessFederated(f.Client)

				svcDNSNames := []string{
					FederatedServiceName,
					fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
					fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
					fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
					fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
				}
				// TODO(mml): This could be much faster.  We can launch all the test
				// pods, perhaps in the BeforeEach, and then just poll until we get
				// successes/failures from them all.
				for i, DNSName := range svcDNSNames {
					discoverService(f, DNSName, true, "federated-service-e2e-discovery-pod-"+strconv.Itoa(i))
				}
			})

			Context("non-local federated service", func() {
				BeforeEach(func() {
					framework.SkipUnlessFederated(f.Client)

					// Delete all the backend pods from the shard which is local to the discovery pod.
					deleteOneBackendPodOrFail(clusters[primaryClusterName])

				})

				It("should be able to discover a non-local federated service", func() {
					framework.SkipUnlessFederated(f.Client)

					svcDNSNames := []string{
						fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
						fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
					}
					for i, name := range svcDNSNames {
						discoverService(f, name, true, "federated-service-e2e-discovery-pod-"+strconv.Itoa(i))
					}
				})

				// TODO(mml): This currently takes 9 minutes.  Consider reducing the
				// TTL and/or running the pods in parallel.
				Context("[Slow] missing local service", func() {
					It("should never find DNS entries for a missing local service", func() {
						framework.SkipUnlessFederated(f.Client)

						localSvcDNSNames := []string{
							FederatedServiceName,
							fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
							fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
						}
						for i, name := range localSvcDNSNames {
							discoverService(f, name, false, "federated-service-e2e-discovery-pod-"+strconv.Itoa(i))
						}
					})
				})
			})
		})
	})
})

/*
   equivalent returns true if the two services are equivalent.  Fields which are expected to differ between
   federated services and the underlying cluster services (e.g. ClusterIP, LoadBalancerIP etc) are ignored.
*/
func equivalent(federationService, clusterService v1.Service) bool {
	// TODO: I think that we need a DeepCopy here to avoid clobbering our parameters.
	clusterService.Spec.ClusterIP = federationService.Spec.ClusterIP
	clusterService.Spec.ExternalIPs = federationService.Spec.ExternalIPs
	clusterService.Spec.DeprecatedPublicIPs = federationService.Spec.DeprecatedPublicIPs
	clusterService.Spec.LoadBalancerIP = federationService.Spec.LoadBalancerIP
	clusterService.Spec.LoadBalancerSourceRanges = federationService.Spec.LoadBalancerSourceRanges
	// N.B. We cannot iterate over the port objects directly, as their values
	// only get copied and our updates will get lost.
	for i := range clusterService.Spec.Ports {
		clusterService.Spec.Ports[i].NodePort = federationService.Spec.Ports[i].NodePort
	}
	return reflect.DeepEqual(clusterService.Spec, federationService.Spec)
}

/*
   waitForServiceOrFail waits until a service is either present or absent in the cluster specified by clientset.
   If the condition is not met within timout, it fails the calling test.
*/
func waitForServiceOrFail(clientset *release_1_3.Clientset, namespace string, service *v1.Service, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated service shard of service %q in namespace %q from cluster", service.Name, namespace))
	var clusterService *v1.Service
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterService, err := clientset.Services(namespace).Get(service.Name)
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated service %q in namespace %q in cluster is absent", service.Name, namespace))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated service %q in namespace %q in cluster is present", service.Name, namespace))
			return true, nil // Success
		}
		By(fmt.Sprintf("Service %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", service.Name, namespace, clusterService != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify service %q in namespace %q in cluster: Present=%v", service.Name, namespace, present)

	if present && clusterService != nil {
		Expect(equivalent(*clusterService, *service))
	}
}

/*
   waitForServiceShardsOrFail waits for the service to appear in all clusters
*/
func waitForServiceShardsOrFail(namespace string, service *v1.Service, clusters map[string]*cluster) {
	framework.Logf("Waiting for service %q in %d clusters", service.Name, len(clusters))
	for _, c := range clusters {
		waitForServiceOrFail(c.Clientset, namespace, service, true, FederatedServiceTimeout)
	}
}

func createService(clientset *federation_release_1_4.Clientset, namespace string) (*v1.Service, error) {
	if clientset == nil || len(namespace) == 0 {
		return nil, fmt.Errorf("Internal error: invalid parameters passed to deleteServiceOrFail: clientset: %v, namespace: %v", clientset, namespace)
	}
	By(fmt.Sprintf("Creating federated service %q in namespace %q", FederatedServiceName, namespace))

	service := &v1.Service{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedServiceName,
		},
		Spec: v1.ServiceSpec{
			Selector: FederatedServiceLabels,
			Type:     "LoadBalancer",
			Ports: []v1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt(8080),
				},
			},
		},
	}
	By(fmt.Sprintf("Trying to create service %q in namespace %q", service.Name, namespace))
	return clientset.Services(namespace).Create(service)
}

func createServiceOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1.Service {
	service, err := createService(clientset, namespace)
	framework.ExpectNoError(err, "Creating service %q in namespace %q", service.Name, namespace)
	By(fmt.Sprintf("Successfully created federated service %q in namespace %q", FederatedServiceName, namespace))
	return service
}

func deleteServiceOrFail(clientset *federation_release_1_4.Clientset, namespace string, serviceName string) {
	if clientset == nil || len(namespace) == 0 || len(serviceName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteServiceOrFail: clientset: %v, namespace: %v, service: %v", clientset, namespace, serviceName))
	}
	err := clientset.Services(namespace).Delete(serviceName, api.NewDeleteOptions(0))
	framework.ExpectNoError(err, "Error deleting service %q from namespace %q", serviceName, namespace)
}

func podExitCodeDetector(f *framework.Framework, name string, code int32) func() error {
	// If we ever get any container logs, stash them here.
	logs := ""

	logerr := func(err error) error {
		if err == nil {
			return nil
		}
		if logs == "" {
			return err
		}
		return fmt.Errorf("%s (%v)", logs, err)
	}

	return func() error {
		pod, err := f.Client.Pods(f.Namespace.Name).Get(name)
		if err != nil {
			return logerr(err)
		}
		if len(pod.Status.ContainerStatuses) < 1 {
			return logerr(fmt.Errorf("no container statuses"))
		}

		// Best effort attempt to grab pod logs for debugging
		logs, err = framework.GetPodLogs(f.Client, f.Namespace.Name, name, pod.Spec.Containers[0].Name)
		if err != nil {
			framework.Logf("Cannot fetch pod logs: %v", err)
		}

		status := pod.Status.ContainerStatuses[0]
		if status.State.Terminated == nil {
			return logerr(fmt.Errorf("container is not in terminated state"))
		}
		if status.State.Terminated.ExitCode == code {
			return nil
		}

		return logerr(fmt.Errorf("exited %d", status.State.Terminated.ExitCode))
	}
}

func discoverService(f *framework.Framework, name string, exists bool, podName string) {
	command := []string{"sh", "-c", fmt.Sprintf("until nslookup '%s'; do sleep 10; done", name)}
	By(fmt.Sprintf("Looking up %q", name))

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "federated-service-discovery-container",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: command,
				},
			},
			RestartPolicy: api.RestartPolicyOnFailure,
		},
	}

	By(fmt.Sprintf("Creating pod %q in namespace %q", pod.Name, f.Namespace.Name))
	_, err := f.Client.Pods(f.Namespace.Name).Create(pod)
	framework.ExpectNoError(err, "Trying to create pod to run %q", command)
	By(fmt.Sprintf("Successfully created pod %q in namespace %q", pod.Name, f.Namespace.Name))
	defer func() {
		By(fmt.Sprintf("Deleting pod %q from namespace %q", podName, f.Namespace.Name))
		err := f.Client.Pods(f.Namespace.Name).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Deleting pod %q from namespace %q", podName, f.Namespace.Name)
		By(fmt.Sprintf("Deleted pod %q from namespace %q", podName, f.Namespace.Name))
	}()

	if exists {
		// TODO(mml): Eventually check the IP address is correct, too.
		Eventually(podExitCodeDetector(f, podName, 0), 3*DNSTTL, time.Second*2).
			Should(BeNil(), "%q should exit 0, but it never did", command)
	} else {
		Eventually(podExitCodeDetector(f, podName, 0), 3*DNSTTL, time.Second*2).
			ShouldNot(BeNil(), "%q should eventually not exit 0, but it always did", command)
	}
}

/*
createBackendPodsOrFail creates one pod in each cluster, and returns the created pods (in the same order as clusterClientSets).
If creation of any pod fails, the test fails (possibly with a partially created set of pods). No retries are attempted.
*/
func createBackendPodsOrFail(clusters map[string]*cluster, namespace string, name string) {
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
			// Namespace: namespace,
			Labels: FederatedServiceLabels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: "gcr.io/google_containers/echoserver:1.4",
				},
			},
			RestartPolicy: v1.RestartPolicyAlways,
		},
	}
	for name, c := range clusters {
		By(fmt.Sprintf("Creating pod %q in namespace %q in cluster %q", pod.Name, namespace, name))
		createdPod, err := c.Clientset.Core().Pods(namespace).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q in cluster %q", name, namespace, name)
		By(fmt.Sprintf("Successfully created pod %q in namespace %q in cluster %q: %v", pod.Name, namespace, name, *createdPod))
		c.backendPod = createdPod
	}
}

/*
deleteOneBackendPodOrFail deletes exactly one backend pod which must not be nil
The test fails if there are any errors.
*/
func deleteOneBackendPodOrFail(c *cluster) {
	pod := c.backendPod
	Expect(pod).ToNot(BeNil())
	err := c.Clientset.Core().Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0))
	if errors.IsNotFound(err) {
		By(fmt.Sprintf("Pod %q in namespace %q in cluster %q does not exist.  No need to delete it.", pod.Name, pod.Namespace, c.name))
	} else {
		framework.ExpectNoError(err, "Deleting pod %q in namespace %q from cluster %q", pod.Name, pod.Namespace, c.name)
	}
	By(fmt.Sprintf("Backend pod %q in namespace %q in cluster %q deleted or does not exist", pod.Name, pod.Namespace, c.name))
}

/*
deleteBackendPodsOrFail deletes one pod from each cluster that has one.
If deletion of any pod fails, the test fails (possibly with a partially deleted set of pods). No retries are attempted.
*/
func deleteBackendPodsOrFail(clusters map[string]*cluster, namespace string) {
	for name, c := range clusters {
		if c.backendPod != nil {
			deleteOneBackendPodOrFail(c)
			c.backendPod = nil
		} else {
			By(fmt.Sprintf("No backend pod to delete for cluster %q", name))
		}
	}
}
