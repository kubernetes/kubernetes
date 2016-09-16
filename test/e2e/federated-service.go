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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"os"
	"reflect"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
)

const (
	FederatedServiceTimeout = 60 * time.Second

	FederatedServiceName    = "federated-service"
	FederatedServicePodName = "federated-service-test-pod"
)

var FederatedServiceLabels = map[string]string{
	"foo": "bar",
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

			clusters = map[string]*cluster{}
			primaryClusterName = registerClusters(clusters, UserAgentName, federationName, f)
		})

		AfterEach(func() {
			unregisterClusters(clusters, f)
		})

		Describe("Service creation", func() {
			var (
				service *v1.Service
				nsName  string
			)

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)
				// Placeholder
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.Client)

				if service != nil {
					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", service.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, service, clusters)
					service = nil
					nsName = ""
				}
			})

			It("should succeed", func() {
				framework.SkipUnlessFederated(f.Client)

				nsName = f.FederationNamespace.Name
				service = createServiceOrFail(f.FederationClientset_1_4, nsName, FederatedServiceName)
				By(fmt.Sprintf("Creation of service %q in namespace %q succeeded.  Deleting service.", service.Name, nsName))

				// Cleanup
				err := f.FederationClientset_1_4.Services(nsName).Delete(service.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
				By(fmt.Sprintf("Deletion of service %q in namespace %q succeeded.", service.Name, nsName))
			})

			It("should create matching services in underlying clusters", func() {
				framework.SkipUnlessFederated(f.Client)

				nsName = f.FederationNamespace.Name
				service = createServiceOrFail(f.FederationClientset_1_4, nsName, FederatedServiceName)
				defer func() { // Cleanup
					By(fmt.Sprintf("Deleting service %q in namespace %q", service.Name, nsName))
					err := f.FederationClientset_1_4.Services(nsName).Delete(service.Name, &api.DeleteOptions{})
					framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, nsName)
				}()
				waitForServiceShardsOrFail(nsName, service, clusters)
			})
		})

		var _ = Describe("DNS", func() {

			var (
				service *v1.Service
			)

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)

				nsName := f.FederationNamespace.Name
				createBackendPodsOrFail(clusters, nsName, FederatedServicePodName)
				service = createServiceOrFail(f.FederationClientset_1_4, nsName, FederatedServiceName)
				waitForServiceShardsOrFail(nsName, service, clusters)
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.Client)

				nsName := f.FederationNamespace.Name
				deleteBackendPodsOrFail(clusters, nsName)

				if service != nil {
					deleteServiceOrFail(f.FederationClientset_1_4, nsName, service.Name)

					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", service.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, service, clusters)

					service = nil
				} else {
					By("No service to delete.  Service is nil")
				}
			})

			It("should be able to discover a federated service", func() {
				framework.SkipUnlessFederated(f.Client)

				nsName := f.FederationNamespace.Name
				svcDNSNames := []string{
					FederatedServiceName,
					fmt.Sprintf("%s.%s", FederatedServiceName, nsName),
					fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, nsName),
					fmt.Sprintf("%s.%s.%s", FederatedServiceName, nsName, federationName),
					fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, nsName, federationName),
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

					nsName := f.FederationNamespace.Name
					svcDNSNames := []string{
						fmt.Sprintf("%s.%s.%s", FederatedServiceName, nsName, federationName),
						fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, nsName, federationName),
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

						nsName := f.FederationNamespace.Name
						localSvcDNSNames := []string{
							FederatedServiceName,
							fmt.Sprintf("%s.%s", FederatedServiceName, nsName),
							fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, nsName),
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
