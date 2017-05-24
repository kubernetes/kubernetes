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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	FederatedServiceTimeout = 60 * time.Second

	FederatedServiceName    = "federated-service"
	FederatedServicePodName = "federated-service-test-pod"

	KubeDNSConfigMapName      = "kube-dns"
	KubeDNSConfigMapNamespace = "kube-system"
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
			framework.SkipUnlessFederated(f.ClientSet)

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
				framework.SkipUnlessFederated(f.ClientSet)
				// Placeholder
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.ClientSet)

				if service != nil {
					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", service.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, service, clusters)
					service = nil
					nsName = ""
				}
			})

			It("should succeed", func() {
				framework.SkipUnlessFederated(f.ClientSet)

				nsName = f.FederationNamespace.Name
				service = createServiceOrFail(f.FederationClientset_1_5, nsName, FederatedServiceName)
				By(fmt.Sprintf("Creation of service %q in namespace %q succeeded.  Deleting service.", service.Name, nsName))

				// Cleanup
				err := f.FederationClientset_1_5.Services(nsName).Delete(service.Name, &v1.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
				By(fmt.Sprintf("Deletion of service %q in namespace %q succeeded.", service.Name, nsName))
			})

			It("should create matching services in underlying clusters", func() {
				framework.SkipUnlessFederated(f.ClientSet)

				nsName = f.FederationNamespace.Name
				service = createServiceOrFail(f.FederationClientset_1_5, nsName, FederatedServiceName)
				defer func() { // Cleanup
					By(fmt.Sprintf("Deleting service %q in namespace %q", service.Name, nsName))
					err := f.FederationClientset_1_5.Services(nsName).Delete(service.Name, &v1.DeleteOptions{})
					framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, nsName)
				}()
				waitForServiceShardsOrFail(nsName, service, clusters)
			})

			It("should not be deleted from underlying clusters when it is deleted", func() {
				framework.SkipUnlessFederated(f.ClientSet)
				nsName = f.FederationNamespace.Name
				service = createServiceOrFail(f.FederationClientset_1_5, nsName, FederatedServiceName)
				By(fmt.Sprintf("Successfully created federated service %q in namespace %q. Waiting for shards to appear in underlying clusters", service.Name, nsName))

				waitForServiceShardsOrFail(nsName, service, clusters)

				By(fmt.Sprintf("Deleting service %s", service.Name))
				err := f.FederationClientset_1_5.Services(nsName).Delete(service.Name, &v1.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
				By(fmt.Sprintf("Deletion of service %q in namespace %q succeeded.", service.Name, nsName))
				By(fmt.Sprintf("Verifying that services in underlying clusters are not deleted"))
				for clusterName, clusterClientset := range clusters {
					_, err := clusterClientset.Core().Services(service.Namespace).Get(service.Name)
					if err != nil {
						framework.Failf("Unexpected error in fetching service %s in cluster %s, %s", service.Name, clusterName, err)
					}
				}
			})
		})

		var _ = Describe("DNS", func() {

			var (
				service      *v1.Service
				serviceShard *v1.Service
			)

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.ClientSet)

				nsName := f.FederationNamespace.Name
				// Create kube-dns configmap for kube-dns to accept federation queries.
				federationsDomainMap := os.Getenv("FEDERATIONS_DOMAIN_MAP")
				if federationsDomainMap == "" {
					framework.Failf("missing required env var FEDERATIONS_DOMAIN_MAP")
				}
				kubeDNSConfigMap := v1.ConfigMap{
					ObjectMeta: v1.ObjectMeta{
						Name:      KubeDNSConfigMapName,
						Namespace: KubeDNSConfigMapNamespace,
					},
					Data: map[string]string{
						"federations": federationsDomainMap,
					},
				}
				// Create this configmap in all clusters.
				for clusterName, cluster := range clusters {
					By(fmt.Sprintf("Creating kube dns config map in cluster: %s", clusterName))
					_, err := cluster.Clientset.Core().ConfigMaps(KubeDNSConfigMapNamespace).Create(&kubeDNSConfigMap)
					framework.ExpectNoError(err, fmt.Sprintf("Error in creating config map in cluster %s", clusterName))
				}

				createBackendPodsOrFail(clusters, nsName, FederatedServicePodName)

				service = createServiceOrFail(f.FederationClientset_1_5, nsName, FederatedServiceName)
				obj, err := conversion.NewCloner().DeepCopy(service)
				// Cloning shouldn't fail. On the off-chance it does, we
				// should shallow copy service to serviceShard before
				// failing. If we don't do this we will never really
				// get a chance to clean up the underlying services
				// when the cloner fails for reasons not in our
				// control. For example, cloner bug. That will cause
				// the resources to leak, which in turn causes the
				// test project to run out of quota and the entire
				// suite starts failing. So we must try as hard as
				// possible to cleanup the underlying services. So
				// if DeepCopy fails, we are going to try with shallow
				// copy as a last resort.
				if err != nil {
					serviceCopy := *service
					serviceShard = &serviceCopy
					framework.ExpectNoError(err, fmt.Sprintf("Error in deep copying service %q", service.Name))
				}
				var ok bool
				serviceShard, ok = obj.(*v1.Service)
				// Same argument as above about using shallow copy
				// as a last resort.
				if !ok {
					serviceCopy := *service
					serviceShard = &serviceCopy
					framework.ExpectNoError(err, fmt.Sprintf("Unexpected service object copied %T", obj))
				}

				waitForServiceShardsOrFail(nsName, serviceShard, clusters)
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.ClientSet)

				nsName := f.FederationNamespace.Name
				deleteBackendPodsOrFail(clusters, nsName)

				if service != nil {
					deleteServiceOrFail(f.FederationClientset_1_5, nsName, service.Name)
					service = nil
				} else {
					By("No service to delete.  Service is nil")
				}

				if serviceShard != nil {
					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", service.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, service, clusters)
					serviceShard = nil
				} else {
					By("No service shards to delete. `serviceShard` is nil")
				}

				// Delete the kube-dns config map from all clusters.
				for clusterName, cluster := range clusters {
					By(fmt.Sprintf("Deleting kube dns config map from cluster: %s", clusterName))
					err := cluster.Clientset.Core().ConfigMaps(KubeDNSConfigMapNamespace).Delete(KubeDNSConfigMapName, nil)
					framework.ExpectNoError(err, fmt.Sprintf("Error in deleting config map from cluster %s", clusterName))
				}
			})

			It("should be able to discover a federated service", func() {
				framework.SkipUnlessFederated(f.ClientSet)

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
				By("Verified that DNS rules are working as expected")

				By("Deleting the service to verify that DNS rules still work")
				err := f.FederationClientset_1_5.Services(nsName).Delete(FederatedServiceName, &v1.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
				// Service is deleted, unset the test block-global service variable.
				service = nil

				for i, DNSName := range svcDNSNames {
					discoverService(f, DNSName, true, "federated-service-e2e-discovery-pod-"+strconv.Itoa(i))
				}
				By("Verified that deleting the service does not affect DNS records")
			})

			Context("non-local federated service", func() {
				BeforeEach(func() {
					framework.SkipUnlessFederated(f.ClientSet)

					// Delete all the backend pods from the shard which is local to the discovery pod.
					deleteOneBackendPodOrFail(clusters[primaryClusterName])

				})

				It("should be able to discover a non-local federated service", func() {
					framework.SkipUnlessFederated(f.ClientSet)

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
						framework.SkipUnlessFederated(f.ClientSet)

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
