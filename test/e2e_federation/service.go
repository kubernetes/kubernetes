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

package e2e_federation

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
)

const (
	FederatedServiceName    = "federated-service"
	FederatedServicePodName = "federated-service-test-pod"
)

var FederatedServiceLabels = map[string]string{
	"foo": "bar",
}

var _ = framework.KubeDescribe("Federated Services [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federated-service")
	var clusters map[string]*cluster // All clusters, keyed by cluster name
	var federationName string
	var primaryClusterName string // The name of the "primary" cluster

	var _ = Describe("Without Clusters [NoCluster]", func() {
		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			// Placeholder
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
		})

		It("should succeed when a service is created", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			service := createServiceOrFail(f.FederationClientset, nsName, FederatedServiceName)
			By(fmt.Sprintf("Creation of service %q in namespace %q succeeded.  Deleting service.", service.Name, nsName))

			// Cleanup
			err := f.FederationClientset.Services(nsName).Delete(service.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, service.Namespace)
			By(fmt.Sprintf("Deletion of service %q in namespace %q succeeded.", service.Name, nsName))
		})
	})

	var _ = Describe("with clusters", func() {
		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			// TODO: Federation API server should be able to answer this.
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}

			clusters, primaryClusterName = getRegisteredClusters(UserAgentName, f)
		})

		Describe("Federated Service", func() {
			var (
				service *v1.Service
				nsName  string
			)

			BeforeEach(func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				nsName = f.FederationNamespace.Name
			})

			AfterEach(func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

				if service != nil {
					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", service.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, service, clusters)
					service = nil
					nsName = ""
				}
			})

			It("should create matching services in underlying clusters", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				service = createServiceOrFail(f.FederationClientset, nsName, FederatedServiceName)
				defer func() { // Cleanup
					By(fmt.Sprintf("Deleting service %q in namespace %q", service.Name, nsName))
					err := f.FederationClientset.Services(nsName).Delete(service.Name, &metav1.DeleteOptions{})
					framework.ExpectNoError(err, "Error deleting service %q in namespace %q", service.Name, nsName)
				}()
				waitForServiceShardsOrFail(nsName, service, clusters)
			})

			It("should be deleted from underlying clusters when OrphanDependents is false", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				orphanDependents := false
				verifyCascadingDeletionForService(f.FederationClientset, clusters, &orphanDependents, nsName)
				By(fmt.Sprintf("Verified that services were deleted from underlying clusters"))
			})

			It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				orphanDependents := true
				verifyCascadingDeletionForService(f.FederationClientset, clusters, &orphanDependents, nsName)
				By(fmt.Sprintf("Verified that services were not deleted from underlying clusters"))
			})

			It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				verifyCascadingDeletionForService(f.FederationClientset, clusters, nil, nsName)
				By(fmt.Sprintf("Verified that services were not deleted from underlying clusters"))
			})
		})

		var _ = Describe("DNS", func() {

			var (
				service      *v1.Service
				serviceShard *v1.Service
			)

			BeforeEach(func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

				nsName := f.FederationNamespace.Name

				createBackendPodsOrFail(clusters, nsName, FederatedServicePodName)

				service = createServiceOrFail(f.FederationClientset, nsName, FederatedServiceName)
				obj, err := api.Scheme.DeepCopy(service)
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
				fedframework.SkipUnlessFederated(f.ClientSet)

				nsName := f.FederationNamespace.Name
				deleteBackendPodsOrFail(clusters, nsName)

				if service != nil {
					deleteServiceOrFail(f.FederationClientset, nsName, service.Name, nil)
					service = nil
				} else {
					By("No service to delete.  Service is nil")
				}

				if serviceShard != nil {
					By(fmt.Sprintf("Deleting service shards and their provider resources in underlying clusters for service %q in namespace %q", serviceShard.Name, nsName))
					cleanupServiceShardsAndProviderResources(nsName, serviceShard, clusters)
					serviceShard = nil
				} else {
					By("No service shards to delete. `serviceShard` is nil")
				}
			})

			It("should be able to discover a federated service", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

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
				err := f.FederationClientset.Services(nsName).Delete(FederatedServiceName, &metav1.DeleteOptions{})
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
					fedframework.SkipUnlessFederated(f.ClientSet)

					// Delete all the backend pods from the shard which is local to the discovery pod.
					deleteOneBackendPodOrFail(clusters[primaryClusterName])

				})

				It("should be able to discover a non-local federated service", func() {
					fedframework.SkipUnlessFederated(f.ClientSet)

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
						fedframework.SkipUnlessFederated(f.ClientSet)

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

// verifyCascadingDeletionForService verifies that services are deleted from
// underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForService(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	service := createServiceOrFail(clientset, nsName, FederatedServiceName)
	serviceName := service.Name
	// Check subclusters if the service was created there.
	By(fmt.Sprintf("Waiting for service %s to be created in all underlying clusters", serviceName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Core().Services(nsName).Get(serviceName, metav1.GetOptions{})
			if err != nil {
				if !errors.IsNotFound(err) {
					return false, err
				}
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all services created")

	By(fmt.Sprintf("Deleting service %s", serviceName))
	deleteServiceOrFail(clientset, nsName, serviceName, orphanDependents)

	By(fmt.Sprintf("Verifying services %s in underlying clusters", serviceName))
	errMessages := []string{}
	// service should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Core().Services(nsName).Get(serviceName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for service %s in cluster %s, expected service to exist", serviceName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for service %s in cluster %s, got error: %v", serviceName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

// equivalent returns true if the two services are equivalent.  Fields which are expected to differ between
// federated services and the underlying cluster services (e.g. ClusterIP, LoadBalancerIP etc) are ignored.
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
