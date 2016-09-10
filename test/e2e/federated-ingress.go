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
	"net/http"
	"os"
	"reflect"
	"strconv"
	"time"

	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	MaxRetriesOnFederatedApiserver = 3
	FederatedIngressTimeout        = 60 * time.Second
	FederatedIngressName           = "federated-ingress"
	FederatedIngressServiceName    = "federated-ingress-service"
	FederatedIngressServicePodName = "federated-ingress-service-test-pod"
)

var _ = framework.KubeDescribe("Federated ingresses [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federated-ingress")

	// Create/delete ingress api objects
	// Validate federation apiserver, does not rely on underlying clusters or federation ingress controller.
	Describe("Ingress objects", func() {
		AfterEach(func() {
			nsName := f.FederationNamespace.Name
			// Delete registered ingresses.
			ingressList, err := f.FederationClientset_1_4.Extensions().Ingresses(nsName).List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, ingress := range ingressList.Items {
				err := f.FederationClientset_1_4.Extensions().Ingresses(nsName).Delete(ingress.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			nsName := f.FederationNamespace.Name
			ingress := createIngressOrFail(f.FederationClientset_1_4, nsName)
			By(fmt.Sprintf("Creation of ingress %q in namespace %q succeeded.  Deleting ingress.", ingress.Name, nsName))
			// Cleanup
			err := f.FederationClientset_1_4.Extensions().Ingresses(nsName).Delete(ingress.Name, &api.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting ingress %q in namespace %q", ingress.Name, ingress.Namespace)
			By(fmt.Sprintf("Deletion of ingress %q in namespace %q succeeded.", ingress.Name, nsName))
		})
	})

	// e2e cases for federation ingress controller
	var _ = Describe("Federated Ingresses", func() {
		var (
			clusters                               map[string]*cluster // All clusters, keyed by cluster name
			primaryClusterName, federationName, ns string
			jig                                    *federationTestJig
		)

		// register clusters in federation apiserver
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}
			jig = newFederationTestJig(f.FederationClientset_1_4)
			clusters = map[string]*cluster{}
			primaryClusterName = registerClusters(clusters, UserAgentName, federationName, f)
			ns = f.FederationNamespace.Name
		})

		AfterEach(func() {
			unregisterClusters(clusters, f)
		})

		It("should create and update matching ingresses in underlying clusters", func() {
			ingress := createIngressOrFail(f.FederationClientset_1_4, ns)
			defer func() { // Cleanup
				By(fmt.Sprintf("Deleting ingress %q in namespace %q", ingress.Name, ns))
				err := f.FederationClientset_1_4.Ingresses(ns).Delete(ingress.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting ingress %q in namespace %q", ingress.Name, ns)
			}()
			// wait for ingress shards being created
			waitForIngressShardsOrFail(ns, ingress, clusters)
			ingress = updateIngressOrFail(f.FederationClientset_1_4, ns)
			waitForIngressShardsUpdatedOrFail(ns, ingress, clusters)
		})

		var _ = Describe("Ingress connectivity and DNS", func() {

			var (
				service *v1.Service
			)

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)
				// create backend pod
				createBackendPodsOrFail(clusters, ns, FederatedIngressServicePodName)
				// create backend service
				service = createServiceOrFail(f.FederationClientset_1_4, ns, FederatedIngressServiceName)
				// create ingress object
				jig.ing = createIngressOrFail(f.FederationClientset_1_4, ns)
				// wait for services objects sync
				waitForServiceShardsOrFail(ns, service, clusters)
				// wait for ingress objects sync
				waitForIngressShardsOrFail(ns, jig.ing, clusters)
			})

			AfterEach(func() {
				deleteBackendPodsOrFail(clusters, ns)
				if service != nil {
					deleteServiceOrFail(f.FederationClientset_1_4, ns, service.Name)
					service = nil
				} else {
					By("No service to delete. Service is nil")
				}
				if jig.ing != nil {
					deleteIngressOrFail(f.FederationClientset_1_4, ns, jig.ing.Name)
					jig.ing = nil
				} else {
					By("No ingress to delete. Ingress is nil")
				}
			})

			PIt("should be able to discover a federated ingress service", func() {
				// we are about the ingress name
				svcDNSNames := []string{
					fmt.Sprintf("%s.%s", FederatedIngressServiceName, ns),
					fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedIngressServiceName, ns),
					// TODO these two entries are not set yet
					//fmt.Sprintf("%s.%s.%s", FederatedIngressServiceName, ns, federationName),
					//fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedIngressServiceName, ns, federationName),
				}
				// check dns records in underlying cluster
				for i, DNSName := range svcDNSNames {
					discoverService(f, DNSName, true, "federated-ingress-e2e-discovery-pod-"+strconv.Itoa(i))
				}
				// TODO check dns record in global dns server

				// check the traffic on federation ingress
				jig.waitForFederatedIngress()
			})

		})
	})
})

/*
   equivalent returns true if the two ingress spec are equivalent.
*/
func equivalentIngress(federatedIngress, clusterIngress v1beta1.Ingress) bool {
	return reflect.DeepEqual(clusterIngress.Spec, federatedIngress.Spec)
}

/*
   waitForIngressOrFail waits until a ingress is either present or absent in the cluster specified by clientset.
   If the condition is not met within timout, it fails the calling test.
*/
func waitForIngressOrFail(clientset *release_1_3.Clientset, namespace string, ingress *v1beta1.Ingress, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated ingress shard of ingress %q in namespace %q from cluster", ingress.Name, namespace))
	var clusterIngress *v1beta1.Ingress
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterIngress, err := clientset.Ingresses(namespace).Get(ingress.Name)
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated ingress %q in namespace %q in cluster is absent", ingress.Name, namespace))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated ingress %q in namespace %q in cluster is present", ingress.Name, namespace))
			return true, nil // Success
		}
		By(fmt.Sprintf("Ingress %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", ingress.Name, namespace, clusterIngress != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify ingress %q in namespace %q in cluster: Present=%v", ingress.Name, namespace, present)

	if present && clusterIngress != nil {
		Expect(equivalentIngress(*clusterIngress, *ingress))
	}
}

/*
   waitForIngressShardsOrFail waits for the ingress to appear in all clusters
*/
func waitForIngressShardsOrFail(namespace string, ingress *v1beta1.Ingress, clusters map[string]*cluster) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressOrFail(c.Clientset, namespace, ingress, true, FederatedIngressTimeout)
	}
}

/*
   waitForIngressShardsUpdatedOrFail waits for the ingress to be updated in all clusters
*/
func waitForIngressShardsUpdatedOrFail(namespace string, ingress *v1beta1.Ingress, clusters map[string]*cluster) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressUpdateOrFail(c.Clientset, namespace, ingress, FederatedIngressTimeout)
	}
}

/*
   waitForIngressUpdateOrFail waits until a ingress is updated in the specified cluster with same spec of federated ingress.
   If the condition is not met within timeout, it fails the calling test.
*/
func waitForIngressUpdateOrFail(clientset *release_1_3.Clientset, namespace string, ingress *v1beta1.Ingress, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated ingress shard of ingress %q in namespace %q from cluster", ingress.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterIngress, err := clientset.Ingresses(namespace).Get(ingress.Name)
		if err == nil { // We want it present, and the Get succeeded, so we're all good.
			if equivalentIngress(*clusterIngress, *ingress) {
				By(fmt.Sprintf("Success: shard of federated ingress %q in namespace %q in cluster is updated", ingress.Name, namespace))
				return true, nil
			}
			By(fmt.Sprintf("Ingress %q in namespace %q in cluster, waiting for service being updated, trying again in %s (err=%v)", ingress.Name, namespace, framework.Poll, err))
			return false, nil
		}
		By(fmt.Sprintf("Ingress %q in namespace %q in cluster, waiting for service being updated, trying again in %s (err=%v)", ingress.Name, namespace, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify ingress %q in namespace %q in cluster", ingress.Name, namespace)
}

/*
   waitForIngressShardsGoneOrFail waits for the ingress to disappear in all clusters
*/
func waitForIngressShardsGoneOrFail(namespace string, ingress *v1beta1.Ingress, clusters map[string]*cluster) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressOrFail(c.Clientset, namespace, ingress, false, FederatedIngressTimeout)
	}
}

func deleteIngressOrFail(clientset *federation_release_1_4.Clientset, namespace string, ingressName string) {
	if clientset == nil || len(namespace) == 0 || len(ingressName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteIngressOrFail: clientset: %v, namespace: %v, ingress: %v", clientset, namespace, ingressName))
	}
	err := clientset.Ingresses(namespace).Delete(ingressName, api.NewDeleteOptions(0))
	framework.ExpectNoError(err, "Error deleting ingress %q from namespace %q", ingressName, namespace)
}

func createIngressOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1beta1.Ingress {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createIngressOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federated ingress %q in namespace %q", FederatedIngressName, namespace))

	ingress := &v1beta1.Ingress{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedIngressName,
		},
		Spec: v1beta1.IngressSpec{
			Backend: &v1beta1.IngressBackend{
				ServiceName: "testingress-service",
				ServicePort: intstr.FromInt(80),
			},
		},
	}

	_, err := clientset.Extensions().Ingresses(namespace).Create(ingress)
	framework.ExpectNoError(err, "Creating ingress %q in namespace %q", ingress.Name, namespace)
	By(fmt.Sprintf("Successfully created federated ingress %q in namespace %q", FederatedIngressName, namespace))
	return ingress
}

func updateIngressOrFail(clientset *federation_release_1_4.Clientset, namespace string) (newIng *v1beta1.Ingress) {
	var err error
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createIngressOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	ingress := &v1beta1.Ingress{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedIngressName,
		},
		Spec: v1beta1.IngressSpec{
			Backend: &v1beta1.IngressBackend{
				ServiceName: "updated-testingress-service",
				ServicePort: intstr.FromInt(80),
			},
		},
	}

	for MaxRetriesOnFederatedApiserver := 0; MaxRetriesOnFederatedApiserver < 3; MaxRetriesOnFederatedApiserver++ {
		_, err = clientset.Extensions().Ingresses(namespace).Get(FederatedIngressName)
		if err != nil {
			framework.Failf("failed to get ingress %q: %v", FederatedIngressName, err)
		}
		newIng, err = clientset.Extensions().Ingresses(namespace).Update(ingress)
		if err == nil {
			describeIng(namespace)
			return
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update ingress %q: %v", FederatedIngressName, err)
		}
	}
	framework.Failf("too many retries updating ingress %q", FederatedIngressName)
	return newIng
}

func (j *federationTestJig) waitForFederatedIngress() {
	// Wait for the loadbalancer IP.
	address, err := WaitForFederatedIngressAddress(j.client, j.ing.Namespace, j.ing.Name, lbPollTimeout)
	if err != nil {
		framework.Failf("Ingress failed to acquire an IP address within %v", lbPollTimeout)
	}
	j.address = address
	framework.Logf("Found address %v for ingress %v", j.address, j.ing.Name)
	timeoutClient := &http.Client{Timeout: reqTimeout}

	// Check that all rules respond to a simple GET.
	for _, rules := range j.ing.Spec.Rules {
		proto := "http"
		for _, p := range rules.IngressRuleValue.HTTP.Paths {
			route := fmt.Sprintf("%v://%v%v", proto, address, p.Path)
			framework.Logf("Testing route %v host %v with simple GET", route, rules.Host)
			ExpectNoError(pollURL(route, rules.Host, lbPollTimeout, timeoutClient, false))
		}
	}
}

type federationTestJig struct {
	// TODO add TLS check later
	rootCAs map[string][]byte
	address string
	ing     *v1beta1.Ingress
	client  *federation_release_1_4.Clientset
}

func newFederationTestJig(c *federation_release_1_4.Clientset) *federationTestJig {
	return &federationTestJig{client: c, rootCAs: map[string][]byte{}}
}

// WaitForFederatedIngressAddress waits for the Ingress to acquire an address.
func WaitForFederatedIngressAddress(c *federation_release_1_4.Clientset, ns, ingName string, timeout time.Duration) (string, error) {
	var address string
	err := wait.PollImmediate(10*time.Second, timeout, func() (bool, error) {
		ipOrNameList, err := getFederatedIngressAddress(c, ns, ingName)
		if err != nil || len(ipOrNameList) == 0 {
			framework.Logf("Waiting for Ingress %v to acquire IP, error %v", ingName, err)
			return false, nil
		}
		address = ipOrNameList[0]
		return true, nil
	})
	return address, err
}

// getFederatedIngressAddress returns the ips/hostnames associated with the Ingress.
func getFederatedIngressAddress(client *federation_release_1_4.Clientset, ns, name string) ([]string, error) {
	ing, err := client.Extensions().Ingresses(ns).Get(name)
	if err != nil {
		return nil, err
	}
	addresses := []string{}
	for _, a := range ing.Status.LoadBalancer.Ingress {
		if a.IP != "" {
			addresses = append(addresses, a.IP)
		}
		if a.Hostname != "" {
			addresses = append(addresses, a.Hostname)
		}
	}
	return addresses, nil
}
