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
	"crypto/tls"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	MaxRetriesOnFederatedApiserver = 3
	FederatedIngressTimeout        = 15 * time.Minute
	FederatedIngressDeleteTimeout  = 2 * time.Minute
	FederatedIngressName           = "federated-ingress"
	FederatedIngressServiceName    = "federated-ingress-service"
	FederatedIngressTLSSecretName  = "federated-ingress-tls-secret"
	FederatedIngressServicePodName = "federated-ingress-service-test-pod"
	FederatedIngressHost           = "test-f8n.k8s.io."

	FederatedSecretTimeout = 60 * time.Second

	// TLS Certificate and Key for the ingress resource
	// Generated using:
	// $ openssl req -nodes -x509 -newkey rsa:2048 -keyout fedingtestkey.pem -out fedingtestcrt.pem -days 2485
	// 2485 days is an arbitrary large number chosen below int32 seconds.
	FederatedIngressTLSCrt = `-----BEGIN CERTIFICATE-----
MIIDaTCCAlGgAwIBAgIJANwsCbwxm9pyMA0GCSqGSIb3DQEBCwUAMEoxCzAJBgNV
BAYTAlVTMRMwEQYDVQQIDApTb21lLVN0YXRlMQswCQYDVQQKDAJOQTEZMBcGA1UE
AwwQdGVzdC1mOG4uazhzLmlvLjAgFw0xNjEyMTYwNjA1NDRaGA8yMDg1MDEwMzA2
MDU0NFowSjELMAkGA1UEBhMCVVMxEzARBgNVBAgMClNvbWUtU3RhdGUxCzAJBgNV
BAoMAk5BMRkwFwYDVQQDDBB0ZXN0LWY4bi5rOHMuaW8uMIIBIjANBgkqhkiG9w0B
AQEFAAOCAQ8AMIIBCgKCAQEAmsHYnLhqSeO1Q6SEjaiPiLUQV8tyGfttwNQiOT5u
ULz6ZWYA40m/1hhla9KH9sJZ515Iq+jTtiVH0rUjryT96SjxitLCAZlxVwQ63B50
aZF2T2OPSzvrmN+J6VGcRIq0N8fUeyp2WTIEdWlpQ7DTmDNArQqFSIvJndkLow3d
hec7O+PErnvZQQC9zqa23rGndDzlgDJ4HJGAQNm3uYVh5WHv+wziP67T/82bEGgO
A6EdDPWzpYxzAA1wsqz9lX5jitlbKdI56698fPR2KRelySf7OXVvZCS4/ED1lF4k
b7fQgtBhAWe1BkuAMUl7vdRjMps7nkxmBSuxBkVQ7sb5AwIDAQABo1AwTjAdBgNV
HQ4EFgQUjf53O/W/iE2mxuJkNjZGUfjJ9RUwHwYDVR0jBBgwFoAUjf53O/W/iE2m
xuJkNjZGUfjJ9RUwDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEABE7B
bAiORDBA3qE5lh6JCs/lEfz93E/gOhD9oDnm9SRND4kjy7qeGxk4Wzsd/Vr+R2mi
EZ40d4MA/mCCPnYsNQoEXMFc8IvwAbzkhh2gqTNgG0/Ks0A1mIPQNpvUcSetS4IV
732DvB3nSnFtlzf6afw+V1Vf5ydRNuM/c9GEOOHSz+rs+9M364d+wNaFD64M72ol
iDMAdtcrhOqkQi0lUING904jlJcyYM5oVNCCtme4F8nkIX9bxP/9Ea6VhDGPeJiX
tVwZuudkoEbrFlEYbyLrbVeVa9oTf4Jn66iz49/+th+bUtEoTt9gk9Cul5TFgfzx
EscdahceC7afheq6zg==
-----END CERTIFICATE-----`

	FederatedIngressTLSKey = `-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCawdicuGpJ47VD
pISNqI+ItRBXy3IZ+23A1CI5Pm5QvPplZgDjSb/WGGVr0of2wlnnXkir6NO2JUfS
tSOvJP3pKPGK0sIBmXFXBDrcHnRpkXZPY49LO+uY34npUZxEirQ3x9R7KnZZMgR1
aWlDsNOYM0CtCoVIi8md2QujDd2F5zs748Sue9lBAL3Oprbesad0POWAMngckYBA
2be5hWHlYe/7DOI/rtP/zZsQaA4DoR0M9bOljHMADXCyrP2VfmOK2Vsp0jnrr3x8
9HYpF6XJJ/s5dW9kJLj8QPWUXiRvt9CC0GEBZ7UGS4AxSXu91GMymzueTGYFK7EG
RVDuxvkDAgMBAAECggEAYrXGPqB6W0r88XpceibL9rzXAcjorJ3s8ZPdiHnDz4fa
hxa69j6yOBMzjcSpqMFqquM+ozhM4d+BomqbqjmEI1ZUSuIHkRGYc5JlIMXkJvn7
ZsPwQGKl8cqTotjFPgrizLmPVEhPWLFImsNzuxNsw6XdWQJe5VkUbrRkccqEQ8Wt
xwq/SlRercIMnRVLOOESq8EyjOY4yDgOdIifq9K9xiI8W6nMiPs0X5AcIJoTMbCe
cX0zUqW317awDWWP8u2GswwDDm4qPeWnXOrDkDx8Eo0dWJbmxw9su0XrM6KMvEMe
2a/Fy/enr5Cc6/jgsh3gO5sa8dJ1Cu+wexcoEbez8QKBgQDMXlXJu/C7djke94s3
vGxati7AGO95bBQHW+cPuN4l0rfPZ8YuUAWD4csW4BOlUPAOukreD/SKdanigR3N
FqVPeI8rXd5kzy8/lPIOGuSkkVEpKsAJ7prFbSUVKjVPYQk2dsOEeR0r7pr2FxC9
SBhVS/LgmPYh++iny9D0aU23hQKBgQDB2t55OE+00vgoauUc10LEY+J6tiwXuNm7
43JtrH5ET4N+TJ2BOUl5f88TY/3QuTu6vYwlxjyn+LFuWQNhShX6lFMjt5zqPTdw
ZPDA+9B6a45cV3YjXjRsYidpWj0D2lJgy0DbucC4f3eIhNGyFUbAQB9npKDzOeUh
7Z+p/Grg5wKBgGUnVCLzySWgUImJUPkXZDJJ9j3SmcVpv0gdLvLTN/FUqPIZlTgb
F3+9ZL4/zrmGpCtF/gSHtSxLLPkVm2CFkvEQ5Rw76/XNrr8zw9NDcGQcISXVKRRB
a43IhhBBwf02NE8m3YNWRyAVi9G+fOSTKKgfXWnZjAoqG2/iK9ytum/ZAoGAYlP8
KIxxkYy5Jvchg4GEck0f4ZJpxxaSCoWR0yN9YHTcg8Gk2pkONbyocnNzmN17+HqQ
jdCBj8nLZedsmXqUr2dwzFskEoQ+jJoGrDyOQKoxqZELcWElQhx/VSbacAvbYRF3
snwDzxGItgx4uNWl73oW8+FDalvhZ1Y6eGR6ad0CgYEAtlNa92Fbvd3r9O2mdyWe
D2SXNMi45+wsNafX2sdkyb+qNN6qZXC9ylUl9h0zdky88JNgtAOgxIaRIdEZajnD
/Zq17sTNtgpm53x16gOAgD8M+/wmBZxA+/IKfFCubuV77MbQoPfcjT5wBMRnFQnY
Ks7c+dzaRlgDKZ6v/L/8iZU=
-----END PRIVATE KEY-----`
)

const (
	// timeout on a single http request.
	reqTimeout = 10 * time.Second
)

var _ = framework.KubeDescribe("Federated ingresses [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federated-ingress")

	// Create/delete ingress api objects
	// Validate federation apiserver, does not rely on underlying clusters or federation ingress controller.
	Describe("Federated Ingresses [NoCluster]", func() {
		AfterEach(func() {
			nsName := f.FederationNamespace.Name
			// Delete all ingresses.
			deleteAllIngressesOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			framework.SkipUnlessProviderIs("gce", "gke") // TODO: Federated ingress is not yet supported on non-GCP platforms.
			nsName := f.FederationNamespace.Name
			ingress := createIngressOrFail(f.FederationClientset, nsName, FederatedIngressServiceName, FederatedIngressTLSSecretName)
			By(fmt.Sprintf("Creation of ingress %q in namespace %q succeeded.  Deleting ingress.", ingress.Name, nsName))
			// Cleanup
			err := f.FederationClientset.Extensions().Ingresses(nsName).Delete(ingress.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting ingress %q in namespace %q", ingress.Name, ingress.Namespace)
			By(fmt.Sprintf("Deletion of ingress %q in namespace %q succeeded.", ingress.Name, nsName))
		})
	})

	// e2e cases for federation ingress controller
	var _ = Describe("Federated Ingresses [Slow]", func() {
		var (
			clusters           fedframework.ClusterSlice
			federationName, ns string
			jig                *federationTestJig
			service            *v1.Service
			secret             *v1.Secret
		)

		// register clusters in federation apiserver
		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			framework.SkipUnlessProviderIs("gce", "gke") // TODO: Federated ingress is not yet supported on non-GCP platforms.
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}
			jig = newFederationTestJig(f.FederationClientset)
			clusters = f.GetRegisteredClusters()
			ns = f.FederationNamespace.Name
			// create backend service
			service = createLBServiceOrFail(f.FederationClientset, ns, FederatedIngressServiceName)
			// create the TLS secret
			secret = createTLSSecretOrFail(f.FederationClientset, ns, FederatedIngressTLSSecretName)
			// wait for services objects sync
			waitForServiceShardsOrFail(ns, service, clusters)
			// wait for TLS secret sync
			waitForSecretShardsOrFail(ns, secret, clusters)
		})

		AfterEach(func() {
			// Delete all ingresses.
			deleteAllIngressesOrFail(f.FederationClientset, ns)
			if secret != nil {
				By("Deleting secret")
				orphanDependents := false
				deleteSecretOrFail(f.FederationClientset, ns, secret.Name, &orphanDependents)
				secret = nil
			} else {
				By("No secret to delete. Secret is nil")
			}
			if service != nil {
				By("Deleting service")
				deleteServiceOrFail(f.FederationClientset, ns, service.Name, nil)
				By("Cleanup service shards and provider resources")
				cleanupServiceShardsAndProviderResources(ns, service, clusters)
				service = nil
			} else {
				By("No service to delete. Service is nil")
			}
		})

		It("should create and update matching ingresses in underlying clusters", func() {
			ingress := createIngressOrFail(f.FederationClientset, ns, FederatedIngressServiceName, FederatedIngressTLSSecretName)
			// wait for ingress shards being created
			waitForIngressShardsOrFail(ns, ingress, clusters)
			ingress = updateIngressOrFail(f.FederationClientset, ns)
			waitForIngressShardsUpdatedOrFail(ns, ingress, clusters)
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			orphanDependents := false
			verifyCascadingDeletionForIngress(f.FederationClientset, clusters, &orphanDependents, ns)
			By(fmt.Sprintf("Verified that ingresses were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			orphanDependents := true
			verifyCascadingDeletionForIngress(f.FederationClientset, clusters, &orphanDependents, ns)
			By(fmt.Sprintf("Verified that ingresses were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			verifyCascadingDeletionForIngress(f.FederationClientset, clusters, nil, ns)
			By(fmt.Sprintf("Verified that ingresses were not deleted from underlying clusters"))
		})

		var _ = Describe("Ingress connectivity and DNS", func() {

			var backendPods BackendPodMap

			BeforeEach(func() {
				fedframework.SkipUnlessFederated(f.ClientSet)
				// create backend pod
				backendPods = createBackendPodsOrFail(clusters, ns, FederatedIngressServicePodName)
				// create ingress object
				jig.ing = createIngressOrFail(f.FederationClientset, ns, service.Name, FederatedIngressTLSSecretName)
				// wait for ingress objects sync
				waitForIngressShardsOrFail(ns, jig.ing, clusters)
				By(fmt.Sprintf("Ingress created as %v", jig.ing.Name))
			})

			AfterEach(func() {
				deleteBackendPodsOrFail(clusters, backendPods)
				backendPods = nil

				if jig.ing != nil {
					By(fmt.Sprintf("Deleting ingress %v on all clusters", jig.ing.Name))
					orphanDependents := false
					deleteIngressOrFail(f.FederationClientset, ns, jig.ing.Name, &orphanDependents)
					jig.ing = nil
				} else {
					By("No ingress to delete. Ingress is nil")
				}
			})

			PIt("should be able to discover a federated ingress service via DNS", func() {
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
			})

			PIt("should be able to connect to a federated ingress via its load balancer", func() {
				By(fmt.Sprintf("Waiting for Federated Ingress on %v", jig.ing.Name))
				// check the traffic on federation ingress
				jig.waitForFederatedIngress()
			})

		})
	})
})

// Deletes all Ingresses in the given namespace name.
func deleteAllIngressesOrFail(clientset *fedclientset.Clientset, nsName string) {
	orphanDependents := false
	err := clientset.Extensions().Ingresses(nsName).DeleteCollection(&metav1.DeleteOptions{OrphanDependents: &orphanDependents}, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Error in deleting ingresses in namespace: %s", nsName))
}

// equivalent returns true if the two ingress spec are equivalent.
func equivalentIngress(federatedIngress, clusterIngress v1beta1.Ingress) bool {
	return reflect.DeepEqual(clusterIngress.Spec, federatedIngress.Spec)
}

// verifyCascadingDeletionForIngress verifies that ingresses are deleted from
// underlying clusters when orphan dependents is false and they are not deleted
// when orphan dependents is true.
func verifyCascadingDeletionForIngress(clientset *fedclientset.Clientset, clusters fedframework.ClusterSlice, orphanDependents *bool, nsName string) {
	ingress := createIngressOrFail(clientset, nsName, FederatedIngressServiceName, FederatedIngressTLSSecretName)
	ingressName := ingress.Name
	// Check subclusters if the ingress was created there.
	By(fmt.Sprintf("Waiting for ingress %s to be created in all underlying clusters", ingressName))
	waitForIngressShardsOrFail(nsName, ingress, clusters)

	By(fmt.Sprintf("Deleting ingress %s", ingressName))
	deleteIngressOrFail(clientset, nsName, ingressName, orphanDependents)

	By(fmt.Sprintf("Verifying ingresses %s in underlying clusters", ingressName))
	errMessages := []string{}
	// ingress should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for _, cluster := range clusters {
		clusterName := cluster.Name
		_, err := cluster.Extensions().Ingresses(nsName).Get(ingressName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for ingress %s in cluster %s, expected ingress to exist", ingressName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for ingress %s in cluster %s, got error: %v", ingressName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

// waitForIngressOrFail waits until a ingress is either present or absent in the cluster specified by clientset.
// If the condition is not met within timout, it fails the calling test.
func waitForIngressOrFail(clientset *kubeclientset.Clientset, namespace string, ingress *v1beta1.Ingress, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated ingress shard of ingress %q in namespace %q from cluster", ingress.Name, namespace))
	var clusterIngress *v1beta1.Ingress
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		var err error
		clusterIngress, err = clientset.Ingresses(namespace).Get(ingress.Name, metav1.GetOptions{})
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

// waitForIngressShardsOrFail waits for the ingress to appear in all clusters
func waitForIngressShardsOrFail(namespace string, ingress *v1beta1.Ingress, clusters fedframework.ClusterSlice) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressOrFail(c.Clientset, namespace, ingress, true, FederatedIngressTimeout)
	}
}

// waitForIngressShardsUpdatedOrFail waits for the ingress to be updated in all clusters
func waitForIngressShardsUpdatedOrFail(namespace string, ingress *v1beta1.Ingress, clusters fedframework.ClusterSlice) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressUpdateOrFail(c.Clientset, namespace, ingress, FederatedIngressTimeout)
	}
}

// waitForIngressUpdateOrFail waits until a ingress is updated in the specified cluster with same spec of federated ingress.
// If the condition is not met within timeout, it fails the calling test.
func waitForIngressUpdateOrFail(clientset *kubeclientset.Clientset, namespace string, ingress *v1beta1.Ingress, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated ingress shard of ingress %q in namespace %q from cluster", ingress.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterIngress, err := clientset.Ingresses(namespace).Get(ingress.Name, metav1.GetOptions{})
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

// waitForIngressShardsGoneOrFail waits for the ingress to disappear in all clusters
func waitForIngressShardsGoneOrFail(namespace string, ingress *v1beta1.Ingress, clusters fedframework.ClusterSlice) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressOrFail(c.Clientset, namespace, ingress, false, FederatedIngressTimeout)
	}
}

func deleteIngressOrFail(clientset *fedclientset.Clientset, namespace string, ingressName string, orphanDependents *bool) {
	if clientset == nil || len(namespace) == 0 || len(ingressName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteIngressOrFail: clientset: %v, namespace: %v, ingress: %v", clientset, namespace, ingressName))
	}
	err := clientset.Ingresses(namespace).Delete(ingressName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	framework.ExpectNoError(err, "Error deleting ingress %q from namespace %q", ingressName, namespace)
	// Wait for the ingress to be deleted.
	err = wait.Poll(framework.Poll, FederatedIngressDeleteTimeout, func() (bool, error) {
		_, err := clientset.Extensions().Ingresses(namespace).Get(ingressName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting ingress %s: %v", ingressName, err)
	}
}

// TODO: quinton: This is largely a cut 'n paste of the above.  Yuck! Refactor as soon as we have a common interface implmented by both fedclientset.Clientset and kubeclientset.Clientset
func deleteClusterIngressOrFail(clusterName string, clientset *kubeclientset.Clientset, namespace string, ingressName string) {
	if clientset == nil || len(namespace) == 0 || len(ingressName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteClusterIngressOrFail: cluster: %q, clientset: %v, namespace: %v, ingress: %v", clusterName, clientset, namespace, ingressName))
	}
	err := clientset.Ingresses(namespace).Delete(ingressName, metav1.NewDeleteOptions(0))
	framework.ExpectNoError(err, "Error deleting cluster ingress %q/%q from cluster %q", namespace, ingressName, clusterName)
}

func createIngressOrFail(clientset *fedclientset.Clientset, namespace, serviceName, secretName string) *v1beta1.Ingress {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createIngressOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federated ingress %q in namespace %q", FederatedIngressName, namespace))

	ingress := &v1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: FederatedIngressName,
		},
		Spec: v1beta1.IngressSpec{
			Backend: &v1beta1.IngressBackend{
				ServiceName: serviceName,
				ServicePort: intstr.FromInt(80),
			},
			TLS: []v1beta1.IngressTLS{
				{
					SecretName: secretName,
				},
			},
		},
	}

	newIng, err := clientset.Extensions().Ingresses(namespace).Create(ingress)
	framework.ExpectNoError(err, "Creating ingress %q in namespace %q", ingress.Name, namespace)
	By(fmt.Sprintf("Successfully created federated ingress %q in namespace %q", FederatedIngressName, namespace))
	return newIng
}

func updateIngressOrFail(clientset *fedclientset.Clientset, namespace string) (newIng *v1beta1.Ingress) {
	var err error
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createIngressOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	ingress := &v1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: FederatedIngressName,
		},
		Spec: v1beta1.IngressSpec{
			Backend: &v1beta1.IngressBackend{
				ServiceName: "updated-testingress-service",
				ServicePort: intstr.FromInt(80),
			},
		},
	}

	err = waitForFederatedIngressExists(clientset, namespace, FederatedIngressName, FederatedIngressTimeout)
	if err != nil {
		framework.Failf("failed to get ingress %q: %v", FederatedIngressName, err)
	}
	for i := 0; i < MaxRetriesOnFederatedApiserver; i++ {
		newIng, err = clientset.Extensions().Ingresses(namespace).Update(ingress)
		if err == nil {
			framework.DescribeIng(namespace)
			return newIng
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update ingress %q: %v", FederatedIngressName, err)
		}
	}
	framework.Failf("too many retries updating ingress %q", FederatedIngressName)
	return nil
}

func (j *federationTestJig) waitForFederatedIngress() {
	// Wait for the loadbalancer IP.
	address, err := waitForFederatedIngressAddress(j.client, j.ing.Namespace, j.ing.Name, framework.LoadBalancerPollTimeout)
	if err != nil {
		framework.Failf("Ingress failed to acquire an IP address within %v", framework.LoadBalancerPollTimeout)
	}
	j.address = address
	framework.Logf("Found address %v for ingress %v", j.address, j.ing.Name)

	client := &http.Client{
		// This is mostly `http.DefaultTransport` except for the
		// `TLSClientConfig`.
		Transport: utilnet.SetTransportDefaults(&http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		}),
		Timeout: reqTimeout,
	}

	// Verify that simple GET works.
	route := fmt.Sprintf("https://%v", address)
	framework.Logf("Testing route %v with simple GET", route)
	framework.ExpectNoError(framework.PollURL(route, FederatedIngressHost, framework.LoadBalancerPollTimeout, framework.LoadBalancerPollInterval, client, false))
}

func createTLSSecretOrFail(clientset *fedclientset.Clientset, namespace, secretName string) *v1.Secret {
	if clientset == nil || len(namespace) == 0 {
		framework.Logf("Internal error: invalid parameters passed to createTLSSecretOrFail: clientset: %v, namespace: %v", clientset, namespace)
	}
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretName,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			"tls.crt": []byte(FederatedIngressTLSCrt),
			"tls.key": []byte(FederatedIngressTLSKey),
		},
	}
	By(fmt.Sprintf("Creating federated secret %q in namespace %q", secretName, namespace))
	newSecret, err := clientset.Core().Secrets(namespace).Create(secret)
	framework.ExpectNoError(err, "creating secret %q in namespace %q", secret.Name, namespace)
	return newSecret
}

type federationTestJig struct {
	// TODO add TLS check later
	rootCAs map[string][]byte
	address string
	ing     *v1beta1.Ingress
	client  *fedclientset.Clientset
}

func newFederationTestJig(c *fedclientset.Clientset) *federationTestJig {
	return &federationTestJig{client: c, rootCAs: map[string][]byte{}}
}

// WaitForFederatedIngressAddress waits for the Ingress to acquire an address.
func waitForFederatedIngressAddress(c *fedclientset.Clientset, ns, ingName string, timeout time.Duration) (string, error) {
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

// waitForFederatedIngressExists waits for the Ingress object exists.
func waitForFederatedIngressExists(c *fedclientset.Clientset, ns, ingName string, timeout time.Duration) error {
	err := wait.PollImmediate(10*time.Second, timeout, func() (bool, error) {
		_, err := c.Extensions().Ingresses(ns).Get(ingName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Waiting for Ingress %v, error %v", ingName, err)
			return false, nil
		}
		return true, nil
	})
	return err
}

// getFederatedIngressAddress returns the ips/hostnames associated with the Ingress.
func getFederatedIngressAddress(client *fedclientset.Clientset, ns, name string) ([]string, error) {
	ing, err := client.Extensions().Ingresses(ns).Get(name, metav1.GetOptions{})
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

func waitForSecretShardsOrFail(nsName string, secret *v1.Secret, clusters fedframework.ClusterSlice) {
	framework.Logf("Waiting for secret %q in %d clusters", secret.Name, len(clusters))
	for _, c := range clusters {
		waitForSecretOrFail(c.Clientset, nsName, secret, true, FederatedSecretTimeout)
	}
}

func waitForSecretOrFail(clientset *kubeclientset.Clientset, nsName string, secret *v1.Secret, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated secret shard of secret %q in namespace %q from cluster", secret.Name, nsName))
	var clusterSecret *v1.Secret
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		var err error
		clusterSecret, err = clientset.Core().Secrets(nsName).Get(secret.Name, metav1.GetOptions{})
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is absent", secret.Name, nsName))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is present", secret.Name, nsName))
			return true, nil // Success
		}
		By(fmt.Sprintf("Secret %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", secret.Name, nsName, clusterSecret != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify secret %q in namespace %q in cluster: Present=%v", secret.Name, nsName, present)

	if present && clusterSecret != nil {
		Expect(util.SecretEquivalent(*clusterSecret, *secret))
	}
}

func deleteSecretOrFail(clientset *fedclientset.Clientset, nsName string, secretName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting secret %q in namespace %q", secretName, nsName))
	err := clientset.Core().Secrets(nsName).Delete(secretName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting secret %q in namespace %q", secretName, nsName)
	}

	// Wait for the secret to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Core().Secrets(nsName).Get(secretName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting secret %s: %v", secretName, err)
	}
}
