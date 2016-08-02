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
	"time"

	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_3"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"path/filepath"
)

const (
	ICUserAgentName = "federation-e2e-ingress-controller"

	FederatedIngressTimeout = 60 * time.Second

	FederatedIngressName    = "federated-ingress"
	FederatedIngressPodName = "federated-ingress-test-pod"
)

var FederatedIngressLabels = map[string]string{
	"foo": "bar",
}

var _ = framework.KubeDescribe("[Feature:Federation]", func() {
	defer GinkgoRecover()

	var (
		clusters                               map[string]*cluster // All clusters, keyed by cluster name
		primaryClusterName, federationName, ns string
		gceController                          *GCEIngressController
		jig                                    *testJig
		conformanceTests                       []conformanceTests
	)

	f := framework.NewDefaultFederatedFramework("federation-ingress")

	var _ = Describe("Federated Ingresses", func() {
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			// TODO: Federation API server should be able to answer this.
			if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
				federationName = DefaultFederationName
			}
			clusters = map[string]*cluster{}
			primaryClusterName = setupClusters(clusters, ICUserAgentName, federationName, f)

			// TODO: do we need run this in federation e2e?
			// f.BeforeEach()
			// TODO: require ingress to be supported by extensions client
			jig = newTestJig(f.FederationClientset.ExtensionsClient.RESTClient)
			ns = f.Namespace.Name

		})

		conformanceTests = createComformanceTests(jig, ns)

		AfterEach(func() {
			teardownClusters(clusters, f)
		})

		Describe("Ingress creation", func() {

			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)
				framework.SkipUnlessProviderIs("gce", "gke")
				By("Initializing gce controller")
				gceController = &GCEIngressController{ns: ns, Project: framework.TestContext.CloudConfig.ProjectID, c: jig.client}
				gceController.init()
			})

			AfterEach(func() {
				framework.SkipUnlessFederated(f.Client)
				if CurrentGinkgoTestDescription().Failed {
					describeIng(ns)
				}
				if jig.ing == nil {
					By("No ingress created, no cleanup necessary")
					return
				}
				By("Deleting ingress")
				jig.deleteIngress()
				//TODO: delete cluster ingresses, or skip as the cluster will be teared down
				//deleteIngressOrFail()
				//TODO: clean up federation GCE(dns name?) and ingress info in all underlying clusters
				By("Cleaning up cloud resources")

				cleanupGCE(gceController)
			})

			It("should succeed", func() {
				framework.SkipUnlessFederated(f.Client)
				for _, t := range conformanceTests {
					By(t.entryLog)
					t.execute()
					By(t.exitLog)
					jig.waitForIngress()
				}
				//TODO: check ingress info in all underlying k8s clusters
				//waitForIngressOrFail()
			})

			It("shoud create ingress with given static-ip ", func() {
				ip := gceController.staticIP(ns)
				By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ns, ip))

				jig.createIngress(filepath.Join(ingressManifestPath, "static-ip"), ns, map[string]string{
					"kubernetes.io/ingress.global-static-ip-name": ns,
					"kubernetes.io/ingress.allow-http":            "false",
				})

				By("waiting for Ingress to come up with ip: " + ip)
				httpClient := buildInsecureClient(reqTimeout)
				ExpectNoError(jig.pollURL(fmt.Sprintf("https://%v/", ip), "", lbPollTimeout, httpClient, false))

				By("should reject HTTP traffic")
				ExpectNoError(jig.pollURL(fmt.Sprintf("http://%v/", ip), "", lbPollTimeout, httpClient, true))
				//TODO: check ingress info in all underlying k8s cluster
			})
		})
	})
})

/*
   equivalent returns true if the two ingresss are equivalent.  Fields which are expected to differ between
   federated ingresss and the underlying cluster ingresss (e.g. ClusterIP, LoadBalancerIP etc) are ignored.
*/
func equivalentIngress(federationIngress, clusterIngress extensions.Ingress) bool {
	// TODO: how to decide two ingresses are equal?
	clusterIngress.Spec = federationIngress.Spec
	//clusterIngress.Spec.ExternalIPs = federationIngress.Spec.ExternalIPs
	//clusterIngress.Spec.DeprecatedPublicIPs = federationIngress.Spec.DeprecatedPublicIPs
	//clusterIngress.Spec.LoadBalancerIP = federationIngress.Spec.LoadBalancerIP
	//clusterIngress.Spec.LoadBalancerSourceRanges = federationIngress.Spec.LoadBalancerSourceRanges
	//// N.B. We cannot iterate over the port objects directly, as their values
	//// only get copied and our updates will get lost.
	//for i := range clusterIngress.Spec.Ports {
	//	clusterIngress.Spec.Ports[i].NodePort = federationIngress.Spec.Ports[i].NodePort
	//}
	return reflect.DeepEqual(clusterIngress.Spec, federationIngress.Spec)
}

/*
   waitForIngressOrFail waits until a ingress is either present or absent in the cluster specified by clientset.
   If the condition is not met within timout, it fails the calling test.
*/
func waitForIngressOrFail(clientset *release_1_3.Clientset, namespace string, ingress *extensions.Ingress, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated ingress shard of ingress %q in namespace %q from cluster", ingress.Name, namespace))
	var clusterIngress *extensions.Ingress
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterIngress, err := clientset.ExtensionsClient.Ingresses(namespace).Get(ingress.Name)
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
func waitForIngressShardsOrFail(namespace string, ingress *extensions.Ingress, clusters map[string]*cluster) {
	framework.Logf("Waiting for ingress %q in %d clusters", ingress.Name, len(clusters))
	for _, c := range clusters {
		waitForIngressOrFail(c.Clientset, namespace, ingress, true, FederatedIngressTimeout)
	}
}

func deleteIngressOrFail(clientset *federation_release_1_3.Clientset, namespace string, ingressName string) {
	var err error
	if clientset == nil || len(namespace) == 0 || len(ingressName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteIngressOrFail: clientset: %v, namespace: %v, ingress: %v", clientset, namespace, ingressName))
	}
	// TODO
	//err := clientset.Ingress(namespace).Delete(ingressName, api.NewDeleteOptions(0))
	framework.ExpectNoError(err, "Error deleting ingress %q from namespace %q", ingressName, namespace)
}
