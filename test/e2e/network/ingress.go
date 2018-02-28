/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	extensions "k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	NEGAnnotation           = "alpha.cloud.google.com/load-balancer-neg"
	NEGUpdateTimeout        = 2 * time.Minute
	instanceGroupAnnotation = "ingress.gcp.kubernetes.io/instance-groups"
)

var _ = SIGDescribe("Loadbalancing: L7", func() {
	defer GinkgoRecover()
	var (
		ns               string
		jig              *framework.IngressTestJig
		conformanceTests []framework.IngressConformanceTests
		cloudConfig      framework.CloudConfig
	)
	f := framework.NewDefaultFramework("ingress")

	BeforeEach(func() {
		jig = framework.NewIngressTestJig(f.ClientSet)
		ns = f.Namespace.Name
		cloudConfig = framework.TestContext.CloudConfig

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		framework.BindClusterRole(jig.Client.RbacV1beta1(), "cluster-admin", f.Namespace.Name,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})

		err := framework.WaitForAuthorizationUpdate(jig.Client.AuthorizationV1beta1(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			"", "create", schema.GroupResource{Resource: "pods"}, true)
		framework.ExpectNoError(err)
	})

	// Before enabling this loadbalancer test in any other test list you must
	// make sure the associated project has enough quota. At the time of this
	// writing a GCE project is allowed 3 backend services by default. This
	// test requires at least 5.
	//
	// Slow by design ~10m for each "It" block dominated by loadbalancer setup time
	// TODO: write similar tests for nginx, haproxy and AWS Ingress.
	Describe("GCE [Slow] [Feature:Ingress]", func() {
		var gceController *framework.GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &framework.GCEIngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init()
			Expect(err).NotTo(HaveOccurred())
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()

			By("Cleaning up cloud resources")
			Expect(gceController.CleanupGCEIngressController()).NotTo(HaveOccurred())
		})

		It("should conform to Ingress spec", func() {
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(true)
			}
		})

		It("should create ingress with given static-ip", func() {
			// ip released when the rest of lb resources are deleted in CleanupGCEIngressController
			ip := gceController.CreateStaticIP(ns)
			By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ns, ip))

			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "static-ip"), ns, map[string]string{
				framework.IngressStaticIPKey:  ns,
				framework.IngressAllowHTTPKey: "false",
			}, map[string]string{})

			By("waiting for Ingress to come up with ip: " + ip)
			httpClient := framework.BuildInsecureClient(framework.IngressReqTimeout)
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%v/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, false))

			By("should reject HTTP traffic")
			framework.ExpectNoError(framework.PollURL(fmt.Sprintf("http://%v/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, true))

			By("should have correct firewall rule for ingress")
			fw := gceController.GetFirewallRule()
			nodeTags := []string{cloudConfig.NodeTag}
			if framework.TestContext.Provider != "gce" {
				// nodeTags would be different in GKE.
				nodeTags = framework.GetNodeTags(jig.Client, cloudConfig)
			}
			expFw := jig.ConstructFirewallForIngress(gceController, nodeTags)
			// Passed the last argument as `true` to verify the backend ports is a subset
			// of the allowed ports in firewall rule, given there may be other existing
			// ingress resources and backends we are not aware of.
			Expect(framework.VerifyFirewallRule(fw, expFw, gceController.Cloud.Network, true)).NotTo(HaveOccurred())

			// TODO: uncomment the restart test once we have a way to synchronize
			// and know that the controller has resumed watching. If we delete
			// the ingress before the controller is ready we will leak.
			// By("restaring glbc")
			// restarter := NewRestartConfig(
			//	 framework.GetMasterHost(), "glbc", glbcHealthzPort, restartPollInterval, restartTimeout)
			// restarter.restart()
			// By("should continue serving on provided static-ip for 30 seconds")
			// framework.ExpectNoError(jig.verifyURL(fmt.Sprintf("https://%v/", ip), "", 30, 1*time.Second, httpClient))
		})

		It("should update ingress while sync failures occur on other ingresses", func() {
			By("Creating ingresses that would fail on sync.")
			ingFailTLSBackend := &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "ing-fail-on-tls-backend",
				},
				Spec: extensions.IngressSpec{
					TLS: []extensions.IngressTLS{
						{SecretName: "tls-secret-notexist"},
					},
					Backend: &extensions.IngressBackend{
						ServiceName: "echoheaders-notexist",
						ServicePort: intstr.IntOrString{
							Type:   intstr.Int,
							IntVal: 80,
						},
					},
				},
			}
			_, err := jig.Client.ExtensionsV1beta1().Ingresses(ns).Create(ingFailTLSBackend)
			defer func() {
				if err := jig.Client.ExtensionsV1beta1().Ingresses(ns).Delete(ingFailTLSBackend.Name, nil); err != nil {
					framework.Logf("Failed to delete ingress %s: %v", ingFailTLSBackend.Name, err)
				}
			}()
			Expect(err).NotTo(HaveOccurred())

			ingFailRules := &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "ing-fail-on-rules",
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path: "/foo",
											Backend: extensions.IngressBackend{
												ServiceName: "echoheaders-notexist",
												ServicePort: intstr.IntOrString{
													Type:   intstr.Int,
													IntVal: 80,
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			_, err = jig.Client.ExtensionsV1beta1().Ingresses(ns).Create(ingFailRules)
			defer func() {
				if err := jig.Client.ExtensionsV1beta1().Ingresses(ns).Delete(ingFailRules.Name, nil); err != nil {
					framework.Logf("Failed to delete ingress %s: %v", ingFailRules.Name, err)
				}
			}()
			Expect(err).NotTo(HaveOccurred())

			By("Creating a basic HTTP ingress and wait for it to come up")
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "http"), ns, nil, nil)
			jig.WaitForIngress(true)

			By("Updating the path on ingress and wait for it to take effect")
			jig.Update(func(ing *extensions.Ingress) {
				updatedRule := extensions.IngressRule{
					Host: "ingress.test.com",
					IngressRuleValue: extensions.IngressRuleValue{
						HTTP: &extensions.HTTPIngressRuleValue{
							Paths: []extensions.HTTPIngressPath{
								{
									Path: "/test",
									// Copy backend from the first rule.
									Backend: ing.Spec.Rules[0].HTTP.Paths[0].Backend,
								},
							},
						},
					},
				}
				// Replace the first rule.
				ing.Spec.Rules[0] = updatedRule
			})
			// Wait for change to take effect on the updated ingress.
			jig.WaitForIngress(false)
		})

		It("should not reconcile manually modified health check for ingress", func() {
			By("Creating a basic HTTP ingress and wait for it to come up.")
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "http"), ns, nil, nil)
			jig.WaitForIngress(true)

			// Get cluster UID.
			clusterID, err := framework.GetClusterID(f.ClientSet)
			Expect(err).NotTo(HaveOccurred())
			// Get the related nodeports.
			nodePorts := jig.GetIngressNodePorts(false)
			Expect(len(nodePorts)).ToNot(Equal(0))

			// Filter health check using cluster UID as the suffix.
			By("Retrieving relevant health check resources from GCE.")
			gceCloud := gceController.Cloud.Provider.(*gcecloud.GCECloud)
			hcs, err := gceCloud.ListHealthChecks()
			Expect(err).NotTo(HaveOccurred())
			var hcToChange *compute.HealthCheck
			for _, hc := range hcs {
				if strings.HasSuffix(hc.Name, clusterID) {
					Expect(hc.HttpHealthCheck).NotTo(BeNil())
					if fmt.Sprintf("%d", hc.HttpHealthCheck.Port) == nodePorts[0] {
						hcToChange = hc
						break
					}
				}
			}
			Expect(hcToChange).NotTo(BeNil())

			By(fmt.Sprintf("Modifying health check %v without involving ingress.", hcToChange.Name))
			// Change timeout from 60s to 25s.
			hcToChange.TimeoutSec = 25
			// Change path from /healthz to /.
			hcToChange.HttpHealthCheck.RequestPath = "/"
			err = gceCloud.UpdateHealthCheck(hcToChange)
			Expect(err).NotTo(HaveOccurred())

			// Add one more path to ingress to trigger resource syncing.
			By("Adding a new path to ingress and wait for it to take effect.")
			jig.Update(func(ing *extensions.Ingress) {
				ing.Spec.Rules = append(ing.Spec.Rules, extensions.IngressRule{
					Host: "ingress.test.com",
					IngressRuleValue: extensions.IngressRuleValue{
						HTTP: &extensions.HTTPIngressRuleValue{
							Paths: []extensions.HTTPIngressPath{
								{
									Path: "/test",
									// Copy backend from the first rule.
									Backend: ing.Spec.Rules[0].HTTP.Paths[0].Backend,
								},
							},
						},
					},
				})
			})
			// Wait for change to take effect before checking the health check resource.
			jig.WaitForIngress(false)

			// Validate the modified fields on health check are intact.
			By("Checking if the modified health check is unchanged.")
			hcAfterSync, err := gceCloud.GetHealthCheck(hcToChange.Name)
			Expect(err).NotTo(HaveOccurred())
			Expect(hcAfterSync.HttpHealthCheck).ToNot(Equal(nil))
			Expect(hcAfterSync.TimeoutSec).To(Equal(hcToChange.TimeoutSec))
			Expect(hcAfterSync.HttpHealthCheck.RequestPath).To(Equal(hcToChange.HttpHealthCheck.RequestPath))
		})

		It("should create ingress with pre-shared certificate", func() {
			preSharedCertName := "test-pre-shared-cert"
			By(fmt.Sprintf("Creating ssl certificate %q on GCE", preSharedCertName))
			testHostname := "test.ingress.com"
			cert, key, err := framework.GenerateRSACerts(testHostname, true)
			Expect(err).NotTo(HaveOccurred())
			gceCloud, err := framework.GetGCECloud()
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				// We would not be able to delete the cert until ingress controller
				// cleans up the target proxy that references it.
				By("Deleting ingress before deleting ssl certificate")
				if jig.Ingress != nil {
					jig.TryDeleteIngress()
				}
				By(fmt.Sprintf("Deleting ssl certificate %q on GCE", preSharedCertName))
				err := wait.Poll(framework.LoadBalancerPollInterval, framework.LoadBalancerCleanupTimeout, func() (bool, error) {
					if err := gceCloud.DeleteSslCertificate(preSharedCertName); err != nil && !errors.IsNotFound(err) {
						framework.Logf("Failed to delete ssl certificate %q: %v. Retrying...", preSharedCertName, err)
						return false, nil
					}
					return true, nil
				})
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to delete ssl certificate %q: %v", preSharedCertName, err))
			}()
			_, err = gceCloud.CreateSslCertificate(&compute.SslCertificate{
				Name:        preSharedCertName,
				Certificate: string(cert),
				PrivateKey:  string(key),
				Description: "pre-shared cert for ingress testing",
			})
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create ssl certificate %q: %v", preSharedCertName, err))

			By("Creating an ingress referencing the pre-shared certificate")
			// Create an ingress referencing this cert using pre-shared-cert annotation.
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "pre-shared-cert"), ns, map[string]string{
				framework.IngressPreSharedCertKey: preSharedCertName,
				framework.IngressAllowHTTPKey:     "false",
			}, map[string]string{})

			By("Test that ingress works with the pre-shared certificate")
			err = jig.WaitForIngressWithCert(true, []string{testHostname}, cert)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Unexpected error while waiting for ingress: %v", err))
		})

		It("should create ingress with backside re-encryption", func() {
			By("Creating a set of ingress, service and deployment that have backside re-encryption configured")
			deployCreated, svcCreated, ingCreated, err := framework.CreateReencryptionIngress(f.ClientSet, f.Namespace.Name)
			defer func() {
				By("Cleaning up re-encryption ingress, service and deployment")
				if errs := framework.CleanupReencryptionIngress(f.ClientSet, deployCreated, svcCreated, ingCreated); len(errs) > 0 {
					framework.Failf("Failed to cleanup re-encryption ingress: %v", errs)
				}
			}()
			Expect(err).NotTo(HaveOccurred(), "Failed to create re-encryption ingress")

			By(fmt.Sprintf("Waiting for ingress %s to come up", ingCreated.Name))
			ingIP, err := jig.WaitForIngressAddress(f.ClientSet, f.Namespace.Name, ingCreated.Name, framework.LoadBalancerPollTimeout)
			Expect(err).NotTo(HaveOccurred(), "Failed to wait for ingress IP")

			By(fmt.Sprintf("Polling on address %s and verify the backend is serving HTTPS", ingIP))
			timeoutClient := &http.Client{Timeout: framework.IngressReqTimeout}
			err = wait.PollImmediate(framework.LoadBalancerPollInterval, framework.LoadBalancerPollTimeout, func() (bool, error) {
				resp, err := framework.SimpleGET(timeoutClient, fmt.Sprintf("http://%s", ingIP), "")
				if err != nil {
					framework.Logf("SimpleGET failed: %v", err)
					return false, nil
				}
				if !strings.Contains(resp, "request_scheme=https") {
					return false, fmt.Errorf("request wasn't served by HTTPS, response body: %s", resp)
				}
				framework.Logf("Poll succeeded, request was served by HTTPS")
				return true, nil
			})
			Expect(err).NotTo(HaveOccurred(), "Failed to verify backside re-encryption ingress")
		})

		It("multicluster ingress should get instance group annotation", func() {
			name := "echomap"
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "http"), ns, map[string]string{
				framework.IngressClassKey: framework.MulticlusterIngressClassValue,
			}, map[string]string{})

			By(fmt.Sprintf("waiting for Ingress %s to come up", name))
			pollErr := wait.Poll(2*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				ing, err := f.ClientSet.ExtensionsV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				annotations := ing.Annotations
				if annotations == nil || annotations[instanceGroupAnnotation] == "" {
					framework.Logf("Waiting for ingress to get %s annotation. Found annotations: %v", instanceGroupAnnotation, annotations)
					return false, nil
				}
				return true, nil
			})
			if pollErr != nil {
				framework.ExpectNoError(fmt.Errorf("Timed out waiting for ingress %s to get %s annotation", name, instanceGroupAnnotation))
			}
			// TODO(nikhiljindal): Check the instance group annotation value and verify with a multizone cluster.
		})

		// TODO: Implement a multizone e2e that verifies traffic reaches each
		// zone based on pod labels.
	})

	Describe("GCE [Slow] [Feature:NEG]", func() {
		var gceController *framework.GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &framework.GCEIngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init()
			Expect(err).NotTo(HaveOccurred())
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()

			By("Cleaning up cloud resources")
			Expect(gceController.CleanupGCEIngressController()).NotTo(HaveOccurred())
		})

		It("should conform to Ingress spec", func() {
			jig.PollInterval = 5 * time.Second
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns, map[string]string{
				NEGAnnotation: "true",
			})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(true)
				usingNeg, err := gceController.BackendServiceUsingNEG(jig.GetIngressNodePorts(false))
				Expect(err).NotTo(HaveOccurred())
				Expect(usingNeg).To(BeTrue())
			}
		})

		It("should be able to switch between IG and NEG modes", func() {
			var err error
			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetIngressNodePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())

			By("Switch backend service to use IG")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[NEGAnnotation] = "false"
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			wait.Poll(5*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				return gceController.BackendServiceUsingIG(jig.GetIngressNodePorts(true))
			})
			jig.WaitForIngress(true)

			By("Switch backend service to use NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[NEGAnnotation] = "true"
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			wait.Poll(5*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				return gceController.BackendServiceUsingNEG(jig.GetIngressNodePorts(false))
			})
			jig.WaitForIngress(true)
		})

		It("should sync endpoints to NEG", func() {
			name := "hostname"
			scaleAndValidateNEG := func(num int) {
				scale, err := f.ClientSet.ExtensionsV1beta1().Deployments(ns).GetScale(name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())
				if scale.Spec.Replicas != int32(num) {
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.ExtensionsV1beta1().Deployments(ns).UpdateScale(name, scale)
					Expect(err).NotTo(HaveOccurred())
				}
				wait.Poll(10*time.Second, NEGUpdateTimeout, func() (bool, error) {
					res, err := jig.GetDistinctResponseFromIngress()
					if err != nil {
						return false, nil
					}
					return res.Len() == num, nil
				})
			}

			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetIngressNodePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())
			// initial replicas number is 1
			scaleAndValidateNEG(1)

			By("Scale up number of backends to 5")
			scaleAndValidateNEG(5)

			By("Scale down number of backends to 3")
			scaleAndValidateNEG(3)

			By("Scale up number of backends to 6")
			scaleAndValidateNEG(6)

			By("Scale down number of backends to 2")
			scaleAndValidateNEG(3)
		})

		It("rolling update backend pods should not cause service disruption", func() {
			name := "hostname"
			replicas := 8
			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(framework.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetIngressNodePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())

			By(fmt.Sprintf("Scale backend replicas to %d", replicas))
			scale, err := f.ClientSet.ExtensionsV1beta1().Deployments(ns).GetScale(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			scale.Spec.Replicas = int32(replicas)
			_, err = f.ClientSet.ExtensionsV1beta1().Deployments(ns).UpdateScale(name, scale)
			Expect(err).NotTo(HaveOccurred())
			wait.Poll(10*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				if err != nil {
					return false, nil
				}
				return res.Len() == replicas, nil
			})

			By("Trigger rolling update and observe service disruption")
			deploy, err := f.ClientSet.ExtensionsV1beta1().Deployments(ns).Get(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			// trigger by changing graceful termination period to 60 seconds
			gracePeriod := int64(60)
			deploy.Spec.Template.Spec.TerminationGracePeriodSeconds = &gracePeriod
			_, err = f.ClientSet.ExtensionsV1beta1().Deployments(ns).Update(deploy)
			Expect(err).NotTo(HaveOccurred())
			wait.Poll(10*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				Expect(err).NotTo(HaveOccurred())
				deploy, err := f.ClientSet.ExtensionsV1beta1().Deployments(ns).Get(name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())
				if int(deploy.Status.UpdatedReplicas) == replicas {
					if res.Len() == replicas {
						return true, nil
					} else {
						framework.Logf("Expecting %d different responses, but got %d.", replicas, res.Len())
						return false, nil
					}

				} else {
					framework.Logf("Waiting for rolling update to finished. Keep sending traffic.")
					return false, nil
				}
			})
		})
	})

	Describe("GCE [Slow] [Feature:kubemci]", func() {
		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			jig.Class = framework.MulticlusterIngressClassValue
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()
		})

		It("should conform to Ingress spec", func() {
			jig.PollInterval = 5 * time.Second
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(true /*waitForNodePort*/)
			}
		})
	})

	// Time: borderline 5m, slow by design
	Describe("[Slow] Nginx", func() {
		var nginxController *framework.NginxIngressController

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing nginx controller")
			jig.Class = "nginx"
			nginxController = &framework.NginxIngressController{Ns: ns, Client: jig.Client}

			// TODO: This test may fail on other platforms. We can simply skip it
			// but we want to allow easy testing where a user might've hand
			// configured firewalls.
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(framework.GcloudComputeResourceCreate("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID, "--allow", "tcp:80,tcp:443", "--network", framework.TestContext.CloudConfig.Network))
			} else {
				framework.Logf("WARNING: Not running on GCE/GKE, cannot create firewall rules for :80, :443. Assuming traffic can reach the external ips of all nodes in cluster on those ports.")
			}

			nginxController.Init()
		})

		AfterEach(func() {
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(framework.GcloudComputeResourceDelete("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID))
			}
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
				return
			}
			By("Deleting ingress")
			jig.TryDeleteIngress()
		})

		It("should conform to Ingress spec", func() {
			// Poll more frequently to reduce e2e completion time.
			// This test runs in presubmit.
			jig.PollInterval = 5 * time.Second
			conformanceTests = framework.CreateIngressComformanceTests(jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(false)
			}
		})
	})
})
