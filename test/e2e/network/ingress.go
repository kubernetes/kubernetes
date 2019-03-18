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
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/ingress"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	negUpdateTimeout        = 2 * time.Minute
	instanceGroupAnnotation = "ingress.gcp.kubernetes.io/instance-groups"
)

var _ = SIGDescribe("Loadbalancing: L7", func() {
	defer GinkgoRecover()
	var (
		ns               string
		jig              *ingress.TestJig
		conformanceTests []ingress.ConformanceTests
		cloudConfig      framework.CloudConfig
	)
	f := framework.NewDefaultFramework("ingress")

	BeforeEach(func() {
		jig = ingress.NewIngressTestJig(f.ClientSet)
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
		var gceController *gce.GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &gce.GCEIngressController{
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
			conformanceTests = ingress.CreateIngressComformanceTests(jig, ns, map[string]string{})
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
			executeStaticIPHttpsOnlyTest(f, jig, ns, ip)

			By("should have correct firewall rule for ingress")
			fw := gceController.GetFirewallRule()
			nodeTags := []string{cloudConfig.NodeTag}
			if framework.TestContext.Provider != "gce" {
				// nodeTags would be different in GKE.
				nodeTags = gce.GetNodeTags(jig.Client, cloudConfig)
			}
			expFw := jig.ConstructFirewallForIngress(gceController.GetFirewallRuleName(), nodeTags)
			// Passed the last argument as `true` to verify the backend ports is a subset
			// of the allowed ports in firewall rule, given there may be other existing
			// ingress resources and backends we are not aware of.
			Expect(gce.VerifyFirewallRule(fw, expFw, gceController.Cloud.Network, true)).NotTo(HaveOccurred())

			// TODO: uncomment the restart test once we have a way to synchronize
			// and know that the controller has resumed watching. If we delete
			// the ingress before the controller is ready we will leak.
			// By("restarting glbc")
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
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "http"), ns, nil, nil)
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
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "http"), ns, nil, nil)
			jig.WaitForIngress(true)

			// Get cluster UID.
			clusterID, err := gce.GetClusterID(f.ClientSet)
			Expect(err).NotTo(HaveOccurred())
			// Get the related nodeports.
			nodePorts := jig.GetIngressNodePorts(false)
			Expect(len(nodePorts)).ToNot(Equal(0))

			// Filter health check using cluster UID as the suffix.
			By("Retrieving relevant health check resources from GCE.")
			gceCloud, err := gce.GetGCECloud()
			Expect(err).NotTo(HaveOccurred())
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
			executePresharedCertTest(f, jig, "")
		})

		It("should create ingress with backend HTTPS", func() {
			executeBacksideBacksideHTTPSTest(f, jig, "")
		})

		It("should support multiple TLS certs", func() {
			By("Creating an ingress with no certs.")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "multiple-certs"), ns, map[string]string{
				ingress.IngressStaticIPKey: ns,
			}, map[string]string{})

			By("Adding multiple certs to the ingress.")
			hosts := []string{"test1.ingress.com", "test2.ingress.com", "test3.ingress.com", "test4.ingress.com"}
			secrets := []string{"tls-secret-1", "tls-secret-2", "tls-secret-3", "tls-secret-4"}
			certs := [][]byte{}
			for i, host := range hosts {
				jig.AddHTTPS(secrets[i], host)
				certs = append(certs, jig.GetRootCA(secrets[i]))
			}
			for i, host := range hosts {
				err := jig.WaitForIngressWithCert(true, []string{host}, certs[i])
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Unexpected error while waiting for ingress: %v", err))
			}

			By("Remove all but one of the certs on the ingress.")
			jig.RemoveHTTPS(secrets[1])
			jig.RemoveHTTPS(secrets[2])
			jig.RemoveHTTPS(secrets[3])

			By("Test that the remaining cert is properly served.")
			err := jig.WaitForIngressWithCert(true, []string{hosts[0]}, certs[0])
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Unexpected error while waiting for ingress: %v", err))

			By("Add back one of the certs that was removed and check that all certs are served.")
			jig.AddHTTPS(secrets[1], hosts[1])
			for i, host := range hosts[:2] {
				err := jig.WaitForIngressWithCert(true, []string{host}, certs[i])
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Unexpected error while waiting for ingress: %v", err))
			}
		})

		It("multicluster ingress should get instance group annotation", func() {
			name := "echomap"
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "http"), ns, map[string]string{
				ingress.IngressClassKey: ingress.MulticlusterIngressClassValue,
			}, map[string]string{})

			By(fmt.Sprintf("waiting for Ingress %s to get instance group annotation", name))
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

			// Verify that the ingress does not get other annotations like url-map, target-proxy, backends, etc.
			// Note: All resources except the firewall rule have an annotation.
			umKey := ingress.StatusPrefix + "/url-map"
			fwKey := ingress.StatusPrefix + "/forwarding-rule"
			tpKey := ingress.StatusPrefix + "/target-proxy"
			fwsKey := ingress.StatusPrefix + "/https-forwarding-rule"
			tpsKey := ingress.StatusPrefix + "/https-target-proxy"
			scKey := ingress.StatusPrefix + "/ssl-cert"
			beKey := ingress.StatusPrefix + "/backends"
			wait.Poll(2*time.Second, time.Minute, func() (bool, error) {
				ing, err := f.ClientSet.ExtensionsV1beta1().Ingresses(ns).Get(name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				annotations := ing.Annotations
				if annotations != nil && (annotations[umKey] != "" || annotations[fwKey] != "" ||
					annotations[tpKey] != "" || annotations[fwsKey] != "" || annotations[tpsKey] != "" ||
					annotations[scKey] != "" || annotations[beKey] != "") {
					framework.Failf("unexpected annotations. Expected to not have annotations for urlmap, forwarding rule, target proxy, ssl cert and backends, got: %v", annotations)
					return true, nil
				}
				return false, nil
			})

			// Verify that the controller does not create any other resource except instance group.
			// TODO(59778): Check GCE resources specific to this ingress instead of listing all resources.
			if len(gceController.ListUrlMaps()) != 0 {
				framework.Failf("unexpected url maps, expected none, got: %v", gceController.ListUrlMaps())
			}
			if len(gceController.ListGlobalForwardingRules()) != 0 {
				framework.Failf("unexpected forwarding rules, expected none, got: %v", gceController.ListGlobalForwardingRules())
			}
			if len(gceController.ListTargetHttpProxies()) != 0 {
				framework.Failf("unexpected target http proxies, expected none, got: %v", gceController.ListTargetHttpProxies())
			}
			if len(gceController.ListTargetHttpsProxies()) != 0 {
				framework.Failf("unexpected target https proxies, expected none, got: %v", gceController.ListTargetHttpProxies())
			}
			if len(gceController.ListSslCertificates()) != 0 {
				framework.Failf("unexpected ssl certificates, expected none, got: %v", gceController.ListSslCertificates())
			}
			if len(gceController.ListGlobalBackendServices()) != 0 {
				framework.Failf("unexpected backend service, expected none, got: %v", gceController.ListGlobalBackendServices())
			}
			// Controller does not have a list command for firewall rule. We use get instead.
			if fw, err := gceController.GetFirewallRuleOrError(); err == nil {
				framework.Failf("unexpected nil error in getting firewall rule, expected firewall NotFound, got firewall: %v", fw)
			}

			// TODO(nikhiljindal): Check the instance group annotation value and verify with a multizone cluster.
		})

		// TODO: remove [Unreleased] tag to once the new GCE API GO client gets revendored in ingress-gce repo
		It("should be able to switch between HTTPS and HTTP2 modes [Unreleased]", func() {
			httpsScheme := "request_scheme=https"

			By("Create a basic HTTP2 ingress")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "http2"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)

			address, err := jig.WaitForIngressAddress(jig.Client, jig.Ingress.Namespace, jig.Ingress.Name, framework.LoadBalancerPollTimeout)

			By(fmt.Sprintf("Polling on address %s and verify the backend is serving HTTP2", address))
			detectHttpVersionAndSchemeTest(f, jig, address, "request_version=2", httpsScheme)

			By("Switch backend service to use HTTPS")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.ServiceApplicationProtocolKey] = `{"http2":"HTTPS"}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectHttpVersionAndSchemeTest(f, jig, address, "request_version=1.1", httpsScheme)

			By("Switch backend service to use HTTP2")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.ServiceApplicationProtocolKey] = `{"http2":"HTTP2"}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectHttpVersionAndSchemeTest(f, jig, address, "request_version=2", httpsScheme)
		})

		// TODO: Implement a multizone e2e that verifies traffic reaches each
		// zone based on pod labels.
	})

	Describe("GCE [Slow] [Feature:NEG]", func() {
		var gceController *gce.GCEIngressController

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing gce controller")
			gceController = &gce.GCEIngressController{
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
			conformanceTests = ingress.CreateIngressComformanceTests(jig, ns, map[string]string{
				ingress.NEGAnnotation: `{"ingress": true}`,
			})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(true)
				usingNeg, err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
				Expect(err).NotTo(HaveOccurred())
				Expect(usingNeg).To(BeTrue())
			}
		})

		It("should be able to switch between IG and NEG modes", func() {
			var err error
			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())

			By("Switch backend service to use IG")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.NEGAnnotation] = `{"ingress": false}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			err = wait.Poll(5*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				return gceController.BackendServiceUsingIG(jig.GetServicePorts(false))
			})
			Expect(err).NotTo(HaveOccurred(), "Expect backend service to target IG, but failed to observe")
			jig.WaitForIngress(true)

			By("Switch backend service to use NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.NEGAnnotation] = `{"ingress": true}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			err = wait.Poll(5*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				return gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
			})
			Expect(err).NotTo(HaveOccurred(), "Expect backend service to target NEG, but failed to observe")
			jig.WaitForIngress(true)
		})

		It("should be able to create a ClusterIP service", func() {
			var err error
			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg-clusterip"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			svcPorts := jig.GetServicePorts(false)
			usingNEG, err := gceController.BackendServiceUsingNEG(svcPorts)
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue(), "Expect backend service to be using NEG. But not.")

			// ClusterIP ServicePorts have no NodePort
			for _, sp := range svcPorts {
				Expect(sp.NodePort).To(Equal(int32(0)))
			}
		})

		It("should sync endpoints to NEG", func() {
			name := "hostname"
			scaleAndValidateNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())
				if scale.Spec.Replicas != int32(num) {
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(name, scale)
					Expect(err).NotTo(HaveOccurred())
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					res, err := jig.GetDistinctResponseFromIngress()
					if err != nil {
						return false, nil
					}
					framework.Logf("Expecting %d backends, got %d", num, res.Len())
					return res.Len() == num, nil
				})
				Expect(err).NotTo(HaveOccurred())
			}

			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			jig.WaitForIngressToStable()
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
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
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			jig.WaitForIngressToStable()
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())

			By(fmt.Sprintf("Scale backend replicas to %d", replicas))
			scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			scale.Spec.Replicas = int32(replicas)
			_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(name, scale)
			Expect(err).NotTo(HaveOccurred())

			err = wait.Poll(10*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				if err != nil {
					return false, nil
				}
				return res.Len() == replicas, nil
			})
			Expect(err).NotTo(HaveOccurred())

			By("Trigger rolling update and observe service disruption")
			deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			// trigger by changing graceful termination period to 60 seconds
			gracePeriod := int64(60)
			deploy.Spec.Template.Spec.TerminationGracePeriodSeconds = &gracePeriod
			_, err = f.ClientSet.AppsV1().Deployments(ns).Update(deploy)
			Expect(err).NotTo(HaveOccurred())
			err = wait.Poll(10*time.Second, framework.LoadBalancerPollTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				Expect(err).NotTo(HaveOccurred())
				deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(name, metav1.GetOptions{})
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
			Expect(err).NotTo(HaveOccurred())
		})

		It("should sync endpoints for both Ingress-referenced NEG and standalone NEG", func() {
			name := "hostname"
			expectedKeys := []int32{80, 443}

			scaleAndValidateExposedNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(name, metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred())
				if scale.Spec.Replicas != int32(num) {
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(name, scale)
					Expect(err).NotTo(HaveOccurred())
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					svc, err := f.ClientSet.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
					Expect(err).NotTo(HaveOccurred())

					var status ingress.NegStatus
					v, ok := svc.Annotations[ingress.NEGStatusAnnotation]
					if !ok {
						// Wait for NEG sync loop to find NEGs
						framework.Logf("Waiting for %v, got: %+v", ingress.NEGStatusAnnotation, svc.Annotations)
						return false, nil
					}
					err = json.Unmarshal([]byte(v), &status)
					if err != nil {
						framework.Logf("Error in parsing Expose NEG annotation: %v", err)
						return false, nil
					}
					framework.Logf("Got %v: %v", ingress.NEGStatusAnnotation, v)

					// Expect 2 NEGs to be created based on the test setup (neg-exposed)
					if len(status.NetworkEndpointGroups) != 2 {
						framework.Logf("Expected 2 NEGs, got %d", len(status.NetworkEndpointGroups))
						return false, nil
					}

					for _, port := range expectedKeys {
						if _, ok := status.NetworkEndpointGroups[port]; !ok {
							framework.Logf("Expected ServicePort key %v, but does not exist", port)
						}
					}

					if len(status.NetworkEndpointGroups) != len(expectedKeys) {
						framework.Logf("Expected length of %+v to equal length of %+v, but does not", status.NetworkEndpointGroups, expectedKeys)
					}

					gceCloud, err := gce.GetGCECloud()
					Expect(err).NotTo(HaveOccurred())
					for _, neg := range status.NetworkEndpointGroups {
						networkEndpoints, err := gceCloud.ListNetworkEndpoints(neg, gceController.Cloud.Zone, false)
						Expect(err).NotTo(HaveOccurred())
						if len(networkEndpoints) != num {
							framework.Logf("Expect number of endpoints to be %d, but got %d", num, len(networkEndpoints))
							return false, nil
						}
					}

					return true, nil
				})
				Expect(err).NotTo(HaveOccurred())
			}

			By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			usingNEG, err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(usingNEG).To(BeTrue())
			// initial replicas number is 1
			scaleAndValidateExposedNEG(1)

			By("Scale up number of backends to 5")
			scaleAndValidateExposedNEG(5)

			By("Scale down number of backends to 3")
			scaleAndValidateExposedNEG(3)

			By("Scale up number of backends to 6")
			scaleAndValidateExposedNEG(6)

			By("Scale down number of backends to 2")
			scaleAndValidateExposedNEG(3)
		})

		It("should create NEGs for all ports with the Ingress annotation, and NEGs for the standalone annotation otherwise", func() {
			By("Create a basic HTTP ingress using standalone NEG")
			jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)

			name := "hostname"
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Add Ingress annotation - NEGs should stay the same.
			By("Adding NEG Ingress annotation")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"80":{},"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Modify exposed NEG annotation, but keep ingress annotation
			By("Modifying exposed NEG annotation, but keep Ingress annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Remove Ingress annotation. Expect 1 NEG
			By("Disabling Ingress annotation, but keeping one standalone NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				svc.Annotations[ingress.NEGAnnotation] = `{"ingress":false,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 1)

			// Remove NEG annotation entirely. Expect 0 NEGs.
			By("Removing NEG annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, svc := range svcList.Items {
				delete(svc.Annotations, ingress.NEGAnnotation)
				// Service cannot be ClusterIP if it's using Instance Groups.
				svc.Spec.Type = v1.ServiceTypeNodePort
				_, err = f.ClientSet.CoreV1().Services(ns).Update(&svc)
				Expect(err).NotTo(HaveOccurred())
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 0)
		})
	})

	Describe("GCE [Slow] [Feature:kubemci]", func() {
		var gceController *gce.GCEIngressController
		var ipName, ipAddress string

		// Platform specific setup
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			jig.Class = ingress.MulticlusterIngressClassValue
			jig.PollInterval = 5 * time.Second
			By("Initializing gce controller")
			gceController = &gce.GCEIngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init()
			Expect(err).NotTo(HaveOccurred())

			// TODO(https://github.com/GoogleCloudPlatform/k8s-multicluster-ingress/issues/19):
			// Kubemci should reserve a static ip if user has not specified one.
			ipName = "kubemci-" + string(uuid.NewUUID())
			// ip released when the rest of lb resources are deleted in CleanupGCEIngressController
			ipAddress = gceController.CreateStaticIP(ipName)
			By(fmt.Sprintf("allocated static ip %v: %v through the GCE cloud provider", ipName, ipAddress))
		})

		// Platform specific cleanup
		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				framework.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				By("No ingress created, no cleanup necessary")
			} else {
				By("Deleting ingress")
				jig.TryDeleteIngress()
			}

			By("Cleaning up cloud resources")
			Expect(gceController.CleanupGCEIngressController()).NotTo(HaveOccurred())
		})

		It("should conform to Ingress spec", func() {
			conformanceTests = ingress.CreateIngressComformanceTests(jig, ns, map[string]string{
				ingress.IngressStaticIPKey: ipName,
			})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(false /*waitForNodePort*/)
			}
		})

		It("should create ingress with pre-shared certificate", func() {
			executePresharedCertTest(f, jig, ipName)
		})

		It("should create ingress with backend HTTPS", func() {
			executeBacksideBacksideHTTPSTest(f, jig, ipName)
		})

		It("should support https-only annotation", func() {
			executeStaticIPHttpsOnlyTest(f, jig, ipName, ipAddress)
		})

		It("should remove clusters as expected", func() {
			ingAnnotations := map[string]string{
				ingress.IngressStaticIPKey: ipName,
			}
			ingFilePath := filepath.Join(ingress.IngressManifestPath, "http")
			jig.CreateIngress(ingFilePath, ns, ingAnnotations, map[string]string{})
			jig.WaitForIngress(false /*waitForNodePort*/)
			name := jig.Ingress.Name
			// Verify that the ingress is spread to 1 cluster as expected.
			verifyKubemciStatusHas(name, "is spread across 1 cluster")
			// Validate that removing the ingress from all clusters throws an error.
			// Reuse the ingress file created while creating the ingress.
			filePath := filepath.Join(framework.TestContext.OutputDir, "mci.yaml")
			output, err := framework.RunKubemciWithKubeconfig("remove-clusters", name, "--ingress="+filePath)
			if err != nil {
				framework.Failf("unexpected error in running kubemci remove-clusters command to remove from all clusters: %s", err)
			}
			if !strings.Contains(output, "You should use kubemci delete to delete the ingress completely") {
				framework.Failf("unexpected output in removing an ingress from all clusters, expected the output to include: You should use kubemci delete to delete the ingress completely, actual output: %s", output)
			}
			// Verify that the ingress is still spread to 1 cluster as expected.
			verifyKubemciStatusHas(name, "is spread across 1 cluster")
			// remove-clusters should succeed with --force=true
			if _, err := framework.RunKubemciWithKubeconfig("remove-clusters", name, "--ingress="+filePath, "--force=true"); err != nil {
				framework.Failf("unexpected error in running kubemci remove-clusters to remove from all clusters with --force=true: %s", err)
			}
			verifyKubemciStatusHas(name, "is spread across 0 cluster")
		})

		It("single and multi-cluster ingresses should be able to exist together", func() {
			By("Creating a single cluster ingress first")
			jig.Class = ""
			singleIngFilePath := filepath.Join(ingress.GCEIngressManifestPath, "static-ip-2")
			jig.CreateIngress(singleIngFilePath, ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(false /*waitForNodePort*/)
			// jig.Ingress will be overwritten when we create MCI, so keep a reference.
			singleIng := jig.Ingress

			// Create the multi-cluster ingress next.
			By("Creating a multi-cluster ingress next")
			jig.Class = ingress.MulticlusterIngressClassValue
			ingAnnotations := map[string]string{
				ingress.IngressStaticIPKey: ipName,
			}
			multiIngFilePath := filepath.Join(ingress.IngressManifestPath, "http")
			jig.CreateIngress(multiIngFilePath, ns, ingAnnotations, map[string]string{})
			jig.WaitForIngress(false /*waitForNodePort*/)
			mciIngress := jig.Ingress

			By("Deleting the single cluster ingress and verifying that multi-cluster ingress continues to work")
			jig.Ingress = singleIng
			jig.Class = ""
			jig.TryDeleteIngress()
			jig.Ingress = mciIngress
			jig.Class = ingress.MulticlusterIngressClassValue
			jig.WaitForIngress(false /*waitForNodePort*/)

			By("Cleanup: Deleting the multi-cluster ingress")
			jig.TryDeleteIngress()
		})
	})

	// Time: borderline 5m, slow by design
	Describe("[Slow] Nginx", func() {
		var nginxController *ingress.NginxIngressController

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			By("Initializing nginx controller")
			jig.Class = "nginx"
			nginxController = &ingress.NginxIngressController{Ns: ns, Client: jig.Client}

			// TODO: This test may fail on other platforms. We can simply skip it
			// but we want to allow easy testing where a user might've hand
			// configured firewalls.
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(gce.GcloudComputeResourceCreate("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID, "--allow", "tcp:80,tcp:443", "--network", framework.TestContext.CloudConfig.Network))
			} else {
				framework.Logf("WARNING: Not running on GCE/GKE, cannot create firewall rules for :80, :443. Assuming traffic can reach the external ips of all nodes in cluster on those ports.")
			}

			nginxController.Init()
		})

		AfterEach(func() {
			if framework.ProviderIs("gce", "gke") {
				framework.ExpectNoError(gce.GcloudComputeResourceDelete("firewall-rules", fmt.Sprintf("ingress-80-443-%v", ns), framework.TestContext.CloudConfig.ProjectID))
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
			conformanceTests = ingress.CreateIngressComformanceTests(jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				By(t.EntryLog)
				t.Execute()
				By(t.ExitLog)
				jig.WaitForIngress(false)
			}
		})
	})
})

// verifyKubemciStatusHas fails if kubemci get-status output for the given mci does not have the given expectedSubStr.
func verifyKubemciStatusHas(name, expectedSubStr string) {
	statusStr, err := framework.RunKubemciCmd("get-status", name)
	if err != nil {
		framework.Failf("unexpected error in running kubemci get-status %s: %s", name, err)
	}
	if !strings.Contains(statusStr, expectedSubStr) {
		framework.Failf("expected status to have sub string %s, actual status: %s", expectedSubStr, statusStr)
	}
}

func executePresharedCertTest(f *framework.Framework, jig *ingress.TestJig, staticIPName string) {
	preSharedCertName := "test-pre-shared-cert"
	By(fmt.Sprintf("Creating ssl certificate %q on GCE", preSharedCertName))
	testHostname := "test.ingress.com"
	cert, key, err := ingress.GenerateRSACerts(testHostname, true)
	Expect(err).NotTo(HaveOccurred())
	gceCloud, err := gce.GetGCECloud()
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
	ingAnnotations := map[string]string{
		ingress.IngressPreSharedCertKey: preSharedCertName,
		// Disallow HTTP to save resources. This is irrelevant to the
		// pre-shared cert test.
		ingress.IngressAllowHTTPKey: "false",
	}
	if staticIPName != "" {
		ingAnnotations[ingress.IngressStaticIPKey] = staticIPName
	}
	jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "pre-shared-cert"), f.Namespace.Name, ingAnnotations, map[string]string{})

	By("Test that ingress works with the pre-shared certificate")
	err = jig.WaitForIngressWithCert(true, []string{testHostname}, cert)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Unexpected error while waiting for ingress: %v", err))
}

func executeStaticIPHttpsOnlyTest(f *framework.Framework, jig *ingress.TestJig, ipName, ip string) {
	jig.CreateIngress(filepath.Join(ingress.IngressManifestPath, "static-ip"), f.Namespace.Name, map[string]string{
		ingress.IngressStaticIPKey:  ipName,
		ingress.IngressAllowHTTPKey: "false",
	}, map[string]string{})

	By("waiting for Ingress to come up with ip: " + ip)
	httpClient := ingress.BuildInsecureClient(ingress.IngressReqTimeout)
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("https://%s/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, false))

	By("should reject HTTP traffic")
	framework.ExpectNoError(framework.PollURL(fmt.Sprintf("http://%s/", ip), "", framework.LoadBalancerPollTimeout, jig.PollInterval, httpClient, true))
}

func executeBacksideBacksideHTTPSTest(f *framework.Framework, jig *ingress.TestJig, staticIPName string) {
	By("Creating a set of ingress, service and deployment that have backside re-encryption configured")
	deployCreated, svcCreated, ingCreated, err := jig.SetUpBacksideHTTPSIngress(f.ClientSet, f.Namespace.Name, staticIPName)
	defer func() {
		By("Cleaning up re-encryption ingress, service and deployment")
		if errs := jig.DeleteTestResource(f.ClientSet, deployCreated, svcCreated, ingCreated); len(errs) > 0 {
			framework.Failf("Failed to cleanup re-encryption ingress: %v", errs)
		}
	}()
	Expect(err).NotTo(HaveOccurred(), "Failed to create re-encryption ingress")

	By(fmt.Sprintf("Waiting for ingress %s to come up", ingCreated.Name))
	ingIP, err := jig.WaitForIngressAddress(f.ClientSet, f.Namespace.Name, ingCreated.Name, framework.LoadBalancerPollTimeout)
	Expect(err).NotTo(HaveOccurred(), "Failed to wait for ingress IP")

	By(fmt.Sprintf("Polling on address %s and verify the backend is serving HTTPS", ingIP))
	timeoutClient := &http.Client{Timeout: ingress.IngressReqTimeout}
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
}

func detectHttpVersionAndSchemeTest(f *framework.Framework, jig *ingress.TestJig, address, version, scheme string) {
	timeoutClient := &http.Client{Timeout: ingress.IngressReqTimeout}
	resp := ""
	err := wait.PollImmediate(framework.LoadBalancerPollInterval, framework.LoadBalancerPollTimeout, func() (bool, error) {
		var err error
		resp, err = framework.SimpleGET(timeoutClient, fmt.Sprintf("http://%s", address), "")
		if err != nil {
			framework.Logf("SimpleGET failed: %v", err)
			return false, nil
		}
		if !strings.Contains(resp, version) {
			framework.Logf("Waiting for transition to HTTP/2")
			return false, nil
		}
		if !strings.Contains(resp, scheme) {
			return false, nil
		}
		framework.Logf("Poll succeeded, request was served by HTTP2")
		return true, nil
	})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to get %s or %s, response body: %s", version, scheme, resp))
}

func detectNegAnnotation(f *framework.Framework, jig *ingress.TestJig, gceController *gce.GCEIngressController, ns, name string, negs int) {
	if err := wait.Poll(5*time.Second, negUpdateTimeout, func() (bool, error) {
		svc, err := f.ClientSet.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		// if we expect no NEGs, then we should be using IGs
		if negs == 0 {
			return gceController.BackendServiceUsingIG(jig.GetServicePorts(false))
		}

		var status ingress.NegStatus
		v, ok := svc.Annotations[ingress.NEGStatusAnnotation]
		if !ok {
			framework.Logf("Waiting for %v, got: %+v", ingress.NEGStatusAnnotation, svc.Annotations)
			return false, nil
		}

		err = json.Unmarshal([]byte(v), &status)
		if err != nil {
			framework.Logf("Error in parsing Expose NEG annotation: %v", err)
			return false, nil
		}
		framework.Logf("Got %v: %v", ingress.NEGStatusAnnotation, v)

		if len(status.NetworkEndpointGroups) != negs {
			framework.Logf("Expected %d NEGs, got %d", negs, len(status.NetworkEndpointGroups))
			return false, nil
		}

		gceCloud, err := gce.GetGCECloud()
		Expect(err).NotTo(HaveOccurred())
		for _, neg := range status.NetworkEndpointGroups {
			networkEndpoints, err := gceCloud.ListNetworkEndpoints(neg, gceController.Cloud.Zone, false)
			Expect(err).NotTo(HaveOccurred())
			if len(networkEndpoints) != 1 {
				framework.Logf("Expect NEG %s to exist, but got %d", neg, len(networkEndpoints))
				return false, nil
			}
		}

		return gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
	}); err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
}
