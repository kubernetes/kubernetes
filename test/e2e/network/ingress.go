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
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2eingress "k8s.io/kubernetes/test/e2e/framework/ingress"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	negUpdateTimeout = 2 * time.Minute
)

var _ = common.SIGDescribe("Loadbalancing: L7", func() {
	defer ginkgo.GinkgoRecover()
	var (
		ns               string
		jig              *e2eingress.TestJig
		conformanceTests []e2eingress.ConformanceTests
	)
	f := framework.NewDefaultFramework("ingress")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		jig = e2eingress.NewIngressTestJig(f.ClientSet)
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		err := e2eauth.BindClusterRole(jig.Client.RbacV1(), "cluster-admin", f.Namespace.Name,
			rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})
		framework.ExpectNoError(err)

		err = e2eauth.WaitForAuthorizationUpdate(jig.Client.AuthorizationV1(),
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
	ginkgo.Describe("GCE [Slow] [Feature:Ingress]", func() {
		var gceController *gce.IngressController

		// Platform specific setup
		ginkgo.BeforeEach(func() {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")
			ginkgo.By("Initializing gce controller")
			gceController = &gce.IngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init()
			framework.ExpectNoError(err)
		})

		// Platform specific cleanup
		ginkgo.AfterEach(func() {
			if ginkgo.CurrentSpecReport().Failed() {
				e2eingress.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				ginkgo.By("No ingress created, no cleanup necessary")
				return
			}
			ginkgo.By("Deleting ingress")
			jig.TryDeleteIngress()

			ginkgo.By("Cleaning up cloud resources")
			err := gceController.CleanupIngressController()
			framework.ExpectNoError(err)
		})

		ginkgo.It("should conform to Ingress spec", func() {
			conformanceTests = e2eingress.CreateIngressComformanceTests(jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				ginkgo.By(t.EntryLog)
				t.Execute()
				ginkgo.By(t.ExitLog)
				jig.WaitForIngress(true)
			}
		})

	})

	ginkgo.Describe("GCE [Slow] [Feature:NEG] [Flaky]", func() {
		var gceController *gce.IngressController

		// Platform specific setup
		ginkgo.BeforeEach(func() {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")
			ginkgo.By("Initializing gce controller")
			gceController = &gce.IngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init()
			framework.ExpectNoError(err)
		})

		// Platform specific cleanup
		ginkgo.AfterEach(func() {
			if ginkgo.CurrentSpecReport().Failed() {
				e2eingress.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				ginkgo.By("No ingress created, no cleanup necessary")
				return
			}
			ginkgo.By("Deleting ingress")
			jig.TryDeleteIngress()

			ginkgo.By("Cleaning up cloud resources")
			err := gceController.CleanupIngressController()
			framework.ExpectNoError(err)
		})

		ginkgo.It("should conform to Ingress spec", func() {
			jig.PollInterval = 5 * time.Second
			conformanceTests = e2eingress.CreateIngressComformanceTests(jig, ns, map[string]string{
				e2eingress.NEGAnnotation: `{"ingress": true}`,
			})
			for _, t := range conformanceTests {
				ginkgo.By(t.EntryLog)
				t.Execute()
				ginkgo.By(t.ExitLog)
				jig.WaitForIngress(true)
				err := gceController.WaitForNegBackendService(jig.GetServicePorts(false))
				framework.ExpectNoError(err)
			}
		})

		ginkgo.It("should be able to switch between IG and NEG modes", func() {
			var err error
			propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(f.ClientSet)
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			err = gceController.WaitForNegBackendService(jig.GetServicePorts(false))
			framework.ExpectNoError(err)

			ginkgo.By("Switch backend service to use IG")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress": false}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = wait.Poll(5*time.Second, propagationTimeout, func() (bool, error) {
				if err := gceController.BackendServiceUsingIG(jig.GetServicePorts(false)); err != nil {
					framework.Logf("ginkgo.Failed to verify IG backend service: %v", err)
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Expect backend service to target IG, but failed to observe")
			jig.WaitForIngress(true)

			ginkgo.By("Switch backend service to use NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress": true}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = wait.Poll(5*time.Second, propagationTimeout, func() (bool, error) {
				if err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(false)); err != nil {
					framework.Logf("ginkgo.Failed to verify NEG backend service: %v", err)
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Expect backend service to target NEG, but failed to observe")
			jig.WaitForIngress(true)
		})

		ginkgo.It("should be able to create a ClusterIP service", func() {
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg-clusterip"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			svcPorts := jig.GetServicePorts(false)
			err := gceController.WaitForNegBackendService(svcPorts)
			framework.ExpectNoError(err)

			// ClusterIP ServicePorts have no NodePort
			for _, sp := range svcPorts {
				framework.ExpectEqual(sp.NodePort, int32(0))
			}
		})

		ginkgo.It("should sync endpoints to NEG", func() {
			name := "hostname"
			scaleAndValidateNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(context.TODO(), name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if scale.Spec.Replicas != int32(num) {
					scale.ResourceVersion = "" // indicate the scale update should be unconditional
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(context.TODO(), name, scale, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					res, err := jig.GetDistinctResponseFromIngress()
					if err != nil {
						return false, nil
					}
					framework.Logf("Expecting %d backends, got %d", num, res.Len())
					return res.Len() == num, nil
				})
				framework.ExpectNoError(err)
			}

			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			jig.WaitForIngressToStable()
			err := gceController.WaitForNegBackendService(jig.GetServicePorts(false))
			framework.ExpectNoError(err)
			// initial replicas number is 1
			scaleAndValidateNEG(1)

			ginkgo.By("Scale up number of backends to 5")
			scaleAndValidateNEG(5)

			ginkgo.By("Scale down number of backends to 3")
			scaleAndValidateNEG(3)

			ginkgo.By("Scale up number of backends to 6")
			scaleAndValidateNEG(6)

			ginkgo.By("Scale down number of backends to 2")
			scaleAndValidateNEG(3)
		})

		ginkgo.It("rolling update backend pods should not cause service disruption", func() {
			name := "hostname"
			replicas := 8
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			jig.WaitForIngressToStable()
			err := gceController.WaitForNegBackendService(jig.GetServicePorts(false))
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("Scale backend replicas to %d", replicas))
			scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(context.TODO(), name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			scale.ResourceVersion = "" // indicate the scale update should be unconditional
			scale.Spec.Replicas = int32(replicas)
			_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(context.TODO(), name, scale, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(f.ClientSet)
			err = wait.Poll(10*time.Second, propagationTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				if err != nil {
					return false, nil
				}
				return res.Len() == replicas, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("Trigger rolling update and observe service disruption")
			deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(context.TODO(), name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			// trigger by changing graceful termination period to 60 seconds
			gracePeriod := int64(60)
			deploy.Spec.Template.Spec.TerminationGracePeriodSeconds = &gracePeriod
			_, err = f.ClientSet.AppsV1().Deployments(ns).Update(context.TODO(), deploy, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
			err = wait.Poll(10*time.Second, propagationTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress()
				framework.ExpectNoError(err)
				deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(context.TODO(), name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if int(deploy.Status.UpdatedReplicas) == replicas {
					if res.Len() == replicas {
						return true, nil
					}
					framework.Logf("Expecting %d different responses, but got %d.", replicas, res.Len())
					return false, nil

				}
				framework.Logf("Waiting for rolling update to finished. Keep sending traffic.")
				return false, nil
			})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should sync endpoints for both Ingress-referenced NEG and standalone NEG", func() {
			name := "hostname"
			expectedKeys := []int32{80, 443}

			scaleAndValidateExposedNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(context.TODO(), name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if scale.Spec.Replicas != int32(num) {
					scale.ResourceVersion = "" // indicate the scale update should be unconditional
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(context.TODO(), name, scale, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					svc, err := f.ClientSet.CoreV1().Services(ns).Get(context.TODO(), name, metav1.GetOptions{})
					framework.ExpectNoError(err)

					var status e2eingress.NegStatus
					v, ok := svc.Annotations[e2eingress.NEGStatusAnnotation]
					if !ok {
						// Wait for NEG sync loop to find NEGs
						framework.Logf("Waiting for %v, got: %+v", e2eingress.NEGStatusAnnotation, svc.Annotations)
						return false, nil
					}
					err = json.Unmarshal([]byte(v), &status)
					if err != nil {
						framework.Logf("Error in parsing Expose NEG annotation: %v", err)
						return false, nil
					}
					framework.Logf("Got %v: %v", e2eingress.NEGStatusAnnotation, v)

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
					framework.ExpectNoError(err)
					for _, neg := range status.NetworkEndpointGroups {
						networkEndpoints, err := gceCloud.ListNetworkEndpoints(neg, gceController.Cloud.Zone, false)
						framework.ExpectNoError(err)
						if len(networkEndpoints) != num {
							framework.Logf("Expect number of endpoints to be %d, but got %d", num, len(networkEndpoints))
							return false, nil
						}
					}

					return true, nil
				})
				framework.ExpectNoError(err)
			}

			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)
			err := gceController.WaitForNegBackendService(jig.GetServicePorts(false))
			framework.ExpectNoError(err)
			// initial replicas number is 1
			scaleAndValidateExposedNEG(1)

			ginkgo.By("Scale up number of backends to 5")
			scaleAndValidateExposedNEG(5)

			ginkgo.By("Scale down number of backends to 3")
			scaleAndValidateExposedNEG(3)

			ginkgo.By("Scale up number of backends to 6")
			scaleAndValidateExposedNEG(6)

			ginkgo.By("Scale down number of backends to 2")
			scaleAndValidateExposedNEG(3)
		})

		ginkgo.It("should create NEGs for all ports with the Ingress annotation, and NEGs for the standalone annotation otherwise", func() {
			ginkgo.By("Create a basic HTTP ingress using standalone NEG")
			jig.CreateIngress(filepath.Join(e2eingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(true)

			name := "hostname"
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Add Ingress annotation - NEGs should stay the same.
			ginkgo.By("Adding NEG Ingress annotation")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"80":{},"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Modify exposed NEG annotation, but keep ingress annotation
			ginkgo.By("Modifying exposed NEG annotation, but keep Ingress annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 2)

			// Remove Ingress annotation. Expect 1 NEG
			ginkgo.By("Disabling Ingress annotation, but keeping one standalone NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":false,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 1)

			// Remove NEG annotation entirely. Expect 0 NEGs.
			ginkgo.By("Removing NEG annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				delete(svc.Annotations, e2eingress.NEGAnnotation)
				// Service cannot be ClusterIP if it's using Instance Groups.
				svc.Spec.Type = v1.ServiceTypeNodePort
				_, err = f.ClientSet.CoreV1().Services(ns).Update(context.TODO(), &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(f, jig, gceController, ns, name, 0)
		})
	})
})

func detectNegAnnotation(f *framework.Framework, jig *e2eingress.TestJig, gceController *gce.IngressController, ns, name string, negs int) {
	if err := wait.Poll(5*time.Second, negUpdateTimeout, func() (bool, error) {
		svc, err := f.ClientSet.CoreV1().Services(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		// if we expect no NEGs, then we should be using IGs
		if negs == 0 {
			err := gceController.BackendServiceUsingIG(jig.GetServicePorts(false))
			if err != nil {
				framework.Logf("ginkgo.Failed to validate IG backend service: %v", err)
				return false, nil
			}
			return true, nil
		}

		var status e2eingress.NegStatus
		v, ok := svc.Annotations[e2eingress.NEGStatusAnnotation]
		if !ok {
			framework.Logf("Waiting for %v, got: %+v", e2eingress.NEGStatusAnnotation, svc.Annotations)
			return false, nil
		}

		err = json.Unmarshal([]byte(v), &status)
		if err != nil {
			framework.Logf("Error in parsing Expose NEG annotation: %v", err)
			return false, nil
		}
		framework.Logf("Got %v: %v", e2eingress.NEGStatusAnnotation, v)

		if len(status.NetworkEndpointGroups) != negs {
			framework.Logf("Expected %d NEGs, got %d", negs, len(status.NetworkEndpointGroups))
			return false, nil
		}

		gceCloud, err := gce.GetGCECloud()
		framework.ExpectNoError(err)
		for _, neg := range status.NetworkEndpointGroups {
			networkEndpoints, err := gceCloud.ListNetworkEndpoints(neg, gceController.Cloud.Zone, false)
			framework.ExpectNoError(err)
			if len(networkEndpoints) != 1 {
				framework.Logf("Expect NEG %s to exist, but got %d", neg, len(networkEndpoints))
				return false, nil
			}
		}

		err = gceController.BackendServiceUsingNEG(jig.GetServicePorts(false))
		if err != nil {
			framework.Logf("ginkgo.Failed to validate NEG backend service: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		framework.ExpectNoError(err)
	}
}

var _ = common.SIGDescribe("Ingress API", func() {
	f := framework.NewDefaultFramework("ingress")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	/*
		Release: v1.19
		Testname: Ingress API
		Description:
		The networking.k8s.io API group MUST exist in the /apis discovery document.
		The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		The ingresses resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		The ingresses resource must support create, get, list, watch, update, patch, delete, and deletecollection.
		The ingresses/status resource must support update and patch
	*/

	framework.ConformanceIt("should support creating Ingress API operations", func() {
		// Setup
		ns := f.Namespace.Name
		ingVersion := "v1"
		ingClient := f.ClientSet.NetworkingV1().Ingresses(ns)

		prefixPathType := networkingv1.PathTypeImplementationSpecific
		serviceBackend := &networkingv1.IngressServiceBackend{
			Name: "default-backend",
			Port: networkingv1.ServiceBackendPort{
				Name:   "",
				Number: 8080,
			},
		}
		defaultBackend := networkingv1.IngressBackend{
			Service: serviceBackend,
		}

		ingTemplate := &networkingv1.Ingress{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-example-ing",
				Labels: map[string]string{
					"special-label": f.UniqueName,
				}},
			Spec: networkingv1.IngressSpec{
				DefaultBackend: &defaultBackend,
				Rules: []networkingv1.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networkingv1.IngressRuleValue{
							HTTP: &networkingv1.HTTPIngressRuleValue{
								Paths: []networkingv1.HTTPIngressPath{{
									Path:     "/",
									PathType: &prefixPathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: "test-backend",
											Port: networkingv1.ServiceBackendPort{
												Number: 8080,
											},
										},
									},
								}},
							},
						},
					},
				},
			},
			Status: networkingv1.IngressStatus{LoadBalancer: v1.LoadBalancerStatus{}},
		}

		ingress1 := ingTemplate.DeepCopy()
		ingress1.Spec.Rules[0].Host = "host1.bar.com"
		ingress2 := ingTemplate.DeepCopy()
		ingress2.Spec.Rules[0].Host = "host2.bar.com"
		ingress3 := ingTemplate.DeepCopy()
		ingress3.Spec.Rules[0].Host = "host3.bar.com"

		// Discovery
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == networkingv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == ingVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected networking API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/networking.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == ingVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected networking API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/networking.k8s.io" + ingVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(networkingv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundIngress := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "ingresses":
					foundIngress = true
				}
			}
			if !foundIngress {
				framework.Failf("expected ingresses, got %#v", resources.APIResources)
			}
		}

		// Ingress resource create/read/update/watch verbs
		ginkgo.By("creating")
		_, err := ingClient.Create(context.TODO(), ingress1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = ingClient.Create(context.TODO(), ingress2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdIngress, err := ingClient.Create(context.TODO(), ingress3, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenIngress, err := ingClient.Get(context.TODO(), createdIngress.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gottenIngress.UID, createdIngress.UID)

		ginkgo.By("listing")
		ings, err := ingClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(ings.Items), 3, "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		ingWatch, err := ingClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: ings.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		// Test cluster-wide list and watch
		clusterIngClient := f.ClientSet.NetworkingV1().Ingresses("")
		ginkgo.By("cluster-wide listing")
		clusterIngs, err := clusterIngClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(clusterIngs.Items), 3, "filtered list should have 3 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterIngClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: ings.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedIngress, err := ingClient.Patch(context.TODO(), createdIngress.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedIngress.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		var ingToUpdate, updatedIngress *networkingv1.Ingress
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			ingToUpdate, err = ingClient.Get(context.TODO(), createdIngress.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			ingToUpdate.Annotations["updated"] = "true"
			updatedIngress, err = ingClient.Update(context.TODO(), ingToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedIngress.Annotations["updated"], "true", "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-ingWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				framework.ExpectEqual(evt.Type, watch.Modified)
				watchedIngress, isIngress := evt.Object.(*networkingv1.Ingress)
				if !isIngress {
					framework.Failf("expected Ingress, got %T", evt.Object)
				}
				if watchedIngress.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					ingWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedIngress.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		// /status subresource operations
		ginkgo.By("patching /status")
		lbStatus := v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{{IP: "169.1.1.1"}},
		}
		lbStatusJSON, err := json.Marshal(lbStatus)
		framework.ExpectNoError(err)
		patchedStatus, err := ingClient.Patch(context.TODO(), createdIngress.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"loadBalancer":`+string(lbStatusJSON)+`}}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedStatus.Status.LoadBalancer, lbStatus, "patched object should have the applied loadBalancer status")
		framework.ExpectEqual(patchedStatus.Annotations["patchedstatus"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		var statusToUpdate, updatedStatus *networkingv1.Ingress
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = ingClient.Get(context.TODO(), createdIngress.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			statusToUpdate.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{{IP: "169.1.1.2"}},
			}
			updatedStatus, err = ingClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedStatus.Status.LoadBalancer, statusToUpdate.Status.LoadBalancer, fmt.Sprintf("updated object expected to have updated loadbalancer status %#v, got %#v", statusToUpdate.Status.LoadBalancer, updatedStatus.Status.LoadBalancer))

		ginkgo.By("get /status")
		ingResource := schema.GroupVersionResource{Group: "networking.k8s.io", Version: ingVersion, Resource: "ingresses"}
		gottenStatus, err := f.DynamicClient.Resource(ingResource).Namespace(ns).Get(context.TODO(), createdIngress.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		statusUID, _, err := unstructured.NestedFieldCopy(gottenStatus.Object, "metadata", "uid")
		framework.ExpectNoError(err)
		framework.ExpectEqual(string(createdIngress.UID), statusUID, fmt.Sprintf("createdIngress.UID: %v expected to match statusUID: %v ", createdIngress.UID, statusUID))

		// Ingress resource delete operations
		ginkgo.By("deleting")

		expectFinalizer := func(ing *networkingv1.Ingress, msg string) {
			framework.ExpectNotEqual(ing.DeletionTimestamp, nil, fmt.Sprintf("expected deletionTimestamp, got nil on step: %q, ingress: %+v", msg, ing))
			if len(ing.Finalizers) == 0 {
				framework.Failf("expected finalizers on ingress, got none on step: %q, ingress: %+v", msg, ing)
			}
		}

		err = ingClient.Delete(context.TODO(), createdIngress.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		ing, err := ingClient.Get(context.TODO(), createdIngress.Name, metav1.GetOptions{})
		// If ingress controller does not support finalizers, we expect a 404.  Otherwise we validate finalizer behavior.
		if err == nil {
			expectFinalizer(ing, "deleting createdIngress")
		} else {
			if !apierrors.IsNotFound(err) {
				framework.Failf("expected 404, got %v", err)
			}
		}
		ings, err = ingClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 3 items since some ingresses might not have been deleted yet due to finalizers
		if len(ings.Items) > 3 {
			framework.Fail("filtered list should have <= 3 items")
		}
		// Validate finalizer on the deleted ingress
		for _, ing := range ings.Items {
			if ing.Namespace == createdIngress.Namespace && ing.Name == createdIngress.Name {
				expectFinalizer(&ing, "listing after deleting createdIngress")
			}
		}

		ginkgo.By("deleting a collection")
		err = ingClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		ings, err = ingClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 3 items since some ingresses might not have been deleted yet due to finalizers
		if len(ings.Items) > 3 {
			framework.Fail("filtered list should have <= 3 items")
		}
		// Validate finalizers
		for _, ing := range ings.Items {
			expectFinalizer(&ing, "deleting ingress collection")
		}
	})
})
