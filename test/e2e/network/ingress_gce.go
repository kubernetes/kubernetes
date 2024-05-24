//go:build !providerless
// +build !providerless

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
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2eingress "k8s.io/kubernetes/test/e2e/framework/ingress"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
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
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func(ctx context.Context) {
		jig = e2eingress.NewIngressTestJig(f.ClientSet)
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		err := e2eauth.BindClusterRole(ctx, jig.Client.RbacV1(), "cluster-admin", f.Namespace.Name,
			rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})
		framework.ExpectNoError(err)

		err = e2eauth.WaitForAuthorizationUpdate(ctx, jig.Client.AuthorizationV1(),
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
	f.Describe("GCE", framework.WithSlow(), feature.Ingress, func() {
		var gceController *gce.IngressController

		// Platform specific setup
		ginkgo.BeforeEach(func(ctx context.Context) {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")
			ginkgo.By("Initializing gce controller")
			gceController = &gce.IngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init(ctx)
			framework.ExpectNoError(err)
		})

		// Platform specific cleanup
		ginkgo.AfterEach(func(ctx context.Context) {
			if ginkgo.CurrentSpecReport().Failed() {
				e2eingress.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				ginkgo.By("No ingress created, no cleanup necessary")
				return
			}
			ginkgo.By("Deleting ingress")
			jig.TryDeleteIngress(ctx)

			ginkgo.By("Cleaning up cloud resources")
			err := gceController.CleanupIngressController(ctx)
			framework.ExpectNoError(err)
		})

		ginkgo.It("should conform to Ingress spec", func(ctx context.Context) {
			conformanceTests = e2eingress.CreateIngressComformanceTests(ctx, jig, ns, map[string]string{})
			for _, t := range conformanceTests {
				ginkgo.By(t.EntryLog)
				t.Execute()
				ginkgo.By(t.ExitLog)
				jig.WaitForIngress(ctx, true)
			}
		})

	})

	f.Describe("GCE", framework.WithSlow(), feature.NEG, func() {
		var gceController *gce.IngressController

		// Platform specific setup
		ginkgo.BeforeEach(func(ctx context.Context) {
			e2eskipper.SkipUnlessProviderIs("gce", "gke")
			ginkgo.By("Initializing gce controller")
			gceController = &gce.IngressController{
				Ns:     ns,
				Client: jig.Client,
				Cloud:  framework.TestContext.CloudConfig,
			}
			err := gceController.Init(ctx)
			framework.ExpectNoError(err)
		})

		// Platform specific cleanup
		ginkgo.AfterEach(func(ctx context.Context) {
			if ginkgo.CurrentSpecReport().Failed() {
				e2eingress.DescribeIng(ns)
			}
			if jig.Ingress == nil {
				ginkgo.By("No ingress created, no cleanup necessary")
				return
			}
			ginkgo.By("Deleting ingress")
			jig.TryDeleteIngress(ctx)

			ginkgo.By("Cleaning up cloud resources")
			err := gceController.CleanupIngressController(ctx)
			framework.ExpectNoError(err)
		})

		ginkgo.It("should conform to Ingress spec", func(ctx context.Context) {
			jig.PollInterval = 5 * time.Second
			conformanceTests = e2eingress.CreateIngressComformanceTests(ctx, jig, ns, map[string]string{
				e2eingress.NEGAnnotation: `{"ingress": true}`,
			})
			for _, t := range conformanceTests {
				ginkgo.By(t.EntryLog)
				t.Execute()
				ginkgo.By(t.ExitLog)
				jig.WaitForIngress(ctx, true)
				err := gceController.WaitForNegBackendService(ctx, jig.GetServicePorts(ctx, false))
				framework.ExpectNoError(err)
			}
		})

		ginkgo.It("should be able to switch between IG and NEG modes", func(ctx context.Context) {
			var err error
			propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, f.ClientSet)
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)
			err = gceController.WaitForNegBackendService(ctx, jig.GetServicePorts(ctx, false))
			framework.ExpectNoError(err)

			ginkgo.By("Switch backend service to use IG")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress": false}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = wait.PollWithContext(ctx, 5*time.Second, propagationTimeout, func(ctx context.Context) (bool, error) {
				if err := gceController.BackendServiceUsingIG(jig.GetServicePorts(ctx, false)); err != nil {
					framework.Logf("ginkgo.Failed to verify IG backend service: %v", err)
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Expect backend service to target IG, but failed to observe")
			jig.WaitForIngress(ctx, true)

			ginkgo.By("Switch backend service to use NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress": true}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = wait.PollWithContext(ctx, 5*time.Second, propagationTimeout, func(ctx context.Context) (bool, error) {
				if err := gceController.BackendServiceUsingNEG(jig.GetServicePorts(ctx, false)); err != nil {
					framework.Logf("ginkgo.Failed to verify NEG backend service: %v", err)
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Expect backend service to target NEG, but failed to observe")
			jig.WaitForIngress(ctx, true)
		})

		ginkgo.It("should be able to create a ClusterIP service", func(ctx context.Context) {
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg-clusterip"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)
			svcPorts := jig.GetServicePorts(ctx, false)
			err := gceController.WaitForNegBackendService(ctx, svcPorts)
			framework.ExpectNoError(err)

			// ClusterIP ServicePorts have no NodePort
			for _, sp := range svcPorts {
				gomega.Expect(sp.NodePort).To(gomega.Equal(int32(0)))
			}
		})

		ginkgo.It("should sync endpoints to NEG", func(ctx context.Context) {
			name := "hostname"
			scaleAndValidateNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(ctx, name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if scale.Spec.Replicas != int32(num) {
					scale.ResourceVersion = "" // indicate the scale update should be unconditional
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					res, err := jig.GetDistinctResponseFromIngress(ctx)
					if err != nil {
						return false, nil
					}
					framework.Logf("Expecting %d backends, got %d", num, res.Len())
					return res.Len() == num, nil
				})
				framework.ExpectNoError(err)
			}

			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)
			jig.WaitForIngressToStable(ctx)
			err := gceController.WaitForNegBackendService(ctx, jig.GetServicePorts(ctx, false))
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

		ginkgo.It("rolling update backend pods should not cause service disruption", func(ctx context.Context) {
			name := "hostname"
			replicas := 8
			ginkgo.By("Create a basic HTTP ingress using NEG")
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)
			jig.WaitForIngressToStable(ctx)
			err := gceController.WaitForNegBackendService(ctx, jig.GetServicePorts(ctx, false))
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("Scale backend replicas to %d", replicas))
			scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(ctx, name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			scale.ResourceVersion = "" // indicate the scale update should be unconditional
			scale.Spec.Replicas = int32(replicas)
			_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			propagationTimeout := e2eservice.GetServiceLoadBalancerPropagationTimeout(ctx, f.ClientSet)
			err = wait.Poll(10*time.Second, propagationTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress(ctx)
				if err != nil {
					return false, nil
				}
				return res.Len() == replicas, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("Trigger rolling update and observe service disruption")
			deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(ctx, name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			// trigger by changing graceful termination period to 60 seconds
			gracePeriod := int64(60)
			deploy.Spec.Template.Spec.TerminationGracePeriodSeconds = &gracePeriod
			_, err = f.ClientSet.AppsV1().Deployments(ns).Update(ctx, deploy, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
			err = wait.Poll(10*time.Second, propagationTimeout, func() (bool, error) {
				res, err := jig.GetDistinctResponseFromIngress(ctx)
				if err != nil {
					return false, err
				}
				deploy, err := f.ClientSet.AppsV1().Deployments(ns).Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
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

		ginkgo.It("should sync endpoints for both Ingress-referenced NEG and standalone NEG", func(ctx context.Context) {
			name := "hostname"
			expectedKeys := []int32{80, 443}

			scaleAndValidateExposedNEG := func(num int) {
				scale, err := f.ClientSet.AppsV1().Deployments(ns).GetScale(ctx, name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if scale.Spec.Replicas != int32(num) {
					scale.ResourceVersion = "" // indicate the scale update should be unconditional
					scale.Spec.Replicas = int32(num)
					_, err = f.ClientSet.AppsV1().Deployments(ns).UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = wait.Poll(10*time.Second, negUpdateTimeout, func() (bool, error) {
					svc, err := f.ClientSet.CoreV1().Services(ns).Get(ctx, name, metav1.GetOptions{})
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
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)
			err := gceController.WaitForNegBackendService(ctx, jig.GetServicePorts(ctx, false))
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

		ginkgo.It("should create NEGs for all ports with the Ingress annotation, and NEGs for the standalone annotation otherwise", func(ctx context.Context) {
			ginkgo.By("Create a basic HTTP ingress using standalone NEG")
			jig.CreateIngress(ctx, filepath.Join(e2eingress.IngressManifestPath, "neg-exposed"), ns, map[string]string{}, map[string]string{})
			jig.WaitForIngress(ctx, true)

			name := "hostname"
			detectNegAnnotation(ctx, f, jig, gceController, ns, name, 2)

			// Add Ingress annotation - NEGs should stay the same.
			ginkgo.By("Adding NEG Ingress annotation")
			svcList, err := f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"80":{},"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(ctx, f, jig, gceController, ns, name, 2)

			// Modify exposed NEG annotation, but keep ingress annotation
			ginkgo.By("Modifying exposed NEG annotation, but keep Ingress annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":true,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(ctx, f, jig, gceController, ns, name, 2)

			// Remove Ingress annotation. Expect 1 NEG
			ginkgo.By("Disabling Ingress annotation, but keeping one standalone NEG")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				svc.Annotations[e2eingress.NEGAnnotation] = `{"ingress":false,"exposed_ports":{"443":{}}}`
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(ctx, f, jig, gceController, ns, name, 1)

			// Remove NEG annotation entirely. Expect 0 NEGs.
			ginkgo.By("Removing NEG annotation")
			svcList, err = f.ClientSet.CoreV1().Services(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			for _, svc := range svcList.Items {
				delete(svc.Annotations, e2eingress.NEGAnnotation)
				// Service cannot be ClusterIP if it's using Instance Groups.
				svc.Spec.Type = v1.ServiceTypeNodePort
				_, err = f.ClientSet.CoreV1().Services(ns).Update(ctx, &svc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			detectNegAnnotation(ctx, f, jig, gceController, ns, name, 0)
		})
	})
})

func detectNegAnnotation(ctx context.Context, f *framework.Framework, jig *e2eingress.TestJig, gceController *gce.IngressController, ns, name string, negs int) {
	if err := wait.Poll(5*time.Second, negUpdateTimeout, func() (bool, error) {
		svc, err := f.ClientSet.CoreV1().Services(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}

		// if we expect no NEGs, then we should be using IGs
		if negs == 0 {
			err := gceController.BackendServiceUsingIG(jig.GetServicePorts(ctx, false))
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

		err = gceController.BackendServiceUsingNEG(jig.GetServicePorts(ctx, false))
		if err != nil {
			framework.Logf("ginkgo.Failed to validate NEG backend service: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		framework.ExpectNoError(err)
	}
}
