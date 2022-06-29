/*
Copyright 2019 The Kubernetes Authors.

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

package windows

import (
	"context"
	"fmt"
	"net"
	"reflect"
	"strconv"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	var (
		cs      clientset.Interface
		testPod *v1.Pod
	)

	ginkgo.BeforeEach(func() {
		//Only for Windows containers
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		cs = f.ClientSet

		//using hybrid_network methods
		ginkgo.By("creating Windows testing Pod")
		testPod = createTestPod(f, windowsBusyBoximage, windowsOS)
		testPod = f.PodClient().CreateSync(testPod)

		ginkgo.By("verifying that pod has the correct nodeSelector")
		// Admission controllers may sometimes do the wrong thing
		framework.ExpectEqual(testPod.Spec.NodeSelector["kubernetes.io/os"], "windows")
	})

	ginkgo.It("should be able to create a functioning Cluster IP service for Windows", func() {
		serviceName := "clusterip-test"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating service " + serviceName + " with type=ClusterIP in namespace " + ns)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)

		svcIP := svc.Spec.ClusterIP

		ginkgo.By("creating Pod to be part of service " + serviceName)
		// tweak the Jig to use windows...
		windowsNodeSelectorTweak := func(rc *v1.ReplicationController) {
			rc.Spec.Template.Spec.NodeSelector = map[string]string{
				"kubernetes.io/os": "windows",
			}
		}
		_, err = jig.Run(windowsNodeSelectorTweak)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("checking connectivity Pod to curl http://%s", svcIP))
		assertConsistentConnectivity(f, testPod.ObjectMeta.Name, windowsOS, windowsCheck(fmt.Sprintf("http://%s", svcIP)))
	})

	ginkgo.It("should be able to create a functioning NodePort service for Windows", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)
		nodeIP, err := e2enode.PickIP(jig.Client)
		framework.ExpectNoError(err)

		ginkgo.By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		svc, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		framework.ExpectNoError(err)

		nodePort := int(svc.Spec.Ports[0].NodePort)

		ginkgo.By("creating Pod to be part of service " + serviceName)
		// tweak the Jig to use windows...
		windowsNodeSelectorTweak := func(rc *v1.ReplicationController) {
			rc.Spec.Template.Spec.NodeSelector = map[string]string{
				"kubernetes.io/os": "windows",
			}
		}
		_, err = jig.Run(windowsNodeSelectorTweak)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("checking connectivity Pod to curl http://%s:%d", nodeIP, nodePort))
		assertConsistentConnectivity(f, testPod.ObjectMeta.Name, windowsOS, windowsCheck(fmt.Sprintf("http://%s", net.JoinHostPort(nodeIP, strconv.Itoa(nodePort)))))

	})

	ginkgo.It("should have the ability to delete and recreate services in such a way that load balancing rules for pods are recovered by whatever chosen service proxy is being utilized", func() {
		serviceName := "clusterip-test"
		ns := f.Namespace.Name
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating service " + serviceName + " with type=ClusterIP in namespace " + ns)
		_, err := jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)

		ginkgo.By("creating Pod to be part of service " + serviceName)
		// tweak the Jig to use windows...
		windowsNodeSelectorTweak := func(rc *v1.ReplicationController) {
			rc.Spec.Template.Spec.NodeSelector = map[string]string{
				"kubernetes.io/os": "windows",
			}
			var replicas int32 = 3
			rc.Spec.Replicas = &replicas
		}
		_, err = jig.Run(windowsNodeSelectorTweak)
		framework.ExpectNoError(err)

		ginkgo.By("getting the endpoints of the " + serviceName + " service")
		epList1, err := cs.CoreV1().Endpoints(ns).Get(context.TODO(), serviceName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("deleting the the " + serviceName + " service")
		err = cs.CoreV1().Services(ns).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for the " + serviceName + " service to disappear")
		if pollErr := wait.PollImmediate(e2eservice.LoadBalancerPollInterval, e2eservice.GetServiceLoadBalancerCreationTimeout(cs), func() (bool, error) {
			_, err := cs.CoreV1().Services(ns).Get(context.TODO(), serviceName, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					framework.Logf("Service %s/%s is gone.", ns, serviceName)
					return true, nil
				}
				return false, err
			}
			framework.Logf("Service %s/%s still exists", ns, serviceName)
			return false, nil
		}); pollErr != nil {
			framework.Failf("Failed to wait for service to disappear: %v", pollErr)
		}

		ginkgo.By("re-creating service " + serviceName + " with type=ClusterIP in namespace " + ns)
		_, err = jig.CreateTCPService(func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
		})
		framework.ExpectNoError(err)

		ginkgo.By("checking the endpoints of the new service should remain the same")
		epList2, err := cs.CoreV1().Endpoints(ns).Get(context.TODO(), serviceName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(reflect.DeepEqual(epList1, epList2), true)
	})
})
