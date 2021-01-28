/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo"
)

const (
	parallelCreateServiceWorkers = 1
	maxServicesPerCluster        = 10000
	maxServicesPerNamespace      = 5000
	checkServicePercent          = 0.05
)

var _ = SIGDescribe("[Feature:PerformanceDNS][Serial]", func() {
	f := framework.NewDefaultFramework("performancedns")

	ginkgo.BeforeEach(func() {
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(f.ClientSet, framework.TestContext.NodeSchedulableTimeout))
		e2enode.WaitForTotalHealthy(f.ClientSet, time.Minute)

		err := framework.CheckTestingNSDeletedExcept(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)
	})

	// answers dns for service - creates the maximum number of services, and then check dns record for one
	ginkgo.It("Should answer DNS query for maximum number of services per cluster", func() {
		// get integer ceiling of maxServicesPerCluster / maxServicesPerNamespace
		numNs := (maxServicesPerCluster + maxServicesPerNamespace - 1) / maxServicesPerNamespace

		var namespaces []string
		for i := 0; i < numNs; i++ {
			ns, _ := f.CreateNamespace(f.BaseName, nil)
			namespaces = append(namespaces, ns.Name)
			f.AddNamespacesToDelete(ns)
		}

		services := generateServicesInNamespaces(namespaces, maxServicesPerCluster)
		createService := func(i int) {
			defer ginkgo.GinkgoRecover()
			framework.ExpectNoError(testutils.CreateServiceWithRetries(f.ClientSet, services[i].Namespace, services[i]))
		}
		framework.Logf("Creating %v test services", maxServicesPerCluster)
		workqueue.ParallelizeUntil(context.TODO(), parallelCreateServiceWorkers, len(services), createService)
		dnsTest := dnsTestCommon{
			f:  f,
			c:  f.ClientSet,
			ns: f.Namespace.Name,
		}
		dnsTest.createUtilPodLabel("e2e-dns-scale-records")
		defer dnsTest.deleteUtilPod()
		framework.Logf("Querying %v%% of service records", checkServicePercent*100)
		for i := 0; i < len(services); i++ {
			if i%(1/checkServicePercent) != 0 {
				continue
			}
			s := services[i]
			svc, err := f.ClientSet.CoreV1().Services(s.Namespace).Get(context.TODO(), s.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			qname := fmt.Sprintf("%v.%v.svc.%v", s.Name, s.Namespace, framework.TestContext.ClusterDNSDomain)
			framework.Logf("Querying %v expecting %v", qname, svc.Spec.ClusterIP)
			dnsTest.checkDNSRecordFrom(
				qname,
				func(actual []string) bool {
					return len(actual) == 1 && actual[0] == svc.Spec.ClusterIP
				},
				"cluster-dns",
				wait.ForeverTestTimeout,
			)
		}
	})
})

func generateServicesInNamespaces(namespaces []string, num int) []*v1.Service {
	services := make([]*v1.Service, num)
	for i := 0; i < num; i++ {
		services[i] = &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "svc-" + strconv.Itoa(i),
				Namespace: namespaces[i%len(namespaces)],
			},
			Spec: v1.ServiceSpec{
				Ports: []v1.ServicePort{{
					Port: 80,
				}},
			},
		}
	}
	return services
}
