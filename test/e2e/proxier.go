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
	"net"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"

	"k8s.io/apimachinery/pkg/types"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/proxy/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	fakeexec "k8s.io/utils/exec/testing"
)

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

type fakeSysctl struct{}

func (f *fakeSysctl) GetSysctl(sysctl string) (int, error) {
	return 1, nil
}
func (f *fakeSysctl) SetSysctl(sysctl string, newVal int) error {
	return nil
}

type fakeEventRecorder struct{}

func (f *fakeEventRecorder) Event(object pkgruntime.Object, eventtype, reason, message string) {}
func (f *fakeEventRecorder) Eventf(object pkgruntime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
}
func (f *fakeEventRecorder) PastEventf(object pkgruntime.Object, timestamp metav1.Time, eventtype, reason, messageFmt string, args ...interface{}) {
}

type fakeHealthzUpdater struct{}

func (f *fakeHealthzUpdater) UpdateTimestamp() {}

func NewFakeProxier(ipt utiliptables.Interface) *iptables.Proxier {
	p, err := iptables.NewProxier(ipt,
		&fakeSysctl{},
		&fakeexec.FakeExec{},
		time.Microsecond,
		time.Microsecond,
		false,
		0,
		"10.0.0.0/24",
		"testHostname",
		net.ParseIP("127.0.0.1"),
		&fakeEventRecorder{},
		&fakeHealthzUpdater{},
	)

	if err != nil {
		panic("can't create proxy!")
	}
	return p
}

func makeEndpointsMap(proxier *iptables.Proxier, allEndpoints ...*api.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsAdd(allEndpoints[i])
	}

	proxier.OnEndpointsSynced()
}

func makeTestService(namespace, name string, svcFunc func(*api.Service)) *api.Service {
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec:   api.ServiceSpec{},
		Status: api.ServiceStatus{},
	}
	svcFunc(svc)
	return svc
}

func makeServiceMap(proxier *iptables.Proxier, allServices ...*api.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.OnServiceSynced()
}

var _ = framework.KubeDescribe("Network", func() {

	fr := framework.NewDefaultFramework("network")

	It("should sync valid iptables rules", func() {
		nodes := framework.GetReadySchedulableNodesOrDie(fr.ClientSet)

		if len(nodes.Items) < 1 {
			framework.Skipf(
				"Test requires >= 1 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		selectNode := &nodes.Items[0]

		ipt := iptablestest.NewFake()
		fp := NewFakeProxier(ipt)

		svcPortName := proxy.ServicePortName{
			NamespacedName: makeNSN("ns1", "svc1"),
			Port:           "p80",
		}

		svcIP := "10.20.30.41"
		svcPort := 80
		svcExternalIPs := "50.60.70.81"

		makeServiceMap(fp,
			makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
				svc.Spec.Type = "ClusterIP"
				svc.Spec.ClusterIP = svcIP
				svc.Spec.ExternalIPs = []string{svcExternalIPs}
				svc.Spec.Ports = []api.ServicePort{{
					Name:       svcPortName.Port,
					Port:       int32(svcPort),
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt(svcPort),
				}}
			}),
		)
		makeEndpointsMap(fp)

		By("Syncing iptables rules")
		fp.Sync()

		// grab the lines from the ipt and test them in the node

		By("Checking rules are valid with iptables restore test")
		// If test flakes occur here, then this check should be performed
		// in a loop as there may be a race with the client connecting.
		result, err := framework.IssueSSHCommandWithResultAndInput(
			string(ipt.Lines), "sudo iptables-restore --test",
			framework.TestContext.Provider,
			selectNode)

		framework.ExpectNoError(err)
		Expect(result.Code).To(Equal(0))

	})
})
