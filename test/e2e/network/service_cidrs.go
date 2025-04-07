/*
Copyright 2023 The Kubernetes Authors.

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
	"net/netip"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/controlplane/controller/defaultservicecidr"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Service CIDRs", func() {

	fr := framework.NewDefaultFramework("servicecidrs")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		cs clientset.Interface
		ns string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = fr.ClientSet
		ns = fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

	})

	ginkgo.It("should create Services and serve on different Service CIDRs", func(ctx context.Context) {
		// use a ServiceCIDR that does not have risk to overlap with other ranges
		serviceCIDR := netip.MustParsePrefix("203.0.113.0/24") // RFC 5737 (TEST-NET-3) is provided for use in documentation.
		// use a random IP inside the range
		serviceIP := netip.MustParseAddr("203.0.113.10")
		if framework.TestContext.ClusterIsIPv6() {
			serviceCIDR = netip.MustParsePrefix("2001:db8:cb00::/64") // RFC 3849 IPv6 Address Prefix Reserved for Documentation.
			serviceIP = netip.MustParseAddr("2001:db8:cb00::a")
		}

		// create a new service CIDR
		svcCIDR := &networkingv1.ServiceCIDR{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-svc-cidr",
			},
			Spec: networkingv1.ServiceCIDRSpec{
				CIDRs: []string{serviceCIDR.String()},
			},
		}
		_, err := cs.NetworkingV1().ServiceCIDRs().Create(context.TODO(), svcCIDR, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating ServiceCIDR")
		ginkgo.DeferCleanup(cs.NetworkingV1().ServiceCIDRs().Delete, svcCIDR.Name, metav1.DeleteOptions{})

		if pollErr := wait.PollUntilContextTimeout(ctx, framework.Poll, e2eservice.RespondingTimeout, false, func(ctx context.Context) (bool, error) {
			svcCIDR, err := cs.NetworkingV1().ServiceCIDRs().Get(ctx, svcCIDR.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return isReady(svcCIDR), nil
		}); pollErr != nil {
			framework.Failf("Failed to wait for serviceCIDR to be ready: %v", pollErr)
		}

		serviceName := "cidr1-test"
		jig := e2eservice.NewTestJig(cs, ns, serviceName)

		ginkgo.By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		// Because this test run in parallel with other test there is a chance that the ClusterIP
		// gets allocated, we also need to ensure the ClusterIP belongs to the new Service range
		// so we need to explicitly set it. Try several times before giving up.
		var nodePortService *v1.Service
		for i := 0; i < 5; i++ {
			nodePortService, err = jig.CreateTCPService(ctx, func(svc *v1.Service) {
				svc.Spec.ClusterIP = serviceIP.String()
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.Ports = []v1.ServicePort{
					{Port: 80, Name: "http", Protocol: v1.ProtocolTCP, TargetPort: intstr.FromInt(9376)},
				}
			})
			if err != nil {
				serviceIP = serviceIP.Next()
			} else {
				break
			}
		}
		framework.ExpectNoError(err, "unable to allocate Service on new ServiceCIDR")

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := cs.CoreV1().Services(ns).Delete(ctx, serviceName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete service: %s in namespace: %s", serviceName, ns)
		})
		err = jig.CreateServicePods(ctx, 2)
		framework.ExpectNoError(err)
		execPod := e2epod.CreateExecPodOrFail(ctx, cs, ns, "execpod", nil)
		err = jig.CheckServiceReachability(ctx, nodePortService, execPod)
		framework.ExpectNoError(err)
	})

})

func isReady(serviceCIDR *networkingv1.ServiceCIDR) bool {
	if serviceCIDR == nil {
		return false
	}

	for _, condition := range serviceCIDR.Status.Conditions {
		if condition.Type == string(networkingv1.ServiceCIDRConditionReady) {
			return condition.Status == metav1.ConditionStatus(metav1.ConditionTrue)
		}
	}
	return false
}

var _ = common.SIGDescribe("ServiceCIDR and IPAddress API", func() {
	f := framework.NewDefaultFramework("servicecidr")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	/*
		Release: v1.34
		Testname: ServiceCIDR API
		Description:
		The networking.k8s.io API group MUST exist in the /apis discovery document.
		The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		The servicecidrs resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		The servicecidrs resource must support create, get, list, watch, update, patch, delete, and deletecollection.
		The servicecidrs/status resource must support update and patch
	*/

	framework.ConformanceIt("should support ServiceCIDR API operations", func(ctx context.Context) {
		ginkgo.By("getting")
		_, err := f.ClientSet.NetworkingV1().ServiceCIDRs().Get(ctx, defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
		if err != nil {
			framework.Failf("unexpected error getting default ServiceCIDR: %v", err)
		}

		ginkgo.By("patching")
		patchedServiceCIDR, err := f.ClientSet.NetworkingV1().ServiceCIDRs().Patch(ctx, defaultservicecidr.DefaultServiceCIDRName, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		if err != nil {
			framework.Failf("unexpected error patching IPAddress: %v", err)
		}
		if v, ok := patchedServiceCIDR.Annotations["patched"]; !ok || v != "true" {
			framework.Failf("patched object should have the applied annotation")
		}

		ginkgo.By("updating")
		var cidrToUpdate, updatedCIDR *networkingv1.ServiceCIDR
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			cidrToUpdate, err = f.ClientSet.NetworkingV1().ServiceCIDRs().Get(ctx, defaultservicecidr.DefaultServiceCIDRName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			cidrToUpdate.Annotations["updated"] = "true"
			updatedCIDR, err = f.ClientSet.NetworkingV1().ServiceCIDRs().Update(ctx, cidrToUpdate, metav1.UpdateOptions{})
			return err
		})
		if err != nil {
			framework.Failf("unexpected error updating IPAddress: %v", err)
		}
		if v, ok := updatedCIDR.Annotations["updated"]; !ok || v != "true" {
			framework.Failf("updated object should have the applied annotation")
		}

		ginkgo.By("listing")
		list, err := f.ClientSet.NetworkingV1().ServiceCIDRs().List(ctx, metav1.ListOptions{})
		if err != nil {
			framework.Failf("unexpected error listing ServiceCIDR: %v", err)
		}
		if len(list.Items) == 0 {
			framework.Failf("expected at list one, the default, ServiceCIDR")
		}

		ginkgo.By("watching")
		_, err = f.ClientSet.NetworkingV1().ServiceCIDRs().Watch(ctx, metav1.ListOptions{})
		if err != nil {
			framework.Failf("unexpected error watching ServiceCIDR: %v", err)
		}

	})

	/*
		Release: v1.34
		Testname: IPAddress API
		Description:
		The networking.k8s.io API group MUST exist in the /apis discovery document.
		The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		The ipaddresses resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		The ipaddresses resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support IPAddress API operations", func(ctx context.Context) {
		ip := &networkingv1.IPAddress{
			ObjectMeta: metav1.ObjectMeta{
				Name: "255.255.255.255",
				Labels: map[string]string{
					"type": "broadcast",
				},
			},
			Spec: networkingv1.IPAddressSpec{
				ParentRef: &networkingv1.ParentReference{
					Group:     "mygroup",
					Resource:  "myresources",
					Namespace: "foo",
					Name:      "bar",
				},
			},
		}

		ipv6 := &networkingv1.IPAddress{
			ObjectMeta: metav1.ObjectMeta{
				Name: "fe80::42:1dff:fe84:f9e2",
				Labels: map[string]string{
					"type": "link-local",
				},
			},
			Spec: networkingv1.IPAddressSpec{
				ParentRef: &networkingv1.ParentReference{
					Group:     "mygroup",
					Resource:  "myresources",
					Namespace: "foo",
					Name:      "bar",
				},
			},
		}

		ginkgo.By("creating")
		_, err := f.ClientSet.NetworkingV1().IPAddresses().Create(ctx, ip, metav1.CreateOptions{})
		if err != nil {
			framework.Failf("unexpected error getting IPAddress: %v", err)
		}
		_, err = f.ClientSet.NetworkingV1().IPAddresses().Create(ctx, ipv6, metav1.CreateOptions{})
		if err != nil {
			framework.Failf("unexpected error getting IPAddress: %v", err)
		}

		ginkgo.By("patching")
		patchedIP, err := f.ClientSet.NetworkingV1().IPAddresses().Patch(ctx, ip.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		if err != nil {
			framework.Failf("unexpected error patching IPAddress: %v", err)
		}
		if v, ok := patchedIP.Annotations["patched"]; !ok || v != "true" {
			framework.Failf("patched object should have the applied annotation")
		}

		ginkgo.By("updating")
		var ipToUpdate, updatedIP *networkingv1.IPAddress
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			ipToUpdate, err = f.ClientSet.NetworkingV1().IPAddresses().Get(ctx, ip.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			ipToUpdate.Annotations["updated"] = "true"
			updatedIP, err = f.ClientSet.NetworkingV1().IPAddresses().Update(ctx, ipToUpdate, metav1.UpdateOptions{})
			return err
		})
		if err != nil {
			framework.Failf("unexpected error updating IPAddress: %v", err)
		}
		if v, ok := updatedIP.Annotations["updated"]; !ok || v != "true" {
			framework.Failf("updated object should have the applied annotation")
		}

		ginkgo.By("getting")
		_, err = f.ClientSet.NetworkingV1().IPAddresses().Get(ctx, ip.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("unexpected error getting IPAddress: %v", err)
		}

		ginkgo.By("listing")
		list, err := f.ClientSet.NetworkingV1().IPAddresses().List(ctx, metav1.ListOptions{
			LabelSelector: "type=broadcast",
		})
		if err != nil {
			framework.Failf("unexpected error listing IPAddress: %v", err)
		}
		if len(list.Items) != 1 {
			framework.Failf("expected one IPAddress")
		}

		ginkgo.By("watching")
		_, err = f.ClientSet.NetworkingV1().IPAddresses().Watch(ctx, metav1.ListOptions{
			LabelSelector: "type=broadcast",
		})
		if err != nil {
			framework.Failf("unexpected error watching IPAddress: %v", err)
		}
		ginkgo.By("deleting")
		err = f.ClientSet.NetworkingV1().IPAddresses().Delete(ctx, ip.Name, metav1.DeleteOptions{})
		if err != nil {
			framework.Failf("unexpected error deleting IPAddress: %v", err)
		}

		ginkgo.By("deleting a collection")
		err = f.ClientSet.NetworkingV1().IPAddresses().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "type=link-local"})
		if err != nil {
			framework.Failf("unexpected error deleting IPAddress Collection: %v", err)
		}
		list, err = f.ClientSet.NetworkingV1().IPAddresses().List(ctx, metav1.ListOptions{
			LabelSelector: "type=link-local",
		})
		if err != nil {
			framework.Failf("unexpected error listing IPAddress: %v", err)
		}
		if len(list.Items) > 0 {
			framework.Failf("expected 0 IPAddresses")
		}
	})
})
