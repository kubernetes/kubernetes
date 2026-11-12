/*
Copyright The Kubernetes Authors.

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

package csimock

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// These tests cover kubelet only. They rely on the csi-attacher of the CSI mock
// driver not calling ControllerGetNodeInfo, so that a driver registered through
// driverRegistrations stays without a drivers entry.
var _ = utils.SIGDescribe("CSI Mock node info from controller", framework.WithFeatureGate(features.CSIControllerGetNodeInfo), func() {
	f := framework.NewDefaultFramework("csi-mock-node-info")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.It("should add the node ID to driverRegistrations when attach is required and remove it on unregistration", func(ctx context.Context) {
		var controllerGetNodeInfoTrue, controllerGetNodeInfoFalse atomic.Int32
		driver := drivers.InitMockCSIDriver(drivers.CSIMockDriverOpts{
			Embedded:                     true,
			RegisterDriver:               true,
			EnableNodeInfoFromController: true,
			Hooks:                        createNodeGetInfoPreHook(&controllerGetNodeInfoTrue, &controllerGetNodeInfoFalse),
		})
		cfg := driver.PrepareTest(ctx, f)
		driverName := cfg.GetUniqueDriverName()
		nodeName := cfg.ClientNodeSelection.Name
		getCSINode := framework.GetObject(f.ClientSet.StorageV1().CSINodes().Get, nodeName, metav1.GetOptions{})
		// The CSI mock driver uses its name as node ID.
		registration := storagev1.CSINodeDriverRegistration{Name: driverName, NodeID: driverName}

		ginkgo.By(fmt.Sprintf("Waiting for kubelet to add driver %s to driverRegistrations of CSINode %s", driverName, nodeName))
		gomega.Eventually(ctx, getCSINode).WithPolling(f.Timeouts.Poll).WithTimeout(f.Timeouts.PodStart).
			Should(gomega.HaveField("Spec.DriverRegistrations", gomega.ContainElement(registration)))
		registrationCalls := controllerGetNodeInfoTrue.Load()
		gomega.Expect(registrationCalls).To(gomega.BeNumerically(">", 0), "NodeGetInfo calls with controller_get_node_info=true")

		// kubelet must skip its periodic NodeGetInfo calls for the driver.
		ginkgo.By(fmt.Sprintf("Enabling periodic node info updates for driver %s", driverName))
		updateCSIDriverWithNodeAllocatableUpdatePeriodSeconds(ctx, f.ClientSet, driverName, updatePeriodSeconds)

		ginkgo.By(fmt.Sprintf("Checking that kubelet does not add driver %s to drivers of CSINode %s", driverName, nodeName))
		gomega.Consistently(ctx, getCSINode).WithPolling(f.Timeouts.Poll).WithTimeout(3 * time.Duration(updatePeriodSeconds) * time.Second).
			Should(gomega.And(
				gomega.HaveField("Spec.DriverRegistrations", gomega.ContainElement(registration)),
				gomega.Not(haveCSINodeDriver(driverName)),
			))

		ginkgo.By("Checking that kubelet did not call NodeGetInfo after the registration")
		gomega.Expect(controllerGetNodeInfoTrue.Load()).To(gomega.Equal(registrationCalls), "NodeGetInfo calls with controller_get_node_info=true")
		gomega.Expect(controllerGetNodeInfoFalse.Load()).To(gomega.BeZero(), "NodeGetInfo calls with controller_get_node_info=false")

		ginkgo.By("Deleting the CSI mock driver to unregister it")
		err := f.ClientSet.AppsV1().StatefulSets(cfg.DriverNamespace.Name).Delete(ctx, "csi-mockplugin", metav1.DeleteOptions{})
		framework.ExpectNoError(err, "delete the CSI mock driver StatefulSet")

		ginkgo.By(fmt.Sprintf("Waiting for kubelet to remove driver %s from driverRegistrations of CSINode %s", driverName, nodeName))
		gomega.Eventually(ctx, getCSINode).WithPolling(f.Timeouts.Poll).WithTimeout(f.Timeouts.PodDelete).
			ShouldNot(haveCSINodeDriverRegistration(driverName))
	})

	f.It("should use the existing NodeGetInfo flow when attach is not required", func(ctx context.Context) {
		var controllerGetNodeInfoTrue, controllerGetNodeInfoFalse atomic.Int32
		driver := drivers.InitMockCSIDriver(drivers.CSIMockDriverOpts{
			Embedded:                     true,
			RegisterDriver:               true,
			DisableAttach:                true,
			EnableNodeInfoFromController: true,
			Hooks:                        createNodeGetInfoPreHook(&controllerGetNodeInfoTrue, &controllerGetNodeInfoFalse),
		})
		cfg := driver.PrepareTest(ctx, f)
		driverName := cfg.GetUniqueDriverName()
		nodeName := cfg.ClientNodeSelection.Name

		ginkgo.By(fmt.Sprintf("Waiting for kubelet to add driver %s to drivers of CSINode %s", driverName, nodeName))
		err := drivers.WaitForCSIDriverRegistrationOnNode(ctx, nodeName, driverName, f.ClientSet)
		framework.ExpectNoError(err, "register driver %s on node %s", driverName, nodeName)

		ginkgo.By(fmt.Sprintf("Checking that driver %s is not in driverRegistrations of CSINode %s", driverName, nodeName))
		csiNode, err := f.ClientSet.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "get CSINode %s", nodeName)
		gomega.Expect(csiNode).NotTo(haveCSINodeDriverRegistration(driverName))

		ginkgo.By("Checking that kubelet called NodeGetInfo only without controller_get_node_info")
		gomega.Expect(controllerGetNodeInfoFalse.Load()).To(gomega.BeNumerically(">", 0), "NodeGetInfo calls with controller_get_node_info=false")
		gomega.Expect(controllerGetNodeInfoTrue.Load()).To(gomega.BeZero(), "NodeGetInfo calls with controller_get_node_info=true")
	})
})

// createNodeGetInfoPreHook creates a hook that counts the NodeGetInfo calls with
// controller_get_node_info set to true and to false.
func createNodeGetInfoPreHook(controllerGetNodeInfoTrue, controllerGetNodeInfoFalse *atomic.Int32) drivers.Hooks {
	return drivers.Hooks{
		Pre: func(ctx context.Context, fullMethod string, request interface{}) (reply interface{}, err error) {
			nodeGetInfoRequest, ok := request.(*csipbv1.NodeGetInfoRequest)
			if !ok {
				return nil, nil
			}
			if nodeGetInfoRequest.GetControllerGetNodeInfo() {
				controllerGetNodeInfoTrue.Add(1)
			} else {
				controllerGetNodeInfoFalse.Add(1)
			}
			return nil, nil
		},
	}
}

// haveCSINodeDriver matches a CSINode that has a drivers entry for the CSI driver.
func haveCSINodeDriver(driverName string) gomegatypes.GomegaMatcher {
	return gomega.HaveField("Spec.Drivers", gomega.ContainElement(gomega.HaveField("Name", driverName)))
}

// haveCSINodeDriverRegistration matches a CSINode that has a driverRegistrations entry for the CSI driver.
func haveCSINodeDriverRegistration(driverName string) gomegatypes.GomegaMatcher {
	return gomega.HaveField("Spec.DriverRegistrations", gomega.ContainElement(gomega.HaveField("Name", driverName)))
}
