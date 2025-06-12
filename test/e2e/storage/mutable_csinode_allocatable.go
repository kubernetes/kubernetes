/*
Copyright 2025 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("MutableCSINodeAllocatableCount", framework.WithFeatureGate(features.MutableCSINodeAllocatableCount), func() {
	f := framework.NewDefaultFramework("dynamic-allocatable")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		driver     storageframework.DynamicPVTestDriver
		testConfig *storageframework.PerTestConfig
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		driver = drivers.InitHostPathCSIDriverWithDynamicAllocatable().(storageframework.DynamicPVTestDriver)
		testConfig = driver.PrepareTest(ctx, f)
	})

	f.It("should observe dynamic changes in CSINode allocatable count", func(ctx context.Context) {
		cs := f.ClientSet

		ginkgo.By("Retrieving node for testing")
		nodeName := testConfig.ClientNodeSelection.Name
		if nodeName == "" {
			node, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
			framework.ExpectNoError(err)
			nodeName = node.Name
		}

		ginkgo.By("Retrieving driver details")
		sc := driver.GetDynamicProvisionStorageClass(ctx, testConfig, "")
		driverName := sc.Provisioner

		ginkgo.By("Retrieving initial allocatable value")
		initialLimit, err := getCSINodeLimits(ctx, cs, testConfig, nodeName, driverName)
		framework.ExpectNoError(err, "error retrieving initial CSINode limit")
		framework.Logf("Initial allocatable count: %d", initialLimit)

		ginkgo.By("Polling until value changes")
		err = wait.PollUntilContextTimeout(ctx, 10*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
			currentLimit, err := getCSINodeLimits(ctx, cs, testConfig, nodeName, driverName)
			if err != nil {
				framework.Logf("Error getting CSINode limits: %v", err)
				return false, nil
			}
			framework.Logf("Current allocatable count: %d", currentLimit)
			if currentLimit != initialLimit {
				framework.Logf("Detected change in allocatable count from %d to %d", initialLimit, currentLimit)
				return true, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "CSINode allocatable count did not change within timeout")
		framework.Logf("Successfully verified that CSINode allocatable count was updated")
	})
})

func getCSINodeLimits(ctx context.Context, cs clientset.Interface, config *storageframework.PerTestConfig, nodeName, driverName string) (int, error) {
	var limit int
	err := wait.PollUntilContextTimeout(ctx, 2*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("%s", err)
			return false, nil
		}
		var csiDriver *storagev1.CSINodeDriver
		for i, c := range csiNode.Spec.Drivers {
			if c.Name == driverName || c.Name == config.GetUniqueDriverName() {
				csiDriver = &csiNode.Spec.Drivers[i]
				break
			}
		}
		if csiDriver == nil {
			framework.Logf("CSINodeInfo does not have driver %s yet", driverName)
			return false, nil
		}
		if csiDriver.Allocatable == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable for driver %s", driverName)
		}
		if csiDriver.Allocatable.Count == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable.Count for driver %s", driverName)
		}
		limit = int(*csiDriver.Allocatable.Count)
		return true, nil
	})
	if err != nil {
		return 0, fmt.Errorf("could not get CSINode limit for driver %s: %w", driverName, err)
	}
	return limit, nil
}
