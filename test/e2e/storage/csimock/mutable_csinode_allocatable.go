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

package csimock

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	initialMaxVolumesPerNode = int64(5)
	updatedMaxVolumesPerNode = int64(8)
	updatePeriodSeconds      = int64(10)
	timeout                  = 30 * time.Second
)

var _ = utils.SIGDescribe("MutableCSINodeAllocatableCount", framework.WithFeatureGate(features.MutableCSINodeAllocatableCount), func() {

	f := framework.NewDefaultFramework("mutable-allocatable-mock")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Describe("Dynamic Allocatable Count", func() {
		var (
			driver     drivers.MockCSITestDriver
			cfg        *storageframework.PerTestConfig
			clientSet  clientset.Interface
			nodeName   string
			driverName string
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			var calls int32
			hook := drivers.Hooks{
				Post: func(_ context.Context, method string, _ interface{}, reply interface{}, err error) (interface{}, error) {
					if strings.Contains(method, "NodeGetInfo") {
						if r, ok := reply.(*csipbv1.NodeGetInfoResponse); ok && err == nil {
							if atomic.AddInt32(&calls, 1) == 1 {
								r.MaxVolumesPerNode = initialMaxVolumesPerNode
							} else {
								r.MaxVolumesPerNode = updatedMaxVolumesPerNode
							}
							framework.Logf("NodeGetInfo called, setting MaxVolumesPerNode to %d", r.MaxVolumesPerNode)
							return r, nil
						}
					}
					return reply, err
				},
			}

			opts := drivers.CSIMockDriverOpts{
				Embedded:       true,
				RegisterDriver: true,
				Hooks:          hook,
			}
			driver = drivers.InitMockCSIDriver(opts)
			cfg = driver.PrepareTest(ctx, f)

			clientSet = f.ClientSet
			driverName = cfg.GetUniqueDriverName()
			nodeName = cfg.ClientNodeSelection.Name

			updateCSIDriverWithNodeAllocatableUpdatePeriodSeconds(ctx, clientSet, driverName, updatePeriodSeconds)

			err := drivers.WaitForCSIDriverRegistrationOnNode(ctx, nodeName, driverName, clientSet)
			framework.ExpectNoError(err)
		})

		f.It("should observe dynamic changes in CSINode allocatable count", func(ctx context.Context) {
			framework.Logf("Testing dynamic changes in CSINode allocatable count")
			initVal, err := readCSINodeLimit(ctx, clientSet, nodeName, driverName)
			framework.ExpectNoError(err)
			framework.Logf("Initial MaxVolumesPerNode limit: %d", initVal)

			err = wait.PollUntilContextTimeout(ctx, time.Duration(updatePeriodSeconds), timeout, true, func(ctx context.Context) (bool, error) {
				cur, err := readCSINodeLimit(ctx, clientSet, nodeName, driverName)
				if err != nil {
					return false, nil
				}
				return int64(cur) == updatedMaxVolumesPerNode, nil
			})

			framework.ExpectNoError(err, "CSINode allocatable count was not updated to %d in time", updatedMaxVolumesPerNode)
			framework.Logf("SUCCESS: MaxVolumesPerNode updated limit %d", updatedMaxVolumesPerNode)
		})
	})

	ginkgo.Describe("Attach Limit Exceeded", func() {
		var (
			driver     drivers.MockCSITestDriver
			cfg        *storageframework.PerTestConfig
			clientSet  clientset.Interface
			nodeName   string
			driverName string
			m          *mockDriverSetup
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			hook := drivers.Hooks{
				Post: func(_ context.Context, method string, _ interface{}, reply interface{}, err error) (interface{}, error) {
					if strings.Contains(method, "ControllerPublishVolume") && err == nil {
						return nil, status.Error(codes.ResourceExhausted, "attachment limit exceeded")
					}
					return reply, err
				},
			}

			opts := drivers.CSIMockDriverOpts{
				Embedded:                             true,
				EnableMutableCSINodeAllocatableCount: true,
				RegisterDriver:                       true,
				Hooks:                                hook,
			}
			driver = drivers.InitMockCSIDriver(opts)
			cfg = driver.PrepareTest(ctx, f)

			clientSet = f.ClientSet
			driverName = cfg.GetUniqueDriverName()
			nodeName = cfg.ClientNodeSelection.Name

			m = newMockDriverSetup(f)
			m.cs = clientSet
			m.driver = driver
			m.config = cfg
			m.provisioner = driverName

			updateCSIDriverWithNodeAllocatableUpdatePeriodSeconds(ctx, clientSet, driverName, updatePeriodSeconds)

			err := drivers.WaitForCSIDriverRegistrationOnNode(ctx, nodeName, driverName, clientSet)
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			m.cleanup(ctx)
		})

		f.It("should transition pod to failed state when attachment limit exceeded", func(ctx context.Context) {
			_, _, pod := m.createPod(ctx, pvcReference)
			if pod == nil {
				return
			}

			ginkgo.By("Waiting for Pod to fail with VolumeAttachmentLimitExceeded")
			err := e2epod.WaitForPodFailedReason(ctx, m.cs, pod, "VolumeAttachmentLimitExceeded", 4*time.Minute)
			framework.ExpectNoError(err, "Pod did not fail with VolumeAttachmentLimitExceeded")
		})
	})
})

func updateCSIDriverWithNodeAllocatableUpdatePeriodSeconds(ctx context.Context, cs clientset.Interface, driverName string, period int64) {
	err := wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		obj, err := cs.StorageV1().CSIDrivers().Get(ctx, driverName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if obj.Spec.NodeAllocatableUpdatePeriodSeconds != nil && *obj.Spec.NodeAllocatableUpdatePeriodSeconds == period {
			return true, nil
		}
		obj.Spec.NodeAllocatableUpdatePeriodSeconds = &period
		_, err = cs.StorageV1().CSIDrivers().Update(ctx, obj, metav1.UpdateOptions{})
		return err == nil, nil
	})
	framework.ExpectNoError(err, "enabling periodic CSINode allocatable updates failed")
}

func readCSINodeLimit(ctx context.Context, cs clientset.Interface, node, drv string) (int32, error) {
	c, err := cs.StorageV1().CSINodes().Get(ctx, node, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	for _, d := range c.Spec.Drivers {
		if d.Name == drv && d.Allocatable != nil && d.Allocatable.Count != nil {
			return *d.Allocatable.Count, nil
		}
	}
	return 0, fmt.Errorf("driver %q not present on CSINode", drv)
}
