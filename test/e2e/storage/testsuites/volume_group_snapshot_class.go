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

package testsuites

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Error substrings emitted by the group-snapshot sidecar controller when default
// VolumeGroupSnapshotClass selection fails. Sourced from:
// https://github.com/kubernetes-csi/external-snapshotter/blob/master/pkg/common-controller/groupsnapshot_controller_helper.go
const (
	errNoDefaultVGSClass         = "cannot find default group snapshot class"
	errMultipleDefaultVGSClasses = "default snapshot classes were found"

	defaultVGSClassAnnotationKey = "groupsnapshot.storage.kubernetes.io/is-default-class"
)

// VolumeGroupSnapshotClassTestSuite tests VolumeGroupSnapshotClass default class
// selection behavior: whether the controller correctly picks a default class when
// no className is specified in the VolumeGroupSnapshot, and whether it rejects
// ambiguous configurations (no default or multiple defaults).
type VolumeGroupSnapshotClassTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitVolumeGroupSnapshotClassTestSuite initializes the test suite for
// VolumeGroupSnapshotClass default class selection tests.
func InitVolumeGroupSnapshotClassTestSuite() storageframework.TestSuite {
	return &VolumeGroupSnapshotClassTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name: "volumegroupsnapshotclass",
			// A single pattern is sufficient: these tests exercise class selection
			// logic and do not depend on snapshot deletion policy.
			TestPatterns: []storageframework.TestPattern{
				storageframework.VolumeGroupSnapshotDelete,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			TestTags: []interface{}{feature.VolumeGroupSnapshotDataSource},
		},
	}
}

// GetTestSuiteInfo returns the test suite information.
func (s *VolumeGroupSnapshotClassTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

// SkipUnsupportedTests skips tests if the driver does not support group snapshots.
func (s *VolumeGroupSnapshotClassTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	_, ok := driver.(storageframework.VolumeGroupSnapshottableTestDriver)
	if !dInfo.Capabilities[storageframework.CapVolumeGroupSnapshot] || !ok {
		e2eskipper.Skipf("Driver %q does not support group snapshots - skipping", dInfo.Name)
	}
}

// DefineTests defines the test cases for VolumeGroupSnapshotClass default class selection.
func (s *VolumeGroupSnapshotClassTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	labelKey := "pvc-group"
	labelValue := "test-vgsclass"

	f := framework.NewDefaultFramework("volumegroupsnapshotclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Describe("VolumeGroupSnapshotClass", func() {
		var (
			snapshottableDriver storageframework.VolumeGroupSnapshottableTestDriver
			cs                  clientset.Interface
			config              *storageframework.PerTestConfig
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			snapshottableDriver = driver.(storageframework.VolumeGroupSnapshottableTestDriver)
			cs = f.ClientSet
			config = driver.PrepareTest(ctx, f)
		})

		// createPVCsWithPod creates two labeled PVCs and a pod that mounts them,
		// ensuring the PVCs enter the Bound state.
		createPVCsWithPod := func(ctx context.Context) (*storageframework.VolumeResource, *storageframework.VolumeResource, *v1.Pod) {
			ginkgo.By("creating two PVCs with labels for the VGS selector")
			vr1 := storageframework.CreateVolumeResource(ctx, driver, config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
			vr2 := storageframework.CreateVolumeResource(ctx, driver, config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
			for _, vr := range []*storageframework.VolumeResource{vr1, vr2} {
				patchData := fmt.Appendf(nil, `{"metadata":{"labels":{"%s":"%s"}}}`, labelKey, labelValue)
				updatedPVC, err := cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Patch(ctx, vr.Pvc.Name, types.MergePatchType, patchData, metav1.PatchOptions{})
				framework.ExpectNoError(err, "failed to add label to PVC %s", vr.Pvc.Name)
				vr.Pvc = updatedPVC
			}

			ginkgo.By("creating a pod to bind the PVCs")
			podConfig := e2epod.Config{
				NS:           f.Namespace.Name,
				PVCs:         []*v1.PersistentVolumeClaim{vr1.Pvc, vr2.Pvc},
				SeLinuxLabel: e2epv.SELinuxLabel,
			}
			pod, err := e2epod.MakeSecPod(&podConfig)
			framework.ExpectNoError(err, "failed to make pod config")
			pod, err = cs.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod")
			framework.ExpectNoError(
				e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow),
				"pod did not become running in time",
			)
			return vr1, vr2, pod
		}

		// createDefaultVGSClass creates a VolumeGroupSnapshotClass with the default
		// annotation and registers a DeferCleanup to delete it.
		createDefaultVGSClass := func(ctx context.Context) *unstructured.Unstructured {
			vgsclassSpec := snapshottableDriver.GetVolumeGroupSnapshotClass(ctx, config, map[string]string{"deletionPolicy": "Delete"})
			annotations := vgsclassSpec.GetAnnotations()
			if annotations == nil {
				annotations = make(map[string]string)
			}
			annotations[defaultVGSClassAnnotationKey] = "true"
			vgsclassSpec.SetAnnotations(annotations)
			vgsclass, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotClassGVR).Create(ctx, vgsclassSpec, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create VolumeGroupSnapshotClass")
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotClassGVR).Delete(ctx, vgsclass.GetName(), metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err, "failed to delete VolumeGroupSnapshotClass %s", vgsclass.GetName())
				}
			})
			return vgsclass
		}

		// createVGSWithoutClassName creates a VolumeGroupSnapshot without specifying
		// volumeGroupSnapshotClassName and registers a DeferCleanup to delete it.
		createVGSWithoutClassName := func(ctx context.Context) *unstructured.Unstructured {
			vgs, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotGVR).Namespace(f.Namespace.Name).Create(ctx,
				storageframework.GetVolumeGroupSnapshot(f.Namespace.Name, map[string]interface{}{labelKey: labelValue}, ""),
				metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create VolumeGroupSnapshot")
			ginkgo.DeferCleanup(func(ctx context.Context) {
				err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotGVR).Namespace(f.Namespace.Name).Delete(ctx, vgs.GetName(), metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err, "failed to delete VolumeGroupSnapshot %s", vgs.GetName())
				}
			})
			return vgs
		}

		// removeAndRestoreDefaultVGSClasses removes the default annotation from any
		// existing VolumeGroupSnapshotClasses for the same CSI driver, and restores
		// them via DeferCleanup after the test. This prevents interference with
		// default class selection tests.
		removeAndRestoreDefaultVGSClasses := func(ctx context.Context) {
			driverName := driver.GetDriverInfo().Name
			list, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotClassGVR).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err, "failed to list VolumeGroupSnapshotClasses")

			for _, item := range list.Items {
				classDriver, _, _ := unstructured.NestedString(item.Object, "driver")
				if classDriver != driverName {
					continue
				}
				if item.GetAnnotations()[defaultVGSClassAnnotationKey] != "true" {
					continue
				}
				name := item.GetName()
				framework.Logf("removing default annotation from VolumeGroupSnapshotClass %s for test isolation", name)
				removePatch := []byte(`{"metadata":{"annotations":{"groupsnapshot.storage.kubernetes.io/is-default-class":null}}}`)
				_, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotClassGVR).Patch(ctx, name, types.MergePatchType, removePatch, metav1.PatchOptions{})
				framework.ExpectNoError(err, "failed to remove default annotation from VolumeGroupSnapshotClass %s", name)
				ginkgo.DeferCleanup(func(ctx context.Context) {
					framework.Logf("restoring default annotation on VolumeGroupSnapshotClass %s", name)
					restorePatch := []byte(`{"metadata":{"annotations":{"groupsnapshot.storage.kubernetes.io/is-default-class":"true"}}}`)
					_, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotClassGVR).Patch(ctx, name, types.MergePatchType, restorePatch, metav1.PatchOptions{})
					framework.ExpectNoError(err, "failed to restore default annotation on VolumeGroupSnapshotClass %s", name)
				})
			}
		}

		// expectVolumeGroupSnapshotClassName fetches the named VolumeGroupSnapshot and
		// asserts spec.volumeGroupSnapshotClassName against expectedClassName.
		// Pass nil to assert the field is absent.
		// Pass a pointer to a string to assert the field equals that exact value,
		// including "" for the edge case where a VolumeGroupSnapshotClass carries a
		// "groupsnapshot.storage.kubernetes.io/is-default-class"="" annotation.
		expectVolumeGroupSnapshotClassName := func(ctx context.Context, vgsName string, expectedClassName *string) {
			ginkgo.GinkgoHelper()
			vgs, err := f.DynamicClient.Resource(utils.VolumeGroupSnapshotGVR).Namespace(f.Namespace.Name).Get(ctx, vgsName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get VolumeGroupSnapshot %s", vgsName)
			spec, ok := vgs.Object["spec"].(map[string]interface{})
			if !ok {
				ginkgo.Fail("failed to get VolumeGroupSnapshot spec: spec field is not a map[string]interface{}")
			}
			if expectedClassName == nil {
				gomega.Expect(spec["volumeGroupSnapshotClassName"]).To(gomega.BeNil(),
					"VolumeGroupSnapshot %s spec.volumeGroupSnapshotClassName should not be set by the controller", vgsName)
			} else {
				gomega.Expect(spec["volumeGroupSnapshotClassName"]).To(gomega.Equal(*expectedClassName),
					"VolumeGroupSnapshot %s spec.volumeGroupSnapshotClassName should be %q", vgsName, *expectedClassName)
			}
		}

		f.It("should use default VolumeGroupSnapshotClass when no className is specified", f.WithSerial(), func(ctx context.Context) {
			removeAndRestoreDefaultVGSClasses(ctx)

			vr1, vr2, pod := createPVCsWithPod(ctx)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
				framework.ExpectNoError(vr1.CleanupResource(ctx))
				framework.ExpectNoError(vr2.CleanupResource(ctx))
			})

			ginkgo.By("creating a VolumeGroupSnapshotClass with the default annotation")
			vgsclass := createDefaultVGSClass(ctx)

			ginkgo.By("creating a VolumeGroupSnapshot without specifying volumeGroupSnapshotClassName")
			vgs := createVGSWithoutClassName(ctx)

			ginkgo.DeferCleanup(func(ctx context.Context) {
				sr := &storageframework.VolumeGroupSnapshotResource{
					Config:  config,
					Pattern: storageframework.VolumeGroupSnapshotDelete,
					VGS:     vgs,
				}
				framework.ExpectNoError(sr.CleanupResource(ctx, f.Timeouts), "failed to clean up VolumeGroupSnapshot resources")
			})

			ginkgo.By("verifying the VolumeGroupSnapshot becomes ready using the default class")
			framework.ExpectNoError(utils.WaitForVolumeGroupSnapshotReady(ctx, f.DynamicClient, f.Namespace.Name, vgs.GetName(), framework.Poll, f.Timeouts.SnapshotCreate))

			ginkgo.By("verifying the default VolumeGroupSnapshotClass name was written back to the VolumeGroupSnapshot spec")
			expectedClassName := vgsclass.GetName()
			expectVolumeGroupSnapshotClassName(ctx, vgs.GetName(), &expectedClassName)
		})

		f.It("should report error when VolumeGroupSnapshot is created without className and no default class exists", f.WithSerial(), func(ctx context.Context) {
			removeAndRestoreDefaultVGSClasses(ctx)

			vr1, vr2, pod := createPVCsWithPod(ctx)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
				framework.ExpectNoError(vr1.CleanupResource(ctx))
				framework.ExpectNoError(vr2.CleanupResource(ctx))
			})

			ginkgo.By("creating a VolumeGroupSnapshot without className and without any default VolumeGroupSnapshotClass")
			vgs := createVGSWithoutClassName(ctx)

			ginkgo.By("verifying the VolumeGroupSnapshot enters an error state")
			framework.ExpectNoError(waitForVolumeGroupSnapshotError(ctx, f.DynamicClient, f.Namespace.Name, vgs.GetName(), framework.Poll, f.Timeouts.SnapshotCreate, errNoDefaultVGSClass))

			ginkgo.By("verifying spec.volumeGroupSnapshotClassName was not set by the controller")
			expectVolumeGroupSnapshotClassName(ctx, vgs.GetName(), nil)
		})

		f.It("should report error when VolumeGroupSnapshot is created without className and multiple default classes exist", f.WithSerial(), func(ctx context.Context) {
			removeAndRestoreDefaultVGSClasses(ctx)

			vr1, vr2, pod := createPVCsWithPod(ctx)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
				framework.ExpectNoError(vr1.CleanupResource(ctx))
				framework.ExpectNoError(vr2.CleanupResource(ctx))
			})

			ginkgo.By("creating two VolumeGroupSnapshotClasses both with the default annotation")
			createDefaultVGSClass(ctx)
			createDefaultVGSClass(ctx)

			ginkgo.By("creating a VolumeGroupSnapshot without specifying className")
			vgs := createVGSWithoutClassName(ctx)

			ginkgo.By("verifying the VolumeGroupSnapshot enters an error state due to multiple default classes")
			framework.ExpectNoError(waitForVolumeGroupSnapshotError(ctx, f.DynamicClient, f.Namespace.Name, vgs.GetName(), framework.Poll, f.Timeouts.SnapshotCreate, errMultipleDefaultVGSClasses))

			ginkgo.By("verifying spec.volumeGroupSnapshotClassName was not set by the controller")
			expectVolumeGroupSnapshotClassName(ctx, vgs.GetName(), nil)
		})
	})
}

// waitForVolumeGroupSnapshotError polls until the given VolumeGroupSnapshot has a
// non-nil error field in its status, then immediately asserts that the message
// contains expectedErrSubstr. Separating the two phases ensures that a controller
// message change surfaces as an immediate assertion failure rather than a silent
// timeout. Use the package-level error constants as expectedErrSubstr.
func waitForVolumeGroupSnapshotError(ctx context.Context, dc dynamic.Interface, ns, name string, poll, timeout time.Duration, expectedErrSubstr string) error {
	framework.Logf("Waiting up to %v for VolumeGroupSnapshot %s/%s to enter an error state", timeout, ns, name)
	var errorMessage string
	if successful := utils.WaitUntil(poll, timeout, func() bool {
		vgs, err := dc.Resource(utils.VolumeGroupSnapshotGVR).Namespace(ns).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get VolumeGroupSnapshot %q, retrying: %v", name, err)
			return false
		}
		status, ok := vgs.Object["status"].(map[string]interface{})
		if !ok || status == nil {
			return false
		}
		errObj, ok := status["error"].(map[string]interface{})
		if !ok || errObj == nil {
			return false
		}
		errorMessage, _ = errObj["message"].(string)
		return true
	}); !successful {
		return fmt.Errorf("VolumeGroupSnapshot %s/%s did not enter an error state within %v", ns, name, timeout)
	}

	gomega.Expect(errorMessage).To(gomega.ContainSubstring(expectedErrSubstr),
		"VolumeGroupSnapshot %s/%s has error status but message %q does not contain expected substring %q",
		ns, name, errorMessage, expectedErrSubstr)
	return nil
}
