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

package testsuites

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	volevents "k8s.io/kubernetes/pkg/controller/volume/events"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	noProvisioner = "kubernetes.io/no-provisioner"
	pvNamePrefix  = "pv"
)

type volumeModeTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &volumeModeTestSuite{}

// InitCustomVolumeModeTestSuite returns volumeModeTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeModeTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeModeTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volumeMode",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitVolumeModeTestSuite returns volumeModeTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeModeTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.FsVolModePreprovisionedPV,
		storageframework.FsVolModeDynamicPV,
		storageframework.BlockVolModePreprovisionedPV,
		storageframework.BlockVolModeDynamicPV,
	}
	return InitCustomVolumeModeTestSuite(patterns)
}

func (t *volumeModeTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeModeTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
}

func (t *volumeModeTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config        *storageframework.PerTestConfig
		driverCleanup func()

		cs clientset.Interface
		ns *v1.Namespace
		// VolumeResource contains pv, pvc, sc, etc., owns cleaning that up
		storageframework.VolumeResource

		migrationCheck *migrationOpCheck
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volumemode", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
	}

	// manualInit initializes l.VolumeResource without creating the PV & PVC objects.
	manualInit := func(ctx context.Context) {
		init(ctx)

		fsType := pattern.FsType
		volBindMode := storagev1.VolumeBindingImmediate

		var (
			scName             string
			pvSource           *v1.PersistentVolumeSource
			volumeNodeAffinity *v1.VolumeNodeAffinity
		)

		l.VolumeResource = storageframework.VolumeResource{
			Config:  l.config,
			Pattern: pattern,
		}

		// Create volume for pre-provisioned volume tests
		l.Volume = storageframework.CreateVolume(ctx, driver, l.config, pattern.VolType)

		switch pattern.VolType {
		case storageframework.PreprovisionedPV:
			if pattern.VolMode == v1.PersistentVolumeBlock {
				scName = fmt.Sprintf("%s-%s-sc-for-block", l.ns.Name, dInfo.Name)
			} else if pattern.VolMode == v1.PersistentVolumeFilesystem {
				scName = fmt.Sprintf("%s-%s-sc-for-file", l.ns.Name, dInfo.Name)
			}
			if pDriver, ok := driver.(storageframework.PreprovisionedPVTestDriver); ok {
				pvSource, volumeNodeAffinity = pDriver.GetPersistentVolumeSource(false, fsType, l.Volume)
				if pvSource == nil {
					e2eskipper.Skipf("Driver %q does not define PersistentVolumeSource - skipping", dInfo.Name)
				}

				storageClass, pvConfig, pvcConfig := generateConfigsForPreprovisionedPVTest(scName, volBindMode, pattern.VolMode, *pvSource, volumeNodeAffinity)
				l.Sc = storageClass
				l.Pv = e2epv.MakePersistentVolume(pvConfig)
				l.Pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, l.ns.Name)
			}
		case storageframework.DynamicPV:
			if dDriver, ok := driver.(storageframework.DynamicPVTestDriver); ok {
				l.Sc = dDriver.GetDynamicProvisionStorageClass(ctx, l.config, fsType)
				if l.Sc == nil {
					e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
				}
				l.Sc.VolumeBindingMode = &volBindMode
				testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
				driverVolumeSizeRange := dInfo.SupportedSizeRange
				claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
				framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)

				l.Pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        claimSize,
					StorageClassName: &(l.Sc.Name),
					VolumeMode:       &pattern.VolMode,
				}, l.ns.Name)
			}
		default:
			framework.Failf("Volume mode test doesn't support: %s", pattern.VolType)
		}
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		errs = append(errs, l.CleanupResource(ctx))
		errs = append(errs, storageutils.TryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
	}

	// We register different tests depending on the drive
	isBlockSupported := dInfo.Capabilities[storageframework.CapBlock]
	switch pattern.VolType {
	case storageframework.PreprovisionedPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			f.It("should fail to create pod by failing to mount volume", f.WithSlow(), func(ctx context.Context) {
				manualInit(ctx)
				ginkgo.DeferCleanup(cleanup)

				var err error

				ginkgo.By("Creating sc")
				l.Sc, err = l.cs.StorageV1().StorageClasses().Create(ctx, l.Sc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create sc")

				ginkgo.By("Creating pv and pvc")
				l.Pv, err = l.cs.CoreV1().PersistentVolumes().Create(ctx, l.Pv, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create pv")

				// Prebind pv
				l.Pvc.Spec.VolumeName = l.Pv.Name
				l.Pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(ctx, l.Pvc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create pvc")

				framework.ExpectNoError(e2epv.WaitOnPVandPVC(ctx, l.cs, f.Timeouts, l.ns.Name, l.Pv, l.Pvc), "Failed to bind pv and pvc")

				ginkgo.By("Creating pod")
				podConfig := e2epod.Config{
					NS:            l.ns.Name,
					PVCs:          []*v1.PersistentVolumeClaim{l.Pvc},
					SeLinuxLabel:  e2epod.GetLinuxLabel(),
					NodeSelection: l.config.ClientNodeSelection,
					ImageID:       e2epod.GetDefaultTestImageID(),
				}
				pod, err := e2epod.MakeSecPod(&podConfig)
				framework.ExpectNoError(err, "Failed to create pod")

				pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(ctx, pod, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create pod")
				defer func() {
					framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, l.cs, pod), "Failed to delete pod")
				}()

				eventSelector := fields.Set{
					"involvedObject.kind":      "Pod",
					"involvedObject.name":      pod.Name,
					"involvedObject.namespace": l.ns.Name,
					"reason":                   events.FailedMountVolume,
				}.AsSelector().String()
				msg := "Unable to attach or mount volumes"

				err = e2eevents.WaitTimeoutForEvent(ctx, l.cs, l.ns.Name, eventSelector, msg, f.Timeouts.PodStart)
				// Events are unreliable, don't depend on the event. It's used only to speed up the test.
				if err != nil {
					framework.Logf("Warning: did not get event about FailedMountVolume")
				}

				// Check the pod is still not running
				p, err := l.cs.CoreV1().Pods(l.ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "could not re-read the pod after event (or timeout)")
				gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodPending), "Pod phase isn't pending")
			})
		}

	case storageframework.DynamicPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			f.It("should fail in binding dynamic provisioned PV to PVC", f.WithSlow(), "[LinuxOnly]", func(ctx context.Context) {
				manualInit(ctx)
				ginkgo.DeferCleanup(cleanup)

				var err error

				ginkgo.By("Creating sc")
				l.Sc, err = l.cs.StorageV1().StorageClasses().Create(ctx, l.Sc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create sc")

				ginkgo.By("Creating pv and pvc")
				l.Pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(ctx, l.Pvc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create pvc")

				eventSelector := fields.Set{
					"involvedObject.kind":      "PersistentVolumeClaim",
					"involvedObject.name":      l.Pvc.Name,
					"involvedObject.namespace": l.ns.Name,
					"reason":                   volevents.ProvisioningFailed,
				}.AsSelector().String()
				// The error message is different for each storage driver
				msg := ""

				err = e2eevents.WaitTimeoutForEvent(ctx, l.cs, l.ns.Name, eventSelector, msg, f.Timeouts.ClaimProvision)
				// Events are unreliable, don't depend on the event. It's used only to speed up the test.
				if err != nil {
					framework.Logf("Warning: did not get event about provisioning failed")
				}

				// Check the pvc is still pending
				pvc, err := l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Get(ctx, l.Pvc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Failed to re-read the pvc after event (or timeout)")
				gomega.Expect(pvc.Status.Phase).To(gomega.Equal(v1.ClaimPending), "PVC phase isn't pending")
			})
		}
	default:
		framework.Failf("Volume mode test doesn't support volType: %v", pattern.VolType)
	}

	f.It("should fail to use a volume in a pod with mismatched mode", f.WithSlow(), func(ctx context.Context) {
		skipTestIfBlockNotSupported(driver)
		init(ctx)
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		l.VolumeResource = *storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
		ginkgo.DeferCleanup(cleanup)

		ginkgo.By("Creating pod")
		var err error
		podConfig := e2epod.Config{
			NS:           l.ns.Name,
			PVCs:         []*v1.PersistentVolumeClaim{l.Pvc},
			SeLinuxLabel: e2epod.GetLinuxLabel(),
			ImageID:      e2epod.GetDefaultTestImageID(),
		}
		pod, err := e2epod.MakeSecPod(&podConfig)
		framework.ExpectNoError(err)

		// Change volumeMounts to volumeDevices and the other way around
		pod = swapVolumeMode(pod)

		// Run the pod
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create pod")
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, l.cs, pod), "Failed to delete pod")
		}()

		ginkgo.By("Waiting for the pod to fail")
		// Wait for an event that the pod is invalid.
		eventSelector := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      pod.Name,
			"involvedObject.namespace": l.ns.Name,
			"reason":                   events.FailedMountVolume,
		}.AsSelector().String()

		var msg string
		if pattern.VolMode == v1.PersistentVolumeBlock {
			msg = "has volumeMode Block, but is specified in volumeMounts"
		} else {
			msg = "has volumeMode Filesystem, but is specified in volumeDevices"
		}
		err = e2eevents.WaitTimeoutForEvent(ctx, l.cs, l.ns.Name, eventSelector, msg, f.Timeouts.PodStart)
		// Events are unreliable, don't depend on them. They're used only to speed up the test.
		if err != nil {
			framework.Logf("Warning: did not get event about mismatched volume use")
		}

		// Check the pod is still not running
		p, err := l.cs.CoreV1().Pods(l.ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "could not re-read the pod after event (or timeout)")
		gomega.Expect(p.Status.Phase).To(gomega.Equal(v1.PodPending), "Pod phase isn't pending")
	})

	ginkgo.It("should not mount / map unused volumes in a pod [LinuxOnly]", func(ctx context.Context) {
		if pattern.VolMode == v1.PersistentVolumeBlock {
			skipTestIfBlockNotSupported(driver)
		}
		init(ctx)
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		l.VolumeResource = *storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
		ginkgo.DeferCleanup(cleanup)

		ginkgo.By("Creating pod")
		var err error
		podConfig := e2epod.Config{
			NS:           l.ns.Name,
			PVCs:         []*v1.PersistentVolumeClaim{l.Pvc},
			SeLinuxLabel: e2epod.GetLinuxLabel(),
			ImageID:      e2epod.GetDefaultTestImageID(),
		}
		pod, err := e2epod.MakeSecPod(&podConfig)
		framework.ExpectNoError(err)

		for i := range pod.Spec.Containers {
			pod.Spec.Containers[i].VolumeDevices = nil
			pod.Spec.Containers[i].VolumeMounts = nil
		}

		// Run the pod
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, l.cs, pod))
		}()

		err = e2epod.WaitForPodNameRunningInNamespace(ctx, l.cs, pod.Name, pod.Namespace)
		framework.ExpectNoError(err)

		// Reload the pod to get its node
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty(), "pod should be scheduled to a node")
		node, err := l.cs.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Listing mounted volumes in the pod")
		hostExec := storageutils.NewHostExec(f)
		ginkgo.DeferCleanup(hostExec.Cleanup)
		volumePaths, devicePaths, err := listPodVolumePluginDirectory(ctx, hostExec, pod, node)
		framework.ExpectNoError(err)

		driverInfo := driver.GetDriverInfo()
		volumePlugin := driverInfo.InTreePluginName
		if len(volumePlugin) == 0 {
			// TODO: check if it's a CSI volume first
			volumePlugin = "kubernetes.io/csi"
		}
		ginkgo.By(fmt.Sprintf("Checking that volume plugin %s is not used in pod directory", volumePlugin))
		safeVolumePlugin := strings.ReplaceAll(volumePlugin, "/", "~")
		for _, path := range volumePaths {
			gomega.Expect(path).NotTo(gomega.ContainSubstring(safeVolumePlugin), fmt.Sprintf("no %s volume should be mounted into pod directory", volumePlugin))
		}
		for _, path := range devicePaths {
			gomega.Expect(path).NotTo(gomega.ContainSubstring(safeVolumePlugin), fmt.Sprintf("no %s volume should be symlinked into pod directory", volumePlugin))
		}
	})
}

func generateConfigsForPreprovisionedPVTest(scName string, volBindMode storagev1.VolumeBindingMode,
	volMode v1.PersistentVolumeMode, pvSource v1.PersistentVolumeSource, volumeNodeAffinity *v1.VolumeNodeAffinity) (*storagev1.StorageClass,
	e2epv.PersistentVolumeConfig, e2epv.PersistentVolumeClaimConfig) {
	// StorageClass
	scConfig := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		Provisioner:       noProvisioner,
		VolumeBindingMode: &volBindMode,
	}
	// PV
	pvConfig := e2epv.PersistentVolumeConfig{
		PVSource:         pvSource,
		NodeAffinity:     volumeNodeAffinity,
		NamePrefix:       pvNamePrefix,
		StorageClassName: scName,
		VolumeMode:       &volMode,
	}
	// PVC
	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &scName,
		VolumeMode:       &volMode,
	}

	return scConfig, pvConfig, pvcConfig
}

// swapVolumeMode changes volumeMounts to volumeDevices and the other way around
func swapVolumeMode(podTemplate *v1.Pod) *v1.Pod {
	pod := podTemplate.DeepCopy()
	for c := range pod.Spec.Containers {
		container := &pod.Spec.Containers[c]
		container.VolumeDevices = []v1.VolumeDevice{}
		container.VolumeMounts = []v1.VolumeMount{}

		// Change VolumeMounts to VolumeDevices
		for _, volumeMount := range podTemplate.Spec.Containers[c].VolumeMounts {
			container.VolumeDevices = append(container.VolumeDevices, v1.VolumeDevice{
				Name:       volumeMount.Name,
				DevicePath: volumeMount.MountPath,
			})
		}
		// Change VolumeDevices to VolumeMounts
		for _, volumeDevice := range podTemplate.Spec.Containers[c].VolumeDevices {
			container.VolumeMounts = append(container.VolumeMounts, v1.VolumeMount{
				Name:      volumeDevice.Name,
				MountPath: volumeDevice.DevicePath,
			})
		}
	}
	return pod
}

// listPodVolumePluginDirectory returns all volumes in /var/lib/kubelet/pods/<pod UID>/volumes/* and
// /var/lib/kubelet/pods/<pod UID>/volumeDevices/*
// Sample output:
//
//	/var/lib/kubelet/pods/a4717a30-000a-4081-a7a8-f51adf280036/volumes/kubernetes.io~secret/default-token-rphdt
//	/var/lib/kubelet/pods/4475b7a3-4a55-4716-9119-fd0053d9d4a6/volumeDevices/kubernetes.io~aws-ebs/pvc-5f9f80f5-c90b-4586-9966-83f91711e1c0
func listPodVolumePluginDirectory(ctx context.Context, h storageutils.HostExec, pod *v1.Pod, node *v1.Node) (mounts []string, devices []string, err error) {
	mountPath := filepath.Join("/var/lib/kubelet/pods/", string(pod.UID), "volumes")
	devicePath := filepath.Join("/var/lib/kubelet/pods/", string(pod.UID), "volumeDevices")

	mounts, err = listPodDirectory(ctx, h, mountPath, node)
	if err != nil {
		return nil, nil, err
	}
	devices, err = listPodDirectory(ctx, h, devicePath, node)
	if err != nil {
		return nil, nil, err
	}
	return mounts, devices, nil
}

func listPodDirectory(ctx context.Context, h storageutils.HostExec, path string, node *v1.Node) ([]string, error) {
	// Return no error if the directory does not exist (e.g. there are no block volumes used)
	_, err := h.IssueCommandWithResult(ctx, "test ! -d "+path, node)
	if err == nil {
		// The directory does not exist
		return nil, nil
	}
	// The directory either exists or a real error happened (e.g. "access denied").
	// Ignore the error, "find" will hit the error again and we report it there.

	// Inside /var/lib/kubelet/pods/<pod>/volumes, look for <volume_plugin>/<volume-name>, hence depth 2
	cmd := fmt.Sprintf("find %s -mindepth 2 -maxdepth 2", path)
	out, err := h.IssueCommandWithResult(ctx, cmd, node)
	if err != nil {
		return nil, fmt.Errorf("error checking directory %s on node %s: %w", path, node.Name, err)
	}
	return strings.Split(out, "\n"), nil
}
