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
	"fmt"
	"strings"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	noProvisioner = "kubernetes.io/no-provisioner"
	pvNamePrefix  = "pv"
)

type volumeModeTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeModeTestSuite{}

// InitVolumeModeTestSuite returns volumeModeTestSuite that implements TestSuite interface
func InitVolumeModeTestSuite() TestSuite {
	return &volumeModeTestSuite{
		tsInfo: TestSuiteInfo{
			name: "volumeMode",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.FsVolModePreprovisionedPV,
				testpatterns.FsVolModeDynamicPV,
				testpatterns.BlockVolModePreprovisionedPV,
				testpatterns.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *volumeModeTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeModeTestSuite) skipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (t *volumeModeTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		cs clientset.Interface
		ns *v1.Namespace
		// genericVolumeTestResource contains pv, pvc, sc, etc., owns cleaning that up
		genericVolumeTestResource

		intreeOps   opCounts
		migratedOps opCounts
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	// No preconditions to test. Normally they would be in a BeforeEach here.

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volumemode")

	init := func() {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet

		// Now do the more expensive test initialization.
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
	}

	// manualInit initializes l.genericVolumeTestResource without creating the PV & PVC objects.
	manualInit := func() {
		init()

		fsType := pattern.FsType
		volBindMode := storagev1.VolumeBindingImmediate

		var (
			scName             string
			pvSource           *v1.PersistentVolumeSource
			volumeNodeAffinity *v1.VolumeNodeAffinity
		)

		l.genericVolumeTestResource = genericVolumeTestResource{
			driver:  driver,
			config:  l.config,
			pattern: pattern,
		}

		// Create volume for pre-provisioned volume tests
		l.volume = CreateVolume(driver, l.config, pattern.VolType)

		switch pattern.VolType {
		case testpatterns.PreprovisionedPV:
			if pattern.VolMode == v1.PersistentVolumeBlock {
				scName = fmt.Sprintf("%s-%s-sc-for-block", l.ns.Name, dInfo.Name)
			} else if pattern.VolMode == v1.PersistentVolumeFilesystem {
				scName = fmt.Sprintf("%s-%s-sc-for-file", l.ns.Name, dInfo.Name)
			}
			if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
				pvSource, volumeNodeAffinity = pDriver.GetPersistentVolumeSource(false, fsType, l.volume)
				if pvSource == nil {
					framework.Skipf("Driver %q does not define PersistentVolumeSource - skipping", dInfo.Name)
				}

				storageClass, pvConfig, pvcConfig := generateConfigsForPreprovisionedPVTest(scName, volBindMode, pattern.VolMode, *pvSource, volumeNodeAffinity)
				l.sc = storageClass
				l.pv = framework.MakePersistentVolume(pvConfig)
				l.pvc = framework.MakePersistentVolumeClaim(pvcConfig, l.ns.Name)
			}
		case testpatterns.DynamicPV:
			if dDriver, ok := driver.(DynamicPVTestDriver); ok {
				l.sc = dDriver.GetDynamicProvisionStorageClass(l.config, fsType)
				if l.sc == nil {
					framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
				}
				l.sc.VolumeBindingMode = &volBindMode

				l.pvc = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
					ClaimSize:        dDriver.GetClaimSize(),
					StorageClassName: &(l.sc.Name),
					VolumeMode:       &pattern.VolMode,
				}, l.ns.Name)
			}
		default:
			e2elog.Failf("Volume mode test doesn't support: %s", pattern.VolType)
		}
	}

	cleanup := func() {
		l.cleanupResource()

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}

		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	// We register different tests depending on the drive
	isBlockSupported := dInfo.Capabilities[CapBlock]
	switch pattern.VolType {
	case testpatterns.PreprovisionedPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			ginkgo.It("should fail to create pod by failing to mount volume [Slow]", func() {
				manualInit()
				defer cleanup()

				var err error

				ginkgo.By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				framework.ExpectNoError(err)

				ginkgo.By("Creating pv and pvc")
				l.pv, err = l.cs.CoreV1().PersistentVolumes().Create(l.pv)
				framework.ExpectNoError(err)

				// Prebind pv
				l.pvc.Spec.VolumeName = l.pv.Name
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				framework.ExpectNoError(err)

				framework.ExpectNoError(framework.WaitOnPVandPVC(l.cs, l.ns.Name, l.pv, l.pvc))

				ginkgo.By("Creating pod")
				pod, err := framework.CreateSecPodWithNodeSelection(l.cs, l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc},
					nil, false, "", false, false, framework.SELinuxLabel,
					nil, framework.NodeSelection{Name: l.config.ClientNodeName}, framework.PodStartTimeout)
				defer func() {
					framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
				}()
				framework.ExpectError(err)
			})
		}

	case testpatterns.DynamicPV:
		if pattern.VolMode == v1.PersistentVolumeBlock && !isBlockSupported {
			ginkgo.It("should fail in binding dynamic provisioned PV to PVC [Slow]", func() {
				manualInit()
				defer cleanup()

				var err error

				ginkgo.By("Creating sc")
				l.sc, err = l.cs.StorageV1().StorageClasses().Create(l.sc)
				framework.ExpectNoError(err)

				ginkgo.By("Creating pv and pvc")
				l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(l.pvc)
				framework.ExpectNoError(err)

				err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, l.cs, l.pvc.Namespace, l.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
				framework.ExpectError(err)
			})
		}
	default:
		e2elog.Failf("Volume mode test doesn't support volType: %v", pattern.VolType)
	}

	ginkgo.It("should fail to use a volume in a pod with mismatched mode [Slow]", func() {
		skipBlockTest(driver)
		init()
		l.genericVolumeTestResource = *createGenericVolumeTestResource(driver, l.config, pattern)
		defer cleanup()

		ginkgo.By("Creating pod")
		var err error
		pod := framework.MakeSecPod(l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc}, nil, false, "", false, false, framework.SELinuxLabel, nil)
		// Change volumeMounts to volumeDevices and the other way around
		pod = swapVolumeMode(pod)

		// Run the pod
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(pod)
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
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
		err = e2epod.WaitTimeoutForPodEvent(l.cs, pod.Name, l.ns.Name, eventSelector, msg, framework.PodStartTimeout)
		// Events are unreliable, don't depend on them. They're used only to speed up the test.
		if err != nil {
			e2elog.Logf("Warning: did not get event about mismatched volume use")
		}

		// Check the pod is still not running
		p, err := l.cs.CoreV1().Pods(l.ns.Name).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "could not re-read the pod after event (or timeout)")
		framework.ExpectEqual(p.Status.Phase, v1.PodPending)
	})

	ginkgo.It("should not mount / map unused volumes in a pod", func() {
		if pattern.VolMode == v1.PersistentVolumeBlock {
			skipBlockTest(driver)
		}
		init()
		l.genericVolumeTestResource = *createGenericVolumeTestResource(driver, l.config, pattern)
		defer cleanup()

		ginkgo.By("Creating pod")
		var err error
		pod := framework.MakeSecPod(l.ns.Name, []*v1.PersistentVolumeClaim{l.pvc}, nil, false, "", false, false, framework.SELinuxLabel, nil)
		for i := range pod.Spec.Containers {
			pod.Spec.Containers[i].VolumeDevices = nil
			pod.Spec.Containers[i].VolumeMounts = nil
		}

		// Run the pod
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(pod)
		framework.ExpectNoError(err)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, l.cs, pod))
		}()

		err = e2epod.WaitForPodNameRunningInNamespace(l.cs, pod.Name, pod.Namespace)
		framework.ExpectNoError(err)

		// Reload the pod to get its node
		pod, err = l.cs.CoreV1().Pods(l.ns.Name).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Listing mounted volumes in the pod")
		volumePaths, devicePaths, err := utils.ListPodVolumePluginDirectory(l.cs, pod)
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
	framework.PersistentVolumeConfig, framework.PersistentVolumeClaimConfig) {
	// StorageClass
	scConfig := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		Provisioner:       noProvisioner,
		VolumeBindingMode: &volBindMode,
	}
	// PV
	pvConfig := framework.PersistentVolumeConfig{
		PVSource:         pvSource,
		NodeAffinity:     volumeNodeAffinity,
		NamePrefix:       pvNamePrefix,
		StorageClassName: scName,
		VolumeMode:       &volMode,
	}
	// PVC
	pvcConfig := framework.PersistentVolumeClaimConfig{
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
