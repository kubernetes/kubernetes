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

package storage

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	cachetools "k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	utilptr "k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	csiNodeLimitUpdateTimeout  = 5 * time.Minute
	csiPodUnschedulableTimeout = 5 * time.Minute
	csiResizeWaitPeriod        = 5 * time.Minute
	csiVolumeAttachmentTimeout = 7 * time.Minute
	// how long to wait for Resizing Condition on PVC to appear
	csiResizingConditionWait = 2 * time.Minute

	// Time for starting a pod with a volume.
	csiPodRunningTimeout = 5 * time.Minute

	// How log to wait for kubelet to unstage a volume after a pod is deleted
	csiUnstageWaitTimeout = 1 * time.Minute
)

// csiCall represents an expected call from Kubernetes to CSI mock driver and
// expected return value.
// When matching expected csiCall with a real CSI mock driver output, one csiCall
// matches *one or more* calls with the same method and error code.
// This is due to exponential backoff in Kubernetes, where the test cannot expect
// exact number of call repetitions.
type csiCall struct {
	expectedMethod string
	expectedError  codes.Code
	// This is a mark for the test itself to delete the tested pod *after*
	// this csiCall is received.
	deletePod bool
}

var _ = utils.SIGDescribe("CSI mock volume", func() {
	type testParameters struct {
		disableAttach       bool
		attachLimit         int
		registerDriver      bool
		lateBinding         bool
		enableTopology      bool
		podInfo             *bool
		storageCapacity     *bool
		scName              string // pre-selected storage class name; must be unique in the cluster
		enableResizing      bool   // enable resizing for both CSI mock driver and storageClass.
		enableNodeExpansion bool   // enable node expansion for CSI mock driver
		// just disable resizing on driver it overrides enableResizing flag for CSI mock driver
		disableResizingOnDriver bool
		enableSnapshot          bool
		enableVolumeMountGroup  bool // enable the VOLUME_MOUNT_GROUP node capability in the CSI mock driver.
		hooks                   *drivers.Hooks
		tokenRequests           []storagev1.TokenRequest
		requiresRepublish       *bool
		fsGroupPolicy           *storagev1.FSGroupPolicy
	}

	type mockDriverSetup struct {
		cs           clientset.Interface
		config       *storageframework.PerTestConfig
		testCleanups []func()
		pods         []*v1.Pod
		pvcs         []*v1.PersistentVolumeClaim
		sc           map[string]*storagev1.StorageClass
		vsc          map[string]*unstructured.Unstructured
		driver       drivers.MockCSITestDriver
		provisioner  string
		tp           testParameters
	}

	var m mockDriverSetup

	f := framework.NewDefaultFramework("csi-mock-volumes")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func(tp testParameters) {
		m = mockDriverSetup{
			cs:  f.ClientSet,
			sc:  make(map[string]*storagev1.StorageClass),
			vsc: make(map[string]*unstructured.Unstructured),
			tp:  tp,
		}
		cs := f.ClientSet
		var err error
		driverOpts := drivers.CSIMockDriverOpts{
			RegisterDriver:         tp.registerDriver,
			PodInfo:                tp.podInfo,
			StorageCapacity:        tp.storageCapacity,
			EnableTopology:         tp.enableTopology,
			AttachLimit:            tp.attachLimit,
			DisableAttach:          tp.disableAttach,
			EnableResizing:         tp.enableResizing,
			EnableNodeExpansion:    tp.enableNodeExpansion,
			EnableSnapshot:         tp.enableSnapshot,
			EnableVolumeMountGroup: tp.enableVolumeMountGroup,
			TokenRequests:          tp.tokenRequests,
			RequiresRepublish:      tp.requiresRepublish,
			FSGroupPolicy:          tp.fsGroupPolicy,
		}

		// At the moment, only tests which need hooks are
		// using the embedded CSI mock driver. The rest run
		// the driver inside the cluster although they could
		// changed to use embedding merely by setting
		// driverOpts.embedded to true.
		//
		// Not enabling it for all tests minimizes
		// the risk that the introduction of embedded breaks
		// some existings tests and avoids a dependency
		// on port forwarding, which is important if some of
		// these tests are supposed to become part of
		// conformance testing (port forwarding isn't
		// currently required).
		if tp.hooks != nil {
			driverOpts.Embedded = true
			driverOpts.Hooks = *tp.hooks
		}

		// this just disable resizing on driver, keeping resizing on SC enabled.
		if tp.disableResizingOnDriver {
			driverOpts.EnableResizing = false
		}

		m.driver = drivers.InitMockCSIDriver(driverOpts)
		config, testCleanup := m.driver.PrepareTest(f)
		m.testCleanups = append(m.testCleanups, testCleanup)
		m.config = config
		m.provisioner = config.GetUniqueDriverName()

		if tp.registerDriver {
			err = waitForCSIDriver(cs, m.config.GetUniqueDriverName())
			framework.ExpectNoError(err, "Failed to get CSIDriver %v", m.config.GetUniqueDriverName())
			m.testCleanups = append(m.testCleanups, func() {
				destroyCSIDriver(cs, m.config.GetUniqueDriverName())
			})
		}

		// Wait for the CSIDriver actually get deployed and CSINode object to be generated.
		// This indicates the mock CSI driver pod is up and running healthy.
		err = drivers.WaitForCSIDriverRegistrationOnNode(m.config.ClientNodeSelection.Name, m.config.GetUniqueDriverName(), cs)
		framework.ExpectNoError(err, "Failed to register CSIDriver %v", m.config.GetUniqueDriverName())
	}

	type volumeType string
	csiEphemeral := volumeType("CSI")
	genericEphemeral := volumeType("Ephemeral")
	pvcReference := volumeType("PVC")
	createPod := func(withVolume volumeType) (class *storagev1.StorageClass, claim *v1.PersistentVolumeClaim, pod *v1.Pod) {
		ginkgo.By("Creating pod")
		sc := m.driver.GetDynamicProvisionStorageClass(m.config, "")
		scTest := testsuites.StorageClassTest{
			Name:                 m.driver.GetDriverInfo().Name,
			Timeouts:             f.Timeouts,
			Provisioner:          sc.Provisioner,
			Parameters:           sc.Parameters,
			ClaimSize:            "1Gi",
			ExpectedSize:         "1Gi",
			DelayBinding:         m.tp.lateBinding,
			AllowVolumeExpansion: m.tp.enableResizing,
		}

		// The mock driver only works when everything runs on a single node.
		nodeSelection := m.config.ClientNodeSelection
		switch withVolume {
		case csiEphemeral:
			pod = startPausePodInline(f.ClientSet, scTest, nodeSelection, f.Namespace.Name)
		case genericEphemeral:
			class, pod = startPausePodGenericEphemeral(f.ClientSet, scTest, nodeSelection, m.tp.scName, f.Namespace.Name)
			if class != nil {
				m.sc[class.Name] = class
			}
			claim = &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pod.Name + "-" + pod.Spec.Volumes[0].Name,
					Namespace: f.Namespace.Name,
				},
			}
		case pvcReference:
			class, claim, pod = startPausePod(f.ClientSet, scTest, nodeSelection, m.tp.scName, f.Namespace.Name)
			if class != nil {
				m.sc[class.Name] = class
			}
			if claim != nil {
				m.pvcs = append(m.pvcs, claim)
			}
		}
		if pod != nil {
			m.pods = append(m.pods, pod)
		}
		return // result variables set above
	}

	createPodWithPVC := func(pvc *v1.PersistentVolumeClaim) (*v1.Pod, error) {
		nodeSelection := m.config.ClientNodeSelection
		pod, err := startPausePodWithClaim(m.cs, pvc, nodeSelection, f.Namespace.Name)
		if pod != nil {
			m.pods = append(m.pods, pod)
		}
		return pod, err
	}

	createPodWithFSGroup := func(fsGroup *int64) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
		ginkgo.By("Creating pod with fsGroup")
		nodeSelection := m.config.ClientNodeSelection
		sc := m.driver.GetDynamicProvisionStorageClass(m.config, "")
		scTest := testsuites.StorageClassTest{
			Name:                 m.driver.GetDriverInfo().Name,
			Provisioner:          sc.Provisioner,
			Parameters:           sc.Parameters,
			ClaimSize:            "1Gi",
			ExpectedSize:         "1Gi",
			DelayBinding:         m.tp.lateBinding,
			AllowVolumeExpansion: m.tp.enableResizing,
		}

		class, claim, pod := startBusyBoxPod(f.ClientSet, scTest, nodeSelection, m.tp.scName, f.Namespace.Name, fsGroup)

		if class != nil {
			m.sc[class.Name] = class
		}
		if claim != nil {
			m.pvcs = append(m.pvcs, claim)
		}

		if pod != nil {
			m.pods = append(m.pods, pod)
		}

		return class, claim, pod
	}

	cleanup := func() {
		cs := f.ClientSet
		var errs []error

		for _, pod := range m.pods {
			ginkgo.By(fmt.Sprintf("Deleting pod %s", pod.Name))
			errs = append(errs, e2epod.DeletePodWithWait(cs, pod))
		}

		for _, claim := range m.pvcs {
			ginkgo.By(fmt.Sprintf("Deleting claim %s", claim.Name))
			claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
			if err == nil {
				if err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{}); err != nil {
					errs = append(errs, err)
				}
				if claim.Spec.VolumeName != "" {
					errs = append(errs, e2epv.WaitForPersistentVolumeDeleted(cs, claim.Spec.VolumeName, framework.Poll, 2*time.Minute))
				}
			}
		}

		for _, sc := range m.sc {
			ginkgo.By(fmt.Sprintf("Deleting storageclass %s", sc.Name))
			cs.StorageV1().StorageClasses().Delete(context.TODO(), sc.Name, metav1.DeleteOptions{})
		}

		for _, vsc := range m.vsc {
			ginkgo.By(fmt.Sprintf("Deleting volumesnapshotclass %s", vsc.GetName()))
			m.config.Framework.DynamicClient.Resource(utils.SnapshotClassGVR).Delete(context.TODO(), vsc.GetName(), metav1.DeleteOptions{})
		}
		ginkgo.By("Cleaning up resources")
		for _, cleanupFunc := range m.testCleanups {
			cleanupFunc()
		}

		err := utilerrors.NewAggregate(errs)
		framework.ExpectNoError(err, "while cleaning up after test")
	}

	// The CSIDriverRegistry feature gate is needed for this test in Kubernetes 1.12.
	ginkgo.Context("CSI attach test using mock driver", func() {
		tests := []struct {
			name                   string
			disableAttach          bool
			deployClusterRegistrar bool
			volumeType             volumeType
		}{
			{
				name:                   "should not require VolumeAttach for drivers without attachment",
				disableAttach:          true,
				deployClusterRegistrar: true,
			},
			{
				name:                   "should require VolumeAttach for drivers with attachment",
				deployClusterRegistrar: true,
			},
			{
				name:                   "should require VolumeAttach for ephemermal volume and drivers with attachment",
				deployClusterRegistrar: true,
				volumeType:             genericEphemeral,
			},
			{
				name:                   "should preserve attachment policy when no CSIDriver present",
				deployClusterRegistrar: false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func() {
				var err error
				init(testParameters{registerDriver: test.deployClusterRegistrar, disableAttach: test.disableAttach})
				defer cleanup()

				volumeType := t.volumeType
				if volumeType == "" {
					volumeType = pvcReference
				}
				_, claim, pod := createPod(volumeType)
				if pod == nil {
					return
				}
				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				ginkgo.By("Checking if VolumeAttachment was created for the pod")
				testConfig := storageframework.ConvertTestConfig(m.config)
				attachmentName := e2evolume.GetVolumeAttachmentName(m.cs, testConfig, m.provisioner, claim.Name, claim.Namespace)
				_, err = m.cs.StorageV1().VolumeAttachments().Get(context.TODO(), attachmentName, metav1.GetOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						if !test.disableAttach {
							framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
						}
					} else {
						framework.ExpectNoError(err, "Failed to find VolumeAttachment")
					}
				}
				if test.disableAttach {
					framework.ExpectError(err, "Unexpected VolumeAttachment found")
				}
			})

		}
	})

	ginkgo.Context("CSI CSIDriver deployment after pod creation using non-attachable mock driver", func() {
		ginkgo.It("should bringup pod after deploying CSIDriver attach=false [Slow]", func() {
			var err error
			init(testParameters{registerDriver: false, disableAttach: true})
			defer cleanup()

			_, claim, pod := createPod(pvcReference) // late binding as specified above
			if pod == nil {
				return
			}

			ginkgo.By("Checking if attaching failed and pod cannot start")
			eventSelector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      pod.Name,
				"involvedObject.namespace": pod.Namespace,
				"reason":                   events.FailedAttachVolume,
			}.AsSelector().String()
			msg := "AttachVolume.Attach failed for volume"

			err = e2eevents.WaitTimeoutForEvent(m.cs, pod.Namespace, eventSelector, msg, f.Timeouts.PodStart)
			if err != nil {
				podErr := e2epod.WaitTimeoutForPodRunningInNamespace(m.cs, pod.Name, pod.Namespace, 10*time.Second)
				framework.ExpectError(podErr, "Pod should not be in running status because attaching should failed")
				// Events are unreliable, don't depend on the event. It's used only to speed up the test.
				framework.Logf("Attach should fail and the corresponding event should show up, error: %v", err)
			}

			// VolumeAttachment should be created because the default value for CSI attachable is true
			ginkgo.By("Checking if VolumeAttachment was created for the pod")
			testConfig := storageframework.ConvertTestConfig(m.config)
			attachmentName := e2evolume.GetVolumeAttachmentName(m.cs, testConfig, m.provisioner, claim.Name, claim.Namespace)
			_, err = m.cs.StorageV1().VolumeAttachments().Get(context.TODO(), attachmentName, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
				} else {
					framework.ExpectNoError(err, "Failed to find VolumeAttachment")
				}
			}

			ginkgo.By("Deploy CSIDriver object with attachRequired=false")
			driverNamespace := m.config.DriverNamespace

			canAttach := false
			o := utils.PatchCSIOptions{
				OldDriverName: "csi-mock",
				NewDriverName: "csi-mock-" + f.UniqueName,
				CanAttach:     &canAttach,
			}
			cleanupCSIDriver, err := utils.CreateFromManifests(f, driverNamespace, func(item interface{}) error {
				return utils.PatchCSIDeployment(f, o, item)
			}, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driverinfo.yaml")
			if err != nil {
				framework.Failf("fail to deploy CSIDriver object: %v", err)
			}
			m.testCleanups = append(m.testCleanups, cleanupCSIDriver)

			ginkgo.By("Wait for the pod in running status")
			err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
			framework.ExpectNoError(err, "Failed to start pod: %v", err)

			ginkgo.By(fmt.Sprintf("Wait for the volumeattachment to be deleted up to %v", csiVolumeAttachmentTimeout))
			// This step can be slow because we have to wait either a NodeUpdate event happens or
			// the detachment for this volume timeout so that we can do a force detach.
			err = e2evolume.WaitForVolumeAttachmentTerminated(attachmentName, m.cs, csiVolumeAttachmentTimeout)
			framework.ExpectNoError(err, "Failed to delete VolumeAttachment: %v", err)
		})
	})

	ginkgo.Context("CSI workload information using mock driver", func() {
		var (
			err          error
			podInfoTrue  = true
			podInfoFalse = false
		)
		tests := []struct {
			name                   string
			podInfoOnMount         *bool
			deployClusterRegistrar bool
			expectPodInfo          bool
			expectEphemeral        bool
		}{
			{
				name:                   "should not be passed when podInfoOnMount=nil",
				podInfoOnMount:         nil,
				deployClusterRegistrar: true,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
			{
				name:                   "should be passed when podInfoOnMount=true",
				podInfoOnMount:         &podInfoTrue,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        false,
			},
			{
				name:                   "contain ephemeral=true when using inline volume",
				podInfoOnMount:         &podInfoTrue,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        true,
			},
			{
				name:                   "should not be passed when podInfoOnMount=false",
				podInfoOnMount:         &podInfoFalse,
				deployClusterRegistrar: true,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
			{
				name:                   "should not be passed when CSIDriver does not exist",
				deployClusterRegistrar: false,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func() {
				init(testParameters{
					registerDriver: test.deployClusterRegistrar,
					podInfo:        test.podInfoOnMount})

				defer cleanup()

				withVolume := pvcReference
				if test.expectEphemeral {
					withVolume = csiEphemeral
				}
				_, _, pod := createPod(withVolume)
				if pod == nil {
					return
				}
				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				// If we expect an ephemeral volume, the feature has to be enabled.
				// Otherwise need to check if we expect pod info, because the content
				// of that depends on whether the feature is enabled or not.
				csiInlineVolumesEnabled := test.expectEphemeral
				if test.expectPodInfo {
					ginkgo.By("checking for CSIInlineVolumes feature")
					csiInlineVolumesEnabled, err = testsuites.CSIInlineVolumesEnabled(m.cs, f.Timeouts, f.Namespace.Name)
					framework.ExpectNoError(err, "failed to test for CSIInlineVolumes")
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting")

				ginkgo.By("Checking CSI driver logs")
				err = checkPodLogs(m.driver.GetCalls, pod, test.expectPodInfo, test.expectEphemeral, csiInlineVolumesEnabled, false, 1)
				framework.ExpectNoError(err)
			})
		}
	})

	ginkgo.Context("CSI volume limit information using mock driver", func() {
		ginkgo.It("should report attach limit when limit is bigger than 0 [Slow]", func() {
			// define volume limit to be 2 for this test
			var err error
			init(testParameters{attachLimit: 2})
			defer cleanup()
			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 2))

			_, _, pod1 := createPod(pvcReference)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating first pod")

			err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := createPod(pvcReference)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating second pod")

			err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod2.Name, pod2.Namespace)
			framework.ExpectNoError(err, "Failed to start pod2: %v", err)

			_, _, pod3 := createPod(pvcReference)
			gomega.Expect(pod3).NotTo(gomega.BeNil(), "while creating third pod")
			err = waitForMaxVolumeCondition(pod3, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod3)
		})

		ginkgo.It("should report attach limit for generic ephemeral volume when persistent volume is attached [Slow]", func() {
			// define volume limit to be 2 for this test
			var err error
			init(testParameters{attachLimit: 1})
			defer cleanup()
			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 1))

			_, _, pod1 := createPod(pvcReference)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating pod with persistent volume")

			err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := createPod(genericEphemeral)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod with ephemeral volume")
			err = waitForMaxVolumeCondition(pod2, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod2)
		})

		ginkgo.It("should report attach limit for persistent volume when generic ephemeral volume is attached [Slow]", func() {
			// define volume limit to be 2 for this test
			var err error
			init(testParameters{attachLimit: 1})
			defer cleanup()
			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 1))

			_, _, pod1 := createPod(genericEphemeral)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating pod with persistent volume")

			err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := createPod(pvcReference)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod with ephemeral volume")
			err = waitForMaxVolumeCondition(pod2, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod2)
		})
	})

	ginkgo.Context("CSI Volume expansion", func() {
		tests := []struct {
			name                    string
			nodeExpansionRequired   bool
			disableAttach           bool
			disableResizingOnDriver bool
			expectFailure           bool
		}{
			{
				name:                  "should expand volume without restarting pod if nodeExpansion=off",
				nodeExpansionRequired: false,
			},
			{
				name:                  "should expand volume by restarting pod if attach=on, nodeExpansion=on",
				nodeExpansionRequired: true,
			},
			{
				name:                  "should expand volume by restarting pod if attach=off, nodeExpansion=on",
				disableAttach:         true,
				nodeExpansionRequired: true,
			},
			{
				name:                    "should not expand volume if resizingOnDriver=off, resizingOnSC=on",
				disableResizingOnDriver: true,
				expectFailure:           true,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func() {
				var err error
				tp := testParameters{
					enableResizing:          true,
					enableNodeExpansion:     test.nodeExpansionRequired,
					disableResizingOnDriver: test.disableResizingOnDriver,
				}
				// disabling attach requires drive registration feature
				if test.disableAttach {
					tp.disableAttach = true
					tp.registerDriver = true
				}

				init(tp)
				defer cleanup()

				sc, pvc, pod := createPod(pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				framework.ExpectEqual(*sc.AllowVolumeExpansion, true, "failed creating sc with allowed expansion")

				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}
				if test.expectFailure {
					err = testsuites.WaitForResizingCondition(pvc, m.cs, csiResizingConditionWait)
					framework.ExpectError(err, "unexpected resizing condition on PVC")
					return
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for CSI PV resize to finish")

				checkPVCSize := func() {
					ginkgo.By("Waiting for PVC resize to finish")
					pvc, err = testsuites.WaitForFSResize(pvc, m.cs)
					framework.ExpectNoError(err, "while waiting for PVC resize to finish")

					pvcConditions := pvc.Status.Conditions
					framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
				}

				// if node expansion is not required PVC should be resized as well
				if !test.nodeExpansionRequired {
					checkPVCSize()
				} else {
					ginkgo.By("Checking for conditions on pvc")
					npvc, err := testsuites.WaitForPendingFSResizeCondition(pvc, m.cs)
					framework.ExpectNoError(err, "While waiting for pvc to have fs resizing condition")
					pvc = npvc

					inProgressConditions := pvc.Status.Conditions
					if len(inProgressConditions) > 0 {
						framework.ExpectEqual(inProgressConditions[0].Type, v1.PersistentVolumeClaimFileSystemResizePending, "pvc must have fs resizing condition")
					}

					ginkgo.By("Deleting the previously created pod")
					err = e2epod.DeletePodWithWait(m.cs, pod)
					framework.ExpectNoError(err, "while deleting pod for resizing")

					ginkgo.By("Creating a new pod with same volume")
					pod2, err := createPodWithPVC(pvc)
					gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod for csi resizing")
					framework.ExpectNoError(err, "while recreating pod for resizing")

					checkPVCSize()
				}
			})
		}
	})
	ginkgo.Context("CSI online volume expansion", func() {
		tests := []struct {
			name          string
			disableAttach bool
		}{
			{
				name: "should expand volume without restarting pod if attach=on, nodeExpansion=on",
			},
			{
				name:          "should expand volume without restarting pod if attach=off, nodeExpansion=on",
				disableAttach: true,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				var err error
				params := testParameters{enableResizing: true, enableNodeExpansion: true}
				if test.disableAttach {
					params.disableAttach = true
					params.registerDriver = true
				}

				init(params)

				defer cleanup()

				sc, pvc, pod := createPod(pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				framework.ExpectEqual(*sc.AllowVolumeExpansion, true, "failed creating sc with allowed expansion")

				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for PV resize to finish")

				ginkgo.By("Waiting for PVC resize to finish")
				pvc, err = testsuites.WaitForFSResize(pvc, m.cs)
				framework.ExpectNoError(err, "while waiting for PVC to finish")

				pvcConditions := pvc.Status.Conditions
				framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")

			})
		}
	})

	ginkgo.Context("CSI NodeStage error cases [Slow]", func() {
		trackedCalls := []string{
			"NodeStageVolume",
			"NodeUnstageVolume",
		}

		tests := []struct {
			name             string
			expectPodRunning bool
			expectedCalls    []csiCall

			// Called for each NodeStateVolume calls, with counter incremented atomically before
			// the invocation (i.e. first value will be 1).
			nodeStageHook func(counter int64) error
		}{
			{
				// This is already tested elsewhere, adding simple good case here to test the test framework.
				name:             "should call NodeUnstage after NodeStage success",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
			},
			{
				// Kubelet should repeat NodeStage as long as the pod exists
				name:             "should retry NodeStage after NodeStage final error",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					// This matches all 3 NodeStage calls with InvalidArgument error
					{expectedMethod: "NodeStageVolume", expectedError: codes.InvalidArgument},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				// Fail first 3 NodeStage requests, 4th succeeds
				nodeStageHook: func(counter int64) error {
					if counter < 4 {
						return status.Error(codes.InvalidArgument, "fake error")
					}
					return nil
				},
			},
			{
				// Kubelet should repeat NodeStage as long as the pod exists
				name:             "should retry NodeStage after NodeStage ephemeral error",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					// This matches all 3 NodeStage calls with DeadlineExceeded error
					{expectedMethod: "NodeStageVolume", expectedError: codes.DeadlineExceeded},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				// Fail first 3 NodeStage requests, 4th succeeds
				nodeStageHook: func(counter int64) error {
					if counter < 4 {
						return status.Error(codes.DeadlineExceeded, "fake error")
					}
					return nil
				},
			},
			{
				// After NodeUnstage with ephemeral error, the driver may continue staging the volume.
				// Kubelet should call NodeUnstage to make sure the volume is really unstaged after
				// the pod is deleted.
				name:             "should call NodeUnstage after NodeStage ephemeral error",
				expectPodRunning: false,
				expectedCalls: []csiCall{
					// Delete the pod before NodeStage succeeds - it should get "uncertain" because of ephemeral error
					// This matches all repeated NodeStage calls with DeadlineExceeded error (due to exp. backoff).
					{expectedMethod: "NodeStageVolume", expectedError: codes.DeadlineExceeded, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeStageHook: func(counter int64) error {
					return status.Error(codes.DeadlineExceeded, "fake error")
				},
			},
			{
				// After NodeUnstage with final error, kubelet can be sure the volume is not staged.
				// The test checks that NodeUnstage is *not* called.
				name:             "should not call NodeUnstage after NodeStage final error",
				expectPodRunning: false,
				expectedCalls: []csiCall{
					// Delete the pod before NodeStage succeeds - it should get "globally unmounted" because of final error.
					// This matches all repeated NodeStage calls with InvalidArgument error (due to exp. backoff).
					{expectedMethod: "NodeStageVolume", expectedError: codes.InvalidArgument, deletePod: true},
				},
				// nodeStageScript: `INVALIDARGUMENT;`,
				nodeStageHook: func(counter int64) error {
					return status.Error(codes.InvalidArgument, "fake error")
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				var hooks *drivers.Hooks
				if test.nodeStageHook != nil {
					hooks = createPreHook("NodeStageVolume", test.nodeStageHook)
				}
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					hooks:          hooks,
				})
				defer cleanup()

				_, claim, pod := createPod(pvcReference)
				if pod == nil {
					return
				}
				// Wait for PVC to get bound to make sure the CSI driver is fully started.
				err := e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, f.ClientSet, f.Namespace.Name, claim.Name, time.Second, framework.ClaimProvisionTimeout)
				framework.ExpectNoError(err, "while waiting for PVC to get provisioned")

				ginkgo.By("Waiting for expected CSI calls")
				// Watch for all calls up to deletePod = true
				ctx, cancel := context.WithTimeout(context.Background(), csiPodRunningTimeout)
				defer cancel()
				for {
					if ctx.Err() != nil {
						framework.Failf("timed out waiting for the CSI call that indicates that the pod can be deleted: %v", test.expectedCalls)
					}
					time.Sleep(1 * time.Second)
					_, index, err := compareCSICalls(trackedCalls, test.expectedCalls, m.driver.GetCalls)
					framework.ExpectNoError(err, "while waiting for initial CSI calls")
					if index == 0 {
						// No CSI call received yet
						continue
					}
					// Check the last *received* call wanted the pod to be deleted
					if test.expectedCalls[index-1].deletePod {
						break
					}
				}

				if test.expectPodRunning {
					ginkgo.By("Waiting for pod to be running")
					err := e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
					framework.ExpectNoError(err, "Failed to start pod: %v", err)
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting")

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.Poll(time.Second, csiUnstageWaitTimeout, func() (done bool, err error) {
					_, index, err := compareCSICalls(trackedCalls, test.expectedCalls, m.driver.GetCalls)
					if err != nil {
						return true, err
					}
					if index == 0 {
						// No CSI call received yet
						return false, nil
					}
					if len(test.expectedCalls) == index {
						// all calls received
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err, "while waiting for all CSI calls")
			})
		}
	})

	ginkgo.Context("CSI NodeUnstage error cases [Slow]", func() {
		trackedCalls := []string{
			"NodeStageVolume",
			"NodeUnstageVolume",
		}

		// Each test starts two pods in sequence.
		// The first pod always runs successfully, but NodeUnstage hook can set various error conditions.
		// The test then checks how NodeStage of the second pod is called.
		tests := []struct {
			name          string
			expectedCalls []csiCall

			// Called for each NodeStageVolume calls, with counter incremented atomically before
			// the invocation (i.e. first value will be 1) and index of deleted pod (the first pod
			// has index 1)
			nodeUnstageHook func(counter, pod int64) error
		}{
			{
				// This is already tested elsewhere, adding simple good case here to test the test framework.
				name: "should call NodeStage after NodeUnstage success",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
			},
			{
				name: "two pods: should call NodeStage after previous NodeUnstage final error",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.InvalidArgument},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeUnstageHook: func(counter, pod int64) error {
					if pod == 1 {
						return status.Error(codes.InvalidArgument, "fake final error")
					}
					return nil
				},
			},
			{
				name: "two pods: should call NodeStage after previous NodeUnstage transient error",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.DeadlineExceeded},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeUnstageHook: func(counter, pod int64) error {
					if pod == 1 {
						return status.Error(codes.DeadlineExceeded, "fake transient error")
					}
					return nil
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				// Index of the last deleted pod. NodeUnstage calls are then related to this pod.
				var deletedPodNumber int64 = 1
				var hooks *drivers.Hooks
				if test.nodeUnstageHook != nil {
					hooks = createPreHook("NodeUnstageVolume", func(counter int64) error {
						pod := atomic.LoadInt64(&deletedPodNumber)
						return test.nodeUnstageHook(counter, pod)
					})
				}
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					hooks:          hooks,
				})
				defer cleanup()

				_, claim, pod := createPod(pvcReference)
				if pod == nil {
					return
				}
				// Wait for PVC to get bound to make sure the CSI driver is fully started.
				err := e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, f.ClientSet, f.Namespace.Name, claim.Name, time.Second, framework.ClaimProvisionTimeout)
				framework.ExpectNoError(err, "while waiting for PVC to get provisioned")
				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "while waiting for the first pod to start")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting the first pod")

				// Create the second pod
				pod, err = createPodWithPVC(claim)
				framework.ExpectNoError(err, "while creating the second pod")
				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "while waiting for the second pod to start")
				// The second pod is running and kubelet can't call NodeUnstage of the first one.
				// Therefore incrementing the pod counter is safe here.
				atomic.AddInt64(&deletedPodNumber, 1)
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting the second pod")

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.Poll(time.Second, csiUnstageWaitTimeout, func() (done bool, err error) {
					_, index, err := compareCSICalls(trackedCalls, test.expectedCalls, m.driver.GetCalls)
					if err != nil {
						return true, err
					}
					if index == 0 {
						// No CSI call received yet
						return false, nil
					}
					if len(test.expectedCalls) == index {
						// all calls received
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err, "while waiting for all CSI calls")
			})
		}
	})

	ginkgo.Context("storage capacity", func() {
		tests := []struct {
			name              string
			resourceExhausted bool
			lateBinding       bool
			topology          bool
		}{
			{
				name: "unlimited",
			},
			{
				name:              "exhausted, immediate binding",
				resourceExhausted: true,
			},
			{
				name:              "exhausted, late binding, no topology",
				resourceExhausted: true,
				lateBinding:       true,
			},
			{
				name:              "exhausted, late binding, with topology",
				resourceExhausted: true,
				lateBinding:       true,
				topology:          true,
			},
		}

		createVolume := "CreateVolume"
		deleteVolume := "DeleteVolume"
		// publishVolume := "NodePublishVolume"
		// unpublishVolume := "NodeUnpublishVolume"
		// stageVolume := "NodeStageVolume"
		// unstageVolume := "NodeUnstageVolume"

		// These calls are assumed to occur in this order for
		// each test run. NodeStageVolume and
		// NodePublishVolume should also be deterministic and
		// only get called once, but sometimes kubelet calls
		// both multiple times, which breaks this test
		// (https://github.com/kubernetes/kubernetes/issues/90250).
		// Therefore they are temporarily commented out until
		// that issue is resolved.
		//
		// NodeUnpublishVolume and NodeUnstageVolume are racing
		// with DeleteVolume, so we cannot assume a deterministic
		// order and have to ignore them
		// (https://github.com/kubernetes/kubernetes/issues/94108).
		deterministicCalls := []string{
			createVolume,
			// stageVolume,
			// publishVolume,
			// unpublishVolume,
			// unstageVolume,
			deleteVolume,
		}

		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				var err error
				params := testParameters{
					lateBinding:    test.lateBinding,
					enableTopology: test.topology,

					// Not strictly necessary, but runs a bit faster this way
					// and for a while there also was a problem with a two minuted delay
					// due to a bug (https://github.com/kubernetes-csi/csi-test/pull/250).
					disableAttach:  true,
					registerDriver: true,
				}

				if test.resourceExhausted {
					params.hooks = createPreHook("CreateVolume", func(counter int64) error {
						if counter%2 != 0 {
							return status.Error(codes.ResourceExhausted, "fake error")
						}
						return nil
					})
				}

				init(params)
				defer cleanup()

				ctx, cancel := context.WithTimeout(context.Background(), csiPodRunningTimeout)
				defer cancel()

				// In contrast to the raw watch, RetryWatcher is expected to deliver all events even
				// when the underlying raw watch gets closed prematurely
				// (https://github.com/kubernetes/kubernetes/pull/93777#discussion_r467932080).
				// This is important because below the test is going to make assertions about the
				// PVC state changes.
				initResource, err := f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).List(ctx, metav1.ListOptions{})
				framework.ExpectNoError(err, "Failed to fetch initial PVC resource")
				listWatcher := &cachetools.ListWatch{
					WatchFunc: func(listOptions metav1.ListOptions) (watch.Interface, error) {
						return f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Watch(ctx, listOptions)
					},
				}
				pvcWatch, err := watchtools.NewRetryWatcher(initResource.GetResourceVersion(), listWatcher)
				framework.ExpectNoError(err, "create PVC watch")
				defer pvcWatch.Stop()

				sc, claim, pod := createPod(pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod")
				bindingMode := storagev1.VolumeBindingImmediate
				if test.lateBinding {
					bindingMode = storagev1.VolumeBindingWaitForFirstConsumer
				}
				framework.ExpectEqual(*sc.VolumeBindingMode, bindingMode, "volume binding mode")

				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "failed to delete pod")
				err = m.cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete claim")

				normal := []csiCall{}
				for _, method := range deterministicCalls {
					normal = append(normal, csiCall{expectedMethod: method})
				}
				expected := normal
				// When simulating limited capacity,
				// we expect exactly two CreateVolume
				// calls because the first one should
				// have failed.
				if test.resourceExhausted {
					expected = []csiCall{
						{expectedMethod: createVolume, expectedError: codes.ResourceExhausted},
					}
					expected = append(expected, normal...)
				}

				var calls []drivers.MockCSICall
				err = wait.PollImmediateUntil(time.Second, func() (done bool, err error) {
					c, index, err := compareCSICalls(deterministicCalls, expected, m.driver.GetCalls)
					if err != nil {
						return true, fmt.Errorf("error waiting for expected CSI calls: %s", err)
					}
					calls = c
					if index == 0 {
						// No CSI call received yet
						return false, nil
					}
					if len(expected) == index {
						// all calls received
						return true, nil
					}
					return false, nil
				}, ctx.Done())
				framework.ExpectNoError(err, "while waiting for all CSI calls")

				// The capacity error is dealt with in two different ways.
				//
				// For delayed binding, the external-provisioner should unset the node annotation
				// to give the scheduler the opportunity to reschedule the pod onto a different
				// node.
				//
				// For immediate binding, the external-scheduler must keep retrying.
				//
				// Unfortunately, the call log is the same in both cases. We have to collect
				// additional evidence that rescheduling really happened. What we have observed
				// above is how the PVC changed over time. Now we can analyze that.
				ginkgo.By("Checking PVC events")
				nodeAnnotationSet := false
				nodeAnnotationReset := false
				watchFailed := false
			loop:
				for {
					select {
					case event, ok := <-pvcWatch.ResultChan():
						if !ok {
							watchFailed = true
							break loop
						}

						framework.Logf("PVC event %s: %#v", event.Type, event.Object)
						switch event.Type {
						case watch.Modified:
							pvc, ok := event.Object.(*v1.PersistentVolumeClaim)
							if !ok {
								framework.Failf("PVC watch sent %#v instead of a PVC", event.Object)
							}
							_, set := pvc.Annotations["volume.kubernetes.io/selected-node"]
							if set {
								nodeAnnotationSet = true
							} else if nodeAnnotationSet {
								nodeAnnotationReset = true
							}
						case watch.Deleted:
							break loop
						case watch.Error:
							watchFailed = true
							break
						}
					case <-ctx.Done():
						framework.Failf("Timeout while waiting to observe PVC list")
					}
				}

				// More tests when capacity is limited.
				if test.resourceExhausted {
					for _, call := range calls {
						if call.Method == createVolume {
							gomega.Expect(call.Error).To(gomega.ContainSubstring("code = ResourceExhausted"), "first CreateVolume error in\n%s", calls)
							break
						}
					}

					switch {
					case watchFailed:
						// If the watch failed or stopped prematurely (which can happen at any time), then we cannot
						// verify whether the annotation was set as expected. This is still considered a successful
						// test.
						framework.Logf("PVC watch delivered incomplete data, cannot check annotation")
					case test.lateBinding:
						gomega.Expect(nodeAnnotationSet).To(gomega.BeTrue(), "selected-node should have been set")
						// Whether it gets reset depends on whether we have topology enabled. Without
						// it, rescheduling is unnecessary.
						if test.topology {
							gomega.Expect(nodeAnnotationReset).To(gomega.BeTrue(), "selected-node should have been set")
						} else {
							gomega.Expect(nodeAnnotationReset).To(gomega.BeFalse(), "selected-node should not have been reset")
						}
					default:
						gomega.Expect(nodeAnnotationSet).To(gomega.BeFalse(), "selected-node should not have been set")
						gomega.Expect(nodeAnnotationReset).To(gomega.BeFalse(), "selected-node should not have been reset")
					}
				}
			})
		}
	})

	// These tests *only* work on a cluster which has the CSIStorageCapacity feature enabled.
	ginkgo.Context("CSIStorageCapacity", func() {
		var (
			err error
			yes = true
			no  = false
		)
		// Tests that expect a failure are slow because we have to wait for a while
		// to be sure that the volume isn't getting created.
		tests := []struct {
			name            string
			storageCapacity *bool
			capacities      []string
			expectFailure   bool
		}{
			{
				name: "CSIStorageCapacity unused",
			},
			{
				name:            "CSIStorageCapacity disabled",
				storageCapacity: &no,
			},
			{
				name:            "CSIStorageCapacity used, no capacity",
				storageCapacity: &yes,
				expectFailure:   true,
			},
			{
				name:            "CSIStorageCapacity used, insufficient capacity",
				storageCapacity: &yes,
				expectFailure:   true,
				capacities:      []string{"1Mi"},
			},
			{
				name:            "CSIStorageCapacity used, have capacity",
				storageCapacity: &yes,
				capacities:      []string{"100Gi"},
			},
			// We could add more test cases here for
			// various situations, but covering those via
			// the scheduler binder unit tests is faster.
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func() {
				scName := "mock-csi-storage-capacity-" + f.UniqueName
				init(testParameters{
					registerDriver:  true,
					scName:          scName,
					storageCapacity: test.storageCapacity,
					lateBinding:     true,
				})
				defer cleanup()

				// The storage class uses a random name, therefore we have to create it first
				// before adding CSIStorageCapacity objects for it.
				for _, capacityStr := range test.capacities {
					capacityQuantity := resource.MustParse(capacityStr)
					capacity := &storagev1.CSIStorageCapacity{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "fake-capacity-",
						},
						// Empty topology, usable by any node.
						StorageClassName: scName,
						NodeTopology:     &metav1.LabelSelector{},
						Capacity:         &capacityQuantity,
					}
					createdCapacity, err := f.ClientSet.StorageV1().CSIStorageCapacities(f.Namespace.Name).Create(context.Background(), capacity, metav1.CreateOptions{})
					framework.ExpectNoError(err, "create CSIStorageCapacity %+v", *capacity)
					m.testCleanups = append(m.testCleanups, func() {
						f.ClientSet.StorageV1().CSIStorageCapacities(f.Namespace.Name).Delete(context.Background(), createdCapacity.Name, metav1.DeleteOptions{})
					})
				}

				// kube-scheduler may need some time before it gets the CSIDriver and CSIStorageCapacity objects.
				// Without them, scheduling doesn't run as expected by the test.
				syncDelay := 5 * time.Second
				time.Sleep(syncDelay)

				sc, _, pod := createPod(pvcReference) // late binding as specified above
				framework.ExpectEqual(sc.Name, scName, "pre-selected storage class name not used")

				waitCtx, cancel := context.WithTimeout(context.Background(), f.Timeouts.PodStart)
				defer cancel()
				condition := anyOf(
					podRunning(waitCtx, f.ClientSet, pod.Name, pod.Namespace),
					// We only just created the CSIStorageCapacity objects, therefore
					// we have to ignore all older events, plus the syncDelay as our
					// safety margin.
					podHasStorage(waitCtx, f.ClientSet, pod.Name, pod.Namespace, time.Now().Add(syncDelay)),
				)
				err = wait.PollImmediateUntil(poll, condition, waitCtx.Done())
				if test.expectFailure {
					switch {
					case errors.Is(err, context.DeadlineExceeded),
						errors.Is(err, wait.ErrWaitTimeout),
						errors.Is(err, errNotEnoughSpace):
						// Okay, we expected that.
					case err == nil:
						framework.Fail("pod unexpectedly started to run")
					default:
						framework.Failf("unexpected error while waiting for pod: %v", err)
					}
				} else {
					framework.ExpectNoError(err, "failed to start pod")
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting")
			})
		}
	})

	ginkgo.Context("CSI Volume Snapshots [Feature:VolumeSnapshotDataSource]", func() {
		tests := []struct {
			name               string
			createSnapshotHook func(counter int64) error
		}{
			{
				name: "volumesnapshotcontent and pvc in Bound state with deletion timestamp set should not get deleted while snapshot finalizer exists",
				createSnapshotHook: func(counter int64) error {
					if counter < 8 {
						return status.Error(codes.DeadlineExceeded, "fake error")
					}
					return nil
				},
			},
		}
		for _, test := range tests {
			test := test
			ginkgo.It(test.name, func() {
				var hooks *drivers.Hooks
				if test.createSnapshotHook != nil {
					hooks = createPreHook("CreateSnapshot", test.createSnapshotHook)
				}
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					enableSnapshot: true,
					hooks:          hooks,
				})
				sDriver, ok := m.driver.(storageframework.SnapshottableTestDriver)
				if !ok {
					e2eskipper.Skipf("mock driver %s does not support snapshots -- skipping", m.driver.GetDriverInfo().Name)

				}
				ctx, cancel := context.WithTimeout(context.Background(), csiPodRunningTimeout)
				defer cancel()
				defer cleanup()

				sc := m.driver.GetDynamicProvisionStorageClass(m.config, "")
				ginkgo.By("Creating storage class")
				class, err := m.cs.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create class: %v", err)
				m.sc[class.Name] = class
				claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					// Use static name so that the volumesnapshot can be created before the pvc.
					Name:             "snapshot-test-pvc",
					StorageClassName: &(class.Name),
				}, f.Namespace.Name)

				ginkgo.By("Creating snapshot")
				// TODO: Test VolumeSnapshots with Retain policy
				parameters := map[string]string{}
				snapshotClass, snapshot := storageframework.CreateSnapshot(sDriver, m.config, storageframework.DynamicSnapshotDelete, claim.Name, claim.Namespace, f.Timeouts, parameters)
				framework.ExpectNoError(err, "failed to create snapshot")
				m.vsc[snapshotClass.GetName()] = snapshotClass
				volumeSnapshotName := snapshot.GetName()

				ginkgo.By(fmt.Sprintf("Creating PVC %s/%s", claim.Namespace, claim.Name))
				claim, err = m.cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(), claim, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				ginkgo.By(fmt.Sprintf("Wait for finalizer to be added to claim %s/%s", claim.Namespace, claim.Name))
				err = e2epv.WaitForPVCFinalizer(ctx, m.cs, claim.Name, claim.Namespace, pvcAsSourceProtectionFinalizer, 1*time.Millisecond, 1*time.Minute)
				framework.ExpectNoError(err)

				ginkgo.By("Wait for PVC to be Bound")
				_, err = e2epv.WaitForPVClaimBoundPhase(m.cs, []*v1.PersistentVolumeClaim{claim}, 1*time.Minute)
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				ginkgo.By(fmt.Sprintf("Delete PVC %s", claim.Name))
				err = e2epv.DeletePersistentVolumeClaim(m.cs, claim.Name, claim.Namespace)
				framework.ExpectNoError(err, "failed to delete pvc")

				ginkgo.By("Get PVC from API server and verify deletion timestamp is set")
				claim, err = m.cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Get(context.TODO(), claim.Name, metav1.GetOptions{})
				if err != nil {
					if !apierrors.IsNotFound(err) {
						framework.ExpectNoError(err, "Failed to get claim: %v", err)
					}
					framework.Logf("PVC not found. Continuing to test VolumeSnapshotContent finalizer")
				} else if claim.DeletionTimestamp == nil {
					framework.Failf("Expected deletion timestamp to be set on PVC: %v", claim)
				}

				ginkgo.By(fmt.Sprintf("Get VolumeSnapshotContent bound to VolumeSnapshot %s", snapshot.GetName()))
				snapshotContent := utils.GetSnapshotContentFromSnapshot(m.config.Framework.DynamicClient, snapshot)
				volumeSnapshotContentName := snapshotContent.GetName()

				ginkgo.By(fmt.Sprintf("Verify VolumeSnapshotContent %s contains finalizer %s", snapshot.GetName(), volumeSnapshotContentFinalizer))
				err = utils.WaitForGVRFinalizer(ctx, m.config.Framework.DynamicClient, utils.SnapshotContentGVR, volumeSnapshotContentName, "", volumeSnapshotContentFinalizer, 1*time.Millisecond, 1*time.Minute)
				framework.ExpectNoError(err)

				ginkgo.By(fmt.Sprintf("Delete VolumeSnapshotContent %s", snapshotContent.GetName()))
				err = m.config.Framework.DynamicClient.Resource(utils.SnapshotContentGVR).Delete(ctx, snapshotContent.GetName(), metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Failed to delete snapshotcontent: %v", err)

				ginkgo.By("Get VolumeSnapshotContent from API server and verify deletion timestamp is set")
				snapshotContent, err = m.config.Framework.DynamicClient.Resource(utils.SnapshotContentGVR).Get(context.TODO(), snapshotContent.GetName(), metav1.GetOptions{})
				framework.ExpectNoError(err)

				if snapshotContent.GetDeletionTimestamp() == nil {
					framework.Failf("Expected deletion timestamp to be set on snapshotcontent")
				}

				// If the claim is non existent, the Get() call on the API server returns
				// an non-nil claim object with all fields unset.
				// Refer https://github.com/kubernetes/kubernetes/pull/99167#issuecomment-781670012
				if claim != nil && claim.Spec.VolumeName != "" {
					ginkgo.By(fmt.Sprintf("Wait for PV %s to be deleted", claim.Spec.VolumeName))
					err = e2epv.WaitForPersistentVolumeDeleted(m.cs, claim.Spec.VolumeName, framework.Poll, 3*time.Minute)
					framework.ExpectNoError(err, fmt.Sprintf("failed to delete PV %s", claim.Spec.VolumeName))
				}

				ginkgo.By(fmt.Sprintf("Verify VolumeSnapshot %s contains finalizer %s", snapshot.GetName(), volumeSnapshotBoundFinalizer))
				err = utils.WaitForGVRFinalizer(ctx, m.config.Framework.DynamicClient, utils.SnapshotGVR, volumeSnapshotName, f.Namespace.Name, volumeSnapshotBoundFinalizer, 1*time.Millisecond, 1*time.Minute)
				framework.ExpectNoError(err)

				ginkgo.By("Delete VolumeSnapshot")
				err = utils.DeleteAndWaitSnapshot(m.config.Framework.DynamicClient, f.Namespace.Name, volumeSnapshotName, framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err, fmt.Sprintf("failed to delete VolumeSnapshot %s", volumeSnapshotName))

				ginkgo.By(fmt.Sprintf("Wait for VolumeSnapshotContent %s to be deleted", volumeSnapshotContentName))
				err = utils.WaitForGVRDeletion(m.config.Framework.DynamicClient, utils.SnapshotContentGVR, volumeSnapshotContentName, framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err, fmt.Sprintf("failed to delete VolumeSnapshotContent %s", volumeSnapshotContentName))
			})
		}
	})

	ginkgo.Context("CSIServiceAccountToken", func() {
		var (
			err error
		)
		tests := []struct {
			desc                  string
			deployCSIDriverObject bool
			tokenRequests         []storagev1.TokenRequest
		}{
			{
				desc:                  "token should not be plumbed down when csiServiceAccountTokenEnabled=false",
				deployCSIDriverObject: true,
				tokenRequests:         nil,
			},
			{
				desc:                  "token should not be plumbed down when CSIDriver is not deployed",
				deployCSIDriverObject: false,
				tokenRequests:         []storagev1.TokenRequest{{}},
			},
			{
				desc:                  "token should be plumbed down when csiServiceAccountTokenEnabled=true",
				deployCSIDriverObject: true,
				tokenRequests:         []storagev1.TokenRequest{{ExpirationSeconds: utilptr.Int64Ptr(60 * 10)}},
			},
		}
		for _, test := range tests {
			test := test
			csiServiceAccountTokenEnabled := test.tokenRequests != nil
			ginkgo.It(test.desc, func() {
				init(testParameters{
					registerDriver:    test.deployCSIDriverObject,
					tokenRequests:     test.tokenRequests,
					requiresRepublish: &csiServiceAccountTokenEnabled,
				})

				defer cleanup()

				_, _, pod := createPod(pvcReference)
				if pod == nil {
					return
				}
				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				// sleep to make sure RequiresRepublish triggers more than 1 NodePublishVolume
				numNodePublishVolume := 1
				if test.deployCSIDriverObject && csiServiceAccountTokenEnabled {
					time.Sleep(5 * time.Second)
					numNodePublishVolume = 2
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(m.cs, pod)
				framework.ExpectNoError(err, "while deleting")

				ginkgo.By("Checking CSI driver logs")
				err = checkPodLogs(m.driver.GetCalls, pod, false, false, false, test.deployCSIDriverObject && csiServiceAccountTokenEnabled, numNodePublishVolume)
				framework.ExpectNoError(err)
			})
		}
	})
	// These tests *only* work on a cluster which has the CSIVolumeFSGroupPolicy feature enabled.
	ginkgo.Context("CSI FSGroupPolicy [LinuxOnly]", func() {
		tests := []struct {
			name          string
			fsGroupPolicy storagev1.FSGroupPolicy
			modified      bool
		}{
			{
				name:          "should modify fsGroup if fsGroupPolicy=default",
				fsGroupPolicy: storagev1.ReadWriteOnceWithFSTypeFSGroupPolicy,
				modified:      true,
			},
			{
				name:          "should modify fsGroup if fsGroupPolicy=File",
				fsGroupPolicy: storagev1.FileFSGroupPolicy,
				modified:      true,
			},
			{
				name:          "should not modify fsGroup if fsGroupPolicy=None",
				fsGroupPolicy: storagev1.NoneFSGroupPolicy,
				modified:      false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("FSGroupPolicy is only applied on linux nodes -- skipping")
				}
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					fsGroupPolicy:  &test.fsGroupPolicy,
				})
				defer cleanup()

				// kube-scheduler may need some time before it gets the CSIDriver object.
				// Without them, scheduling doesn't run as expected by the test.
				syncDelay := 5 * time.Second
				time.Sleep(syncDelay)

				fsGroupVal := int64(rand.Int63n(20000) + 1024)
				fsGroup := &fsGroupVal

				_, _, pod := createPodWithFSGroup(fsGroup) /* persistent volume */

				mountPath := pod.Spec.Containers[0].VolumeMounts[0].MountPath
				dirName := mountPath + "/" + f.UniqueName
				fileName := dirName + "/" + f.UniqueName

				err := e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")

				// Create the subdirectory to ensure that fsGroup propagates
				createDirectory := fmt.Sprintf("mkdir %s", dirName)
				_, _, err = e2evolume.PodExec(f, pod, createDirectory)
				framework.ExpectNoError(err, "failed: creating the directory: %s", err)

				// Inject the contents onto the mount
				createFile := fmt.Sprintf("echo '%s' > '%s'; sync", "filecontents", fileName)
				_, _, err = e2evolume.PodExec(f, pod, createFile)
				framework.ExpectNoError(err, "failed: writing the contents: %s", err)

				// Delete the created file. This step is mandatory, as the mock driver
				// won't clean up the contents automatically.
				defer func() {
					delete := fmt.Sprintf("rm -fr %s", dirName)
					_, _, err = e2evolume.PodExec(f, pod, delete)
					framework.ExpectNoError(err, "failed: deleting the directory: %s", err)
				}()

				// Ensure that the fsGroup matches what we expect
				if test.modified {
					utils.VerifyFSGroupInPod(f, fileName, strconv.FormatInt(*fsGroup, 10), pod)
				} else {
					utils.VerifyFSGroupInPod(f, fileName, "root", pod)
				}

				// The created resources will be removed by the cleanup() function,
				// so need to delete anything here.
			})
		}
	})

	ginkgo.Context("Delegate FSGroup to CSI driver [LinuxOnly]", func() {
		tests := []struct {
			name                   string
			enableVolumeMountGroup bool
		}{
			{
				name:                   "should pass FSGroup to CSI driver if it is set in pod and driver supports VOLUME_MOUNT_GROUP",
				enableVolumeMountGroup: true,
			},
			{
				name:                   "should not pass FSGroup to CSI driver if it is set in pod and driver supports VOLUME_MOUNT_GROUP",
				enableVolumeMountGroup: false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func() {
				var nodeStageFsGroup, nodePublishFsGroup string
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("FSGroupPolicy is only applied on linux nodes -- skipping")
				}
				init(testParameters{
					disableAttach:          true,
					registerDriver:         true,
					enableVolumeMountGroup: t.enableVolumeMountGroup,
					hooks:                  createFSGroupRequestPreHook(&nodeStageFsGroup, &nodePublishFsGroup),
				})
				defer cleanup()

				fsGroupVal := int64(rand.Int63n(20000) + 1024)
				fsGroup := &fsGroupVal
				fsGroupStr := strconv.FormatInt(fsGroupVal, 10 /* base */)

				_, _, pod := createPodWithFSGroup(fsGroup) /* persistent volume */

				err := e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")

				if t.enableVolumeMountGroup {
					framework.ExpectEqual(nodeStageFsGroup, fsGroupStr, "Expect NodeStageVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to equal %q; got: %q", fsGroupStr, nodeStageFsGroup)
					framework.ExpectEqual(nodePublishFsGroup, fsGroupStr, "Expect NodePublishVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to equal %q; got: %q", fsGroupStr, nodePublishFsGroup)
				} else {
					framework.ExpectEmpty(nodeStageFsGroup, "Expect NodeStageVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to be empty; got: %q", nodeStageFsGroup)
					framework.ExpectEmpty(nodePublishFsGroup, "Expect NodePublishVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to empty; got: %q", nodePublishFsGroup)
				}
			})
		}
	})

	ginkgo.Context("CSI Volume Snapshots secrets [Feature:VolumeSnapshotDataSource]", func() {

		var (
			// CSISnapshotterSecretName is the name of the secret to be created
			CSISnapshotterSecretName string = "snapshot-secret"

			// CSISnapshotterSecretNameAnnotation is the annotation key for the CSI snapshotter secret name in VolumeSnapshotClass.parameters
			CSISnapshotterSecretNameAnnotation string = "csi.storage.k8s.io/snapshotter-secret-name"

			// CSISnapshotterSecretNamespaceAnnotation is the annotation key for the CSI snapshotter secret namespace in VolumeSnapshotClass.parameters
			CSISnapshotterSecretNamespaceAnnotation string = "csi.storage.k8s.io/snapshotter-secret-namespace"

			// anotations holds the annotations object
			annotations interface{}
		)

		tests := []struct {
			name               string
			createSnapshotHook func(counter int64) error
		}{
			{
				// volume snapshot should be created using secrets successfully even if there is a failure in the first few attempts,
				name: "volume snapshot create/delete with secrets",
				// Fail the first 8 calls to create snapshot and succeed the  9th call.
				createSnapshotHook: func(counter int64) error {
					if counter < 8 {
						return status.Error(codes.DeadlineExceeded, "fake error")
					}
					return nil
				},
			},
		}
		for _, test := range tests {
			ginkgo.It(test.name, func() {
				hooks := createPreHook("CreateSnapshot", test.createSnapshotHook)
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					enableSnapshot: true,
					hooks:          hooks,
				})

				sDriver, ok := m.driver.(storageframework.SnapshottableTestDriver)
				if !ok {
					e2eskipper.Skipf("mock driver does not support snapshots -- skipping")
				}
				defer cleanup()

				var sc *storagev1.StorageClass
				if dDriver, ok := m.driver.(storageframework.DynamicPVTestDriver); ok {
					sc = dDriver.GetDynamicProvisionStorageClass(m.config, "")
				}
				ginkgo.By("Creating storage class")
				class, err := m.cs.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create storage class: %v", err)
				m.sc[class.Name] = class
				pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					Name:             "snapshot-test-pvc",
					StorageClassName: &(class.Name),
				}, f.Namespace.Name)

				ginkgo.By(fmt.Sprintf("Creating PVC %s/%s", pvc.Namespace, pvc.Name))
				pvc, err = m.cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				ginkgo.By("Wait for PVC to be Bound")
				_, err = e2epv.WaitForPVClaimBoundPhase(m.cs, []*v1.PersistentVolumeClaim{pvc}, 1*time.Minute)
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				m.pvcs = append(m.pvcs, pvc)

				ginkgo.By("Creating Secret")
				secret := &v1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: f.Namespace.Name,
						Name:      CSISnapshotterSecretName,
					},
					Data: map[string][]byte{
						"secret-data": []byte("secret-value-1"),
					},
				}

				if secret, err := m.cs.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
					framework.Failf("unable to create test secret %s: %v", secret.Name, err)
				}

				ginkgo.By("Creating snapshot with secrets")
				parameters := map[string]string{
					CSISnapshotterSecretNameAnnotation:      CSISnapshotterSecretName,
					CSISnapshotterSecretNamespaceAnnotation: f.Namespace.Name,
				}

				_, snapshot := storageframework.CreateSnapshot(sDriver, m.config, storageframework.DynamicSnapshotDelete, pvc.Name, pvc.Namespace, f.Timeouts, parameters)
				framework.ExpectNoError(err, "failed to create snapshot")
				snapshotcontent := utils.GetSnapshotContentFromSnapshot(m.config.Framework.DynamicClient, snapshot)
				if annotations, ok = snapshotcontent.Object["metadata"].(map[string]interface{})["annotations"]; !ok {
					framework.Failf("Unable to get volume snapshot content annotations")
				}

				// checks if delete snapshot secrets annotation is applied to the VolumeSnapshotContent.
				checkDeleteSnapshotSecrets(m.cs, annotations)

				// delete the snapshot and check if the snapshot is deleted.
				deleteSnapshot(m.cs, m.config, snapshot)
			})
		}
	})

	ginkgo.Context("CSI Snapshot Controller metrics [Feature:VolumeSnapshotDataSource]", func() {
		tests := []struct {
			name    string
			pattern storageframework.TestPattern
		}{
			{
				name:    "snapshot controller should emit dynamic CreateSnapshot, CreateSnapshotAndReady, and DeleteSnapshot metrics",
				pattern: storageframework.DynamicSnapshotDelete,
			},
			{
				name:    "snapshot controller should emit pre-provisioned CreateSnapshot, CreateSnapshotAndReady, and DeleteSnapshot metrics",
				pattern: storageframework.PreprovisionedSnapshotDelete,
			},
		}
		for _, test := range tests {
			ginkgo.It(test.name, func() {
				init(testParameters{
					disableAttach:  true,
					registerDriver: true,
					enableSnapshot: true,
				})

				sDriver, ok := m.driver.(storageframework.SnapshottableTestDriver)
				if !ok {
					e2eskipper.Skipf("mock driver does not support snapshots -- skipping")
				}
				defer cleanup()

				metricsGrabber, err := e2emetrics.NewMetricsGrabber(m.config.Framework.ClientSet, nil, f.ClientConfig(), false, false, false, false, false, true)
				if err != nil {
					framework.Failf("Error creating metrics grabber : %v", err)
				}

				// Grab initial metrics - if this fails, snapshot controller metrics are not setup. Skip in this case.
				_, err = metricsGrabber.GrabFromSnapshotController(framework.TestContext.SnapshotControllerPodName, framework.TestContext.SnapshotControllerHTTPPort)
				if err != nil {
					e2eskipper.Skipf("Snapshot controller metrics not found -- skipping")
				}

				ginkgo.By("getting all initial metric values")
				metricsTestConfig := newSnapshotMetricsTestConfig("snapshot_controller_operation_total_seconds_count",
					"count",
					m.config.GetUniqueDriverName(),
					"CreateSnapshot",
					"success",
					"",
					test.pattern)
				createSnapshotMetrics := newSnapshotControllerMetrics(metricsTestConfig, metricsGrabber)
				originalCreateSnapshotCount, _ := createSnapshotMetrics.getSnapshotControllerMetricValue()
				metricsTestConfig.operationName = "CreateSnapshotAndReady"
				createSnapshotAndReadyMetrics := newSnapshotControllerMetrics(metricsTestConfig, metricsGrabber)
				originalCreateSnapshotAndReadyCount, _ := createSnapshotAndReadyMetrics.getSnapshotControllerMetricValue()

				metricsTestConfig.operationName = "DeleteSnapshot"
				deleteSnapshotMetrics := newSnapshotControllerMetrics(metricsTestConfig, metricsGrabber)
				originalDeleteSnapshotCount, _ := deleteSnapshotMetrics.getSnapshotControllerMetricValue()

				ginkgo.By("Creating storage class")
				var sc *storagev1.StorageClass
				if dDriver, ok := m.driver.(storageframework.DynamicPVTestDriver); ok {
					sc = dDriver.GetDynamicProvisionStorageClass(m.config, "")
				}
				class, err := m.cs.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create storage class: %v", err)
				m.sc[class.Name] = class
				pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					Name:             "snapshot-test-pvc",
					StorageClassName: &(class.Name),
				}, f.Namespace.Name)

				ginkgo.By(fmt.Sprintf("Creating PVC %s/%s", pvc.Namespace, pvc.Name))
				pvc, err = m.cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				ginkgo.By("Wait for PVC to be Bound")
				_, err = e2epv.WaitForPVClaimBoundPhase(m.cs, []*v1.PersistentVolumeClaim{pvc}, 1*time.Minute)
				framework.ExpectNoError(err, "Failed to create claim: %v", err)

				ginkgo.By("Creating snapshot")
				parameters := map[string]string{}
				sr := storageframework.CreateSnapshotResource(sDriver, m.config, test.pattern, pvc.Name, pvc.Namespace, f.Timeouts, parameters)
				framework.ExpectNoError(err, "failed to create snapshot")

				ginkgo.By("Checking for CreateSnapshot metrics")
				createSnapshotMetrics.waitForSnapshotControllerMetric(originalCreateSnapshotCount+1.0, f.Timeouts.SnapshotControllerMetrics)

				ginkgo.By("Checking for CreateSnapshotAndReady metrics")
				err = utils.WaitForSnapshotReady(m.config.Framework.DynamicClient, pvc.Namespace, sr.Vs.GetName(), framework.Poll, f.Timeouts.SnapshotCreate)
				framework.ExpectNoError(err, "failed to wait for snapshot ready")
				createSnapshotAndReadyMetrics.waitForSnapshotControllerMetric(originalCreateSnapshotAndReadyCount+1.0, f.Timeouts.SnapshotControllerMetrics)

				// delete the snapshot and check if the snapshot is deleted
				deleteSnapshot(m.cs, m.config, sr.Vs)

				ginkgo.By("check for delete metrics")
				metricsTestConfig.operationName = "DeleteSnapshot"
				deleteSnapshotMetrics.waitForSnapshotControllerMetric(originalDeleteSnapshotCount+1.0, f.Timeouts.SnapshotControllerMetrics)
			})
		}
	})
})

func deleteSnapshot(cs clientset.Interface, config *storageframework.PerTestConfig, snapshot *unstructured.Unstructured) {
	// delete the given snapshot
	dc := config.Framework.DynamicClient
	err := dc.Resource(utils.SnapshotGVR).Namespace(snapshot.GetNamespace()).Delete(context.TODO(), snapshot.GetName(), metav1.DeleteOptions{})
	framework.ExpectNoError(err)

	// check if the snapshot is deleted
	_, err = dc.Resource(utils.SnapshotGVR).Get(context.TODO(), snapshot.GetName(), metav1.GetOptions{})
	framework.ExpectError(err)
}

// A lot of this code was copied from e2e/framework. It would be nicer
// if it could be reused - see https://github.com/kubernetes/kubernetes/issues/92754
func podRunning(ctx context.Context, c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodRunning:
			return true, nil
		case v1.PodFailed, v1.PodSucceeded:
			return false, errPodCompleted
		}
		return false, nil
	}
}

const (
	poll                           = 2 * time.Second
	pvcAsSourceProtectionFinalizer = "snapshot.storage.kubernetes.io/pvc-as-source-protection"
	volumeSnapshotContentFinalizer = "snapshot.storage.kubernetes.io/volumesnapshotcontent-bound-protection"
	volumeSnapshotBoundFinalizer   = "snapshot.storage.kubernetes.io/volumesnapshot-bound-protection"
	errReasonNotEnoughSpace        = "node(s) did not have enough free storage"
)

var (
	errPodCompleted   = fmt.Errorf("pod ran to completion")
	errNotEnoughSpace = errors.New(errReasonNotEnoughSpace)
)

func podHasStorage(ctx context.Context, c clientset.Interface, podName, namespace string, when time.Time) wait.ConditionFunc {
	// Check for events of this pod. Copied from test/e2e/common/container_probe.go.
	expectedEvent := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      podName,
		"involvedObject.namespace": namespace,
		"reason":                   "FailedScheduling",
	}.AsSelector().String()
	options := metav1.ListOptions{
		FieldSelector: expectedEvent,
	}
	// copied from test/e2e/framework/events/events.go
	return func() (bool, error) {
		// We cannot be sure here whether it has enough storage, only when
		// it hasn't. In that case we abort waiting with a special error.
		events, err := c.CoreV1().Events(namespace).List(ctx, options)
		if err != nil {
			return false, fmt.Errorf("got error while getting events: %w", err)
		}
		for _, event := range events.Items {
			if /* event.CreationTimestamp.After(when) &&
			 */strings.Contains(event.Message, errReasonNotEnoughSpace) {
				return false, errNotEnoughSpace
			}
		}
		return false, nil
	}
}

func anyOf(conditions ...wait.ConditionFunc) wait.ConditionFunc {
	return func() (bool, error) {
		for _, condition := range conditions {
			done, err := condition()
			if err != nil {
				return false, err
			}
			if done {
				return true, nil
			}
		}
		return false, nil
	}
}

func waitForMaxVolumeCondition(pod *v1.Pod, cs clientset.Interface) error {
	waitErr := wait.PollImmediate(10*time.Second, csiPodUnschedulableTimeout, func() (bool, error) {
		pod, err := cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, c := range pod.Status.Conditions {
			// Conformance tests cannot rely on specific output of optional fields (e.g., Reason
			// and Message) because these fields are not suject to the deprecation policy.
			if c.Type == v1.PodScheduled && c.Status == v1.ConditionFalse && c.Reason != "" && c.Message != "" {
				return true, nil
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return fmt.Errorf("error waiting for pod %s/%s to have max volume condition: %v", pod.Namespace, pod.Name, waitErr)
	}
	return nil
}

func checkCSINodeForLimits(nodeName string, driverName string, cs clientset.Interface) (int32, error) {
	var attachLimit int32

	waitErr := wait.PollImmediate(10*time.Second, csiNodeLimitUpdateTimeout, func() (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		attachLimit = getVolumeLimitFromCSINode(csiNode, driverName)
		if attachLimit > 0 {
			return true, nil
		}
		return false, nil
	})
	if waitErr != nil {
		return 0, fmt.Errorf("error waiting for non-zero volume limit of driver %s on node %s: %v", driverName, nodeName, waitErr)
	}
	return attachLimit, nil
}

func createSC(cs clientset.Interface, t testsuites.StorageClassTest, scName, ns string) *storagev1.StorageClass {
	class := newStorageClass(t, ns, "")
	if scName != "" {
		class.Name = scName
	}
	var err error
	_, err = cs.StorageV1().StorageClasses().Get(context.TODO(), class.Name, metav1.GetOptions{})
	if err != nil {
		class, err = cs.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create class: %v", err)
	}

	return class
}

func createClaim(cs clientset.Interface, t testsuites.StorageClassTest, node e2epod.NodeSelection, scName, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim) {
	class := createSC(cs, t, scName, ns)
	claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
		ClaimSize:        t.ClaimSize,
		StorageClassName: &(class.Name),
		VolumeMode:       &t.VolumeMode,
	}, ns)
	claim, err := cs.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), claim, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create claim: %v", err)

	if !t.DelayBinding {
		pvcClaims := []*v1.PersistentVolumeClaim{claim}
		_, err = e2epv.WaitForPVClaimBoundPhase(cs, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound: %v", err)
	}
	return class, claim
}

func startPausePod(cs clientset.Interface, t testsuites.StorageClassTest, node e2epod.NodeSelection, scName, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
	class, claim := createClaim(cs, t, node, scName, ns)

	pod, err := startPausePodWithClaim(cs, claim, node, ns)
	framework.ExpectNoError(err, "Failed to create pause pod: %v", err)
	return class, claim, pod
}

func startBusyBoxPod(cs clientset.Interface, t testsuites.StorageClassTest, node e2epod.NodeSelection, scName, ns string, fsGroup *int64) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
	class, claim := createClaim(cs, t, node, scName, ns)
	pod, err := startBusyBoxPodWithClaim(cs, claim, node, ns, fsGroup)
	framework.ExpectNoError(err, "Failed to create busybox pod: %v", err)
	return class, claim, pod
}

func startPausePodInline(cs clientset.Interface, t testsuites.StorageClassTest, node e2epod.NodeSelection, ns string) *v1.Pod {
	pod, err := startPausePodWithInlineVolume(cs,
		&v1.CSIVolumeSource{
			Driver: t.Provisioner,
		},
		node, ns)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return pod
}

func startPausePodGenericEphemeral(cs clientset.Interface, t testsuites.StorageClassTest, node e2epod.NodeSelection, scName, ns string) (*storagev1.StorageClass, *v1.Pod) {
	class := createSC(cs, t, scName, ns)
	claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
		ClaimSize:        t.ClaimSize,
		StorageClassName: &(class.Name),
		VolumeMode:       &t.VolumeMode,
	}, ns)
	pod, err := startPausePodWithVolumeSource(cs, v1.VolumeSource{
		Ephemeral: &v1.EphemeralVolumeSource{
			VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{Spec: claim.Spec}},
	}, node, ns)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return class, pod
}

func startPausePodWithClaim(cs clientset.Interface, pvc *v1.PersistentVolumeClaim, node e2epod.NodeSelection, ns string) (*v1.Pod, error) {
	return startPausePodWithVolumeSource(cs,
		v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvc.Name,
				ReadOnly:  false,
			},
		},
		node, ns)
}

func startBusyBoxPodWithClaim(cs clientset.Interface, pvc *v1.PersistentVolumeClaim, node e2epod.NodeSelection, ns string, fsGroup *int64) (*v1.Pod, error) {
	return startBusyBoxPodWithVolumeSource(cs,
		v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvc.Name,
				ReadOnly:  false,
			},
		},
		node, ns, fsGroup)
}

func startPausePodWithInlineVolume(cs clientset.Interface, inlineVolume *v1.CSIVolumeSource, node e2epod.NodeSelection, ns string) (*v1.Pod, error) {
	return startPausePodWithVolumeSource(cs,
		v1.VolumeSource{
			CSI: inlineVolume,
		},
		node, ns)
}

func startPausePodWithVolumeSource(cs clientset.Interface, volumeSource v1.VolumeSource, node e2epod.NodeSelection, ns string) (*v1.Pod, error) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: imageutils.GetE2EImage(imageutils.Pause),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         "my-volume",
					VolumeSource: volumeSource,
				},
			},
		},
	}
	e2epod.SetNodeSelection(&pod.Spec, node)
	return cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
}

func startBusyBoxPodWithVolumeSource(cs clientset.Interface, volumeSource v1.VolumeSource, node e2epod.NodeSelection, ns string, fsGroup *int64) (*v1.Pod, error) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: framework.BusyBoxImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
					Command: e2epod.GenerateScriptCmd("while true ; do sleep 2; done"),
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				FSGroup: fsGroup,
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         "my-volume",
					VolumeSource: volumeSource,
				},
			},
		},
	}
	e2epod.SetNodeSelection(&pod.Spec, node)
	return cs.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
}

// checkPodLogs tests that NodePublish was called with expected volume_context and (for ephemeral inline volumes)
// has the matching NodeUnpublish
func checkPodLogs(getCalls func() ([]drivers.MockCSICall, error), pod *v1.Pod, expectPodInfo, ephemeralVolume, csiInlineVolumesEnabled, csiServiceAccountTokenEnabled bool, expectedNumNodePublish int) error {
	expectedAttributes := map[string]string{}
	if expectPodInfo {
		expectedAttributes["csi.storage.k8s.io/pod.name"] = pod.Name
		expectedAttributes["csi.storage.k8s.io/pod.namespace"] = pod.Namespace
		expectedAttributes["csi.storage.k8s.io/pod.uid"] = string(pod.UID)
		expectedAttributes["csi.storage.k8s.io/serviceAccount.name"] = "default"

	}
	if csiInlineVolumesEnabled {
		// This is only passed in 1.15 when the CSIInlineVolume feature gate is set.
		expectedAttributes["csi.storage.k8s.io/ephemeral"] = strconv.FormatBool(ephemeralVolume)
	}

	if csiServiceAccountTokenEnabled {
		expectedAttributes["csi.storage.k8s.io/serviceAccount.tokens"] = "<nonempty>"
	}

	// Find NodePublish in the GRPC calls.
	foundAttributes := sets.NewString()
	numNodePublishVolume := 0
	numNodeUnpublishVolume := 0
	calls, err := getCalls()
	if err != nil {
		return err
	}

	for _, call := range calls {
		switch call.Method {
		case "NodePublishVolume":
			numNodePublishVolume++
			if numNodePublishVolume == 1 {
				// Check that NodePublish had expected attributes for first volume
				for k, v := range expectedAttributes {
					vv, found := call.Request.VolumeContext[k]
					if found && (v == vv || (v == "<nonempty>" && len(vv) != 0)) {
						foundAttributes.Insert(k)
						framework.Logf("Found volume attribute %s: %s", k, vv)
					}
				}
			}
		case "NodeUnpublishVolume":
			framework.Logf("Found NodeUnpublishVolume: %+v", call)
			numNodeUnpublishVolume++
		}
	}
	if numNodePublishVolume < expectedNumNodePublish {
		return fmt.Errorf("NodePublish should be called at least %d", expectedNumNodePublish)
	}

	if numNodeUnpublishVolume == 0 {
		return fmt.Errorf("NodeUnpublish was never called")
	}
	if foundAttributes.Len() != len(expectedAttributes) {
		return fmt.Errorf("number of found volume attributes does not match, expected %d, got %d", len(expectedAttributes), foundAttributes.Len())
	}
	return nil
}

// compareCSICalls compares expectedCalls with logs of the mock driver.
// It returns index of the first expectedCall that was *not* received
// yet or error when calls do not match.
// All repeated calls to the CSI mock driver (e.g. due to exponential backoff)
// are squashed and checked against single expectedCallSequence item.
//
// Only permanent errors are returned. Other errors are logged and no
// calls are returned. The caller is expected to retry.
func compareCSICalls(trackedCalls []string, expectedCallSequence []csiCall, getCalls func() ([]drivers.MockCSICall, error)) ([]drivers.MockCSICall, int, error) {
	allCalls, err := getCalls()
	if err != nil {
		framework.Logf("intermittent (?) log retrieval error, proceeding without output: %v", err)
		return nil, 0, nil
	}

	// Remove all repeated and ignored calls
	tracked := sets.NewString(trackedCalls...)
	var calls []drivers.MockCSICall
	var last drivers.MockCSICall
	for _, c := range allCalls {
		if !tracked.Has(c.Method) {
			continue
		}
		if c.Method != last.Method || c.FullError.Code != last.FullError.Code {
			last = c
			calls = append(calls, c)
		}
		// This call is the same as the last one, ignore it.
	}

	for i, c := range calls {
		if i >= len(expectedCallSequence) {
			// Log all unexpected calls first, return error below outside the loop.
			framework.Logf("Unexpected CSI driver call: %s (%d)", c.Method, c.FullError)
			continue
		}

		// Compare current call with expected call
		expectedCall := expectedCallSequence[i]
		if c.Method != expectedCall.expectedMethod || c.FullError.Code != expectedCall.expectedError {
			return allCalls, i, fmt.Errorf("Unexpected CSI call %d: expected %s (%d), got %s (%d)", i, expectedCall.expectedMethod, expectedCall.expectedError, c.Method, c.FullError.Code)
		}
	}
	if len(calls) > len(expectedCallSequence) {
		return allCalls, len(expectedCallSequence), fmt.Errorf("Received %d unexpected CSI driver calls", len(calls)-len(expectedCallSequence))
	}
	// All calls were correct
	return allCalls, len(calls), nil

}

func waitForCSIDriver(cs clientset.Interface, driverName string) error {
	timeout := 4 * time.Minute

	framework.Logf("waiting up to %v for CSIDriver %q", timeout, driverName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(framework.Poll) {
		_, err := cs.StorageV1().CSIDrivers().Get(context.TODO(), driverName, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			return err
		}
	}
	return fmt.Errorf("gave up after waiting %v for CSIDriver %q", timeout, driverName)
}

func destroyCSIDriver(cs clientset.Interface, driverName string) {
	driverGet, err := cs.StorageV1().CSIDrivers().Get(context.TODO(), driverName, metav1.GetOptions{})
	if err == nil {
		framework.Logf("deleting %s.%s: %s", driverGet.TypeMeta.APIVersion, driverGet.TypeMeta.Kind, driverGet.ObjectMeta.Name)
		// Uncomment the following line to get full dump of CSIDriver object
		// framework.Logf("%s", framework.PrettyPrint(driverGet))
		cs.StorageV1().CSIDrivers().Delete(context.TODO(), driverName, metav1.DeleteOptions{})
	}
}

func getVolumeLimitFromCSINode(csiNode *storagev1.CSINode, driverName string) int32 {
	for _, d := range csiNode.Spec.Drivers {
		if d.Name != driverName {
			continue
		}
		if d.Allocatable != nil && d.Allocatable.Count != nil {
			return *d.Allocatable.Count
		}
	}
	return 0
}

// checkDeleteSnapshotSecrets checks if delete snapshot secrets annotation is applied to the VolumeSnapshotContent.
func checkDeleteSnapshotSecrets(cs clientset.Interface, annotations interface{}) error {
	ginkgo.By("checking if delete snapshot secrets annotation is applied to the VolumeSnapshotContent")

	var (
		annDeletionSecretName      string
		annDeletionSecretNamespace string
		ok                         bool
		err                        error

		// CSISnapshotterDeleteSecretNameAnnotation is the annotation key for the CSI snapshotter delete secret name in VolumeSnapshotClass.parameters
		CSISnapshotterDeleteSecretNameAnnotation string = "snapshot.storage.kubernetes.io/deletion-secret-name"

		// CSISnapshotterDeleteSecretNamespaceAnnotation is the annotation key for the CSI snapshotter delete secret namespace in VolumeSnapshotClass.parameters
		CSISnapshotterDeleteSecretNamespaceAnnotation string = "snapshot.storage.kubernetes.io/deletion-secret-namespace"
	)

	annotationsObj, ok := annotations.(map[string]interface{})
	if !ok {
		framework.Failf("failed to get annotations from annotations object")
	}

	if annDeletionSecretName, ok = annotationsObj[CSISnapshotterDeleteSecretNameAnnotation].(string); !ok {
		framework.Failf("unable to get secret annotation name")
	}
	if annDeletionSecretNamespace, ok = annotationsObj[CSISnapshotterDeleteSecretNamespaceAnnotation].(string); !ok {
		framework.Failf("unable to get secret annotation namespace")
	}

	// verify if secrets exists
	if _, err = cs.CoreV1().Secrets(annDeletionSecretNamespace).Get(context.TODO(), annDeletionSecretName, metav1.GetOptions{}); err != nil {
		framework.Failf("unable to get test secret %s: %v", annDeletionSecretName, err)
	}

	return err
}

// createPreHook counts invocations of a certain method (identified by a substring in the full gRPC method name).
func createPreHook(method string, callback func(counter int64) error) *drivers.Hooks {
	var counter int64

	return &drivers.Hooks{
		Pre: func() func(ctx context.Context, fullMethod string, request interface{}) (reply interface{}, err error) {
			return func(ctx context.Context, fullMethod string, request interface{}) (reply interface{}, err error) {
				if strings.Contains(fullMethod, method) {
					counter := atomic.AddInt64(&counter, 1)
					return nil, callback(counter)
				}
				return nil, nil
			}
		}(),
	}
}

// createFSGroupRequestPreHook creates a hook that records the fsGroup passed in
// through NodeStageVolume and NodePublishVolume calls.
func createFSGroupRequestPreHook(nodeStageFsGroup, nodePublishFsGroup *string) *drivers.Hooks {
	return &drivers.Hooks{
		Pre: func(ctx context.Context, fullMethod string, request interface{}) (reply interface{}, err error) {
			nodeStageRequest, ok := request.(csipbv1.NodeStageVolumeRequest)
			if ok {
				mountVolume := nodeStageRequest.GetVolumeCapability().GetMount()
				if mountVolume != nil {
					*nodeStageFsGroup = mountVolume.VolumeMountGroup
				}
			}
			nodePublishRequest, ok := request.(csipbv1.NodePublishVolumeRequest)
			if ok {
				mountVolume := nodePublishRequest.GetVolumeCapability().GetMount()
				if mountVolume != nil {
					*nodePublishFsGroup = mountVolume.VolumeMountGroup
				}
			}
			return nil, nil
		},
	}
}

type snapshotMetricsTestConfig struct {
	// expected values
	metricName      string
	metricType      string
	driverName      string
	operationName   string
	operationStatus string
	snapshotType    string
	le              string
}

type snapshotControllerMetrics struct {
	// configuration for metric
	cfg            snapshotMetricsTestConfig
	metricsGrabber *e2emetrics.Grabber

	// results
	countMetrics  map[string]float64
	sumMetrics    map[string]float64
	bucketMetrics map[string]float64
}

func newSnapshotMetricsTestConfig(metricName, metricType, driverName, operationName, operationStatus, le string, pattern storageframework.TestPattern) snapshotMetricsTestConfig {
	var snapshotType string
	switch pattern.SnapshotType {
	case storageframework.DynamicCreatedSnapshot:
		snapshotType = "dynamic"

	case storageframework.PreprovisionedCreatedSnapshot:
		snapshotType = "pre-provisioned"

	default:
		framework.Failf("invalid snapshotType: %v", pattern.SnapshotType)
	}

	return snapshotMetricsTestConfig{
		metricName:      metricName,
		metricType:      metricType,
		driverName:      driverName,
		operationName:   operationName,
		operationStatus: operationStatus,
		snapshotType:    snapshotType,
		le:              le,
	}
}

func newSnapshotControllerMetrics(cfg snapshotMetricsTestConfig, metricsGrabber *e2emetrics.Grabber) *snapshotControllerMetrics {
	return &snapshotControllerMetrics{
		cfg:            cfg,
		metricsGrabber: metricsGrabber,

		countMetrics:  make(map[string]float64),
		sumMetrics:    make(map[string]float64),
		bucketMetrics: make(map[string]float64),
	}
}

func (scm *snapshotControllerMetrics) waitForSnapshotControllerMetric(expectedValue float64, timeout time.Duration) {
	metricKey := scm.getMetricKey()
	if successful := utils.WaitUntil(10*time.Second, timeout, func() bool {
		// get metric value
		actualValue, err := scm.getSnapshotControllerMetricValue()
		if err != nil {
			return false
		}

		// Another operation could have finished from a previous test,
		// so we check if we have at least the expected value.
		if actualValue < expectedValue {
			return false
		}

		return true
	}); successful {
		return
	}

	scm.showMetricsFailure(metricKey)
	framework.Failf("Unable to get valid snapshot controller metrics after %v", timeout)
}

func (scm *snapshotControllerMetrics) getSnapshotControllerMetricValue() (float64, error) {
	metricKey := scm.getMetricKey()

	// grab and parse into readable format
	err := scm.grabSnapshotControllerMetrics()
	if err != nil {
		return 0, err
	}

	metrics := scm.getMetricsTable()
	actual, ok := metrics[metricKey]
	if !ok {
		return 0, fmt.Errorf("did not find metric for key %s", metricKey)
	}

	return actual, nil
}

func (scm *snapshotControllerMetrics) getMetricsTable() map[string]float64 {
	var metrics map[string]float64
	switch scm.cfg.metricType {
	case "count":
		metrics = scm.countMetrics

	case "sum":
		metrics = scm.sumMetrics

	case "bucket":
		metrics = scm.bucketMetrics
	}

	return metrics
}

func (scm *snapshotControllerMetrics) showMetricsFailure(metricKey string) {
	framework.Logf("failed to find metric key %s inside of the following metrics:", metricKey)

	metrics := scm.getMetricsTable()
	for k, v := range metrics {
		framework.Logf("%s: %v", k, v)
	}
}

func (scm *snapshotControllerMetrics) grabSnapshotControllerMetrics() error {
	// pull all metrics
	metrics, err := scm.metricsGrabber.GrabFromSnapshotController(framework.TestContext.SnapshotControllerPodName, framework.TestContext.SnapshotControllerHTTPPort)
	if err != nil {
		return err
	}

	for method, samples := range metrics {

		for _, sample := range samples {
			operationName := string(sample.Metric["operation_name"])
			driverName := string(sample.Metric["driver_name"])
			operationStatus := string(sample.Metric["operation_status"])
			snapshotType := string(sample.Metric["snapshot_type"])
			le := string(sample.Metric["le"])
			key := snapshotMetricKey(scm.cfg.metricName, driverName, operationName, operationStatus, snapshotType, le)

			switch method {
			case "snapshot_controller_operation_total_seconds_count":
				for _, sample := range samples {
					scm.countMetrics[key] = float64(sample.Value)
				}

			case "snapshot_controller_operation_total_seconds_sum":
				for _, sample := range samples {
					scm.sumMetrics[key] = float64(sample.Value)
				}

			case "snapshot_controller_operation_total_seconds_bucket":
				for _, sample := range samples {
					scm.bucketMetrics[key] = float64(sample.Value)
				}
			}
		}
	}

	return nil
}

func (scm *snapshotControllerMetrics) getMetricKey() string {
	return snapshotMetricKey(scm.cfg.metricName, scm.cfg.driverName, scm.cfg.operationName, scm.cfg.operationStatus, scm.cfg.snapshotType, scm.cfg.le)
}

func snapshotMetricKey(metricName, driverName, operationName, operationStatus, snapshotType, le string) string {
	key := driverName

	// build key for shorthand metrics storage
	for _, s := range []string{metricName, operationName, operationStatus, snapshotType, le} {
		if s != "" {
			key = fmt.Sprintf("%s_%s", key, s)
		}
	}

	return key
}
