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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-helpers/storage/ephemeral"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// data file name
const datapath = "/mnt/test/data"

type snapshottableTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var (
	sDriver storageframework.SnapshottableTestDriver
	dDriver storageframework.DynamicPVTestDriver
)

// InitCustomSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomSnapshottableTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &snapshottableTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "snapshottable",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			TestTags: []interface{}{feature.VolumeSnapshotDataSource},
		},
	}
}

// InitSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitSnapshottableTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DynamicSnapshotDelete,
		storageframework.DynamicSnapshotRetain,
		storageframework.EphemeralSnapshotDelete,
		storageframework.EphemeralSnapshotRetain,
		storageframework.PreprovisionedSnapshotDelete,
		storageframework.PreprovisionedSnapshotRetain,
	}
	return InitCustomSnapshottableTestSuite(patterns)
}

func (s *snapshottableTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *snapshottableTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	dInfo := driver.GetDriverInfo()
	ok := false
	_, ok = driver.(storageframework.SnapshottableTestDriver)
	if !dInfo.Capabilities[storageframework.CapSnapshotDataSource] || !ok {
		e2eskipper.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
	}
	_, ok = driver.(storageframework.DynamicPVTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %q does not support dynamic provisioning - skipping", driver.GetDriverInfo().Name)
	}
}

func (s *snapshottableTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("snapshotting")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Describe("volume snapshot controller", func() {
		var (
			err    error
			config *storageframework.PerTestConfig

			cs                  clientset.Interface
			dc                  dynamic.Interface
			pvc                 *v1.PersistentVolumeClaim
			sc                  *storagev1.StorageClass
			volumeResource      *storageframework.VolumeResource
			pod                 *v1.Pod
			claimSize           string
			originalMntTestData string
		)
		init := func(ctx context.Context) {
			sDriver, _ = driver.(storageframework.SnapshottableTestDriver)
			dDriver, _ = driver.(storageframework.DynamicPVTestDriver)
			// init snap class, create a source PV, PVC, Pod
			cs = f.ClientSet
			dc = f.DynamicClient

			// Now do the more expensive test initialization.
			config = driver.PrepareTest(ctx, f)

			volumeResource = storageframework.CreateVolumeResource(ctx, dDriver, config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
			ginkgo.DeferCleanup(volumeResource.CleanupResource)

			ginkgo.By("[init] starting a pod to use the claim")
			originalMntTestData = fmt.Sprintf("hello from %s namespace", f.Namespace.Name)
			// After writing data to a file `sync` flushes the data from memory to disk.
			// sync is available in the Linux and Windows versions of agnhost.
			command := fmt.Sprintf("echo '%s' > %s; sync", originalMntTestData, datapath)

			pod = StartInPodWithVolumeSource(ctx, cs, *volumeResource.VolSource, f.Namespace.Name, "pvc-snapshottable-tester", command, config.ClientNodeSelection)

			// At this point a pod is created with a PVC. How to proceed depends on which test is running.
		}

		ginkgo.Context("", func() {
			ginkgo.It("should check snapshot fields, check restore correctly works, check deletion (ephemeral)", func(ctx context.Context) {
				if pattern.VolType != storageframework.GenericEphemeralVolume {
					e2eskipper.Skipf("volume type %q is not ephemeral", pattern.VolType)
				}
				init(ctx)

				// delete the pod at the end of the test
				ginkgo.DeferCleanup(e2epod.DeletePodWithWait, cs, pod)

				// We can test snapshotting of generic
				// ephemeral volumes by creating the snapshot
				// while the pod is running (online). We cannot do it after pod deletion,
				// because then the PVC also gets marked and snapshotting no longer works
				// (even when a finalizer prevents actual removal of the PVC).
				//
				// Because data consistency cannot be
				// guaranteed, this flavor of the test doesn't
				// check the content of the snapshot.

				framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, cs, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow))
				pod, err = cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "check pod after it terminated")

				// Get new copy of the claim
				ginkgo.By("[init] checking the claim")
				pvcName := ephemeral.VolumeClaimName(pod, &pod.Spec.Volumes[0])
				pvcNamespace := pod.Namespace

				parameters := map[string]string{}
				sr := storageframework.CreateSnapshotResource(ctx, sDriver, config, pattern, pvcName, pvcNamespace, f.Timeouts, parameters)
				ginkgo.DeferCleanup(sr.CleanupResource, f.Timeouts)

				err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, pvcNamespace, pvcName, framework.Poll, f.Timeouts.ClaimProvision)
				framework.ExpectNoError(err)

				pvc, err = cs.CoreV1().PersistentVolumeClaims(pvcNamespace).Get(ctx, pvcName, metav1.GetOptions{})
				framework.ExpectNoError(err, "get PVC")
				claimSize = pvc.Spec.Resources.Requests.Storage().String()
				sc = volumeResource.Sc

				// Get the bound PV
				ginkgo.By("[init] checking the PV")
				_, err := cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
				framework.ExpectNoError(err)

				vs := sr.Vs
				// get the snapshot and check SnapshotContent properties
				vscontent := checkSnapshot(ctx, dc, sr, pattern)

				var restoredPVC *v1.PersistentVolumeClaim
				var restoredPod *v1.Pod

				ginkgo.By("creating a pvc from the snapshot")
				restoredPVC = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        claimSize,
					StorageClassName: &(sc.Name),
				}, config.Framework.Namespace.Name)

				group := "snapshot.storage.k8s.io"

				restoredPVC.Spec.DataSource = &v1.TypedLocalObjectReference{
					APIGroup: &group,
					Kind:     "VolumeSnapshot",
					Name:     vs.GetName(),
				}

				ginkgo.By("starting a pod to use the snapshot")
				volSrc := v1.VolumeSource{
					Ephemeral: &v1.EphemeralVolumeSource{
						VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
							Spec: restoredPVC.Spec,
						},
					},
				}

				restoredPod = StartInPodWithVolumeSource(ctx, cs, volSrc, restoredPVC.Namespace, "restored-pvc-tester", "sleep 300", config.ClientNodeSelection)
				ginkgo.DeferCleanup(e2epod.DeletePodWithWait, cs, restoredPod)

				framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, restoredPod.Name, restoredPod.Namespace, f.Timeouts.PodStartSlow))
				if pattern.VolType != storageframework.GenericEphemeralVolume {
					commands := e2evolume.GenerateReadFileCmd(datapath)
					_, err = e2eoutput.LookForStringInPodExec(restoredPod.Namespace, restoredPod.Name, commands, originalMntTestData, time.Minute)
					framework.ExpectNoError(err)
				}

				ginkgo.By("should delete the VolumeSnapshotContent according to its deletion policy")
				// Delete both Snapshot and restored Pod/PVC at the same time because different storage systems
				// have different ordering of deletion. Some may require delete the restored PVC first before
				// Snapshot deletion and some are opposite.
				err = storageutils.DeleteSnapshotWithoutWaiting(ctx, dc, vs.GetNamespace(), vs.GetName())
				framework.ExpectNoError(err)
				framework.Logf("deleting restored pod %q/%q", restoredPod.Namespace, restoredPod.Name)
				err = cs.CoreV1().Pods(restoredPod.Namespace).Delete(context.TODO(), restoredPod.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err)
				deleteVolumeSnapshot(ctx, f, dc, sr, pattern, vscontent)
			})

			ginkgo.It("should check snapshot fields, check restore correctly works after modifying source data, check deletion (persistent)", func(ctx context.Context) {
				if pattern.VolType == storageframework.GenericEphemeralVolume {
					e2eskipper.Skipf("volume type %q is ephemeral", pattern.VolType)
				}
				init(ctx)

				pvc = volumeResource.Pvc
				sc = volumeResource.Sc

				// The pod should be in the Success state.
				ginkgo.By("[init] check pod success")
				pod, err = cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Failed to fetch pod: %v", err)
				framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, cs, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow))
				// Sync the pod to know additional fields.
				pod, err = cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "Failed to fetch pod: %v", err)

				ginkgo.By("[init] checking the claim")
				err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, f.Timeouts.ClaimProvision)
				framework.ExpectNoError(err)
				// Get new copy of the claim.
				pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// Get the bound PV.
				ginkgo.By("[init] checking the PV")
				pv, err := cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// Delete the pod to force NodeUnpublishVolume (unlike the ephemeral case where the pod is deleted at the end of the test).
				ginkgo.By("[init] deleting the pod")
				StopPod(ctx, cs, pod)

				// At this point we know that:
				// - a pod was created with a PV that's supposed to have data
				//
				// However there's a caching issue that @jinxu97 explained and it's related with the pod & volume
				// lifecycle, to understand it we first analyze what the volumemanager does:
				// - when a pod is delete the volumemanager will try to cleanup the volume mounts
				//   - NodeUnpublishVolume: unbinds the bind mount from the container
				//     - Linux: the bind mount is removed, which does not flush any cache
				//     - Windows: we delete a symlink, data's not flushed yet to disk
				//   - NodeUnstageVolume: unmount the global mount
				//     - Linux: disk is unmounted and all caches flushed.
				//     - Windows: data is flushed to disk and the disk is detached
				//
				// Pod deletion might not guarantee a data flush to disk, however NodeUnstageVolume adds the logic
				// to flush the data to disk (see #81690 for details). We need to wait for NodeUnstageVolume, as
				// NodeUnpublishVolume only removes the bind mount, which doesn't force the caches to flush.
				// It's possible to create empty snapshots if we don't wait (see #101279 for details).
				//
				// In the following code by checking if the PV is not in the node.Status.VolumesInUse field we
				// ensure that the volume is not used by the node anymore (an indicator that NodeUnstageVolume has
				// already finished)
				nodeName := pod.Spec.NodeName
				gomega.Expect(nodeName).NotTo(gomega.BeEmpty(), "pod.Spec.NodeName must not be empty")

				// Snapshot tests are only executed for CSI drivers. When CSI drivers
				// are attached to the node they use VolumeHandle instead of the pv.Name.
				volumeName := pv.Spec.PersistentVolumeSource.CSI.VolumeHandle

				ginkgo.By(fmt.Sprintf("[init] waiting until the node=%s is not using the volume=%s", nodeName, volumeName))
				success := storageutils.WaitUntil(framework.Poll, f.Timeouts.PVDelete, func() bool {
					node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
					framework.ExpectNoError(err)
					volumesInUse := node.Status.VolumesInUse
					framework.Logf("current volumes in use: %+v", volumesInUse)
					for i := 0; i < len(volumesInUse); i++ {
						if strings.HasSuffix(string(volumesInUse[i]), volumeName) {
							return false
						}
					}
					return true
				})
				if !success {
					framework.Failf("timed out waiting for node=%s to not use the volume=%s", nodeName, volumeName)
				}

				// Take the snapshot.
				parameters := map[string]string{}
				sr := storageframework.CreateSnapshotResource(ctx, sDriver, config, pattern, pvc.Name, pvc.Namespace, f.Timeouts, parameters)
				ginkgo.DeferCleanup(sr.CleanupResource, f.Timeouts)
				vs := sr.Vs
				// get the snapshot and check SnapshotContent properties
				vscontent := checkSnapshot(ctx, dc, sr, pattern)

				ginkgo.By("Modifying source data test")
				var restoredPVC *v1.PersistentVolumeClaim
				var restoredPod *v1.Pod
				modifiedMntTestData := fmt.Sprintf("modified data from %s namespace", pvc.GetNamespace())

				ginkgo.By("modifying the data in the source PVC")

				// After writing data to a file `sync` flushes the data from memory to disk.
				// sync is available in the Linux and Windows versions of agnhost.
				command := fmt.Sprintf("echo '%s' > %s; sync", modifiedMntTestData, datapath)
				RunInPodWithVolume(ctx, cs, f.Timeouts, pvc.Namespace, pvc.Name, "pvc-snapshottable-data-tester", command, config.ClientNodeSelection)

				ginkgo.By("creating a pvc from the snapshot")
				claimSize = pvc.Spec.Resources.Requests.Storage().String()
				restoredPVC = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        claimSize,
					StorageClassName: &(sc.Name),
				}, config.Framework.Namespace.Name)

				group := "snapshot.storage.k8s.io"

				restoredPVC.Spec.DataSource = &v1.TypedLocalObjectReference{
					APIGroup: &group,
					Kind:     "VolumeSnapshot",
					Name:     vs.GetName(),
				}

				restoredPVC, err = cs.CoreV1().PersistentVolumeClaims(restoredPVC.Namespace).Create(ctx, restoredPVC, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				ginkgo.DeferCleanup(func(ctx context.Context) {
					framework.Logf("deleting claim %q/%q", restoredPVC.Namespace, restoredPVC.Name)
					// typically this claim has already been deleted
					err = cs.CoreV1().PersistentVolumeClaims(restoredPVC.Namespace).Delete(ctx, restoredPVC.Name, metav1.DeleteOptions{})
					if err != nil && !apierrors.IsNotFound(err) {
						framework.Failf("Error deleting claim %q. Error: %v", restoredPVC.Name, err)
					}
				})

				ginkgo.By("starting a pod to use the snapshot")
				restoredPod = StartInPodWithVolume(ctx, cs, restoredPVC.Namespace, restoredPVC.Name, "restored-pvc-tester", "sleep 300", config.ClientNodeSelection)
				ginkgo.DeferCleanup(StopPod, cs, restoredPod)
				framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, restoredPod.Name, restoredPod.Namespace, f.Timeouts.PodStartSlow))
				commands := e2evolume.GenerateReadFileCmd(datapath)
				_, err = e2eoutput.LookForStringInPodExec(restoredPod.Namespace, restoredPod.Name, commands, originalMntTestData, time.Minute)
				framework.ExpectNoError(err)

				ginkgo.By("should delete the VolumeSnapshotContent according to its deletion policy")

				// Delete both Snapshot and restored Pod/PVC at the same time because different storage systems
				// have different ordering of deletion. Some may require delete the restored PVC first before
				// Snapshot deletion and some are opposite.
				err = storageutils.DeleteSnapshotWithoutWaiting(ctx, dc, vs.GetNamespace(), vs.GetName())
				framework.ExpectNoError(err)
				framework.Logf("deleting restored pod %q/%q", restoredPod.Namespace, restoredPod.Name)
				err = cs.CoreV1().Pods(restoredPod.Namespace).Delete(ctx, restoredPod.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err)
				framework.Logf("deleting restored PVC %q/%q", restoredPVC.Namespace, restoredPVC.Name)
				err = cs.CoreV1().PersistentVolumeClaims(restoredPVC.Namespace).Delete(ctx, restoredPVC.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err)

				deleteVolumeSnapshot(ctx, f, dc, sr, pattern, vscontent)
			})
		})
	})
}

func deleteVolumeSnapshot(ctx context.Context, f *framework.Framework, dc dynamic.Interface, sr *storageframework.SnapshotResource, pattern storageframework.TestPattern, vscontent *unstructured.Unstructured) {
	vs := sr.Vs

	// Wait for the Snapshot to be actually deleted from API server
	err := storageutils.WaitForNamespacedGVRDeletion(ctx, dc, storageutils.SnapshotGVR, vs.GetNamespace(), vs.GetNamespace(), framework.Poll, f.Timeouts.SnapshotDelete)
	framework.ExpectNoError(err)

	switch pattern.SnapshotDeletionPolicy {
	case storageframework.DeleteSnapshot:
		ginkgo.By("checking the SnapshotContent has been deleted")
		err = storageutils.EnsureGVRDeletion(ctx, dc, storageutils.SnapshotContentGVR, vscontent.GetName(), framework.Poll, f.Timeouts.SnapshotDelete, "")
		framework.ExpectNoError(err)
	case storageframework.RetainSnapshot:
		ginkgo.By("checking the SnapshotContent has not been deleted")
		err = storageutils.EnsureNoGVRDeletion(ctx, dc, storageutils.SnapshotContentGVR, vscontent.GetName(), 1*time.Second /* poll */, 30*time.Second /* timeout */, "")
		framework.ExpectNoError(err)
	}
}

func checkSnapshot(ctx context.Context, dc dynamic.Interface, sr *storageframework.SnapshotResource, pattern storageframework.TestPattern) *unstructured.Unstructured {
	vs := sr.Vs
	vsc := sr.Vsclass

	// Get new copy of the snapshot
	ginkgo.By("checking the snapshot")
	vs, err := dc.Resource(storageutils.SnapshotGVR).Namespace(vs.GetNamespace()).Get(ctx, vs.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	// Get the bound snapshotContent
	snapshotStatus := vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	vscontent, err := dc.Resource(storageutils.SnapshotContentGVR).Get(ctx, snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	snapshotContentSpec := vscontent.Object["spec"].(map[string]interface{})
	volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})

	// Check SnapshotContent properties
	ginkgo.By("checking the SnapshotContent")
	// PreprovisionedCreatedSnapshot do not need to set volume snapshot class name
	if pattern.SnapshotType != storageframework.PreprovisionedCreatedSnapshot {
		gomega.Expect(snapshotContentSpec["volumeSnapshotClassName"]).To(gomega.Equal(vsc.GetName()))
	}
	gomega.Expect(volumeSnapshotRef).To(gomega.HaveKeyWithValue("name", vs.GetName()))
	gomega.Expect(volumeSnapshotRef).To(gomega.HaveKeyWithValue("namespace", vs.GetNamespace()))
	return vscontent
}
