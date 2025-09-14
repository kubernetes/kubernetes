/*
Copyright 2024 The Kubernetes Authors.

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
	"crypto/sha256"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumeGroupSnapshottableTest struct {
	config      *storageframework.PerTestConfig
	pods        []*v1.Pod
	volumeGroup [3][]*storageframework.VolumeResource
	snapshots   []*storageframework.VolumeGroupSnapshotResource
	numPods     int
	numVolumes  int
}

type VolumeGroupSnapshottableTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

func InitVolumeGroupSnapshottableTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.VolumeGroupSnapshotDelete,
	}
	return InitCustomGroupSnapshottableTestSuite(patterns)
}

func InitCustomGroupSnapshottableTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &VolumeGroupSnapshottableTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volumegroupsnapshottable",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			TestTags: []interface{}{feature.VolumeGroupSnapshotDataSource},
		},
	}
}

func (s *VolumeGroupSnapshottableTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	dInfo := driver.GetDriverInfo()
	ok := false
	_, ok = driver.(storageframework.VoulmeGroupSnapshottableTestDriver)
	if !dInfo.Capabilities[storageframework.CapVolumeGroupSnapshot] || !ok {
		e2eskipper.Skipf("Driver %q does not support group snapshots - skipping", dInfo.Name)
	}
}

func (s *VolumeGroupSnapshottableTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *VolumeGroupSnapshottableTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	labelKey := "group"
	labelValue := "test-group"
	f := framework.NewDefaultFramework("volumegroupsnapshottable")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Describe("VolumeGroupSnapshottable", func() {

		ginkgo.Context("", func() {
			var (
				snapshottableDriver storageframework.VoulmeGroupSnapshottableTestDriver
				cs                  clientset.Interface
				groupTest           *volumeGroupSnapshottableTest
			)
			init := func(ctx context.Context) {
				snapshottableDriver = driver.(storageframework.VoulmeGroupSnapshottableTestDriver)
				cs = f.ClientSet
				config := driver.PrepareTest(ctx, f)

				groupTest = &volumeGroupSnapshottableTest{
					config:      config,
					volumeGroup: [3][]*storageframework.VolumeResource{},
					snapshots:   []*storageframework.VolumeGroupSnapshotResource{},
					pods:        []*v1.Pod{},
					numPods:     1,
					numVolumes:  3,
				}
			}

			createGroupLabel := func(ctx context.Context, pvc *v1.PersistentVolumeClaim, labelKey, labelValue string) {
				if pvc.Labels == nil {
					pvc.Labels = map[string]string{}
				}
				pvc.Labels[labelKey] = labelValue
				_, err := cs.CoreV1().PersistentVolumeClaims(pvc.GetNamespace()).Update(ctx, pvc, metav1.UpdateOptions{})
				framework.ExpectNoError(err, "failed to update PVC %s", pvc.Name)
			}

			createPodsAndVolumes := func(ctx context.Context) {
				for i := 0; i < groupTest.numPods; i++ {
					framework.Logf("Creating resources for pod %d/%d", i, groupTest.numPods-1)
					for j := 0; j < groupTest.numVolumes; j++ {
						volume := storageframework.CreateVolumeResource(ctx, driver, groupTest.config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
						groupTest.volumeGroup[i] = append(groupTest.volumeGroup[i], volume)
						createGroupLabel(ctx, volume.Pvc, labelKey, labelValue)

					}
					pvcs := []*v1.PersistentVolumeClaim{}
					for _, volume := range groupTest.volumeGroup[i] {
						pvcs = append(pvcs, volume.Pvc)
					}
					// Create a pod with multiple volumes
					podConfig := e2epod.Config{
						NS:           f.Namespace.Name,
						PVCs:         pvcs,
						SeLinuxLabel: e2epv.SELinuxLabel,
					}
					pod, err := e2epod.MakeSecPod(&podConfig)
					framework.ExpectNoError(err, "failed to create pod")
					groupTest.pods = append(groupTest.pods, pod)
				}
				for i, pod := range groupTest.pods {
					pod, err := cs.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
					if err != nil {
						framework.Failf("Failed to create pod-%d [%+v]. Error: %v", i, pod, err)
					}
					if err = e2epod.WaitForPodRunningInNamespace(ctx, cs, pod); err != nil {
						framework.Failf("Failed to wait for pod-%d [%+v] to turn into running status. Error: %v", i, pod, err)
					}
				}
			}

			cleanup := func(ctx context.Context) {
				for _, pod := range groupTest.pods {
					framework.Logf("Deleting pod %s", pod.Name)
					err := e2epod.DeletePodWithWait(ctx, cs, pod)
					framework.ExpectNoError(err, "failed to delete pod %s", pod.Name)
				}
				for _, group := range groupTest.volumeGroup {
					for _, volume := range group {
						framework.Logf("Deleting volume %s", volume.Pvc.Name)
						err := volume.CleanupResource(ctx)
						framework.ExpectNoError(err, "failed to delete volume %s", volume.Pvc.Name)
					}
				}

			}

			ginkgo.It("should create snapshots for multiple volumes in a pod", func(ctx context.Context) {
				init(ctx)
				createPodsAndVolumes(ctx)
				ginkgo.DeferCleanup(cleanup)

				snapshot := storageframework.CreateVolumeGroupSnapshotResource(ctx, snapshottableDriver, groupTest.config, pattern, labelValue, groupTest.volumeGroup[0][0].Pvc.GetNamespace(), f.Timeouts, map[string]string{"deletionPolicy": pattern.SnapshotDeletionPolicy.String()})
				groupTest.snapshots = append(groupTest.snapshots, snapshot)
				ginkgo.By("verifying the snapshots in the group are ready to use")
				status := snapshot.VGS.Object["status"]
				err := framework.Gomega().Expect(status).NotTo(gomega.BeNil())
				framework.ExpectNoError(err, "failed to get status of group snapshot")

				volumeListMap := snapshot.VGSContent.Object["status"].(map[string]interface{})
				err = framework.Gomega().Expect(volumeListMap).NotTo(gomega.BeNil())
				framework.ExpectNoError(err, "failed to get volume snapshot list")
				volumeSnapshotHandlePairList := volumeListMap["volumeSnapshotHandlePairList"].([]interface{})
				err = framework.Gomega().Expect(volumeSnapshotHandlePairList).NotTo(gomega.BeNil())
				framework.ExpectNoError(err, "failed to get volume snapshot list")
				err = framework.Gomega().Expect(len(volumeSnapshotHandlePairList)).To(gomega.Equal(groupTest.numVolumes))
				framework.ExpectNoError(err, "failed to get volume snapshot list")
				claimSize := groupTest.volumeGroup[0][0].Pvc.Spec.Resources.Requests.Storage().String()
				for _, volume := range volumeSnapshotHandlePairList {
					// Create a PVC from the snapshot
					volumeHandle := volume.(map[string]interface{})["volumeHandle"].(string)
					err = framework.Gomega().Expect(volumeHandle).NotTo(gomega.BeNil())
					framework.ExpectNoError(err, "failed to get volume handle from volume")
					uid := snapshot.VGS.Object["metadata"].(map[string]interface{})["uid"].(string)
					err = framework.Gomega().Expect(uid).NotTo(gomega.BeNil())
					framework.ExpectNoError(err, "failed to get uuid from content")
					volumeSnapshotName := fmt.Sprintf("snapshot-%x", sha256.Sum256([]byte(
						uid+volumeHandle)))

					pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
						StorageClassName: &groupTest.volumeGroup[0][0].Sc.Name,
						ClaimSize:        claimSize,
					}, f.Namespace.Name)

					group := "snapshot.storage.k8s.io"

					pvc.Spec.DataSource = &v1.TypedLocalObjectReference{
						APIGroup: &group,
						Kind:     "VolumeSnapshot",
						Name:     volumeSnapshotName,
					}

					volSrc := v1.VolumeSource{
						Ephemeral: &v1.EphemeralVolumeSource{
							VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
								Spec: pvc.Spec,
							},
						},
					}
					pvc, err = cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(ctx, pvc, metav1.CreateOptions{})
					framework.ExpectNoError(err, "failed to create PVC from snapshot")

					pod := StartInPodWithVolumeSource(ctx, cs, volSrc, pvc.Namespace, "snapshot-pod", "sleep 300", groupTest.config.ClientNodeSelection)
					ginkgo.DeferCleanup(e2epod.DeletePodWithWait, cs, pod)
					framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, pod.Name, pod.Namespace, f.Timeouts.PodStartSlow), "Pod did not start in expected time")
				}

			})
		})
	})
}
