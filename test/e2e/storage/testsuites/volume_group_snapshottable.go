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
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumeGroupSnapshottableTest struct {
	config          *storageframework.PerTestConfig
	statefulSet     *appsv1.StatefulSet
	pods            []*v1.Pod
	volumeResources []*storageframework.VolumeResource
	snapshots       []*storageframework.VolumeGroupSnapshotResource
	numReplicas     int
}

// VolumeGroupSnapshottableTestSuite represents a test suite for testing volume group snapshot functionality.
type VolumeGroupSnapshottableTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitVolumeGroupSnapshottableTestSuite initializes the test suite for volume group snapshottable functionality.
func InitVolumeGroupSnapshottableTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.VolumeGroupSnapshotDelete,
	}
	return InitCustomGroupSnapshottableTestSuite(patterns)
}

// InitCustomGroupSnapshottableTestSuite initializes a custom test suite for volume group snapshottable tests
// with the provided test patterns.
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

// SkipUnsupportedTests skips tests if the driver does not support group snapshots.
func (s *VolumeGroupSnapshottableTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	dInfo := driver.GetDriverInfo()
	ok := false
	_, ok = driver.(storageframework.VolumeGroupSnapshottableTestDriver)
	if !dInfo.Capabilities[storageframework.CapVolumeGroupSnapshot] || !ok {
		e2eskipper.Skipf("Driver %q does not support group snapshots - skipping", dInfo.Name)
	}
}

// GetTestSuiteInfo returns the test suite information for the VolumeGroupSnapshottableTestSuite.
func (s *VolumeGroupSnapshottableTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

// DefineTests defines the test cases for the VolumeGroupSnapshottableTestSuite.
func (s *VolumeGroupSnapshottableTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	labelKey := "group"
	labelValue := "test-group"
	f := framework.NewDefaultFramework("volumegroupsnapshottable")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Describe("VolumeGroupSnapshottable", func() {
		var tk *e2ekubectl.TestKubeconfig
		ginkgo.BeforeEach(func() {
			tk = e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, f.Namespace.Name)
		})

		ginkgo.Context("", func() {
			var (
				snapshottableDriver storageframework.VolumeGroupSnapshottableTestDriver
				cs                  clientset.Interface
				groupTest           *volumeGroupSnapshottableTest
			)
			init := func(ctx context.Context) {
				snapshottableDriver = driver.(storageframework.VolumeGroupSnapshottableTestDriver)
				cs = f.ClientSet
				config := driver.PrepareTest(ctx, f)

				groupTest = &volumeGroupSnapshottableTest{
					config:          config,
					volumeResources: []*storageframework.VolumeResource{},
					snapshots:       []*storageframework.VolumeGroupSnapshotResource{},
					pods:            []*v1.Pod{},
					numReplicas:     3,
				}
			}

			createStatefulSetAndVolumes := func(ctx context.Context) {
				// Create volume resource which includes storage class
				volumeResource := storageframework.CreateVolumeResource(ctx, driver, groupTest.config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
				groupTest.volumeResources = append(groupTest.volumeResources, volumeResource)

				// Create StatefulSet with volumeClaimTemplates
				statefulSetName := fmt.Sprintf("statefulset-vgs-%s", f.Namespace.Name)
				replicas := int32(groupTest.numReplicas)

				statefulSet := &appsv1.StatefulSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      statefulSetName,
						Namespace: f.Namespace.Name,
					},
					Spec: appsv1.StatefulSetSpec{
						Replicas: &replicas,
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"app": statefulSetName,
							},
						},
						PersistentVolumeClaimRetentionPolicy: &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
							WhenDeleted: appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
							WhenScaled:  appsv1.DeletePersistentVolumeClaimRetentionPolicyType,
						},
						Template: v1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									"app": statefulSetName,
								},
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Name:    "test-container",
										Image:   e2epod.GetDefaultTestImage(),
										Command: []string{"sleep", "3600"},
										VolumeMounts: []v1.VolumeMount{
											{
												Name:      "data",
												MountPath: "/mnt/data",
											},
										},
									},
								},
							},
						},
						VolumeClaimTemplates: []v1.PersistentVolumeClaim{
							{
								ObjectMeta: metav1.ObjectMeta{
									Name: "data",
									Labels: map[string]string{
										labelKey: labelValue,
									},
								},
								Spec: v1.PersistentVolumeClaimSpec{
									AccessModes: []v1.PersistentVolumeAccessMode{
										v1.ReadWriteOnce,
									},
									Resources: v1.VolumeResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceStorage: resource.MustParse(s.GetTestSuiteInfo().SupportedSizeRange.Min),
										},
									},
									StorageClassName: &volumeResource.Sc.Name,
								},
							},
						},
					},
				}

				var err error
				groupTest.statefulSet, err = cs.AppsV1().StatefulSets(f.Namespace.Name).Create(ctx, statefulSet, metav1.CreateOptions{})
				framework.ExpectNoError(err, "failed to create StatefulSet")

				// Wait for StatefulSet to be ready
				e2estatefulset.WaitForRunningAndReady(ctx, cs, replicas, groupTest.statefulSet)

				// Get the pods created by StatefulSet
				for i := 0; i < groupTest.numReplicas; i++ {
					podName := fmt.Sprintf("%s-%d", statefulSetName, i)
					pod, err := cs.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get StatefulSet pod %s", podName)
					groupTest.pods = append(groupTest.pods, pod)
				}
			}

			writeTestDataToVolumes := func(ctx context.Context) map[string]string {
				mountPath := "/mnt/data"
				writePath := fmt.Sprintf("%s/testfile", mountPath)
				originalMntTestData := make(map[string]string)

				ginkgo.By("writing test data to all StatefulSet volumes")
				for i, pod := range groupTest.pods {
					// For StatefulSet, each pod has one PVC named "data-{statefulset-name}-{index}"
					pvcName := fmt.Sprintf("data-%s-%d", groupTest.statefulSet.Name, i)
					testData := fmt.Sprintf("HelloFromStatefulSetPVC%d", i)
					originalMntTestData[pvcName] = testData
					err := tk.WriteFileViaContainer(pod.Name, pod.Spec.Containers[0].Name, writePath, testData)
					framework.ExpectNoError(err, "failed to write test data to StatefulSet pod %s", pod.Name)
				}
				return originalMntTestData
			}

			cleanupResources := func(ctx context.Context) {
				if groupTest.statefulSet != nil {
					framework.Logf("deleting StatefulSet %s", groupTest.statefulSet.Name)
					e2estatefulset.DeleteAllStatefulSets(ctx, cs, groupTest.statefulSet.Namespace)
				}

				var cleanupVGSErrs []error
				for _, vgsr := range groupTest.snapshots {
					if vgsr == nil || vgsr.VGS == nil {
						framework.Logf("Skipping cleanup: VolumeGroupSnapshotResource or VGS is nil")
						continue
					}

					vgsName := vgsr.VGS.GetName()
					vgsNamespace := vgsr.VGS.GetNamespace()

					framework.Logf("deleting VolumeGroupSnapshotResource %s/%s", vgsNamespace, vgsName)
					err := vgsr.CleanupResource(ctx, f.Timeouts)
					if err != nil {
						cleanupVGSErrs = append(cleanupVGSErrs, err)
						framework.Logf("Warning: failed to delete VolumeGroupSnapshotResource %s/%s: %v", vgsNamespace, vgsName, err)
					} else {
						framework.Logf("deleted VolumeGroupSnapshotResource %s/%s", vgsNamespace, vgsName)
					}
				}
				framework.ExpectNoError(utilerrors.NewAggregate(cleanupVGSErrs), "failed to delete VGS resources")

				var cleanupVolumeErrs []error
				for _, volumeResource := range groupTest.volumeResources {
					if volumeResource == nil || volumeResource.Pvc == nil {
						continue
					}

					ns, name := volumeResource.Pvc.Namespace, volumeResource.Pvc.Name
					framework.Logf("deleting volume resource %s/%s", ns, name)
					if err := volumeResource.CleanupResource(ctx); err != nil {
						cleanupVolumeErrs = append(cleanupVolumeErrs, fmt.Errorf("%s/%s: %w", ns, name, err))
						framework.Logf("Warning: failed to delete volume resource %s/%s: %v", ns, name, err)
					} else {
						framework.Logf("deleted volume resource %s/%s", ns, name)
					}
				}
				framework.ExpectNoError(utilerrors.NewAggregate(cleanupVolumeErrs), "failed to delete volume resources")
			}

			ginkgo.It("should create snapshots for StatefulSet volumes and verify data consistency after restore", func(ctx context.Context) {
				init(ctx)
				createStatefulSetAndVolumes(ctx)
				ginkgo.DeferCleanup(cleanupResources)

				originalMntTestData := writeTestDataToVolumes(ctx)
				snapshot := storageframework.CreateVolumeGroupSnapshotResource(ctx, snapshottableDriver, groupTest.config, pattern, labelValue, f.Namespace.Name, f.Timeouts, map[string]string{"deletionPolicy": pattern.SnapshotDeletionPolicy.String()})
				groupTest.snapshots = append(groupTest.snapshots, snapshot)

				ginkgo.By("verifying the snapshots in the group are ready to use")
				status := snapshot.VGS.Object["status"]
				gomega.Expect(status).ShouldNot(gomega.BeNil(), "failed to get status of group snapshot")
				volumeListMap := snapshot.VGSContent.Object["status"].(map[string]interface{})
				gomega.Expect(volumeListMap).ShouldNot(gomega.BeNil(), "failed to get group snapshot list")
				volumeSnapshotInfoList := volumeListMap["volumeSnapshotInfoList"].([]interface{})
				gomega.Expect(volumeSnapshotInfoList).ShouldNot(gomega.BeNil(), "failed to get group snapshot handle list")
				gomega.Expect(volumeSnapshotInfoList).Should(gomega.HaveLen(groupTest.numReplicas), "failed to verify snapshot handle list length")
				claimSize := s.GetTestSuiteInfo().SupportedSizeRange.Min

				ginkgo.By("creating restored PVCs from snapshots")
				restoredPVCs := []*v1.PersistentVolumeClaim{}
				volumeHandleToPVCName := make(map[string]string)

				// First, create mapping from volume handles to original PVC names for StatefulSet
				for i := 0; i < groupTest.numReplicas; i++ {
					pvcName := fmt.Sprintf("data-%s-%d", groupTest.statefulSet.Name, i)
					pvc, err := cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Get(ctx, pvcName, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get PVC %s", pvcName)

					pv, err := cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get PV for PVC %s", pvcName)
					volumeHandle := pv.Spec.CSI.VolumeHandle
					volumeHandleToPVCName[volumeHandle] = pvcName
				}

				for _, info := range volumeSnapshotInfoList {
					// Create a PVC from the snapshot
					volumeHandle := info.(map[string]interface{})["volumeHandle"].(string)
					if volumeHandle == "" {
						framework.Failf("volumeHandle missing for volume snapshot %v", info)
					}

					uid := snapshot.VGS.Object["metadata"].(map[string]interface{})["uid"].(string)
					gomega.Expect(uid).NotTo(gomega.BeNil(), "failed to get uuid from content")
					volumeSnapshotName := fmt.Sprintf("snapshot-%x", sha256.Sum256([]byte(
						uid+volumeHandle)))

					// Use original PVC name as base for restored PVC name
					originalPVCName := volumeHandleToPVCName[volumeHandle]
					restoredPVCName := fmt.Sprintf("restored-%s", originalPVCName)

					pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
						StorageClassName: &groupTest.volumeResources[0].Sc.Name,
						ClaimSize:        claimSize,
						Name:             restoredPVCName,
					}, f.Namespace.Name)

					group := "snapshot.storage.k8s.io"

					pvc.Spec.DataSource = &v1.TypedLocalObjectReference{
						APIGroup: &group,
						Kind:     "VolumeSnapshot",
						Name:     volumeSnapshotName,
					}

					pvc, err := cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(ctx, pvc, metav1.CreateOptions{})
					framework.ExpectNoError(err, "failed to create PVC from snapshot")
					restoredPVCs = append(restoredPVCs, pvc)
				}

				ginkgo.DeferCleanup(func(ctx context.Context) {
					for _, pvc := range restoredPVCs {
						framework.Logf("Deleting restored PVC %s", pvc.Name)
						err := cs.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Delete(ctx, pvc.Name, metav1.DeleteOptions{})
						framework.ExpectNoError(err, "failed to delete restored PVC %s", pvc.Name)
					}
				})

				ginkgo.By("creating single pod with all restored volumes")
				restoredPodConfig := e2epod.Config{
					NS:           f.Namespace.Name,
					PVCs:         restoredPVCs,
					SeLinuxLabel: e2epv.SELinuxLabel,
				}
				restoredPod, err := e2epod.MakeSecPod(&restoredPodConfig)
				framework.ExpectNoError(err, "failed to create restored pod config")

				restoredPod, err = cs.CoreV1().Pods(f.Namespace.Name).Create(ctx, restoredPod, metav1.CreateOptions{})
				framework.ExpectNoError(err, "failed to create restored pod")
				ginkgo.DeferCleanup(e2epod.DeletePodWithWait, cs, restoredPod)
				framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, restoredPod.Name, restoredPod.Namespace, f.Timeouts.PodStartSlow), "Restored pod did not start in expected time")

				ginkgo.By("verifying data consistency for all restored StatefulSet volumes")
				var dataConsistencyCheckErrors []string
				for volumeIndex, pvc := range restoredPVCs {
					dataPath := "/mnt/volume"
					mountPath := fmt.Sprintf("%s%d", dataPath, volumeIndex+1)
					readPath := fmt.Sprintf("%s/testfile", mountPath)

					// Extract original PVC name from restored PVC name
					const restoredPrefix = "restored-"
					if !strings.HasPrefix(pvc.Name, restoredPrefix) {
						framework.Failf("unexpected restored PVC name format: %s, expected prefix %s", pvc.Name, restoredPrefix)
					}
					originalPVCName := pvc.Name[len(restoredPrefix):]
					expectedData, ok := originalMntTestData[originalPVCName]
					if !ok {
						framework.Failf("no test data found for original PVC %s", originalPVCName)
					}

					restoredData, err := tk.ReadFileViaContainer(restoredPod.Name, restoredPod.Spec.Containers[0].Name, readPath)
					if err != nil || !strings.Contains(restoredData, expectedData) {
						dataConsistencyCheckErrors = append(dataConsistencyCheckErrors, fmt.Sprintf("volume %s (from %s), expectedData: %s, restoredData: %s : err: %v", pvc.Name, originalPVCName, expectedData, restoredData, err))
					} else {
						framework.Logf("data consistency verified for StatefulSet volume %s (from %s): found expected data '%s'", pvc.Name, originalPVCName, expectedData)
					}
				}
				if len(dataConsistencyCheckErrors) > 0 {
					framework.Logf("data verification failed for one or more volumes: %v", dataConsistencyCheckErrors)
				}
				gomega.Expect(dataConsistencyCheckErrors).Should(gomega.BeEmpty(), "failed to check data consistency")
			})
		})
	})
}
