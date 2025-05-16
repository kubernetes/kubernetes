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

package testsuites

import (
	"context"
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"

	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type SnapshotMetadataTest struct{}

type snapshotMetadataTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

func InitCustomSnapshotMetadataTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &snapshotMetadataTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "snapshotmetadata",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				// Min: "1Gi",
				Max: "1Gi",
			},
			TestTags: []interface{}{feature.SnapshotMetadata},
		},
	}
}

func InitSnapshotMetadataTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.SnapshotMetadata,
	}

	return InitCustomSnapshotMetadataTestSuite(patterns)
}

// GetTestSuiteInfo implements framework.TestSuite.
func (s *snapshotMetadataTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

// SkipUnsupportedTests implements framework.TestSuite.
func (s *snapshotMetadataTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	_, ok := driver.(storageframework.SnapshotMetadataTestDriver)
	driverInfo := driver.GetDriverInfo()
	if !driverInfo.Capabilities[storageframework.CapSnapshotMetadata] || !ok {
		e2eskipper.Skipf("Driver %q does not support snapshot metadata", driverInfo.Name)
	}
}

func constructVerifierCommand(
	namespace, snapshotName, previousSnapshotName,
	sourceDevicePath, targetDevicePath string,
) string {
	command := []string{
		"/tools/snapshot-metadata-verifier",
		"-max-results 10",
		"-starting-offset 0",
		"-kubeconfig=\"\"",
		fmt.Sprintf("-namespace=%s", namespace),
	}
	if snapshotName != "" {
		command = append(command, fmt.Sprintf("-snapshot=%s", snapshotName))
	}
	if previousSnapshotName != "" {
		command = append(command, fmt.Sprintf("-previous-snapshot=%s", previousSnapshotName))
	}
	if sourceDevicePath != "" {
		command = append(command, fmt.Sprintf("-source-device-path=%s", sourceDevicePath))
	}
	if targetDevicePath != "" {
		command = append(command, fmt.Sprintf("-target-device-path=%s", targetDevicePath))
	}

	// return command
	return strings.Join(command, " ")
}

func runSnapshotMetadataVerifier(pod *v1.Pod, toolCommand string) {
	ginkgo.By("run snapshot-metadata-verifier")
	stdout, stderr, err := e2ekubectl.RunKubectlWithFullOutput(
		pod.Namespace,
		strings.Split(toolCommand, " ")...,
	)
	framework.ExpectNoError(err, "failed to run snapshot-metadata-verifier tool")
	if stderr != "" {
		framework.Failf("failed to run snapshot-metadata-verifier tool:\nstdout:%s\nstderr:%s\n", stdout, stderr)
	}
}

func createBackupClientResources(ctx context.Context, f *framework.Framework) {
	ginkgo.By("creating backup client resources")
	// create from manifest
	backupClientManifests := []string{
		"test/e2e/testing-manifests/storage-csi/external-snapshot-metadata/backup-client-rbac.yaml",
	}
	err := storageutils.CreateFromManifests(ctx, f, f.Namespace,
		func(item interface{}) error { return nil },
		backupClientManifests...)
	framework.ExpectNoError(err)
}

func runSyncInPod(pod *v1.Pod) {
	ginkgo.By(fmt.Sprintf("running sync in pod %s", pod.Name))
	writeCmd := fmt.Sprintf("exec %s -c write-pod -- sync", pod.Name)
	stdout, stderr, err := e2ekubectl.RunKubectlWithFullOutput(
		pod.Namespace,
		strings.Split(writeCmd, " ")...,
	)
	framework.ExpectNoError(err, "failed running sync command")
	if stderr != "" {
		framework.Failf("failed running sync command:\nstdout:%s\nstderr:%s\n", stdout, stderr)
	}
}

func writeToDeviceInPod(pod *v1.Pod, writeCmd string) {
	ginkgo.By(fmt.Sprintf("writing content into pod %s", pod.Name))
	stdout, stderr, err := e2ekubectl.RunKubectlWithFullOutput(
		pod.Namespace,
		strings.Split(writeCmd, " ")...,
	)
	framework.ExpectNoError(err, "failed writing the contents")
	if stderr != "" {
		framework.Failf("failed writing the contents:\nstdout:%s\nstderr:%s\n", stdout, stderr)
	}
	runSyncInPod(pod)
}

const (
	backupClientServiceAccountName = "backup-app-service-account"
	sourceDevicePvcName            = "source-device"
	targetDevicePvcName            = "target-device"
	installToolContainerName       = "install-tool"
	installToolImage               = "golang:1.23.6"
	installToolCommand             = "/bin/sh -c 'go install github.com/kubernetes-csi/external-snapshot-metadata/examples/snapshot-metadata-verifier@main && cp $(go env GOPATH)/bin/snapshot-metadata-verifier /output'"
	sharedVolumeName               = "shared-volume"
	sharedVolumeMountPath          = "/tools"
)

func createBackupClientPod(ctx context.Context, f *framework.Framework, clientSet clientset.Interface, source, target *v1.PersistentVolumeClaim) (*v1.Pod, error) {
	ginkgo.By("creating backup client pod")
	backupClientPodConfig := e2epod.Config{
		NS:   f.Namespace.Name,
		PVCs: []*v1.PersistentVolumeClaim{source, target},
	}

	backupClientPodObject, err := e2epod.MakeSecPod(&backupClientPodConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod config: %w", err)
	}

	backupClientPodObject.Spec.InitContainers = append(backupClientPodObject.Spec.InitContainers, v1.Container{
		Name:    installToolContainerName,
		Image:   installToolImage,
		Command: []string{"/bin/sh", "-c", installToolCommand},
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      sharedVolumeName,
				MountPath: "/output",
			},
		},
	})

	backupClientPodObject.Spec.Containers[0].VolumeMounts = append(
		backupClientPodObject.Spec.Containers[0].VolumeMounts,
		v1.VolumeMount{
			Name:      sharedVolumeName,
			MountPath: sharedVolumeMountPath,
		},
	)
	backupClientPodObject.Spec.Volumes = append(
		backupClientPodObject.Spec.Volumes,
		v1.Volume{
			Name: sharedVolumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		},
	)

	backupClientPodObject.Spec.ServiceAccountName = backupClientServiceAccountName
	backupClientPod, err := clientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, backupClientPodObject, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create backup client pod")
	err = e2epod.WaitForPodNameRunningInNamespace(ctx, clientSet, backupClientPod.Name, f.Namespace.Name)
	framework.ExpectNoError(err, "Failed to wait for pod %s to turn into running status", backupClientPod.Name)
	backupClientPod, err = clientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, backupClientPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get pod %s", backupClientPod.Name)

	return backupClientPod, nil
}

func createPVCFromSnapshot(ctx context.Context, clientSet clientset.Interface, f *framework.Framework, pvcName, snapshotName string, testPVC *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	ginkgo.By(fmt.Sprintf("creating pvc %s from the snapshot %s", pvcName, snapshotName))
	sourceDevicePvcObject := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
		Name:             pvcName,
		ClaimSize:        testPVC.Spec.Resources.Requests.Storage().String(),
		StorageClassName: testPVC.Spec.StorageClassName,
		VolumeMode:       testPVC.Spec.VolumeMode,
	}, f.Namespace.Name)

	group := "snapshot.storage.k8s.io"
	sourceDevicePvcObject.Spec.DataSource = &v1.TypedLocalObjectReference{
		APIGroup: &group,
		Kind:     "VolumeSnapshot",
		Name:     snapshotName,
	}
	restorePvc, err := clientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(ctx, sourceDevicePvcObject, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pvc: %w", err)
	}

	return restorePvc, nil
}

// DefineTests implements framework.TestSuite.
func (s *snapshotMetadataTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		smDriver  storageframework.SnapshotMetadataTestDriver
		config    *storageframework.PerTestConfig
		clientSet clientset.Interface

		//
		volume         *storageframework.VolumeResource
		testPVC        *v1.PersistentVolumeClaim
		testPod        *v1.Pod
		testDevicePath string
		err            error

		// backup client
		sourceDeviceName string
		sourveDevicePath string
		targetDevicePath string
		targetDeviceName string
	)

	f := framework.NewDefaultFramework("snapshotmetadata")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	createBackupClientPod := func(ctx context.Context, source, target *v1.PersistentVolumeClaim) *v1.Pod {
		createBackupClientResources(ctx, f)
		backupClientPod, err := createBackupClientPod(ctx, f, clientSet, source, target)
		framework.ExpectNoError(err, "Failed to create backup client pod")

		// Update source and target device path
		for _, volume := range backupClientPod.Spec.Volumes {
			if volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == sourceDevicePvcName {
				sourceDeviceName = volume.Name
			}
			if volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == targetDevicePvcName {
				targetDeviceName = volume.Name
			}
		}
		for _, device := range backupClientPod.Spec.Containers[0].VolumeDevices {
			if device.Name == sourceDeviceName {
				sourveDevicePath = device.DevicePath
			}
			if device.Name == targetDeviceName {
				targetDevicePath = device.DevicePath
			}
		}

		err = framework.Gomega().Expect(sourveDevicePath).NotTo(gomega.BeEmpty())
		framework.ExpectNoError(err, "Failed to get source device path")
		err = framework.Gomega().Expect(targetDevicePath).NotTo(gomega.BeEmpty())
		framework.ExpectNoError(err, "Failed to get target device path")

		return backupClientPod
	}

	init := func(ctx context.Context) {
		framework.Logf("Initializing test")
		smDriver = driver.(storageframework.SnapshotMetadataTestDriver)
		clientSet = f.ClientSet

		config = smDriver.PrepareTest(ctx, f)

		pattern.VolMode = v1.PersistentVolumeBlock
		volume = storageframework.CreateVolumeResource(ctx, smDriver, config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)
		testPVC = volume.Pvc
		podConfig := e2epod.Config{
			NS:           f.Namespace.Name,
			PVCs:         []*v1.PersistentVolumeClaim{testPVC},
			SeLinuxLabel: e2epv.SELinuxLabel,
		}

		testPod, err = e2epod.MakeSecPod(&podConfig)
		framework.ExpectNoError(err, "Failed to get pod config for %s", testPod.Name)
		testPod, err = clientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, testPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create pod %s", testPod.Name)
		err = e2epod.WaitForPodRunningInNamespace(ctx, clientSet, testPod)
		framework.ExpectNoError(err, "Failed to wait for pod %s to turn into running status. Error", testPod.Name)
		testPod, err = clientSet.CoreV1().Pods(testPod.Namespace).Get(ctx, testPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get pod %s", testPod.Name)
		testDevicePath = testPod.Spec.Containers[0].VolumeDevices[0].DevicePath // Other way?
	}

	createPVCFromSnapshot := func(ctx context.Context, pvcName, snaphotName string) *v1.PersistentVolumeClaim {
		restorePvc, err := createPVCFromSnapshot(ctx, clientSet, f, pvcName, snaphotName, testPVC)
		framework.ExpectNoError(err, "Failed to create pvc from snapshot")
		return restorePvc
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		// framework.Logf("Don't cleanup")
		// ginkgo.DeferCleanup(nil)
	})

	ginkgo.It("should verify GetMetadataDelta", func(ctx context.Context) {
		init(ctx)

		parameters := map[string]string{}
		sDriver := smDriver.(storageframework.SnapshottableTestDriver)

		// Write content
		writeCmd := fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=6 oflag=direct status=none", testPod.Name, testDevicePath)
		writeToDeviceInPod(testPod, writeCmd)

		// Create snap-1
		ginkgo.By("taking snapshot snap-1")
		snapResource1 := storageframework.CreateSnapshotResource(ctx, sDriver, config, pattern, testPVC.Name, testPVC.Namespace, f.Timeouts, parameters)
		ginkgo.DeferCleanup(snapResource1.CleanupResource, f.Timeouts)

		// Write more content
		writeCmd = fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=3 oflag=direct status=none", testPod.Name, testDevicePath)
		writeToDeviceInPod(testPod, writeCmd)

		// Create snap-2
		ginkgo.By("taking snapshot snap-2")
		snapResource2 := storageframework.CreateSnapshotResource(ctx, sDriver, config, pattern, testPVC.Name, testPVC.Namespace, f.Timeouts, parameters)
		ginkgo.DeferCleanup(snapResource2.CleanupResource, f.Timeouts)

		// Restore PVC (source device) from snapshot snap-2
		sourceDevicePvc := createPVCFromSnapshot(ctx, sourceDevicePvcName, snapResource2.Vs.GetName())

		// Restore PVC (target device) from snapshot snap-1
		targetDevicePvc := createPVCFromSnapshot(ctx, targetDevicePvcName, snapResource1.Vs.GetName())

		// create backup client
		backupClientPod := createBackupClientPod(ctx, sourceDevicePvc, targetDevicePvc)

		// Run snapshot-metadata-verifier
		ginkgo.By("run snapshot-metadata-verifier")
		toolCommand := fmt.Sprintf("exec %s -c write-pod -- %s",
			backupClientPod.Name,
			constructVerifierCommand(f.Namespace.Name, snapResource2.Vs.GetName(), snapResource1.Vs.GetName(), sourveDevicePath, targetDevicePath))
		runSnapshotMetadataVerifier(backupClientPod, toolCommand)
	})

	ginkgo.It("should verify GetAllocatedMetadata", func(ctx context.Context) {
		init(ctx)

		parameters := map[string]string{}
		sDriver := smDriver.(storageframework.SnapshottableTestDriver)

		// Write content
		writeCmd := fmt.Sprintf("exec %s -c write-pod -- dd if=/dev/urandom of=%s bs=4K count=6 oflag=direct status=none", testPod.Name, testDevicePath)
		writeToDeviceInPod(testPod, writeCmd)

		// Create snap-1
		ginkgo.By("taking snapshot snap-1")
		snapResource1 := storageframework.CreateSnapshotResource(ctx, sDriver, config, pattern, testPVC.Name, testPVC.Namespace, f.Timeouts, parameters)
		ginkgo.DeferCleanup(snapResource1.CleanupResource, f.Timeouts)

		// Restore PVC (source device) from snapshot snap-1
		sourceDevicePvc := createPVCFromSnapshot(ctx, sourceDevicePvcName, snapResource1.Vs.GetName())

		// create new PVC (target device)
		ginkgo.By("creating pvc target-device")
		targetPvcClaim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			Name:             "target-device",
			ClaimSize:        testPVC.Spec.Resources.Requests.Storage().String(),
			StorageClassName: testPVC.Spec.StorageClassName,
			VolumeMode:       testPVC.Spec.VolumeMode,
		}, f.Namespace.Name)

		targetDevicePvc, err := e2epv.CreatePVC(ctx, f.ClientSet, f.Namespace.Name, targetPvcClaim)
		framework.ExpectNoError(err)

		// create backup client
		backupClientPod := createBackupClientPod(ctx, sourceDevicePvc, targetDevicePvc)

		// Run snapshot-metadata-verifier
		ginkgo.By("run snapshot-metadata-verifier")
		toolCommand := fmt.Sprintf("exec %s -c write-pod -- %s",
			backupClientPod.Name,
			constructVerifierCommand(f.Namespace.Name, snapResource1.Vs.GetName(), "", sourveDevicePath, targetDevicePath))
		runSnapshotMetadataVerifier(backupClientPod, toolCommand)
	})
}
