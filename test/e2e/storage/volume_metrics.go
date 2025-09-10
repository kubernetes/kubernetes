/*
Copyright 2017 The Kubernetes Authors.

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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/component-helpers/storage/ephemeral"
	"k8s.io/kubernetes/pkg/features"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

func createPod(ctx context.Context, c clientset.Interface, pod *v1.Pod) *v1.Pod {
	ns := pod.Namespace
	pod, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.DeferCleanup(func(ctx context.Context) {
		framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, c, pod))
	})
	return pod
}

func createPVC(ctx context.Context, c clientset.Interface, pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {
	pvc, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	gomega.Expect(pvc).ToNot(gomega.BeNil())

	ginkgo.DeferCleanup(func(ctx context.Context) {
		newPvc, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to get pvc %s/%s: %v", pvc.Namespace, pvc.Name, err)
		} else {
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(ctx, c, newPvc.Name, newPvc.Namespace))
			if newPvc.Spec.VolumeName != "" {
				err = e2epv.WaitForPersistentVolumeDeleted(ctx, c, newPvc.Spec.VolumeName, 5*time.Second, 5*time.Minute)
				framework.ExpectNoError(err, "Persistent Volume %v not deleted by dynamic provisioner", newPvc.Spec.VolumeName)
			}
		}
	})
	return pvc
}

// This test needs to run in serial because other tests could interfere
// with metrics being tested here.
var _ = utils.SIGDescribe("Volume metrics", func() {
	var (
		c              clientset.Interface
		ns             string
		pvc            *v1.PersistentVolumeClaim
		pvcBlock       *v1.PersistentVolumeClaim
		metricsGrabber *e2emetrics.Grabber
		sc             *storagev1.StorageClass
		err            error
	)
	f := framework.NewDefaultFramework("volume-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	driver := drivers.InitHostPathCSIDriver()

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		ns = f.Namespace.Name
		var err error

		cfg := driver.PrepareTest(ctx, f)
		sc = driver.(storageframework.DynamicPVTestDriver).GetDynamicProvisionStorageClass(ctx, cfg, "")
		sc = testsuites.SetupStorageClass(ctx, f.ClientSet, sc)

		test := testsuites.StorageClassTest{
			Name:      "default",
			Timeouts:  f.Timeouts,
			ClaimSize: "2Gi",
		}

		pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			StorageClassName: &sc.Name,
			ClaimSize:        test.ClaimSize,
			VolumeMode:       ptr.To(v1.PersistentVolumeFilesystem),
		}, ns)

		pvcBlock = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			StorageClassName: &sc.Name,
			ClaimSize:        test.ClaimSize,
			VolumeMode:       ptr.To(v1.PersistentVolumeBlock),
		}, ns)

		metricsGrabber, err = e2emetrics.NewMetricsGrabber(ctx, c, nil, f.ClientConfig(), true, false, true, false, false, false)

		if err != nil {
			framework.Failf("Error creating metrics grabber : %v", err)
		}
	})

	provisioning := func(ctx context.Context, ephemeral bool) {
		if !metricsGrabber.HasControlPlanePods() {
			e2eskipper.Skipf("Environment does not support getting controller-manager metrics - skipping")
		}

		pluginName := sc.Provisioner

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)

		framework.ExpectNoError(err, "Error getting c-m metrics : %v", err)

		storageOpMetrics := getControllerStorageMetrics(controllerMetrics, pluginName)

		if !ephemeral {
			pvc = createPVC(ctx, c, pvc)
		}

		pod := makePod(f, pvc, ephemeral)
		pod, err = c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, c, pod))

		updatedStorageMetrics := waitForDetachAndGrabMetrics(ctx, storageOpMetrics, metricsGrabber, pluginName)

		gomega.Expect(updatedStorageMetrics.latencyMetrics).ToNot(gomega.BeEmpty(), "Error fetching c-m updated storage metrics")
		gomega.Expect(updatedStorageMetrics.statusMetrics).ToNot(gomega.BeEmpty(), "Error fetching c-m updated storage metrics")

		volumeOperations := []string{"volume_detach", "volume_attach"}

		for _, volumeOp := range volumeOperations {
			verifyMetricCount(storageOpMetrics, updatedStorageMetrics, volumeOp, false)
		}
	}

	filesystemMode := func(ctx context.Context, isEphemeral bool) {
		if !isEphemeral {
			pvc = createPVC(ctx, c, pvc)
		}

		pod := makePod(f, pvc, isEphemeral)
		pod = createPod(ctx, c, pod)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		pvcName := pvc.Name
		if isEphemeral {
			pvcName = ephemeral.VolumeClaimName(pod, &pod.Spec.Volumes[0])
		}
		pvcNamespace := pod.Namespace

		// Verify volume stat metrics were collected for the referenced PVC
		volumeStatKeys := []string{
			kubeletmetrics.VolumeStatsUsedBytesKey,
			kubeletmetrics.VolumeStatsCapacityBytesKey,
			kubeletmetrics.VolumeStatsAvailableBytesKey,
			kubeletmetrics.VolumeStatsInodesKey,
			kubeletmetrics.VolumeStatsInodesFreeKey,
			kubeletmetrics.VolumeStatsInodesUsedKey,
		}
		key := volumeStatKeys[0]
		kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
		// Poll kubelet metrics waiting for the volume to be picked up
		// by the volume stats collector
		var kubeMetrics e2emetrics.KubeletMetrics
		waitErr := wait.PollUntilContextTimeout(ctx, 30*time.Second, 5*time.Minute, false, func(ctx context.Context) (bool, error) {
			framework.Logf("Grabbing Kubelet metrics")
			// Grab kubelet metrics from the node the pod was scheduled on
			var err error
			kubeMetrics, err = metricsGrabber.GrabFromKubelet(ctx, pod.Spec.NodeName)
			if err != nil {
				framework.Logf("Error fetching kubelet metrics")
				return false, err
			}
			if !findVolumeStatMetric(kubeletKeyName, pvcNamespace, pvcName, kubeMetrics) {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(waitErr, "Unable to find metric %s for PVC %s/%s", kubeletKeyName, pvcNamespace, pvcName)

		for _, key := range volumeStatKeys {
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			found := findVolumeStatMetric(kubeletKeyName, pvcNamespace, pvcName, kubeMetrics)
			if !found {
				framework.Failf("PVC %s, Namespace %s not found for %s", pvcName, pvcNamespace, kubeletKeyName)
			}
		}
	}

	blockmode := func(ctx context.Context, isEphemeral bool) {
		if !isEphemeral {
			pvcBlock, err = c.CoreV1().PersistentVolumeClaims(pvcBlock.Namespace).Create(ctx, pvcBlock, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pvcBlock).ToNot(gomega.BeNil())
		}

		pod := makePod(f, pvcBlock, isEphemeral)
		pod.Spec.Containers[0].VolumeDevices = []v1.VolumeDevice{{
			Name:       pod.Spec.Volumes[0].Name,
			DevicePath: "/mnt/" + pod.Spec.Volumes[0].Name,
		}}
		pod.Spec.Containers[0].VolumeMounts = nil
		pod = createPod(ctx, c, pod)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Verify volume stat metrics were collected for the referenced PVC
		volumeStatKeys := []string{
			// BlockMode PVCs only support capacity (for now)
			kubeletmetrics.VolumeStatsCapacityBytesKey,
		}
		key := volumeStatKeys[0]
		kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
		pvcName := pvcBlock.Name
		pvcNamespace := pvcBlock.Namespace
		if isEphemeral {
			pvcName = ephemeral.VolumeClaimName(pod, &pod.Spec.Volumes[0])
			pvcNamespace = pod.Namespace
		}
		// Poll kubelet metrics waiting for the volume to be picked up
		// by the volume stats collector
		var kubeMetrics e2emetrics.KubeletMetrics
		waitErr := wait.PollUntilContextTimeout(ctx, 30*time.Second, 5*time.Minute, false, func(ctx context.Context) (bool, error) {
			framework.Logf("Grabbing Kubelet metrics")
			// Grab kubelet metrics from the node the pod was scheduled on
			var err error
			kubeMetrics, err = metricsGrabber.GrabFromKubelet(ctx, pod.Spec.NodeName)
			if err != nil {
				framework.Logf("Error fetching kubelet metrics")
				return false, err
			}
			if !findVolumeStatMetric(kubeletKeyName, pvcNamespace, pvcName, kubeMetrics) {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(waitErr, "Unable to find metric %s for PVC %s/%s", kubeletKeyName, pvcNamespace, pvcName)

		for _, key := range volumeStatKeys {
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			found := findVolumeStatMetric(kubeletKeyName, pvcNamespace, pvcName, kubeMetrics)
			if !found {
				framework.Failf("PVC %s, Namespace %s not found for %s", pvcName, pvcNamespace, kubeletKeyName)
			}
		}
	}

	totalTime := func(ctx context.Context, isEphemeral bool) {
		if !isEphemeral {
			pvc = createPVC(ctx, c, pvc)
		}

		pod := makePod(f, pvc, isEphemeral)
		pod = createPod(ctx, c, pod)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
		if err != nil {
			e2eskipper.Skipf("Could not get controller-manager metrics - skipping")
		}

		metricKey := "volume_operation_total_seconds_count"
		verifyMetric(metricKey, map[testutil.LabelName]testutil.LabelValue{
			"operation_name": "provision",
			"plugin_name":    testutil.LabelValue(sc.Provisioner),
		}, testutil.Metrics(controllerMetrics))
	}

	volumeManager := func(ctx context.Context, isEphemeral bool) {
		if !isEphemeral {
			pvc = createPVC(ctx, c, pvc)
		}

		pod := makePod(f, pvc, isEphemeral)
		pod = createPod(ctx, c, pod)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		kubeMetrics, err := metricsGrabber.GrabFromKubelet(ctx, pod.Spec.NodeName)
		framework.ExpectNoError(err)

		// Metrics should have dimensions plugin_name and state available
		totalVolumesKey := "volume_manager_total_volumes"
		verifyMetric(totalVolumesKey, map[testutil.LabelName]testutil.LabelValue{
			"state":       "actual_state_of_world",
			"plugin_name": testutil.LabelValue("kubernetes.io/csi:" + sc.Provisioner),
		}, testutil.Metrics(kubeMetrics))
	}

	adController := func(ctx context.Context, isEphemeral bool) {
		if !isEphemeral {
			pvc = createPVC(ctx, c, pvc)
		}

		pod := makePod(f, pvc, isEphemeral)

		// Get metrics
		controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
		if err != nil {
			e2eskipper.Skipf("Could not get controller-manager metrics - skipping")
		}

		// Create pod
		pod = createPod(ctx, c, pod)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, c, pod.Name, pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Error starting pod %s", pod.Name)

		// Get updated metrics
		updatedControllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
		if err != nil {
			e2eskipper.Skipf("Could not get controller-manager metrics - skipping")
		}

		// Wait and validate
		totalVolumesKey := "attachdetach_controller_total_volumes"
		states := []string{"actual_state_of_world", "desired_state_of_world"}
		pluginName := "kubernetes.io/csi:" + sc.Provisioner

		// Total number of volumes in both ActualStateofWorld and DesiredStateOfWorld
		// states should be 1 plus it used to be
		oldStates := getStatesMetrics(totalVolumesKey, testutil.Metrics(controllerMetrics), pluginName)
		updatedStates := getStatesMetrics(totalVolumesKey, testutil.Metrics(updatedControllerMetrics), pluginName)
		for _, stateName := range states {
			gomega.Expect(updatedStates[stateName]).To(gomega.Equal(oldStates[stateName] + 1))
		}
	}

	testAll := func(isEphemeral bool) {
		ginkgo.It("should create prometheus metrics for volume provisioning and attach/detach", func(ctx context.Context) {
			provisioning(ctx, isEphemeral)
		})
		ginkgo.It("should create volume metrics with the correct FilesystemMode PVC ref", func(ctx context.Context) {
			filesystemMode(ctx, isEphemeral)
		})
		ginkgo.It("should create volume metrics with the correct BlockMode PVC ref", func(ctx context.Context) {
			blockmode(ctx, isEphemeral)
		})
		ginkgo.It("should create metrics for total time taken in volume operations in P/V Controller", func(ctx context.Context) {
			totalTime(ctx, isEphemeral)
		})
		ginkgo.It("should create volume metrics in Volume Manager", func(ctx context.Context) {
			volumeManager(ctx, isEphemeral)
		})
		ginkgo.It("should create metrics for total number of volumes in A/D Controller", func(ctx context.Context) {
			adController(ctx, isEphemeral)
		})
	}

	ginkgo.Context("PVC", func() {
		testAll(false)
	})

	ginkgo.Context("Ephemeral", func() {
		testAll(true)
	})

	// Test for pv controller metrics, concretely: bound/unbound pv/pvc count.
	ginkgo.Describe("PVController", func() {
		const (
			namespaceKey            = "namespace"
			pluginNameKey           = "plugin_name"
			volumeModeKey           = "volume_mode"
			storageClassKey         = "storage_class"
			volumeAttributeClassKey = "volume_attributes_class"

			totalPVKey    = "pv_collector_total_pv_count"
			boundPVKey    = "pv_collector_bound_pv_count"
			unboundPVKey  = "pv_collector_unbound_pv_count"
			boundPVCKey   = "pv_collector_bound_pvc_count"
			unboundPVCKey = "pv_collector_unbound_pvc_count"
		)

		type mvs struct {
			boundPV, unboundPV, boundPVC, unboundPVC int64
		}

		var (
			pv  *v1.PersistentVolume
			pvc *v1.PersistentVolumeClaim

			storageClassName          string
			pvConfig                  e2epv.PersistentVolumeConfig
			volumeAttributesClassName string
			pvcConfig                 e2epv.PersistentVolumeClaimConfig
		)

		// validator used to validate each metric's values, the length of metricValues
		validator := func(ctx context.Context, metric mvs) {
			gomega.Eventually(ctx, func(ctx context.Context) mvs {
				controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
				framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)
				ms := testutil.Metrics(controllerMetrics)
				return mvs{
					testutil.GetMetricValuesForLabel(ms, boundPVKey, storageClassKey)[storageClassName],
					testutil.GetMetricValuesForLabel(ms, unboundPVKey, storageClassKey)[storageClassName],
					testutil.GetMetricValuesForLabel(ms, boundPVCKey, namespaceKey)[ns],
					testutil.GetMetricValuesForLabel(ms, unboundPVCKey, namespaceKey)[ns],
				}
			}).WithPolling(2 * time.Second).WithTimeout(1 * time.Minute).Should(gomega.Equal(metric))
		}

		ginkgo.BeforeEach(func(ctx context.Context) {
			storageClassName = "bound-unbound-count-test-sc-" + f.UniqueName
			pvConfig = e2epv.PersistentVolumeConfig{
				PVSource: v1.PersistentVolumeSource{
					HostPath: &v1.HostPathVolumeSource{Path: "/data"},
				},
				NamePrefix:       "pv-test-",
				StorageClassName: storageClassName,
			}
			// TODO: Insert volumeAttributesClassName into pvcConfig when "VolumeAttributesClass" is GA
			volumeAttributesClassName = "bound-unbound-count-test-vac-" + f.UniqueName
			pvcConfig = e2epv.PersistentVolumeClaimConfig{StorageClassName: &storageClassName}

			if !metricsGrabber.HasControlPlanePods() {
				e2eskipper.Skipf("Environment does not support getting controller-manager metrics - skipping")
			}

			pv = e2epv.MakePersistentVolume(pvConfig)
			pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, ns)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if err := e2epv.DeletePersistentVolume(ctx, c, pv.Name); err != nil {
				framework.Failf("Error deleting pv: %v", err)
			}
			if err := e2epv.DeletePersistentVolumeClaim(ctx, c, pvc.Name, pvc.Namespace); err != nil {
				framework.Failf("Error deleting pvc: %v", err)
			}
		})

		ginkgo.It("should create none metrics for pvc controller before creating any PV or PVC", func(ctx context.Context) {
			validator(ctx, mvs{0, 0, 0, 0})
		})

		ginkgo.It("should create unbound pv count metrics for pvc controller after creating pv only",
			func(ctx context.Context) {
				var err error
				pv, err = e2epv.CreatePV(ctx, c, f.Timeouts, pv)
				framework.ExpectNoError(err, "Error creating pv: %v", err)
				validator(ctx, mvs{0, 1, 0, 0})
			})

		ginkgo.It("should create unbound pvc count metrics for pvc controller after creating pvc only",
			func(ctx context.Context) {
				var err error
				pvc, err = e2epv.CreatePVC(ctx, c, ns, pvc)
				framework.ExpectNoError(err, "Error creating pvc: %v", err)
				validator(ctx, mvs{0, 0, 0, 1})
			})

		ginkgo.It("should create bound pv/pvc count metrics for pvc controller after creating both pv and pvc",
			func(ctx context.Context) {
				var err error
				pv, pvc, err = e2epv.CreatePVPVC(ctx, c, f.Timeouts, pvConfig, pvcConfig, ns, true)
				framework.ExpectNoError(err, "Error creating pv pvc: %v", err)
				validator(ctx, mvs{1, 0, 1, 0})
			})

		// TODO: Merge with bound/unbound tests when "VolumeAttributesClass" feature is enabled by default.
		f.It("should create unbound pvc count metrics for pvc controller with volume attributes class dimension after creating pvc only", framework.WithFeatureGate(features.VolumeAttributesClass), feature.VolumeAttributesClass, func(ctx context.Context) {
			var err error
			dimensions := []string{namespaceKey, storageClassKey, volumeAttributeClassKey}
			pvcConfigWithVAC := pvcConfig
			pvcConfigWithVAC.VolumeAttributesClassName = &volumeAttributesClassName
			pvcWithVAC := e2epv.MakePersistentVolumeClaim(pvcConfigWithVAC, ns)
			pvc, err = e2epv.CreatePVC(ctx, c, ns, pvcWithVAC)
			framework.ExpectNoError(err, "Error creating pvc: %v", err)
			waitForPVControllerSync(ctx, metricsGrabber, unboundPVCKey, volumeAttributeClassKey)
			controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
			framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)
			err = testutil.ValidateMetrics(testutil.Metrics(controllerMetrics), unboundPVCKey, dimensions...)
			framework.ExpectNoError(err, "Invalid metric in Controller Manager metrics: %q", unboundPVCKey)
		})

		// TODO: Merge with bound/unbound tests when "VolumeAttributesClass" feature is enabled by default
		f.It("should create bound pv/pvc count metrics for pvc controller with volume attributes class dimension after creating both pv and pvc", framework.WithFeatureGate(features.VolumeAttributesClass), feature.VolumeAttributesClass, func(ctx context.Context) {
			var err error
			dimensions := []string{namespaceKey, storageClassKey, volumeAttributeClassKey}
			pvcConfigWithVAC := pvcConfig
			pvcConfigWithVAC.VolumeAttributesClassName = &volumeAttributesClassName
			pv, pvc, err = e2epv.CreatePVPVC(ctx, c, f.Timeouts, pvConfig, pvcConfigWithVAC, ns, true)
			framework.ExpectNoError(err, "Error creating pv pvc: %v", err)
			waitForPVControllerSync(ctx, metricsGrabber, boundPVKey, storageClassKey)
			waitForPVControllerSync(ctx, metricsGrabber, boundPVCKey, volumeAttributeClassKey)
			controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
			framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)
			err = testutil.ValidateMetrics(testutil.Metrics(controllerMetrics), boundPVCKey, dimensions...)
			framework.ExpectNoError(err, "Invalid metric in Controller Manager metrics: %q", boundPVCKey)
		})

		ginkgo.It("should create total pv count metrics for with plugin and volume mode labels after creating pv",
			func(ctx context.Context) {
				var err error
				dimensions := []string{pluginNameKey, volumeModeKey}
				pv, err = e2epv.CreatePV(ctx, c, f.Timeouts, pv)
				framework.ExpectNoError(err, "Error creating pv: %v", err)
				waitForPVControllerSync(ctx, metricsGrabber, totalPVKey, pluginNameKey)
				controllerMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
				framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)
				err = testutil.ValidateMetrics(testutil.Metrics(controllerMetrics), totalPVKey, dimensions...)
				framework.ExpectNoError(err, "Invalid metric in Controller Manager metrics: %q", totalPVKey)
			})
	})
})

type storageControllerMetrics struct {
	latencyMetrics map[string]int64
	statusMetrics  map[string]statusMetricCounts
}

type statusMetricCounts struct {
	successCount int64
	failCount    int64
	otherCount   int64
}

func newStorageControllerMetrics() *storageControllerMetrics {
	return &storageControllerMetrics{
		latencyMetrics: make(map[string]int64),
		statusMetrics:  make(map[string]statusMetricCounts),
	}
}

func waitForDetachAndGrabMetrics(ctx context.Context, oldMetrics *storageControllerMetrics, metricsGrabber *e2emetrics.Grabber, pluginName string) *storageControllerMetrics {
	backoff := wait.Backoff{
		Duration: 10 * time.Second,
		Factor:   1.2,
		Steps:    21,
	}

	updatedStorageMetrics := newStorageControllerMetrics()
	oldDetachCount, ok := oldMetrics.latencyMetrics["volume_detach"]
	if !ok {
		oldDetachCount = 0
	}

	verifyMetricFunc := func(ctx context.Context) (bool, error) {
		updatedMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)

		if err != nil {
			framework.Logf("Error fetching controller-manager metrics")
			return false, err
		}

		updatedStorageMetrics = getControllerStorageMetrics(updatedMetrics, pluginName)
		newDetachCount, ok := updatedStorageMetrics.latencyMetrics["volume_detach"]

		// if detach metrics are not yet there, we need to retry
		if !ok {
			framework.Logf("Detach metrics not found yet")
			return false, nil
		}

		// if old Detach count is more or equal to new detach count, that means detach
		// event has not been observed yet.
		if oldDetachCount >= newDetachCount {
			return false, nil
		}

		return true, nil
	}

	waitErr := wait.ExponentialBackoffWithContext(ctx, backoff, verifyMetricFunc)
	framework.ExpectNoError(waitErr, "Unable to get updated metrics for plugin %s", pluginName)
	return updatedStorageMetrics
}

func verifyMetricCount(oldMetrics, newMetrics *storageControllerMetrics, metricName string, expectFailure bool) {
	oldLatencyCount, ok := oldMetrics.latencyMetrics[metricName]
	// if metric does not exist in oldMap, it probably hasn't been emitted yet.
	if !ok {
		oldLatencyCount = 0
	}

	oldStatusCount := int64(0)
	if oldStatusCounts, ok := oldMetrics.statusMetrics[metricName]; ok {
		if expectFailure {
			oldStatusCount = oldStatusCounts.failCount
		} else {
			oldStatusCount = oldStatusCounts.successCount
		}
	}

	newLatencyCount, ok := newMetrics.latencyMetrics[metricName]
	if !expectFailure {
		if !ok {
			framework.Failf("Error getting updated latency metrics for %s", metricName)
		}
	}
	newStatusCounts, ok := newMetrics.statusMetrics[metricName]
	if !ok {
		framework.Failf("Error getting updated status metrics for %s", metricName)
	}

	newStatusCount := int64(0)
	if expectFailure {
		newStatusCount = newStatusCounts.failCount
	} else {
		newStatusCount = newStatusCounts.successCount
	}

	// It appears that in a busy cluster some spurious detaches are unavoidable
	// even if the test is run serially.  We really just verify if new count
	// is greater than old count
	if !expectFailure {
		gomega.Expect(newLatencyCount).To(gomega.BeNumerically(">", oldLatencyCount), "New latency count %d should be more than old count %d for action %s", newLatencyCount, oldLatencyCount, metricName)
	}
	gomega.Expect(newStatusCount).To(gomega.BeNumerically(">", oldStatusCount), "New status count %d should be more than old count %d for action %s", newStatusCount, oldStatusCount, metricName)
}

func getControllerStorageMetrics(ms e2emetrics.ControllerManagerMetrics, pluginName string) *storageControllerMetrics {
	result := newStorageControllerMetrics()

	for method, samples := range ms {
		switch method {
		// from the base metric name "storage_operation_duration_seconds"
		case "storage_operation_duration_seconds_count":
			for _, sample := range samples {
				count := int64(sample.Value)
				operation := string(sample.Metric["operation_name"])
				// if the volumes were provisioned with a CSI Driver
				// the metric operation name will be prefixed with
				// "kubernetes.io/csi:"
				metricPluginName := string(sample.Metric["volume_plugin"])
				status := string(sample.Metric["status"])
				if !strings.Contains(metricPluginName, pluginName) {
					// the metric volume plugin field doesn't match
					// the default storageClass.Provisioner field
					continue
				}

				statusCounts := result.statusMetrics[operation]
				switch status {
				case "success":
					statusCounts.successCount = count
				case "fail-unknown":
					statusCounts.failCount = count
				default:
					statusCounts.otherCount = count
				}
				result.statusMetrics[operation] = statusCounts
				result.latencyMetrics[operation] = count
			}
		}
	}
	return result
}

func findMetric(metricKeyName string, labels map[testutil.LabelName]testutil.LabelValue, metrics testutil.Metrics) bool {
	found := false
	errCount := 0
	if samples, ok := metrics[metricKeyName]; ok {
	samples:
		for _, sample := range samples {
			for k, v := range labels {
				got, ok := sample.Metric[k]
				if !ok {
					framework.Logf("sample %v missing key %s", sample, k)
					errCount++
					continue samples
				} else if got != v {
					continue samples
				}
			}
			found = true
			framework.Logf("Found expected sample: %v", sample)
			break
		}
	}
	gomega.Expect(errCount).To(gomega.Equal(0), "Found invalid samples")
	return found
}

func verifyMetric(metricKeyName string, labels map[testutil.LabelName]testutil.LabelValue, metrics testutil.Metrics) {
	ginkgo.GinkgoHelper()
	found := findMetric(metricKeyName, labels, metrics)
	gomega.Expect(found).To(gomega.BeTrueBecause("Failed to find metric %s with labels %v. Got %v", metricKeyName, labels, metrics[metricKeyName]))
}

// Finds the sample in the specified metric from `KubeletMetrics` tagged with
// the specified namespace and pvc name
func findVolumeStatMetric(metricKeyName string, namespace string, pvcName string, kubeletMetrics e2emetrics.KubeletMetrics) bool {
	framework.Logf("Looking for sample in metric `%s` tagged with namespace `%s`, PVC `%s`", metricKeyName, namespace, pvcName)
	return findMetric(metricKeyName, map[testutil.LabelName]testutil.LabelValue{
		"namespace":             testutil.LabelValue(namespace),
		"persistentvolumeclaim": testutil.LabelValue(pvcName),
	}, testutil.Metrics(kubeletMetrics))
}

// Wait for the count of a pv controller's metric specified by metricName and dimension bigger than zero.
func waitForPVControllerSync(ctx context.Context, metricsGrabber *e2emetrics.Grabber, metricName, dimension string) {
	backoff := wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   1.2,
		Steps:    21,
	}
	verifyMetricFunc := func(ctx context.Context) (bool, error) {
		updatedMetrics, err := metricsGrabber.GrabFromControllerManager(ctx)
		if err != nil {
			framework.Logf("Error fetching controller-manager metrics")
			return false, err
		}
		return len(testutil.GetMetricValuesForLabel(testutil.Metrics(updatedMetrics), metricName, dimension)) > 0, nil
	}
	waitErr := wait.ExponentialBackoffWithContext(ctx, backoff, verifyMetricFunc)
	framework.ExpectNoError(waitErr, "Unable to get pv controller metrics")
}

func getStatesMetrics(metricKey string, givenMetrics testutil.Metrics, pluginName string) map[string]int64 {
	states := make(map[string]int64)
	for _, sample := range givenMetrics[metricKey] {
		if string(sample.Metric["plugin_name"]) != pluginName {
			continue
		}
		framework.Logf("Found sample %q", sample.String())
		state := string(sample.Metric["state"])
		states[state] = int64(sample.Value)
	}
	return states
}

// makePod creates a pod which either references the PVC or creates it via a
// generic ephemeral volume claim template.
func makePod(f *framework.Framework, pvc *v1.PersistentVolumeClaim, isEphemeral bool) *v1.Pod {
	claims := []*v1.PersistentVolumeClaim{pvc}
	pod := e2epod.MakePod(f.Namespace.Name, nil, claims, f.NamespacePodSecurityLevel, "")
	if isEphemeral {
		volSrc := pod.Spec.Volumes[0]
		volSrc.PersistentVolumeClaim = nil
		volSrc.Ephemeral = &v1.EphemeralVolumeSource{
			VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
				Spec: pvc.Spec,
			},
		}
		pod.Spec.Volumes[0] = volSrc
	}
	return pod
}
