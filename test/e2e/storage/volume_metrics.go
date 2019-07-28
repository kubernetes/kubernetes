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
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/model"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// This test needs to run in serial because other tests could interfere
// with metrics being tested here.
var _ = utils.SIGDescribe("[Serial] Volume metrics", func() {
	var (
		c              clientset.Interface
		ns             string
		pvc            *v1.PersistentVolumeClaim
		metricsGrabber *metrics.Grabber
		invalidSc      *storagev1.StorageClass
		defaultScName  string
	)
	f := framework.NewDefaultFramework("pv")

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		var err error
		framework.SkipUnlessProviderIs("gce", "gke", "aws")
		defaultScName, err = framework.GetDefaultStorageClassName(c)
		if err != nil {
			e2elog.Failf(err.Error())
		}
		test := testsuites.StorageClassTest{
			Name:      "default",
			ClaimSize: "2Gi",
		}

		pvc = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
			ClaimSize:  test.ClaimSize,
			VolumeMode: &test.VolumeMode,
		}, ns)

		metricsGrabber, err = metrics.NewMetricsGrabber(c, nil, true, false, true, false, false)

		if err != nil {
			e2elog.Failf("Error creating metrics grabber : %v", err)
		}
	})

	ginkgo.AfterEach(func() {
		newPvc, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Failed to get pvc %s/%s: %v", pvc.Namespace, pvc.Name, err)
		} else {
			framework.DeletePersistentVolumeClaim(c, newPvc.Name, newPvc.Namespace)
			if newPvc.Spec.VolumeName != "" {
				err = framework.WaitForPersistentVolumeDeleted(c, newPvc.Spec.VolumeName, 5*time.Second, 5*time.Minute)
				framework.ExpectNoError(err, "Persistent Volume %v not deleted by dynamic provisioner", newPvc.Spec.VolumeName)
			}
		}

		if invalidSc != nil {
			err := c.StorageV1().StorageClasses().Delete(invalidSc.Name, nil)
			framework.ExpectNoError(err, "Error deleting storageclass %v: %v", invalidSc.Name, err)
			invalidSc = nil
		}
	})

	ginkgo.It("should create prometheus metrics for volume provisioning and attach/detach", func() {
		var err error

		if !metricsGrabber.HasRegisteredMaster() {
			framework.Skipf("Environment does not support getting controller-manager metrics - skipping")
		}

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()

		framework.ExpectNoError(err, "Error getting c-m metrics : %v", err)

		storageOpMetrics := getControllerStorageMetrics(controllerMetrics)

		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}

		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, pod), "Error starting pod %s", pod.Name)

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))

		updatedStorageMetrics := waitForDetachAndGrabMetrics(storageOpMetrics, metricsGrabber)

		framework.ExpectNotEqual(len(updatedStorageMetrics.latencyMetrics), 0, "Error fetching c-m updated storage metrics")
		framework.ExpectNotEqual(len(updatedStorageMetrics.statusMetrics), 0, "Error fetching c-m updated storage metrics")

		volumeOperations := []string{"volume_provision", "volume_detach", "volume_attach"}

		for _, volumeOp := range volumeOperations {
			verifyMetricCount(storageOpMetrics, updatedStorageMetrics, volumeOp, false)
		}
	})

	ginkgo.It("should create prometheus metrics for volume provisioning errors [Slow]", func() {
		var err error

		if !metricsGrabber.HasRegisteredMaster() {
			framework.Skipf("Environment does not support getting controller-manager metrics - skipping")
		}

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()

		framework.ExpectNoError(err, "Error getting c-m metrics : %v", err)

		storageOpMetrics := getControllerStorageMetrics(controllerMetrics)

		ginkgo.By("Creating an invalid storageclass")
		defaultClass, err := c.StorageV1().StorageClasses().Get(defaultScName, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error getting default storageclass: %v", err)

		invalidSc = &storagev1.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("fail-metrics-invalid-sc-%s", pvc.Namespace),
			},
			Provisioner: defaultClass.Provisioner,
			Parameters: map[string]string{
				"invalidparam": "invalidvalue",
			},
		}
		_, err = c.StorageV1().StorageClasses().Create(invalidSc)
		framework.ExpectNoError(err, "Error creating new storageclass: %v", err)

		pvc.Spec.StorageClassName = &invalidSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err, "failed to create PVC %s/%s", pvc.Namespace, pvc.Name)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}

		ginkgo.By("Creating a pod and expecting it to fail")
		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err, "failed to create Pod %s/%s", pod.Namespace, pod.Name)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(c, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
		framework.ExpectError(err)

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))

		ginkgo.By("Checking failure metrics")
		updatedControllerMetrics, err := metricsGrabber.GrabFromControllerManager()
		framework.ExpectNoError(err, "failed to get controller manager metrics")
		updatedStorageMetrics := getControllerStorageMetrics(updatedControllerMetrics)

		framework.ExpectNotEqual(len(updatedStorageMetrics.statusMetrics), 0, "Error fetching c-m updated storage metrics")
		verifyMetricCount(storageOpMetrics, updatedStorageMetrics, "volume_provision", true)
	})

	ginkgo.It("should create volume metrics with the correct PVC ref", func() {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Verify volume stat metrics were collected for the referenced PVC
		volumeStatKeys := []string{
			kubeletmetrics.VolumeStatsUsedBytesKey,
			kubeletmetrics.VolumeStatsCapacityBytesKey,
			kubeletmetrics.VolumeStatsAvailableBytesKey,
			kubeletmetrics.VolumeStatsUsedBytesKey,
			kubeletmetrics.VolumeStatsInodesFreeKey,
			kubeletmetrics.VolumeStatsInodesUsedKey,
		}
		// Poll kubelet metrics waiting for the volume to be picked up
		// by the volume stats collector
		var kubeMetrics metrics.KubeletMetrics
		waitErr := wait.Poll(30*time.Second, 5*time.Minute, func() (bool, error) {
			e2elog.Logf("Grabbing Kubelet metrics")
			// Grab kubelet metrics from the node the pod was scheduled on
			var err error
			kubeMetrics, err = metricsGrabber.GrabFromKubelet(pod.Spec.NodeName)
			if err != nil {
				e2elog.Logf("Error fetching kubelet metrics")
				return false, err
			}
			key := volumeStatKeys[0]
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			if !findVolumeStatMetric(kubeletKeyName, pvc.Namespace, pvc.Name, kubeMetrics) {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(waitErr, "Error finding volume metrics : %v", waitErr)

		for _, key := range volumeStatKeys {
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			found := findVolumeStatMetric(kubeletKeyName, pvc.Namespace, pvc.Name, kubeMetrics)
			gomega.Expect(found).To(gomega.BeTrue(), "PVC %s, Namespace %s not found for %s", pvc.Name, pvc.Namespace, kubeletKeyName)
		}

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
	})

	ginkgo.It("should create metrics for total time taken in volume operations in P/V Controller", func() {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
		if err != nil {
			framework.Skipf("Could not get controller-manager metrics - skipping")
		}

		metricKey := "volume_operation_total_seconds_count"
		dimensions := []string{"operation_name", "plugin_name"}
		valid := hasValidMetrics(metrics.Metrics(controllerMetrics), metricKey, dimensions...)
		gomega.Expect(valid).To(gomega.BeTrue(), "Invalid metric in P/V Controller metrics: %q", metricKey)

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
	})

	ginkgo.It("should create volume metrics in Volume Manager", func() {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err)

		err = e2epod.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		kubeMetrics, err := metricsGrabber.GrabFromKubelet(pod.Spec.NodeName)
		framework.ExpectNoError(err)

		// Metrics should have dimensions plugin_name and state available
		totalVolumesKey := "volume_manager_total_volumes"
		dimensions := []string{"state", "plugin_name"}
		valid := hasValidMetrics(metrics.Metrics(kubeMetrics), totalVolumesKey, dimensions...)
		gomega.Expect(valid).To(gomega.BeTrue(), "Invalid metric in Volume Manager metrics: %q", totalVolumesKey)

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
	})

	ginkgo.It("should create metrics for total number of volumes in A/D Controller", func() {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		framework.ExpectNotEqual(pvc, nil)

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, nil, claims, false, "")

		// Get metrics
		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
		if err != nil {
			framework.Skipf("Could not get controller-manager metrics - skipping")
		}

		// Create pod
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err)
		err = e2epod.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)
		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get updated metrics
		updatedControllerMetrics, err := metricsGrabber.GrabFromControllerManager()
		if err != nil {
			framework.Skipf("Could not get controller-manager metrics - skipping")
		}

		// Forced detach metric should be present
		forceDetachKey := "attachdetach_controller_forced_detaches"
		_, ok := updatedControllerMetrics[forceDetachKey]
		gomega.Expect(ok).To(gomega.BeTrue(), "Key %q not found in A/D Controller metrics", forceDetachKey)

		// Wait and validate
		totalVolumesKey := "attachdetach_controller_total_volumes"
		states := []string{"actual_state_of_world", "desired_state_of_world"}
		dimensions := []string{"state", "plugin_name"}
		waitForADControllerStatesMetrics(metricsGrabber, totalVolumesKey, dimensions, states)

		// Total number of volumes in both ActualStateofWorld and DesiredStateOfWorld
		// states should be higher or equal than it used to be
		oldStates := getStatesMetrics(totalVolumesKey, metrics.Metrics(controllerMetrics))
		updatedStates := getStatesMetrics(totalVolumesKey, metrics.Metrics(updatedControllerMetrics))
		for _, stateName := range states {
			if _, ok := oldStates[stateName]; !ok {
				continue
			}
			for pluginName, numVolumes := range updatedStates[stateName] {
				oldNumVolumes := oldStates[stateName][pluginName]
				gomega.Expect(numVolumes).To(gomega.BeNumerically(">=", oldNumVolumes),
					"Wrong number of volumes in state %q, plugin %q: wanted >=%d, got %d",
					stateName, pluginName, oldNumVolumes, numVolumes)
			}
		}

		e2elog.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
	})

	// Test for pv controller metrics, concretely: bound/unbound pv/pvc count.
	ginkgo.Describe("PVController", func() {
		const (
			classKey     = "storage_class"
			namespaceKey = "namespace"

			boundPVKey    = "pv_collector_bound_pv_count"
			unboundPVKey  = "pv_collector_unbound_pv_count"
			boundPVCKey   = "pv_collector_bound_pvc_count"
			unboundPVCKey = "pv_collector_unbound_pvc_count"
		)

		var (
			pv  *v1.PersistentVolume
			pvc *v1.PersistentVolumeClaim

			className = "bound-unbound-count-test-sc"
			pvConfig  = framework.PersistentVolumeConfig{
				PVSource: v1.PersistentVolumeSource{
					HostPath: &v1.HostPathVolumeSource{Path: "/data"},
				},
				NamePrefix:       "pv-test-",
				StorageClassName: className,
			}
			pvcConfig = framework.PersistentVolumeClaimConfig{StorageClassName: &className}

			metrics = []struct {
				name      string
				dimension string
			}{
				{boundPVKey, classKey},
				{unboundPVKey, classKey},
				{boundPVCKey, namespaceKey},
				{unboundPVCKey, namespaceKey},
			}

			// Original metric values before we create any PV/PVCs. The length should be 4,
			// and the elements should be bound pv count, unbound pv count, bound pvc count,
			// unbound pvc count in turn.
			// We use these values to calculate relative increment of each test.
			originMetricValues []map[string]int64
		)

		// validator used to validate each metric's values, the length of metricValues
		// should be 4, and the elements should be bound pv count, unbound pv count, bound
		// pvc count, unbound pvc count in turn.
		validator := func(metricValues []map[string]int64) {
			framework.ExpectEqual(len(metricValues), 4, "Wrong metric size: %d", len(metricValues))

			controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
			framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)

			for i, metric := range metrics {
				expectValues := metricValues[i]
				if expectValues == nil {
					expectValues = make(map[string]int64)
				}
				// We using relative increment value instead of absolute value to reduce unexpected flakes.
				// Concretely, we expect the difference of the updated values and original values for each
				// test suit are equal to expectValues.
				actualValues := calculateRelativeValues(originMetricValues[i],
					getPVControllerMetrics(controllerMetrics, metric.name, metric.dimension))
				framework.ExpectEqual(actualValues, expectValues, "Wrong pv controller metric %s(%s): wanted %v, got %v",
					metric.name, metric.dimension, expectValues, actualValues)
			}
		}

		ginkgo.BeforeEach(func() {
			if !metricsGrabber.HasRegisteredMaster() {
				framework.Skipf("Environment does not support getting controller-manager metrics - skipping")
			}

			pv = framework.MakePersistentVolume(pvConfig)
			pvc = framework.MakePersistentVolumeClaim(pvcConfig, ns)

			// Initializes all original metric values.
			controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
			framework.ExpectNoError(err, "Error getting c-m metricValues: %v", err)
			for _, metric := range metrics {
				originMetricValues = append(originMetricValues,
					getPVControllerMetrics(controllerMetrics, metric.name, metric.dimension))
			}
		})

		ginkgo.AfterEach(func() {
			if err := framework.DeletePersistentVolume(c, pv.Name); err != nil {
				e2elog.Failf("Error deleting pv: %v", err)
			}
			if err := framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace); err != nil {
				e2elog.Failf("Error deleting pvc: %v", err)
			}

			// Clear original metric values.
			originMetricValues = nil
		})

		ginkgo.It("should create none metrics for pvc controller before creating any PV or PVC", func() {
			validator([]map[string]int64{nil, nil, nil, nil})
		})

		ginkgo.It("should create unbound pv count metrics for pvc controller after creating pv only",
			func() {
				var err error
				pv, err = framework.CreatePV(c, pv)
				framework.ExpectNoError(err, "Error creating pv: %v", err)
				waitForPVControllerSync(metricsGrabber, unboundPVKey, classKey)
				validator([]map[string]int64{nil, {className: 1}, nil, nil})
			})

		ginkgo.It("should create unbound pvc count metrics for pvc controller after creating pvc only",
			func() {
				var err error
				pvc, err = framework.CreatePVC(c, ns, pvc)
				framework.ExpectNoError(err, "Error creating pvc: %v", err)
				waitForPVControllerSync(metricsGrabber, unboundPVCKey, namespaceKey)
				validator([]map[string]int64{nil, nil, nil, {ns: 1}})
			})

		ginkgo.It("should create bound pv/pvc count metrics for pvc controller after creating both pv and pvc",
			func() {
				var err error
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, true)
				framework.ExpectNoError(err, "Error creating pv pvc: %v", err)
				waitForPVControllerSync(metricsGrabber, boundPVKey, classKey)
				waitForPVControllerSync(metricsGrabber, boundPVCKey, namespaceKey)
				validator([]map[string]int64{{className: 1}, nil, {ns: 1}, nil})

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

func waitForDetachAndGrabMetrics(oldMetrics *storageControllerMetrics, metricsGrabber *metrics.Grabber) *storageControllerMetrics {
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

	verifyMetricFunc := func() (bool, error) {
		updatedMetrics, err := metricsGrabber.GrabFromControllerManager()

		if err != nil {
			e2elog.Logf("Error fetching controller-manager metrics")
			return false, err
		}

		updatedStorageMetrics = getControllerStorageMetrics(updatedMetrics)
		newDetachCount, ok := updatedStorageMetrics.latencyMetrics["volume_detach"]

		// if detach metrics are not yet there, we need to retry
		if !ok {
			return false, nil
		}

		// if old Detach count is more or equal to new detach count, that means detach
		// event has not been observed yet.
		if oldDetachCount >= newDetachCount {
			return false, nil
		}
		return true, nil
	}

	waitErr := wait.ExponentialBackoff(backoff, verifyMetricFunc)
	framework.ExpectNoError(waitErr, "Timeout error fetching storage c-m metrics : %v", waitErr)
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
		gomega.Expect(ok).To(gomega.BeTrue(), "Error getting updated latency metrics for %s", metricName)
	}
	newStatusCounts, ok := newMetrics.statusMetrics[metricName]
	gomega.Expect(ok).To(gomega.BeTrue(), "Error getting updated status metrics for %s", metricName)

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

func getControllerStorageMetrics(ms metrics.ControllerManagerMetrics) *storageControllerMetrics {
	result := newStorageControllerMetrics()

	for method, samples := range ms {
		switch method {

		case "storage_operation_duration_seconds_count":
			for _, sample := range samples {
				count := int64(sample.Value)
				operation := string(sample.Metric["operation_name"])
				result.latencyMetrics[operation] = count
			}
		case "storage_operation_status_count":
			for _, sample := range samples {
				count := int64(sample.Value)
				operation := string(sample.Metric["operation_name"])
				status := string(sample.Metric["status"])
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
			}

		}
	}
	return result
}

// Finds the sample in the specified metric from `KubeletMetrics` tagged with
// the specified namespace and pvc name
func findVolumeStatMetric(metricKeyName string, namespace string, pvcName string, kubeletMetrics metrics.KubeletMetrics) bool {
	found := false
	errCount := 0
	e2elog.Logf("Looking for sample in metric `%s` tagged with namespace `%s`, PVC `%s`", metricKeyName, namespace, pvcName)
	if samples, ok := kubeletMetrics[metricKeyName]; ok {
		for _, sample := range samples {
			e2elog.Logf("Found sample %s", sample.String())
			samplePVC, ok := sample.Metric["persistentvolumeclaim"]
			if !ok {
				e2elog.Logf("Error getting pvc for metric %s, sample %s", metricKeyName, sample.String())
				errCount++
			}
			sampleNS, ok := sample.Metric["namespace"]
			if !ok {
				e2elog.Logf("Error getting namespace for metric %s, sample %s", metricKeyName, sample.String())
				errCount++
			}

			if string(samplePVC) == pvcName && string(sampleNS) == namespace {
				found = true
				break
			}
		}
	}
	framework.ExpectEqual(errCount, 0, "Found invalid samples")
	return found
}

// Wait for the count of a pv controller's metric specified by metricName and dimension bigger than zero.
func waitForPVControllerSync(metricsGrabber *metrics.Grabber, metricName, dimension string) {
	backoff := wait.Backoff{
		Duration: 10 * time.Second,
		Factor:   1.2,
		Steps:    21,
	}
	verifyMetricFunc := func() (bool, error) {
		updatedMetrics, err := metricsGrabber.GrabFromControllerManager()
		if err != nil {
			e2elog.Logf("Error fetching controller-manager metrics")
			return false, err
		}
		return len(getPVControllerMetrics(updatedMetrics, metricName, dimension)) > 0, nil
	}
	waitErr := wait.ExponentialBackoff(backoff, verifyMetricFunc)
	framework.ExpectNoError(waitErr,
		"Timeout error fetching pv controller metrics : %v", waitErr)
}

func getPVControllerMetrics(ms metrics.ControllerManagerMetrics, metricName, dimension string) map[string]int64 {
	result := make(map[string]int64)
	for method, samples := range ms {
		if method != metricName {
			continue
		}
		for _, sample := range samples {
			count := int64(sample.Value)
			dimensionName := string(sample.Metric[model.LabelName(dimension)])
			result[dimensionName] = count
		}
	}
	return result
}

func calculateRelativeValues(originValues, updatedValues map[string]int64) map[string]int64 {
	relativeValues := make(map[string]int64)
	for key, value := range updatedValues {
		relativeValue := value - originValues[key]
		if relativeValue != 0 {
			relativeValues[key] = relativeValue
		}
	}
	for key, value := range originValues {
		if _, exist := updatedValues[key]; !exist && value > 0 {
			relativeValues[key] = -value
		}
	}
	return relativeValues
}

func hasValidMetrics(metrics metrics.Metrics, metricKey string, dimensions ...string) bool {
	var errCount int
	e2elog.Logf("Looking for sample in metric %q", metricKey)
	samples, ok := metrics[metricKey]
	if !ok {
		e2elog.Logf("Key %q was not found in metrics", metricKey)
		return false
	}
	for _, sample := range samples {
		e2elog.Logf("Found sample %q", sample.String())
		for _, d := range dimensions {
			if _, ok := sample.Metric[model.LabelName(d)]; !ok {
				e2elog.Logf("Error getting dimension %q for metric %q, sample %q", d, metricKey, sample.String())
				errCount++
			}
		}
	}
	return errCount == 0
}

func getStatesMetrics(metricKey string, givenMetrics metrics.Metrics) map[string]map[string]int64 {
	states := make(map[string]map[string]int64)
	for _, sample := range givenMetrics[metricKey] {
		e2elog.Logf("Found sample %q", sample.String())
		state := string(sample.Metric["state"])
		pluginName := string(sample.Metric["plugin_name"])
		states[state] = map[string]int64{pluginName: int64(sample.Value)}
	}
	return states
}

func waitForADControllerStatesMetrics(metricsGrabber *metrics.Grabber, metricName string, dimensions []string, stateNames []string) {
	backoff := wait.Backoff{
		Duration: 10 * time.Second,
		Factor:   1.2,
		Steps:    21,
	}
	verifyMetricFunc := func() (bool, error) {
		updatedMetrics, err := metricsGrabber.GrabFromControllerManager()
		if err != nil {
			framework.Skipf("Could not get controller-manager metrics - skipping")
			return false, err
		}
		if !hasValidMetrics(metrics.Metrics(updatedMetrics), metricName, dimensions...) {
			return false, fmt.Errorf("could not get valid metrics for %q", metricName)
		}
		states := getStatesMetrics(metricName, metrics.Metrics(updatedMetrics))
		for _, name := range stateNames {
			if _, ok := states[name]; !ok {
				return false, fmt.Errorf("could not get state %q from A/D Controller metrics", name)
			}
		}
		return true, nil
	}
	waitErr := wait.ExponentialBackoff(backoff, verifyMetricFunc)
	framework.ExpectNoError(waitErr, "Timeout error fetching A/D controller metrics : %v", waitErr)
}
