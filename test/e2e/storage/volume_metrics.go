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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
)

// This test needs to run in serial because other tests could interfere
// with metrics being tested here.
var _ = SIGDescribe("[Serial] Volume metrics", func() {
	var (
		c              clientset.Interface
		ns             string
		pvc            *v1.PersistentVolumeClaim
		metricsGrabber *metrics.MetricsGrabber
	)
	f := framework.NewDefaultFramework("pv")

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.SkipUnlessProviderIs("gce", "gke", "aws")
		defaultScName := getDefaultStorageClassName(c)
		verifyDefaultStorageClass(c, defaultScName, true)

		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}

		pvc = newClaim(test, ns, "default")
		var err error
		metricsGrabber, err = metrics.NewMetricsGrabber(c, nil, true, false, true, false, false)

		if err != nil {
			framework.Failf("Error creating metrics grabber : %v", err)
		}

		if !metricsGrabber.HasRegisteredMaster() {
			framework.Skipf("Environment does not support getting controller-manager metrics - skipping")
		}
	})

	AfterEach(func() {
		framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace)
	})

	It("should create prometheus metrics for volume provisioning and attach/detach", func() {
		var err error

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()

		Expect(err).NotTo(HaveOccurred(), "Error getting c-m metrics : %v", err)

		storageOpMetrics := getControllerStorageMetrics(controllerMetrics)

		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred())
		Expect(pvc).ToNot(Equal(nil))

		claims := []*v1.PersistentVolumeClaim{pvc}

		pod := framework.MakePod(ns, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))

		backoff := wait.Backoff{
			Duration: 10 * time.Second,
			Factor:   1.2,
			Steps:    3,
		}

		updatedStorageMetrics := make(map[string]int64)

		waitErr := wait.ExponentialBackoff(backoff, func() (bool, error) {
			updatedMetrics, err := metricsGrabber.GrabFromControllerManager()

			if err != nil {
				framework.Logf("Error fetching controller-manager metrics")
				return false, err
			}
			updatedStorageMetrics = getControllerStorageMetrics(updatedMetrics)
			if len(updatedStorageMetrics) == 0 {
				framework.Logf("Volume metrics not collected yet, going to retry")
				return false, nil
			}
			return true, nil
		})
		Expect(waitErr).NotTo(HaveOccurred(), "Error fetching storage c-m metrics : %v", waitErr)

		volumeOperations := []string{"volume_provision", "volume_detach", "volume_attach"}

		for _, volumeOp := range volumeOperations {
			verifyMetricCount(storageOpMetrics, updatedStorageMetrics, volumeOp)
		}
	})

	It("should create volume metrics with the correct PVC ref", func() {
		var err error
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred())
		Expect(pvc).ToNot(Equal(nil))

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// Wait for `VolumeStatsAggPeriod' to grab metrics
		time.Sleep(1 * time.Minute)

		// Grab kubelet metrics from the node the pod was scheduled on
		kubeMetrics, err := metricsGrabber.GrabFromKubelet(pod.Spec.NodeName)
		Expect(err).NotTo(HaveOccurred(), "Error getting kubelet metrics : %v", err)

		framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))

		// Verify volume stat metrics were collected for the referenced PVC
		volumeStatKeys := []string{
			kubeletmetrics.VolumeStatsUsedBytesKey,
			kubeletmetrics.VolumeStatsCapacityBytesKey,
			kubeletmetrics.VolumeStatsAvailableBytesKey,
			kubeletmetrics.VolumeStatsUsedBytesKey,
			kubeletmetrics.VolumeStatsInodesFreeKey,
			kubeletmetrics.VolumeStatsInodesUsedKey,
		}

		for _, key := range volumeStatKeys {
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			verifyVolumeStatMetric(kubeletKeyName, pvc.Namespace, pvc.Name, kubeMetrics)
		}
	})
})

func verifyMetricCount(oldMetrics map[string]int64, newMetrics map[string]int64, metricName string) {
	oldCount, ok := oldMetrics[metricName]
	// if metric does not exist in oldMap, it probably hasn't been emitted yet.
	if !ok {
		oldCount = 0
	}

	newCount, ok := newMetrics[metricName]
	Expect(ok).To(BeTrue(), "Error getting updated metrics for %s", metricName)

	Expect(oldCount + 1).To(Equal(newCount))
}

func getControllerStorageMetrics(ms metrics.ControllerManagerMetrics) map[string]int64 {
	result := make(map[string]int64)

	for method, samples := range ms {
		if method != "storage_operation_duration_seconds_count" {
			continue
		}

		for _, sample := range samples {
			count := int64(sample.Value)
			operation := string(sample.Metric["operation_name"])
			result[operation] = count
		}
	}
	return result
}

// Verifies the specified metrics are in `kubeletMetrics`
func verifyVolumeStatMetric(metricKeyName string, namespace string, pvcName string, kubeletMetrics metrics.KubeletMetrics) {
	found := false
	errCount := 0
	if samples, ok := kubeletMetrics[metricKeyName]; ok {
		for _, sample := range samples {
			samplePVC, ok := sample.Metric["persistentvolumeclaim"]
			if !ok {
				framework.Logf("Error getting pvc for metric %s, sample %s", metricKeyName, sample.String())
				errCount++
			}
			sampleNS, ok := sample.Metric["namespace"]
			if !ok {
				framework.Logf("Error getting namespace for metric %s, sample %s", metricKeyName, sample.String())
				errCount++
			}

			if string(samplePVC) == pvcName && string(sampleNS) == namespace {
				found = true
				break
			}
		}
	}
	Expect(errCount).To(Equal(0), "Found invalid samples")
	Expect(found).To(BeTrue(), "PVC %s, Namespace %s not found for %s", pvcName, namespace, metricKeyName)
}
