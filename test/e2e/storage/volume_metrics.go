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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
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
		metricsGrabber, err = metrics.NewMetricsGrabber(c, nil, false, false, true, false, false)

		if err != nil {
			framework.Failf("Error creating metrics grabber : %v", err)
		}
	})

	It("should create prometheus metrics for volume provisioning and attach/detach", func() {
		var err error

		controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
		Expect(err).NotTo(HaveOccurred(), "Error getting c-m metrics : %v", err)

		storageOpMetrics := getControllerStorageMetrics(controllerMetrics)

		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred())
		Expect(pvc).ToNot(Equal(nil))
		defer func() {
			framework.Logf("Deleting claim %q/%q", pvc.Namespace, pvc.Name)
			framework.ExpectNoError(c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, nil))
		}()

		claims := []*v1.PersistentVolumeClaim{pvc}

		pod := framework.MakePod(ns, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))

		updatedMetrics, err := metricsGrabber.GrabFromControllerManager()
		Expect(err).NotTo(HaveOccurred(), "Error getting c-m metrics : %v", err)
		updatedStorageMetrics := getControllerStorageMetrics(updatedMetrics)
		volumeOperations := []string{"volume_provision", "volume_detach", "volume_attach"}

		for _, volumeOp := range volumeOperations {
			verifyMetricCount(storageOpMetrics, updatedStorageMetrics, volumeOp)
		}
	})
})

func verifyMetricCount(oldMetrics map[string]int64, newMetrics map[string]int64, metricName string) {
	oldCount, ok := oldMetrics[metricName]
	Expect(ok).To(BeTrue(), "Error getting metrics for %s", metricName)

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
