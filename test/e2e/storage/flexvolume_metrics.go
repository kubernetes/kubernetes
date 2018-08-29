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
	"path"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	metricspkg "k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// This test needs to run in serial because other tests could interfere
// with metrics being tested here.
var _ = utils.SIGDescribe("[Serial] Flexvolume metrics", func() {
	const (
		classKey      = "storage_class"
		namespaceKey  = "namespace"
		boundPVKey    = "pv_collector_bound_pv_count"
		unboundPVKey  = "pv_collector_unbound_pv_count"
		boundPVCKey   = "pv_collector_bound_pvc_count"
		unboundPVCKey = "pv_collector_unbound_pvc_count"
	)
	var (
		c                 clientset.Interface
		ns                string
		pv                *v1.PersistentVolume
		pvc               *v1.PersistentVolumeClaim
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
		node              v1.Node
		suffix            string
		pvcClaims         []*v1.PersistentVolumeClaim
		metricsGrabber    *metricspkg.MetricsGrabber
	)

	f := framework.NewDefaultFramework("pv")

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.SkipUnlessProviderIs("gce", "gke", "aws", "local")

		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) != 0 {
			nodeName = nodeList.Items[0].Name
		} else {
			framework.Failf("Unable to find ready and schedulable Node")
		}

		nodeKey = "mounted_volume_expand"
		node = nodeList.Items[0]

		if !isNodeLabeled {
			nodeLabelValue = ns
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel[nodeKey] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
			isNodeLabeled = true
		}

		suffix = ns

		driver := "dummy-attachable-metrics"
		driverInstallAs := driver + "-" + suffix

		By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(c, &node, "k8s", driverInstallAs, path.Join(driverDir, driver))

		pv = &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "flex-volume-0",
				Annotations: map[string]string{
					"volume.beta.kubernetes.io/storage-class": "",
				},
			},
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("6Gi"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					FlexVolume: &v1.FlexPersistentVolumeSource{
						Driver:   "k8s/" + driverInstallAs,
						ReadOnly: true,
					},
				},
			},
		}
		var err error
		pv, err = c.CoreV1().PersistentVolumes().Create(pv)
		framework.ExpectNoError(err)

		pvc = &v1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "my-claim",
				Namespace: ns,
				Annotations: map[string]string{
					"volume.beta.kubernetes.io/storage-class": "",
				},
			},
			Spec: v1.PersistentVolumeClaimSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						"storage": resource.MustParse("2Gi"),
					},
				},
				VolumeName: "flex-volume-0",
			},
		}

		//pvc = newClaim(test, ns.Name, "default")
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc")

		pvcClaims = []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionShortTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		metricsGrabber, err = metricspkg.NewMetricsGrabber(c, nil, true, false, true, true, false)
		if err != nil {
			framework.Failf("Error creating metrics grabber : %v", err)
		}
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up resources for mounted volume resize")

		if c != nil {
			if errs := framework.PVPVCCleanup(c, ns, pv, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}
	})

	It("should create volume metrics with the correct PVC", func() {
		var err error

		claims := []*v1.PersistentVolumeClaim{pvc}
		pod := framework.MakePod(ns, nil, claims, false, "")
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForPodRunningInNamespace(c, pod)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod), "Error starting pod ", pod.Name)

		pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

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
		var kubeMetrics metricspkg.KubeletMetrics
		waitErr := wait.Poll(30*time.Second, 1*time.Minute, func() (bool, error) {
			framework.Logf("Grabbing Kubelet metrics")
			// Grab kubelet metrics from the node the pod was scheduled on
			var err error
			kubeMetrics, err = metricsGrabber.GrabFromKubelet(pod.Spec.NodeName)
			if err != nil {
				framework.Logf("Error fetching kubelet metrics")
				return false, err
			}
			key := volumeStatKeys[0]
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			if !findVolumeStatMetric(kubeletKeyName, pvc.Namespace, pvc.Name, kubeMetrics) {
				return false, nil
			}
			return true, nil
		})
		Expect(waitErr).NotTo(HaveOccurred(), "Error finding volume metrics : %v", waitErr)

		for _, key := range volumeStatKeys {
			kubeletKeyName := fmt.Sprintf("%s_%s", kubeletmetrics.KubeletSubsystem, key)
			found := findVolumeStatMetric(kubeletKeyName, pvc.Namespace, pvc.Name, kubeMetrics)
			Expect(found).To(BeTrue(), "PVC %s, Namespace %s not found for %s", pvc.Name, pvc.Namespace, kubeletKeyName)
		}

		framework.Logf("Deleting pod %q/%q", pod.Namespace, pod.Name)
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod))
	})
})
