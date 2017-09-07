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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("Fast pod add delete and then add back", func() {
	var (
		c                 clientset.Interface
		ns                string
		pvc               *v1.PersistentVolumeClaim
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
	)

	f := framework.NewDefaultFramework("pv")

	// We are going to pause between pod creation and deletion for this duration because
	// while we do want attach/detach controller to initiate action based on events
	// we do not want pod to be fully running or fully deleted because in either will
	// cause volume to be attached or detached.
	pauseBetweenAction := 10 * time.Second

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) != 0 {
			nodeName = nodeList.Items[0].Name
		} else {
			framework.Failf("Unable to find ready and schedulable Node")
		}

		if !isNodeLabeled {
			nodeLabelValue := "pod_vol_add_del_add_" + string(uuid.NewUUID())
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel["pod_vol_add_del_add_label"] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, "pod_vol_add_del_add_label", nodeLabelValue)
			isNodeLabeled = true
		}

		// Make sure there is a default storageclass
		defaultScName := getDefaultStorageClassName(c)
		verifyDefaultStorageClass(c, defaultScName, true)
		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}

		pvc = newClaim(test, ns, "default")
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		pvc = updatedPVC
		Expect(err).NotTo(HaveOccurred())
		Expect(pvc).ToNot(Equal(nil))
	})

	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(c, nodeName, "pod_vol_add_del_add_label")
		}
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up resources for fast add delete")

		if c != nil {
			if errs := framework.PVPVCCleanup(c, ns, nil, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}

	})

	It("should test that quickly adding, deleting and then adding back a pod does not stop pod creation", func() {
		podCommand := "while true; do date; date >>/mnt/volume1; sleep 1; done"
		pod1 := createPodWithNodeAffinity(pvc.Name, nodeKeyValueLabel, podCommand)
		pod1, err := c.CoreV1().Pods(ns).Create(pod1)

		Expect(err).NotTo(HaveOccurred())

		time.Sleep(pauseBetweenAction)

		// now lets delete the pod
		podDeleteErr := c.CoreV1().Pods(ns).Delete(pod1.Name, nil)
		Expect(podDeleteErr).NotTo(HaveOccurred())

		time.Sleep(pauseBetweenAction)

		pod2 := createPodWithNodeAffinity(pvc.Name, nodeKeyValueLabel, podCommand)
		pod2, pod2Err := c.CoreV1().Pods(ns).Create(pod2)
		defer func() {
			framework.DeletePodWithWait(f, c, pod2)
		}()

		Expect(pod2Err).NotTo(HaveOccurred())
		podWaitErr := framework.WaitForPodNameRunningInNamespace(c, pod2.Name, ns)
		Expect(podWaitErr).NotTo(HaveOccurred())
	})

})

func createPodWithNodeAffinity(claimName string, nodeSelectorKV map[string]string, command string) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pod-add-del-add-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   imageutils.GetBusyBoxImage(),
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
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
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: claimName,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}
	if nodeSelectorKV != nil {
		pod.Spec.NodeSelector = nodeSelectorKV
	}
	return pod
}
