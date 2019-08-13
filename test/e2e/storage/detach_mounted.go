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
	"math/rand"
	"path"

	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var (
	// BusyBoxImage is the image URI of BusyBox.
	BusyBoxImage          = imageutils.GetE2EImage(imageutils.BusyBox)
	durationForStuckMount = 110 * time.Second
)

var _ = utils.SIGDescribe("Detaching volumes", func() {
	f := framework.NewDefaultFramework("flexvolume")

	// note that namespace deletion is handled by delete-namespace flag

	var cs clientset.Interface
	var ns *v1.Namespace
	var node v1.Node
	var suffix string

	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "local")
		framework.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessSSHKeyPresent()

		cs = f.ClientSet
		ns = f.Namespace
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		node = nodes.Items[rand.Intn(len(nodes.Items))]
		suffix = ns.Name
	})

	ginkgo.It("should not work when mount is in progress [Slow]", func() {
		framework.SkipUnlessSSHKeyPresent()

		driver := "attachable-with-long-mount"
		driverInstallAs := driver + "-" + suffix

		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(cs, &node, "k8s", driverInstallAs, path.Join(driverDir, driver))
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on master as %s", path.Join(driverDir, driver), driverInstallAs))
		installFlex(cs, nil, "k8s", driverInstallAs, path.Join(driverDir, driver))
		volumeSource := v1.VolumeSource{
			FlexVolume: &v1.FlexVolumeSource{
				Driver: "k8s/" + driverInstallAs,
			},
		}

		clientPod := getFlexVolumePod(volumeSource, node.Name)
		ginkgo.By("Creating pod that uses slow format volume")
		pod, err := cs.CoreV1().Pods(ns.Name).Create(clientPod)
		framework.ExpectNoError(err)

		uniqueVolumeName := getUniqueVolumeName(pod, driverInstallAs)

		ginkgo.By("waiting for volumes to be attached to node")
		err = waitForVolumesAttached(cs, node.Name, uniqueVolumeName)
		framework.ExpectNoError(err, "while waiting for volume to attach to %s node", node.Name)

		ginkgo.By("waiting for volume-in-use on the node after pod creation")
		err = waitForVolumesInUse(cs, node.Name, uniqueVolumeName)
		framework.ExpectNoError(err, "while waiting for volume in use")

		ginkgo.By("waiting for kubelet to start mounting the volume")
		time.Sleep(20 * time.Second)

		ginkgo.By("Deleting the flexvolume pod")
		err = framework.DeletePodWithWait(f, cs, pod)
		framework.ExpectNoError(err, "in deleting the pod")

		// Wait a bit for node to sync the volume status
		time.Sleep(30 * time.Second)

		ginkgo.By("waiting for volume-in-use on the node after pod deletion")
		err = waitForVolumesInUse(cs, node.Name, uniqueVolumeName)
		framework.ExpectNoError(err, "while waiting for volume in use")

		// Wait for 110s because mount device operation has a sleep of 120 seconds
		// we previously already waited for 30s.
		time.Sleep(durationForStuckMount)

		ginkgo.By("waiting for volume to disappear from node in-use")
		err = waitForVolumesNotInUse(cs, node.Name, uniqueVolumeName)
		framework.ExpectNoError(err, "while waiting for volume to be removed from in-use")

		ginkgo.By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(cs, &node, "k8s", driverInstallAs)
		ginkgo.By(fmt.Sprintf("uninstalling flexvolume %s from master", driverInstallAs))
		uninstallFlex(cs, nil, "k8s", driverInstallAs)
	})
})

func getUniqueVolumeName(pod *v1.Pod, driverName string) string {
	return fmt.Sprintf("flexvolume-k8s/%s/%s", driverName, pod.Spec.Volumes[0].Name)
}

func waitForVolumesNotInUse(client clientset.Interface, nodeName, volumeName string) error {
	return wait.PollImmediate(10*time.Second, 60*time.Second, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching node %s with %v", nodeName, err)
		}
		volumeInUSe := node.Status.VolumesInUse
		for _, volume := range volumeInUSe {
			if string(volume) == volumeName {
				return false, nil
			}
		}
		return true, nil
	})
}

func waitForVolumesAttached(client clientset.Interface, nodeName, volumeName string) error {
	return wait.PollImmediate(2*time.Second, 2*time.Minute, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching node %s with %v", nodeName, err)
		}
		volumeAttached := node.Status.VolumesAttached
		for _, volume := range volumeAttached {
			if string(volume.Name) == volumeName {
				return true, nil
			}
		}
		return false, nil
	})
}

func waitForVolumesInUse(client clientset.Interface, nodeName, volumeName string) error {
	return wait.PollImmediate(10*time.Second, 60*time.Second, func() (bool, error) {
		node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching node %s with %v", nodeName, err)
		}
		volumeInUSe := node.Status.VolumesInUse
		for _, volume := range volumeInUSe {
			if string(volume) == volumeName {
				return true, nil
			}
		}
		return false, nil
	})
}

func getFlexVolumePod(volumeSource v1.VolumeSource, nodeName string) *v1.Pod {
	var gracePeriod int64
	clientPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "flexvolume-detach-test" + "-client",
			Labels: map[string]string{
				"role": "flexvolume-detach-test" + "-client",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:       "flexvolume-detach-test" + "-client",
					Image:      BusyBoxImage,
					WorkingDir: "/opt",
					// An imperative and easily debuggable container which reads vol contents for
					// us to scan in the tests or by eye.
					// We expect that /opt is empty in the minimal containers which we use in this test.
					Command: []string{
						"/bin/sh",
						"-c",
						"while true ; do cat /opt/foo/index.html ; sleep 2 ; ls -altrh /opt/  ; sleep 2 ; done ",
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "test-long-detach-flex",
							MountPath: "/opt/foo",
						},
					},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &v1.SELinuxOptions{
					Level: "s0:c0,c1",
				},
			},
			Volumes: []v1.Volume{
				{
					Name:         "test-long-detach-flex",
					VolumeSource: volumeSource,
				},
			},
			NodeName: nodeName,
		},
	}
	return clientPod
}
