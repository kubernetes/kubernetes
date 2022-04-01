/*
Copyright 2016 The Kubernetes Authors.

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
	"os/exec"

	v1 "k8s.io/api/core/v1"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	"github.com/onsi/ginkgo"
)

var _ = utils.SIGDescribe("GKE local SSD [Feature:GKELocalSSD]", func() {

	f := framework.NewDefaultFramework("localssd")

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gke")
	})

	ginkgo.It("should write and read from node local SSD [Feature:GKELocalSSD]", func() {
		e2eutils.Logf("Start local SSD test")
		createNodePoolWithLocalSsds("np-ssd")
		doTestWriteAndReadToLocalSsd(f)
	})
})

func createNodePoolWithLocalSsds(nodePoolName string) {
	e2eutils.Logf("Create node pool: %s with local SSDs in cluster: %s ",
		nodePoolName, e2econfig.TestContext.CloudConfig.Cluster)
	out, err := exec.Command("gcloud", "alpha", "container", "node-pools", "create",
		nodePoolName,
		fmt.Sprintf("--cluster=%s", e2econfig.TestContext.CloudConfig.Cluster),
		"--local-ssd-count=1").CombinedOutput()
	if err != nil {
		e2eutils.Failf("Failed to create node pool %s: Err: %v\n%v", nodePoolName, err, string(out))
	}
	e2eutils.Logf("Successfully created node pool %s:\n%v", nodePoolName, string(out))
}

func doTestWriteAndReadToLocalSsd(f *framework.Framework) {
	var pod = testPodWithSsd("echo 'hello world' > /mnt/disks/ssd0/data  && sleep 1 && cat /mnt/disks/ssd0/data")
	var msg string
	var out = []string{"hello world"}

	f.TestContainerOutput(msg, pod, 0, out)
}

func testPodWithSsd(command string) *v1.Pod {
	containerName := "test-container"
	volumeName := "test-ssd-volume"
	path := "/mnt/disks/ssd0"
	podName := "pod-" + string(uuid.NewUUID())
	image := "ubuntu:14.04"
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    containerName,
					Image:   image,
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: path,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: path,
						},
					},
				},
			},
			NodeSelector: map[string]string{"cloud.google.com/gke-local-ssd": "true"},
		},
	}
}
