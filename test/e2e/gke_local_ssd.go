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

package e2e

import (
	"fmt"
	"os/exec"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("GKE local SSD [Feature:GKELocalSSD]", func() {

	f := framework.NewDefaultFramework("localssd")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gke")
	})

	It("should write and read from node local SSD [Feature:GKELocalSSD]", func() {
		framework.Logf("Start local SSD test")
		createNodePoolWithLocalSsds("np-ssd")
		doTestWriteAndReadToLocalSsd(f)
	})
})

func createNodePoolWithLocalSsds(nodePoolName string) {
	framework.Logf("Create node pool: %s with local SSDs in cluster: %s ",
		nodePoolName, framework.TestContext.CloudConfig.Cluster)
	out, err := exec.Command("gcloud", "alpha", "container", "node-pools", "create",
		nodePoolName,
		fmt.Sprintf("--cluster=%s", framework.TestContext.CloudConfig.Cluster),
		"--local-ssd-count=1").CombinedOutput()
	if err != nil {
		framework.Failf("Failed to create node pool %s: Err: %v\n%v", nodePoolName, err, string(out))
	}
	framework.Logf("Successfully created node pool %s:\n%v", nodePoolName, string(out))
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
			APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
		},
		ObjectMeta: v1.ObjectMeta{
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
