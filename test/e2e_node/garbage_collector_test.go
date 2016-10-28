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

package e2e_node

import (
	"fmt"
	"strings"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	docker "k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

/*
GarbageCollect tests that the Kubelet conforms to the Kubelet Garbage Collection Policy, found here:
http://kubernetes.io/docs/admin/garbage-collection/
TODO (dashpole): Once the Container Runtime Interface (CRI) is complete, generalize to other runtimes (other than docker)
TODO (dashpole): Once dynamic config is possible, test different values for maxPerPodContainer and maxContainers
*/
const (
	defaultDockerEndpoint                = "unix:///var/run/docker.sock"
	defaultRuntimeRequestTimeoutDuration = 2 * time.Minute
	maxPerPodContainer                   = 2
	maxTotalContainers                   = -1

	runtimePollInterval = 10 * time.Second
	deleteTimeout       = 2 * time.Minute

	containerNamePrefix = "gc-test-container"
)

var _ = framework.KubeDescribe("GarbageCollect [Serial]", func() {
	f := framework.NewDefaultFramework("garbage-collect-test")
	runtime := docker.ConnectToDockerOrDie(defaultDockerEndpoint, defaultRuntimeRequestTimeoutDuration)

	Context("", func() {
		AfterEach(func() {
			glog.Infof("Summary of node events during the garbage collection test:")
			err := framework.ListNamespaceEvents(f.ClientSet, f.Namespace.Name)
			framework.ExpectNoError(err)
			glog.Infof("Summary of pod events during the garbage collection test:")
			err = framework.ListNamespaceEvents(f.ClientSet, "")
			framework.ExpectNoError(err)
		})
		maxPerPodContainerGCTest(f, runtime, 6, 1)
		maxPerPodContainerGCTest(f, runtime, 2, 4)
		maxPerPodContainerGCTest(f, runtime, 3, 6)
		maxPerPodContainerGCTest(f, runtime, 4, 1)
	})

})

// Tests that maxPerPodContainer limits the number of non-garbage collected containers for each container created.
func maxPerPodContainerGCTest(f *framework.Framework, runtime docker.DockerInterface, restartCount int, numContainers int) {
	Context(fmt.Sprintf("when we create and delete pods with containers %v times", restartCount), func() {
		podName := fmt.Sprintf("gc-test-pod%d.%d", numContainers, restartCount)
		BeforeEach(func() {
			f.PodClient().CreateSync(getPodWithRestartCount(podName, numContainers, restartCount))
			time.Sleep(10 * time.Second)
		})
		It(fmt.Sprintf("should eventually garbage collect containers that have been recreated more than %v times", maxPerPodContainer), func() {
			Eventually(func() error {
				dockerContainers, err := getDockerContainers(runtime)
				glog.Infof("checking to see if containers have been garbage collected... We currently have %v containers", len(dockerContainers))
				if err != nil {
					return err
				}
				if maxTotalContainers > 0 && len(dockerContainers) > maxTotalContainers {
					return fmt.Errorf("expected total number of containers: %v, to be <= maxTotalContainers: %v", len(dockerContainers), maxTotalContainers)
				}
				if len(dockerContainers) > maxPerPodContainer*numContainers {
					return fmt.Errorf("expected total number of containers: %v, to be <= maxPerPodContainer * numContainers %v", len(dockerContainers), maxPerPodContainer*numContainers)
				}
				return nil

			}, defaultRuntimeRequestTimeoutDuration, runtimePollInterval).Should(BeNil())
		})
		AfterEach(func() {
			f.PodClient().DeleteSync(podName, &api.DeleteOptions{}, defaultRuntimeRequestTimeoutDuration)
		})
	})
}

func getDockerContainers(runtime docker.DockerInterface) ([]*dockertypes.Container, error) {
	relevantContainers := []*dockertypes.Container{}
	dockerContainers, err := docker.GetKubeletDockerContainers(runtime, true)
	if err != nil {
		return relevantContainers, err
	}
	for _, container := range dockerContainers {
		if strings.Contains(container.Names[0], containerNamePrefix) { // only look for containers from this test
			relevantContainers = append(relevantContainers, container)
		}
	}
	return relevantContainers, nil
}

func getPodWithRestartCount(podName string, numContainers int, restartCount int) *api.Pod {
	By(fmt.Sprintf("creating %v containers with restartCount: %v", numContainers, restartCount))
	containers := []api.Container{}
	for i := 0; i < numContainers; i++ {
		containers = append(containers, api.Container{
			Image:           "gcr.io/google_containers/busybox:1.24",
			ImagePullPolicy: "Always",
			Name:            fmt.Sprintf("%s%d", containerNamePrefix, i),
			Command: []string{
				"sh",
				"-c",
				fmt.Sprintf(`
					f=/test-empty-dir-mnt/countfile%v
					count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
					if [ $count -lt %v ]; then
						exit 0
					fi
					while true; do sleep 1; done
				`, i, restartCount),
			},
			VolumeMounts: []api.VolumeMount{
				{MountPath: "/test-empty-dir-mnt", Name: "test-empty-dir"},
			},
		})
	}
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: podName},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			Containers:    containers,
			Volumes: []api.Volume{
				{Name: "test-empty-dir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
		},
	}
}
