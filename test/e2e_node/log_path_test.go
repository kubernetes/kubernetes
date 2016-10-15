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
	"bufio"
	"os"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("ContainerLogPath", func() {
	f := framework.NewDefaultFramework("kubelet-container-log-path")
	Describe("Pod containers", func() {
		Context("with log printed to stdout", func() {
			var ns, logDir string
			var podClient *framework.PodClient
			BeforeEach(func() {
				podClient = f.PodClient()
				ns = f.Namespace.Name
				// logDir is the same with kubelet.containerLogsDir
				logDir = "/var/log/containers/"
			})
			// NOTE: We can not do this check within a pod because container will try to find the source file of a symlink.
			It("should print log to correct log path", func() {
				podName := "log-pod" + string(uuid.NewUUID())
				logContName := "log-container" + string(uuid.NewUUID())

				podObject := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      podName,
						Namespace: ns,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   "gcr.io/google_containers/busybox:1.24",
								Name:    logContName,
								Command: []string{"sh", "-c", "for var in 1 2 3 4; do echo \"$(hostname)\"; sleep 1; done"},
							},
						},
					},
				}

				podClient.Create(podObject)
				err := framework.WaitForPodSuccessInNamespace(f.Client, podName, ns)
				Expect(err).NotTo(HaveOccurred(), "Failed waiting for pod %s to enter success state", podName)

				// Get the newly created pod
				pod, err := podClient.Get(podName)
				Expect(err).NotTo(HaveOccurred(), "Failed to get newly created pod %s : %v", podName, err)

				// Get container id of log container
				var containerID string
				for _, cStatus := range pod.Status.ContainerStatuses {
					if cStatus.Name == logContName {
						if cStatus.ContainerID != "" {
							if idParts := strings.Split(cStatus.ContainerID, "://"); len(idParts) != 0 {
								// remove the docker:// prefix
								containerID = idParts[1]
							}
							break
						}
					}
				}
				// Should be able to get container ID
				Expect(containerID == "").NotTo(Equal(true), "Failed get container ID from statues: %v", pod.Status.ContainerStatuses)

				// Read expected log file, which name is: <pod_name>_<pod_namespace>_<container_name>-<container_id>.log
				var logLines []string
				containerLogFile := logDir + podName + "_" + ns + "_" + logContName + "-" + containerID + ".log"
				inFile, err := os.Open(containerLogFile)
				defer inFile.Close()
				Expect(err).NotTo(HaveOccurred(), "Failed read expected log file %s : %v", containerLogFile, err)

				scanner := bufio.NewScanner(inFile)
				scanner.Split(bufio.ScanLines)

				for scanner.Scan() {
					// Should contain the right log content
					Expect(strings.Contains(scanner.Text(), pod.Spec.Hostname)).To(Equal(true))
					logLines = append(logLines, scanner.Text())
				}
				// Should only contain right number of log lines
				Expect(len(logLines)).To(Equal(4), "Unexpected number of lines in log file")
			})
		})
	})
})
