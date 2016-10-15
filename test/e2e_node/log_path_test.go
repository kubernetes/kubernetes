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
	"io/ioutil"
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
			BeforeEach(func() {
				ns = f.Namespace.Name
				logDir = "/var/log/containers/"
			})
			It("Correct log path on the node should have been created", func() {
				podName := "log-pod" + string(uuid.NewUUID())
				logContName := "log-container" + string(uuid.NewUUID())
				// The expected container's log file name on host is:
				// <pod_name>_<pod_namespace>_<container_name>-<container_id>.log
				logPathPrefix := podName + "_" + ns + "_" + logContName + "-"

				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      podName,
						Namespace: ns,
					},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image:   "gcr.io/google_containers/busybox:1.24",
								Name:    logContName,
								Command: []string{"sh", "-c", "while true; do echo \"$(hostname)\"; sleep 1; done"},
							},
						},
					},
				}

				pod, err := f.Client.Pods(ns).Create(pod)
				if err != nil {
					framework.Failf("unable to create pod %v: %v", podName, err)
				}

				err = f.WaitForPodRunning(podName)
				Expect(err).NotTo(HaveOccurred(), "Failed waiting for pod %s to enter running state", podName)

				files, err := ioutil.ReadDir(logDir)
				if err != nil {
					framework.Failf("unable to read container log directory %v: %v", logDir, err)
				}
				var found bool
				for _, f := range files {
					if strings.HasPrefix(f.Name(), logPathPrefix) {
						logBytes, err := ioutil.ReadFile(logDir + f.Name())
						if err != nil {
							framework.Failf("unable to read container log file %v: %v", f.Name(), err)
						}
						logStr := string(logBytes)
						// Should contains right log content
						Expect(strings.Contains(logStr, pod.Spec.Hostname)).To(Equal(true))
						found = true
						break
					}
				}
				// Should found the log file
				Expect(found).To(Equal(true), "Failed to find container log files with prefix %s", logPathPrefix)
			})
		})
	})
})
