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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	logString = "This is the expected log content of this node e2e test"
)

var _ = framework.KubeDescribe("ContainerLogPath", func() {
	f := framework.NewDefaultFramework("kubelet-container-log-path")
	Describe("Pod with a container", func() {
		Context("printed log to stdout", func() {
			// NOTE: We can not do this check within a pod because container will try to find the source file of a symlink.
			It("should print log to correct log path", func() {
				podClient := f.PodClient()
				ns := f.Namespace.Name

				rootfsDirVolumeName := "docker-dir-vol"

				rootfsDir := "/root"
				rootDir := "/"
				logDir := kubelet.ContainerLogsDir

				podName := "logger-checker" + string(uuid.NewUUID())
				logContName := "logger-" + string(uuid.NewUUID())
				checkContName := "checker-" + string(uuid.NewUUID())

				// we use a wildcard here because we cannot get containerID before it's created, and logContName is unique enough.
				expectedlogFile := logDir + "/" + podName + "_" + ns + "_" + logContName + "-*.log"

				podObject := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						// this pod is expected to exit successfully
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   "gcr.io/google_containers/busybox:1.24",
								Name:    logContName,
								Command: []string{"sh", "-c", "echo " + logString},
							},
							{
								Image: "gcr.io/google_containers/busybox:1.24",
								Name:  checkContName,
								// if we find expected log file and contains right content, exit 0
								// else, keep checking until test timeout
								Command: []string{"sh", "-c", "chroot " + rootfsDir + " while true; do for f in " + expectedlogFile + " ; do if [ -e \"$f\" ] && grep -q " + logString + " $f; then exit 0; fi; done; sleep 1; done"},
								VolumeMounts: []api.VolumeMount{
									{
										Name:      rootfsDirVolumeName,
										MountPath: rootfsDir,
										ReadOnly:  true,
									},
								},
							},
						},
						Volumes: []api.Volume{
							{
								Name: rootfsDirVolumeName,
								VolumeSource: api.VolumeSource{
									HostPath: &api.HostPathVolumeSource{
										Path: rootDir,
									},
								},
							},
						},
					},
				}

				podClient.Create(podObject)
				err := framework.WaitForPodSuccessInNamespace(f.Client, podName, ns)
				framework.ExpectNoError(err, "Failed waiting for pod: %s to enter success state", podName)
			})
		})
	})
})
