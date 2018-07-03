/*
Copyright 2018 The Kubernetes Authors.

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
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types/filters"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/docker/docker/api/types"
	docker "github.com/docker/docker/client"
	"github.com/google/uuid"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	toolboxPodName          = "perf-toolbox-pod"
	toolboxContainer        = "perf-toolbox"
	graphingPodName         = "perf-graphing-pod"
	workloadPodPrefix       = "workload-pod-"
	workloadContainerPrefix = "workload-container-"
)

type Workload struct{}

var _ = framework.KubeDescribe("PerfAnalysis [Slow]", func() {
	f := framework.NewDefaultFramework("perf-test")
	workload := &Workload{}

	Context("pulling the toolbox image and create the container", func() {
		BeforeEach(func() {
			By("creating toolbox container")
			toolboxPod := makeToolboxPod()
			toolboxPod = f.PodClient().CreateSync(toolboxPod)
			Expect(len(toolboxPod.Status.ContainerStatuses)).To(Equal(1))
			err := f.WaitForPodRunning(toolboxPodName)
			Expect(err).NotTo(HaveOccurred(), "failed to get toolbox pod running")
		})

		It("starts a workload pod and perf records the data", func() {
			By("creating and run workload pod")
			workloadPod := makeWorkloadPod(workload.Image(), workload.Cmd())
			workloadPod = f.PodClient().CreateSync(workloadPod)

			err := f.WaitForPodRunning(workloadPod.Name)
			Expect(err).NotTo(HaveOccurred(), "failed to get workload pod running")
			cgroup, err := getCgroupOfPod(f, workloadPod.Name)
			Expect(err).NotTo(HaveOccurred(), "failed to get cgroup of the first container in the workload pod")

			By("starting perf record command on toolbox pod")
			perfDone := make(chan bool)
			// the exec command will block if there's any background task running
			// when finished, it sends a signal to perfDone channel
			go func() {
				f.ExecCommandInPod(toolboxPodName, "sh", "bootstrap.sh", cgroup)
				perfDone <- true
			}()

			// wait for workload pod exiting and perf.data to be closed
			f.WaitForPodNoLongerRunning(workloadPod.Name)
			f.ExecCommandInPod(toolboxPodName, "sh", "kill.sh")
			<-perfDone

			By("parsing perf record command on toolbox pod")
			// parse shows the missing files that contain symbols in stderr
			_, stderr, err := f.ExecCommandInPodWithFullOutput(toolboxPodName, "sh", "parse.sh")
			Expect(err).NotTo(HaveOccurred(), "failed to parse perf.data")

			By("creating another toolbox pod with files from workload pod mounted")
			userLocalPath, err := getLocalPathOfPod(f, workloadPod.Name)
			Expect(err).NotTo(HaveOccurred(), "failed to get local path of the workload pod")
			toolboxLocalPath, err := getLocalPathOfPod(f, toolboxPodName)
			Expect(err).NotTo(HaveOccurred(), "failed to get local path of the toolbox pod")
			By("copying missing files from the workload pod to the toolbox pod")
			missing, err := getMissingFiles(stderr)
			graphingPod := makeGraphingPod(missing, userLocalPath, toolboxLocalPath)
			graphingPod = f.PodClient().CreateSync(graphingPod)
			err = f.WaitForPodRunning(graphingPodName)
			Expect(err).NotTo(HaveOccurred(), "failed to get graphing pod running")

			By("generating flame graph by perf.data on graphing pod")
			f.ExecCommandInPod(graphingPodName, "sh", "flamegraph.sh")

			By("copying the perf.data and generated flame graph back to the host")
			reportDir := framework.TestContext.ReportDir
			_, err = f.CopyFromPodToHost(f.Namespace.Name, graphingPodName, "/perf.data", filepath.Join(reportDir, "perf.data"))
			Expect(err).NotTo(HaveOccurred(), "failed to copy perf.data to the host")
			_, err = f.CopyFromPodToHost(f.Namespace.Name, graphingPodName, "/perf.svg", filepath.Join(reportDir, "perf.svg"))
			Expect(err).NotTo(HaveOccurred(), "failed to copy flame graph to the host")

			f.PodClient().DeleteSync(workloadPod.Name, &metav1.DeleteOptions{}, 3*time.Minute)
			f.PodClient().DeleteSync(graphingPod.Name, &metav1.DeleteOptions{}, 3*time.Minute)
		})

		AfterEach(func() {
			By("removing the toolbox container")
			f.PodClient().Delete(toolboxPodName, &metav1.DeleteOptions{})
		})
	})
})

func makeToolboxPod() *v1.Pod {
	privileged := true
	volumeName := "cgroupfs"
	volumePath := "/sys/fs/cgroup"
	hostPathType := v1.HostPathDirectory

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: toolboxPodName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  toolboxContainer,
					Image: imageutils.GetE2EImage(imageutils.PerfTestToolbox),
					SecurityContext: &v1.SecurityContext{
						Capabilities: &v1.Capabilities{
							Add: []v1.Capability{"SYS_ADMIN"},
						},
						Privileged: &privileged,
					},
					// mount the cgroup fs so that we can use perf record on user container within the cgroup
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: volumePath,
							Type: &hostPathType,
						},
					},
				},
			},
			HostPID: true,
		},
	}
	return pod
}

func makeGraphingPod(files []string, userLocalPath, toolboxLocalPath string) *v1.Pod {
	hostPathType := v1.HostPathFile
	perfDataName := "perfdata"

	// mount the missing files for symbols
	volumes := []v1.Volume{}
	volumeMounts := []v1.VolumeMount{}
	for _, path := range files {
		volumes = append(volumes, v1.Volume{
			Name: path,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: filepath.Join(userLocalPath, path),
					Type: &hostPathType,
				},
			},
		})
		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      path,
			MountPath: path,
		})
	}
	// mount the perf.data for flame graph generating
	volumes = append(volumes, v1.Volume{
		Name: perfDataName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: filepath.Join(toolboxLocalPath, "perf.data"),
				Type: &hostPathType,
			},
		},
	})
	volumeMounts = append(volumeMounts, v1.VolumeMount{
		Name:      perfDataName,
		MountPath: "/perf.data",
	})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: graphingPodName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:         toolboxContainer,
					Image:        imageutils.GetE2EImage(imageutils.PerfTestToolbox),
					VolumeMounts: volumeMounts,
				},
			},
			Volumes: volumes,
		},
	}
	return pod
}

func makeWorkloadPod(image string, cmd []string) *v1.Pod {
	randSuffix := uuid.New().String()[:8]
	container := v1.Container{
		Name:  workloadContainerPrefix + randSuffix,
		Image: image,
	}
	if cmd != nil {
		container.Command = cmd
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: workloadPodPrefix + randSuffix,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				container,
			},
		},
	}
	return pod
}

func getCgroupOfContainer(containerName string) (string, error) {
	ctx := context.Background()
	cli, err := docker.NewEnvClient()
	listFilters := filters.NewArgs()
	listFilters.Add("name", containerName)
	containerList, err := cli.ContainerList(ctx, types.ContainerListOptions{
		All:     true,
		Filters: listFilters,
	})
	if err != nil {
		return "", err
	}
	Expect(containerList).NotTo(BeEmpty())
	containerJSON, err := cli.ContainerInspect(ctx, containerList[0].ID)
	if err != nil {
		return "", err
	}
	return "kubepods/besteffort/pod" + containerJSON.Config.Labels["io.kubernetes.pod.uid"] + "/" + containerList[0].ID, nil
}

func getCgroupOfPod(f *framework.Framework, podName string) (string, error) {
	pod, err := f.PodClient().Get(podName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "failed to get pod")
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	return getCgroupOfContainer(pod.Spec.Containers[0].Name)
}

func getLocalPathOfContainer(containerName string) (string, error) {
	ctx := context.Background()
	cli, err := docker.NewEnvClient()
	listFilters := filters.NewArgs()
	listFilters.Add("name", containerName)
	containerList, err := cli.ContainerList(ctx, types.ContainerListOptions{
		All:     true,
		Filters: listFilters,
	})
	if err != nil {
		return "", err
	}
	Expect(containerList).NotTo(BeEmpty())
	containerJSON, err := cli.ContainerInspect(ctx, containerList[0].ID)
	if err != nil {
		return "", err
	}
	return containerJSON.GraphDriver.Data["MergedDir"], nil
}

func getLocalPathOfPod(f *framework.Framework, podName string) (string, error) {
	pod, err := f.PodClient().Get(podName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "failed to get pod")
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	return getLocalPathOfContainer(pod.Spec.Containers[0].Name)
}

// Parsing the error message when calling perf script.
// The error message tells which symbols are missing.
func getMissingFiles(input string) ([]string, error) {
	result := []string{}
	for _, line := range strings.Split(input, "\n") {
		if strings.HasPrefix(line, "Failed to open") {
			result = append(result, line[15:strings.Index(line, ", continuing without symbols")])
		}
	}
	return result, nil
}

func (w Workload) Name() string {
	return "dd_4gb_md5sum"
}

func (w Workload) Image() string {
	return imageutils.GetE2EImage(imageutils.PerfTestToolbox)
}

func (w Workload) Cmd() []string {
	return []string{
		"sh",
		"-c",
		"dd if=/dev/zero bs=1M count=4096 | md5sum",
	}
}
