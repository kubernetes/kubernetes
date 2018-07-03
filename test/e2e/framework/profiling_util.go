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

package framework

import (
	"context"
	"errors"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	docker "github.com/docker/docker/client"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	profilingPodName   = "profiling-pod"
	profilingContainer = "profiling-container"
	graphingPodPrefix  = "graphing-pod-"
	graphingContainer  = "graphing-container"
)

type ProfilingToolbox struct {
	profilingPod ProfilingPod
	graphingPod  GraphingPod
	framework    *Framework
	perfDone     chan bool
}

// ProfilingPod is a type of pod that has functions to start and stop profiling commands
type ProfilingPod *v1.Pod

// GraphingPod is a type of pod that has function to draw flamegraph
// Use MakeGraphingPod to create it with necessary files from workload pod mounted
type GraphingPod *v1.Pod

// ProfilingToolbox returns an instance of profiling toolbox
// The instance should be used across the whole test since it's not singleton
func (f *Framework) ProfilingToolbox() *ProfilingToolbox {
	return &ProfilingToolbox{
		framework: f,
	}
}

// StartProfilingPod starts a profiling pod
func (p *ProfilingToolbox) StartProfilingPod() {
	By("creating and starting a profiling pod")
	p.profilingPod = makeProfilingPod(profilingPodName)
	p.profilingPod = p.framework.PodClient().CreateSync(p.profilingPod)
	err := p.framework.WaitForPodRunning(p.profilingPod.Name)
	Expect(err).NotTo(HaveOccurred(), "failed to get profiling pod running")
}

// StartPerf executes start_perf command on a given workload pod.
// The command will start perf record in the cgroup of the workload pod.
// It uses Docker client to get the container ID and the cgroup name, hence only supports Docker environment.
func (p *ProfilingToolbox) StartPerf(workloadPod *v1.Pod) {
	p.perfDone = make(chan bool)
	cgroup, err := p.framework.GetCgroupOfPod(workloadPod.Name)
	Expect(err).NotTo(HaveOccurred(), "failed to get cgroup of the first container in the workload pod")
	go func() {
		p.framework.ExecCommandInPod(p.profilingPod.Name, "start_perf", cgroup)
		p.perfDone <- true
	}()
}

// StopPerf executes stop_perf command in the profiling toolbox pod.
// The command will send a SIGINT signal to the running perf command.
func (p *ProfilingToolbox) StopPerf() {
	p.framework.ExecCommandInPod(p.profilingPod.Name, "stop_perf")
	<-p.perfDone
}

// StartGraphingPod creates and starts a graphing pod with files in workload pod mounted
func (p *ProfilingToolbox) StartGraphingPod(workloadPod *v1.Pod) {
	By("copying result files from the profiling pod to the host")
	reportDir := TestContext.ReportDir
	_, err := p.framework.CopyFromPodToHost(p.framework.Namespace.Name, p.profilingPod.Name, "/perf.data", filepath.Join(reportDir, "perf-"+workloadPod.Name+".data"))
	Expect(err).NotTo(HaveOccurred(), "failed to copy perf.data to the host")
	_, err = p.framework.CopyFromPodToHost(p.framework.Namespace.Name, p.profilingPod.Name, "/perf.err", filepath.Join(reportDir, "perf-"+workloadPod.Name+".err"))
	Expect(err).NotTo(HaveOccurred(), "failed to copy perf.err to the host")
	_, err = p.framework.CopyFromPodToHost(p.framework.Namespace.Name, p.profilingPod.Name, "/perf.out", filepath.Join(reportDir, "perf-"+workloadPod.Name+".out"))
	Expect(err).NotTo(HaveOccurred(), "failed to copy perf.out to the host")

	By("getting the host paths of workload pod")
	workloadHostPaths, err := p.framework.GetHostPathsOfPod(workloadPod.Name)
	Expect(err).NotTo(HaveOccurred(), "failed to get local path of the workload pod")

	graphingPodName := graphingPodPrefix + workloadPod.Name
	p.graphingPod = makeGraphingPod(graphingPodName, workloadHostPaths, filepath.Join(reportDir, "perf-"+workloadPod.Name+".data"))
	p.graphingPod = p.framework.PodClient().CreateSync(p.graphingPod)
	err = p.framework.WaitForPodRunning(p.graphingPod.Name)
	Expect(err).NotTo(HaveOccurred(), "failed to get graphing pod running")

	By("copying all files from workload layers to root dir of graphing pod")
	for i := range workloadHostPaths {
		p.framework.ExecCommandInPod(p.graphingPod.Name, "sh", "-c", "cp -r /overlay/"+strconv.Itoa(i)+"/* / || :")
	}
}

// GenerateFlamegraph executes graph command in the graphing pod.
// The command will draw a flamegraph using perf.data.
func (p *ProfilingToolbox) GenerateFlamegraph() {
	p.framework.ExecCommandInPod(p.graphingPod.Name, "graph")
}

// CopyFlamegraph copies the flamegraph file to <dst> path on the host machine
func (p *ProfilingToolbox) CopyFlamegraph(dst string) {
	_, err := p.framework.CopyFromPodToHost(p.framework.Namespace.Name, p.graphingPod.Name, "/perf.svg", dst)
	Expect(err).NotTo(HaveOccurred(), "failed to copy flame graph to the host")
}

// DeleteProfilingPod synchronously delete the profiling pod
func (p *ProfilingToolbox) DeleteProfilingPod() {
	p.framework.PodClient().DeleteSync(p.profilingPod.Name, &metav1.DeleteOptions{}, 3*time.Minute)
}

// DeleteGraphingPod synchronously delete the graphing pod
func (p *ProfilingToolbox) DeleteGraphingPod() {
	p.framework.PodClient().DeleteSync(p.graphingPod.Name, &metav1.DeleteOptions{}, 3*time.Minute)
}

// GetCgroupOfContainer returns the cgroup name of the given container name
// Only work with Docker environment.
// The test using this function should run framework.RunIfContainerRuntimeIs("docker") in BeforeEach().
func (f *Framework) GetCgroupOfContainer(containerName string) (string, error) {
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
	// TODO hard coded for now
	return "kubepods/besteffort/pod" + containerJSON.Config.Labels["io.kubernetes.pod.uid"] + "/" + containerList[0].ID, nil
}

// GetCgroupOfPod returns the cgroup name of the first container of the pod
// Only work with Docker environment.
// The test using this function should run framework.RunIfContainerRuntimeIs("docker") in BeforeEach().
func (f *Framework) GetCgroupOfPod(podName string) (string, error) {
	pod, err := f.PodClient().Get(podName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "failed to get pod")
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	return f.GetCgroupOfContainer(pod.Spec.Containers[0].Name)
}

// GetHostPathsOfContainer gets the the host path of all docker layers of the given container.
// Only work with Docker environment.
// The test using this function should run framework.RunIfContainerRuntimeIs("docker") in BeforeEach().
func (f *Framework) GetHostPathsOfContainer(containerName string) ([]string, error) {
	ctx := context.Background()
	cli, err := docker.NewEnvClient()
	listFilters := filters.NewArgs()
	listFilters.Add("name", containerName)
	containerList, err := cli.ContainerList(ctx, types.ContainerListOptions{
		All:     true,
		Filters: listFilters,
	})
	if err != nil {
		return nil, err
	}
	Expect(containerList).NotTo(BeEmpty())
	containerJSON, err := cli.ContainerInspect(ctx, containerList[0].ID)
	if err != nil {
		return nil, err
	}
	// only tested with overlay2 as graph driver
	if containerJSON.GraphDriver.Name != "overlay2" {
		return nil, errors.New("Docker not using overlay2")
	}
	lowers := strings.Split(containerJSON.GraphDriver.Data["LowerDir"], ":")
	return append(lowers, containerJSON.GraphDriver.Data["UpperDir"], containerJSON.GraphDriver.Data["WorkDir"]), nil
}

// GetHostPathsOfPod gets the the host path of all docker layers of the first container of the given pod.
// Only work with Docker environment.
// The test using this function should run framework.RunIfContainerRuntimeIs("docker") in BeforeEach().
func (f *Framework) GetHostPathsOfPod(podName string) ([]string, error) {
	pod, err := f.PodClient().Get(podName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "failed to get pod")
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	return f.GetHostPathsOfContainer(pod.Spec.Containers[0].Name)
}

func makeProfilingPod(podName string) *v1.Pod {
	privileged := true
	volumeName := "cgroupfs"
	volumePath := "/sys/fs/cgroup"
	hostPathType := v1.HostPathDirectory

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  profilingContainer,
					Image: imageutils.GetE2EImage(imageutils.ProfilingToolbox),
					SecurityContext: &v1.SecurityContext{
						// capability and privilege are required for perf recording
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

func makeGraphingPod(podName string, workloadHostDirs []string, perfDataPath string) *v1.Pod {
	hostPathFile := v1.HostPathFile
	hostPathDir := v1.HostPathDirectory

	// mount the all layers from workload pod to /overlay/
	volumes := []v1.Volume{}
	volumeMounts := []v1.VolumeMount{}
	for i, path := range workloadHostDirs {
		volumes = append(volumes, v1.Volume{
			Name: "overlay-" + strconv.Itoa(i),
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: path,
					Type: &hostPathDir,
				},
			},
		})
		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      "overlay-" + strconv.Itoa(i),
			MountPath: "/overlay/" + strconv.Itoa(i),
			ReadOnly:  true,
		})
	}

	// mount perf.data for flame graph generating
	volumes = append(volumes, v1.Volume{
		Name: "perfdata-" + podName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: perfDataPath,
				Type: &hostPathFile,
			},
		},
	})
	volumeMounts = append(volumeMounts, v1.VolumeMount{
		Name:      "perfdata-" + podName,
		MountPath: "/perf.data",
	})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:         graphingContainer,
					Image:        imageutils.GetE2EImage(imageutils.ProfilingToolbox),
					VolumeMounts: volumeMounts,
				},
			},
			Volumes: volumes,
		},
	}
	return pod
}
