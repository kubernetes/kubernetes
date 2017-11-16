// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	cadvisorclient "github.com/google/cadvisor/client/v2"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/util/procfs"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/perftype"

	. "github.com/onsi/gomega"
)

const (
	// resource monitoring
	cadvisorImageName = "google/cadvisor:latest"
	cadvisorPodName   = "cadvisor"
	cadvisorPort      = 8090
	// housekeeping interval of Cadvisor (second)
	houseKeepingInterval = 1
)

var (
	systemContainers map[string]string
)

type ResourceCollector struct {
	client  *cadvisorclient.Client
	request *cadvisorapiv2.RequestOptions

	pollingInterval time.Duration
	buffers         map[string][]*framework.ContainerResourceUsage
	lock            sync.RWMutex
	stopCh          chan struct{}
}

// NewResourceCollector creates a resource collector object which collects
// resource usage periodically from Cadvisor
func NewResourceCollector(interval time.Duration) *ResourceCollector {
	buffers := make(map[string][]*framework.ContainerResourceUsage)
	return &ResourceCollector{
		pollingInterval: interval,
		buffers:         buffers,
	}
}

// Start starts resource collector and connects to the standalone Cadvisor pod
// then repeatedly runs collectStats.
func (r *ResourceCollector) Start() {
	// Get the cgroup container names for kubelet and docker
	kubeletContainer, err := getContainerNameForProcess(kubeletProcessName, "")
	dockerContainer, err := getContainerNameForProcess(dockerProcessName, dockerPidFile)
	if err == nil {
		systemContainers = map[string]string{
			stats.SystemContainerKubelet: kubeletContainer,
			stats.SystemContainerRuntime: dockerContainer,
		}
	} else {
		framework.Failf("Failed to get docker container name in test-e2e-node resource collector.")
	}

	wait.Poll(1*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		r.client, err = cadvisorclient.NewClient(fmt.Sprintf("http://localhost:%d/", cadvisorPort))
		if err == nil {
			return true, nil
		}
		return false, err
	})

	Expect(r.client).NotTo(BeNil(), "cadvisor client not ready")

	r.request = &cadvisorapiv2.RequestOptions{IdType: "name", Count: 1, Recursive: false}
	r.stopCh = make(chan struct{})

	oldStatsMap := make(map[string]*cadvisorapiv2.ContainerStats)
	go wait.Until(func() { r.collectStats(oldStatsMap) }, r.pollingInterval, r.stopCh)
}

// Stop stops resource collector collecting stats. It does not clear the buffer
func (r *ResourceCollector) Stop() {
	close(r.stopCh)
}

// Reset clears the stats buffer of resource collector.
func (r *ResourceCollector) Reset() {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range systemContainers {
		r.buffers[name] = []*framework.ContainerResourceUsage{}
	}
}

// GetCPUSummary gets CPU usage in percentile.
func (r *ResourceCollector) GetCPUSummary() framework.ContainersCPUSummary {
	result := make(framework.ContainersCPUSummary)
	for key, name := range systemContainers {
		data := r.GetBasicCPUStats(name)
		result[key] = data
	}
	return result
}

// LogLatest logs the latest resource usage.
func (r *ResourceCollector) LogLatest() {
	summary, err := r.GetLatest()
	if err != nil {
		framework.Logf("%v", err)
	}
	framework.Logf("%s", formatResourceUsageStats(summary))
}

// collectStats collects resource usage from Cadvisor.
func (r *ResourceCollector) collectStats(oldStatsMap map[string]*cadvisorapiv2.ContainerStats) {
	for _, name := range systemContainers {
		ret, err := r.client.Stats(name, r.request)
		if err != nil {
			framework.Logf("Error getting container stats, err: %v", err)
			return
		}
		cStats, ok := ret[name]
		if !ok {
			framework.Logf("Missing info/stats for container %q", name)
			return
		}

		newStats := cStats.Stats[0]

		if oldStats, ok := oldStatsMap[name]; ok && oldStats.Timestamp.Before(newStats.Timestamp) {
			if oldStats.Timestamp.Equal(newStats.Timestamp) {
				continue
			}
			r.buffers[name] = append(r.buffers[name], computeContainerResourceUsage(name, oldStats, newStats))
		}
		oldStatsMap[name] = newStats
	}
}

// computeContainerResourceUsage computes resource usage based on new data sample.
func computeContainerResourceUsage(name string, oldStats, newStats *cadvisorapiv2.ContainerStats) *framework.ContainerResourceUsage {
	return &framework.ContainerResourceUsage{
		Name:                    name,
		Timestamp:               newStats.Timestamp,
		CPUUsageInCores:         float64(newStats.Cpu.Usage.Total-oldStats.Cpu.Usage.Total) / float64(newStats.Timestamp.Sub(oldStats.Timestamp).Nanoseconds()),
		MemoryUsageInBytes:      newStats.Memory.Usage,
		MemoryWorkingSetInBytes: newStats.Memory.WorkingSet,
		MemoryRSSInBytes:        newStats.Memory.RSS,
		CPUInterval:             newStats.Timestamp.Sub(oldStats.Timestamp),
	}
}

// GetLatest gets the latest resource usage from stats buffer.
func (r *ResourceCollector) GetLatest() (framework.ResourceUsagePerContainer, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()
	stats := make(framework.ResourceUsagePerContainer)
	for key, name := range systemContainers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			return nil, fmt.Errorf("No resource usage data for %s container (%s)", key, name)
		}
		stats[key] = contStats[len(contStats)-1]
	}
	return stats, nil
}

type resourceUsageByCPU []*framework.ContainerResourceUsage

func (r resourceUsageByCPU) Len() int           { return len(r) }
func (r resourceUsageByCPU) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r resourceUsageByCPU) Less(i, j int) bool { return r[i].CPUUsageInCores < r[j].CPUUsageInCores }

// The percentiles to report.
var percentiles = [...]float64{0.50, 0.90, 0.95, 0.99, 1.00}

// GetBasicCPUStats returns the percentiles the cpu usage in cores for
// containerName. This method examines all data currently in the buffer.
func (r *ResourceCollector) GetBasicCPUStats(containerName string) map[float64]float64 {
	r.lock.RLock()
	defer r.lock.RUnlock()
	result := make(map[float64]float64, len(percentiles))

	// We must make a copy of array, otherwise the timeseries order is changed.
	usages := make([]*framework.ContainerResourceUsage, 0)
	for _, usage := range r.buffers[containerName] {
		usages = append(usages, usage)
	}

	sort.Sort(resourceUsageByCPU(usages))
	for _, q := range percentiles {
		index := int(float64(len(usages))*q) - 1
		if index < 0 {
			// We don't have enough data.
			result[q] = 0
			continue
		}
		result[q] = usages[index].CPUUsageInCores
	}
	return result
}

func formatResourceUsageStats(containerStats framework.ResourceUsagePerContainer) string {
	// Example output:
	//
	// Resource usage for node "e2e-test-foo-node-abcde":
	// container        cpu(cores)  memory(MB)
	// "/"              0.363       2942.09
	// "/docker-daemon" 0.088       521.80
	// "/kubelet"       0.086       424.37
	// "/system"        0.007       119.88
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "container\tcpu(cores)\tmemory_working_set(MB)\tmemory_rss(MB)\n")
	for name, s := range containerStats {
		fmt.Fprintf(w, "%q\t%.3f\t%.2f\t%.2f\n", name, s.CPUUsageInCores, float64(s.MemoryWorkingSetInBytes)/(1024*1024), float64(s.MemoryRSSInBytes)/(1024*1024))
	}
	w.Flush()
	return fmt.Sprintf("Resource usage:\n%s", buf.String())
}

func formatCPUSummary(summary framework.ContainersCPUSummary) string {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers on node "e2e-test-foo-node-0vj7":
	// container        5th%  50th% 90th% 95th%
	// "/"              0.051 0.159 0.387 0.455
	// "/runtime        0.000 0.000 0.146 0.166
	// "/kubelet"       0.036 0.053 0.091 0.154
	// "/misc"          0.001 0.001 0.001 0.002
	var summaryStrings []string
	var header []string
	header = append(header, "container")
	for _, p := range percentiles {
		header = append(header, fmt.Sprintf("%.0fth%%", p*100))
	}

	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "%s\n", strings.Join(header, "\t"))

	for _, containerName := range framework.TargetContainers() {
		var s []string
		s = append(s, fmt.Sprintf("%q", containerName))
		data, ok := summary[containerName]
		for _, p := range percentiles {
			value := "N/A"
			if ok {
				value = fmt.Sprintf("%.3f", data[p])
			}
			s = append(s, value)
		}
		fmt.Fprintf(w, "%s\n", strings.Join(s, "\t"))
	}
	w.Flush()
	summaryStrings = append(summaryStrings, fmt.Sprintf("CPU usage of containers:\n%s", buf.String()))

	return strings.Join(summaryStrings, "\n")
}

// createCadvisorPod creates a standalone cadvisor pod for fine-grain resource monitoring.
func getCadvisorPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: cadvisorPodName,
		},
		Spec: v1.PodSpec{
			// It uses a host port for the tests to collect data.
			// Currently we can not use port mapping in test-e2e-node.
			HostNetwork:     true,
			SecurityContext: &v1.PodSecurityContext{},
			Containers: []v1.Container{
				{
					Image: cadvisorImageName,
					Name:  cadvisorPodName,
					Ports: []v1.ContainerPort{
						{
							Name:          "http",
							HostPort:      cadvisorPort,
							ContainerPort: cadvisorPort,
							Protocol:      v1.ProtocolTCP,
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "sys",
							ReadOnly:  true,
							MountPath: "/sys",
						},
						{
							Name:      "var-run",
							ReadOnly:  false,
							MountPath: "/var/run",
						},
						{
							Name:      "docker",
							ReadOnly:  true,
							MountPath: "/var/lib/docker/",
						},
						{
							Name:      "rootfs",
							ReadOnly:  true,
							MountPath: "/rootfs",
						},
					},
					Args: []string{
						"--profiling",
						fmt.Sprintf("--housekeeping_interval=%ds", houseKeepingInterval),
						fmt.Sprintf("--port=%d", cadvisorPort),
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name:         "rootfs",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/"}},
				},
				{
					Name:         "var-run",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/var/run"}},
				},
				{
					Name:         "sys",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/sys"}},
				},
				{
					Name:         "docker",
					VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/var/lib/docker"}},
				},
			},
		},
	}
}

// deletePodsSync deletes a list of pods and block until pods disappear.
func deletePodsSync(f *framework.Framework, pods []*v1.Pod) {
	var wg sync.WaitGroup
	for _, pod := range pods {
		wg.Add(1)
		go func(pod *v1.Pod) {
			defer wg.Done()

			err := f.PodClient().Delete(pod.ObjectMeta.Name, metav1.NewDeleteOptions(30))
			Expect(err).NotTo(HaveOccurred())

			Expect(framework.WaitForPodToDisappear(f.ClientSet, f.Namespace.Name, pod.ObjectMeta.Name, labels.Everything(),
				30*time.Second, 10*time.Minute)).NotTo(HaveOccurred())
		}(pod)
	}
	wg.Wait()
	return
}

// newTestPods creates a list of pods (specification) for test.
func newTestPods(numPods int, volume bool, imageName, podType string) []*v1.Pod {
	var pods []*v1.Pod
	for i := 0; i < numPods; i++ {
		podName := "test-" + string(uuid.NewUUID())
		labels := map[string]string{
			"type": podType,
			"name": podName,
		}
		if volume {
			pods = append(pods,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   podName,
						Labels: labels,
					},
					Spec: v1.PodSpec{
						// Restart policy is always (default).
						Containers: []v1.Container{
							{
								Image: imageName,
								Name:  podName,
								VolumeMounts: []v1.VolumeMount{
									{MountPath: "/test-volume-mnt", Name: podName + "-volume"},
								},
							},
						},
						Volumes: []v1.Volume{
							{Name: podName + "-volume", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
						},
					},
				})
		} else {
			pods = append(pods,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   podName,
						Labels: labels,
					},
					Spec: v1.PodSpec{
						// Restart policy is always (default).
						Containers: []v1.Container{
							{
								Image: imageName,
								Name:  podName,
							},
						},
					},
				})
		}

	}
	return pods
}

// GetResourceSeriesWithLabels gets the time series of resource usage of each container.
func (r *ResourceCollector) GetResourceTimeSeries() map[string]*perftype.ResourceSeries {
	resourceSeries := make(map[string]*perftype.ResourceSeries)
	for key, name := range systemContainers {
		newSeries := &perftype.ResourceSeries{Units: map[string]string{
			"cpu":    "mCPU",
			"memory": "MB",
		}}
		resourceSeries[key] = newSeries
		for _, usage := range r.buffers[name] {
			newSeries.Timestamp = append(newSeries.Timestamp, usage.Timestamp.UnixNano())
			newSeries.CPUUsageInMilliCores = append(newSeries.CPUUsageInMilliCores, int64(usage.CPUUsageInCores*1000))
			newSeries.MemoryRSSInMegaBytes = append(newSeries.MemoryRSSInMegaBytes, int64(float64(usage.MemoryUsageInBytes)/(1024*1024)))
		}
	}
	return resourceSeries
}

// Code for getting container name of docker, copied from pkg/kubelet/cm/container_manager_linux.go
// since they are not exposed
const (
	kubeletProcessName    = "kubelet"
	dockerProcessName     = "docker"
	dockerPidFile         = "/var/run/docker.pid"
	containerdProcessName = "docker-containerd"
	containerdPidFile     = "/run/docker/libcontainerd/docker-containerd.pid"
)

func getPidsForProcess(name, pidFile string) ([]int, error) {
	if len(pidFile) > 0 {
		if pid, err := getPidFromPidFile(pidFile); err == nil {
			return []int{pid}, nil
		} else {
			// log the error and fall back to pidof
			runtime.HandleError(err)
		}
	}
	return procfs.PidOf(name)
}

func getPidFromPidFile(pidFile string) (int, error) {
	file, err := os.Open(pidFile)
	if err != nil {
		return 0, fmt.Errorf("error opening pid file %s: %v", pidFile, err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return 0, fmt.Errorf("error reading pid file %s: %v", pidFile, err)
	}

	pid, err := strconv.Atoi(string(data))
	if err != nil {
		return 0, fmt.Errorf("error parsing %s as a number: %v", string(data), err)
	}

	return pid, nil
}

func getContainerNameForProcess(name, pidFile string) (string, error) {
	pids, err := getPidsForProcess(name, pidFile)
	if err != nil {
		return "", fmt.Errorf("failed to detect process id for %q - %v", name, err)
	}
	if len(pids) == 0 {
		return "", nil
	}
	cont, err := getContainer(pids[0])
	if err != nil {
		return "", err
	}
	return cont, nil
}

// getContainer returns the cgroup associated with the specified pid.
// It enforces a unified hierarchy for memory and cpu cgroups.
// On systemd environments, it uses the name=systemd cgroup for the specified pid.
func getContainer(pid int) (string, error) {
	cgs, err := cgroups.ParseCgroupFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}

	cpu, found := cgs["cpu"]
	if !found {
		return "", cgroups.NewNotFoundError("cpu")
	}
	memory, found := cgs["memory"]
	if !found {
		return "", cgroups.NewNotFoundError("memory")
	}

	// since we use this container for accounting, we need to ensure it is a unified hierarchy.
	if cpu != memory {
		return "", fmt.Errorf("cpu and memory cgroup hierarchy not unified.  cpu: %s, memory: %s", cpu, memory)
	}

	// on systemd, every pid is in a unified cgroup hierarchy (name=systemd as seen in systemd-cgls)
	// cpu and memory accounting is off by default, users may choose to enable it per unit or globally.
	// users could enable CPU and memory accounting globally via /etc/systemd/system.conf (DefaultCPUAccounting=true DefaultMemoryAccounting=true).
	// users could also enable CPU and memory accounting per unit via CPUAccounting=true and MemoryAccounting=true
	// we only warn if accounting is not enabled for CPU or memory so as to not break local development flows where kubelet is launched in a terminal.
	// for example, the cgroup for the user session will be something like /user.slice/user-X.slice/session-X.scope, but the cpu and memory
	// cgroup will be the closest ancestor where accounting is performed (most likely /) on systems that launch docker containers.
	// as a result, on those systems, you will not get cpu or memory accounting statistics for kubelet.
	// in addition, you would not get memory or cpu accounting for the runtime unless accounting was enabled on its unit (or globally).
	if systemd, found := cgs["name=systemd"]; found {
		if systemd != cpu {
			log.Printf("CPUAccounting not enabled for pid: %d", pid)
		}
		if systemd != memory {
			log.Printf("MemoryAccounting not enabled for pid: %d", pid)
		}
		return systemd, nil
	}

	return cpu, nil
}
