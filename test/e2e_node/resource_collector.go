//go:build linux
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

package e2enode

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	cadvisorclient "github.com/google/cadvisor/client/v2"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeletstatsv1alpha1 "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/perftype"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	// resource monitoring
	cadvisorImageName = "gcr.io/cadvisor/cadvisor:v0.47.2"
	cadvisorPodName   = "cadvisor"
	cadvisorPort      = 8090
	// housekeeping interval of Cadvisor (second)
	houseKeepingInterval = 1
)

var (
	systemContainers map[string]string
)

// ResourceCollector is a collector object which collects
// resource usage periodically from Cadvisor.
type ResourceCollector struct {
	client  *cadvisorclient.Client
	request *cadvisorapiv2.RequestOptions

	pollingInterval time.Duration
	buffers         map[string][]*e2ekubelet.ContainerResourceUsage
	lock            sync.RWMutex
	stopCh          chan struct{}
}

// NewResourceCollector creates a resource collector object which collects
// resource usage periodically from Cadvisor
func NewResourceCollector(interval time.Duration) *ResourceCollector {
	buffers := make(map[string][]*e2ekubelet.ContainerResourceUsage)
	return &ResourceCollector{
		pollingInterval: interval,
		buffers:         buffers,
	}
}

// Start starts resource collector and connects to the standalone Cadvisor pod
// then repeatedly runs collectStats.
func (r *ResourceCollector) Start() {
	// Get the cgroup container names for kubelet and runtime
	kubeletContainer, err1 := getContainerNameForProcess(kubeletProcessName, "")
	runtimeContainer, err2 := getContainerNameForProcess(framework.TestContext.ContainerRuntimeProcessName, framework.TestContext.ContainerRuntimePidFile)
	if err1 == nil && err2 == nil && kubeletContainer != "" && runtimeContainer != "" {
		systemContainers = map[string]string{
			kubeletstatsv1alpha1.SystemContainerKubelet: kubeletContainer,
			kubeletstatsv1alpha1.SystemContainerRuntime: runtimeContainer,
		}
	} else {
		framework.Failf("Failed to get runtime container name in test-e2e-node resource collector.")
	}

	wait.Poll(1*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		r.client, err = cadvisorclient.NewClient(fmt.Sprintf("http://localhost:%d/", cadvisorPort))
		if err == nil {
			return true, nil
		}
		return false, err
	})

	gomega.Expect(r.client).NotTo(gomega.BeNil(), "cadvisor client not ready")

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
		r.buffers[name] = []*e2ekubelet.ContainerResourceUsage{}
	}
}

// GetCPUSummary gets CPU usage in percentile.
func (r *ResourceCollector) GetCPUSummary() e2ekubelet.ContainersCPUSummary {
	result := make(e2ekubelet.ContainersCPUSummary)
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
			r.buffers[name] = append(r.buffers[name], computeContainerResourceUsage(name, oldStats, newStats))
		}
		oldStatsMap[name] = newStats
	}
}

// computeContainerResourceUsage computes resource usage based on new data sample.
func computeContainerResourceUsage(name string, oldStats, newStats *cadvisorapiv2.ContainerStats) *e2ekubelet.ContainerResourceUsage {
	return &e2ekubelet.ContainerResourceUsage{
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
func (r *ResourceCollector) GetLatest() (e2ekubelet.ResourceUsagePerContainer, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()
	resourceUsage := make(e2ekubelet.ResourceUsagePerContainer)
	for key, name := range systemContainers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			return nil, fmt.Errorf("No resource usage data for %s container (%s)", key, name)
		}
		resourceUsage[key] = contStats[len(contStats)-1]
	}
	return resourceUsage, nil
}

type resourceUsageByCPU []*e2ekubelet.ContainerResourceUsage

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
	usages := make([]*e2ekubelet.ContainerResourceUsage, 0)
	usages = append(usages, r.buffers[containerName]...)

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

func formatResourceUsageStats(containerStats e2ekubelet.ResourceUsagePerContainer) string {
	// Example output:
	//
	// Resource usage:
	//container cpu(cores) memory_working_set(MB) memory_rss(MB)
	//"kubelet" 0.068      27.92                  15.43
	//"runtime" 0.664      89.88                  68.13

	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "container\tcpu(cores)\tmemory_working_set(MB)\tmemory_rss(MB)\n")
	for name, s := range containerStats {
		fmt.Fprintf(w, "%q\t%.3f\t%.2f\t%.2f\n", name, s.CPUUsageInCores, float64(s.MemoryWorkingSetInBytes)/(1024*1024), float64(s.MemoryRSSInBytes)/(1024*1024))
	}
	w.Flush()
	return fmt.Sprintf("Resource usage:\n%s", buf.String())
}

func formatCPUSummary(summary e2ekubelet.ContainersCPUSummary) string {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers:
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

	for _, containerName := range e2ekubelet.TargetContainers() {
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
func deletePodsSync(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	var wg sync.WaitGroup
	for i := range pods {
		pod := pods[i]
		wg.Add(1)
		go func() {
			defer ginkgo.GinkgoRecover()
			defer wg.Done()

			err := e2epod.NewPodClient(f).Delete(ctx, pod.ObjectMeta.Name, *metav1.NewDeleteOptions(30))
			if apierrors.IsNotFound(err) {
				framework.Failf("Unexpected error trying to delete pod %s: %v", pod.Name, err)
			}

			framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.ObjectMeta.Name, f.Namespace.Name, 10*time.Minute))
		}()
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

// GetResourceTimeSeries gets the time series of resource usage of each container.
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

const kubeletProcessName = "kubelet"

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

	if cgroups.IsCgroup2UnifiedMode() {
		unified, found := cgs[""]
		if !found {
			return "", cgroups.NewNotFoundError("unified")
		}
		return unified, nil
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
