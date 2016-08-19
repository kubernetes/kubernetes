// +build cgo,linux

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

package cadvisor

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/cache/memory"
	cadvisorMetrics "github.com/google/cadvisor/container"
	"github.com/google/cadvisor/events"
	cadvisorfs "github.com/google/cadvisor/fs"
	cadvisorhttp "github.com/google/cadvisor/http"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/utils/sysfs"
	"k8s.io/kubernetes/pkg/api/unversioned"
	statsApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/runtime"
)

type cadvisorClient struct {
	runtime string
	manager.Manager
}

var _ Interface = new(cadvisorClient)

// TODO(vmarmol): Make configurable.
// The amount of time for which to keep stats in memory.
const statsCacheDuration = 2 * time.Minute
const maxHousekeepingInterval = 15 * time.Second
const defaultHousekeepingInterval = 10 * time.Second
const allowDynamicHousekeeping = true

func init() {
	// Override cAdvisor flag defaults.
	flagOverrides := map[string]string{
		// Override the default cAdvisor housekeeping interval.
		"housekeeping_interval": defaultHousekeepingInterval.String(),
		// Disable event storage by default.
		"event_storage_event_limit": "default=0",
		"event_storage_age_limit":   "default=0",
	}
	for name, defaultValue := range flagOverrides {
		if f := flag.Lookup(name); f != nil {
			f.DefValue = defaultValue
			f.Value.Set(defaultValue)
		} else {
			glog.Errorf("Expected cAdvisor flag %q not found", name)
		}
	}
}

// Creates a cAdvisor and exports its API on the specified port if port > 0.
func New(port uint, runtime string) (Interface, error) {
	sysFs, err := sysfs.NewRealSysFs()
	if err != nil {
		return nil, err
	}

	// Create and start the cAdvisor container manager.
	m, err := manager.New(memory.New(statsCacheDuration, nil), sysFs, maxHousekeepingInterval, allowDynamicHousekeeping, cadvisorMetrics.MetricSet{cadvisorMetrics.NetworkTcpUsageMetrics: struct{}{}}, http.DefaultClient)
	if err != nil {
		return nil, err
	}

	cadvisorClient := &cadvisorClient{
		runtime: runtime,
		Manager: m,
	}

	err = cadvisorClient.exportHTTP(port)
	if err != nil {
		return nil, err
	}
	return cadvisorClient, nil
}

func (cc *cadvisorClient) Start() error {
	return cc.Manager.Start()
}

func (cc *cadvisorClient) exportHTTP(port uint) error {
	// Register the handlers regardless as this registers the prometheus
	// collector properly.
	mux := http.NewServeMux()
	err := cadvisorhttp.RegisterHandlers(mux, cc, "", "", "", "")
	if err != nil {
		return err
	}

	cadvisorhttp.RegisterPrometheusHandler(mux, cc, "/metrics", nil)

	// Only start the http server if port > 0
	if port > 0 {
		serv := &http.Server{
			Addr:    fmt.Sprintf(":%d", port),
			Handler: mux,
		}

		// TODO(vmarmol): Remove this when the cAdvisor port is once again free.
		// If export failed, retry in the background until we are able to bind.
		// This allows an existing cAdvisor to be killed before this one registers.
		go func() {
			defer runtime.HandleCrash()

			err := serv.ListenAndServe()
			for err != nil {
				glog.Infof("Failed to register cAdvisor on port %d, retrying. Error: %v", port, err)
				time.Sleep(time.Minute)
				err = serv.ListenAndServe()
			}
		}()
	}

	return nil
}

func (cc *cadvisorClient) ContainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {
	return cc.GetContainerInfo(name, req)
}

func (cc *cadvisorClient) ContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return cc.GetContainerInfoV2(name, options)
}

func (cc *cadvisorClient) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return cc.GetVersionInfo()
}

func (cc *cadvisorClient) SubcontainerInfo(name string, req *cadvisorapi.ContainerInfoRequest) (map[string]*cadvisorapi.ContainerInfo, error) {
	infos, err := cc.SubcontainersInfo(name, req)
	if err != nil && len(infos) == 0 {
		return nil, err
	}

	result := make(map[string]*cadvisorapi.ContainerInfo, len(infos))
	for _, info := range infos {
		result[info.Name] = info
	}
	return result, err
}

func (cc *cadvisorClient) MachineInfo() (*cadvisorapi.MachineInfo, error) {
	return cc.GetMachineInfo()
}

func (cc *cadvisorClient) ImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	var label string

	switch cc.runtime {
	case "docker":
		label = cadvisorfs.LabelDockerImages
	case "rkt":
		label = cadvisorfs.LabelRktImages
	default:
		return cadvisorapiv2.FsInfo{}, fmt.Errorf("ImagesFsInfo: unknown runtime: %v", cc.runtime)
	}

	return cc.getFsInfo(label)
}

func (cc *cadvisorClient) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return cc.getFsInfo(cadvisorfs.LabelSystemRoot)
}

func (cc *cadvisorClient) getFsInfo(label string) (cadvisorapiv2.FsInfo, error) {
	res, err := cc.GetFsInfo(label)
	if err != nil {
		return cadvisorapiv2.FsInfo{}, err
	}
	if len(res) == 0 {
		return cadvisorapiv2.FsInfo{}, fmt.Errorf("failed to find information for the filesystem labeled %q", label)
	}
	// TODO(vmarmol): Handle this better when a label has more than one image filesystem.
	if len(res) > 1 {
		glog.Warningf("More than one filesystem labeled %q: %#v. Only using the first one", label, res)
	}

	return res[0], nil
}

func (cc *cadvisorClient) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	return cc.WatchForEvents(request)
}

func (cc *cadvisorClient) GetAllContainerStats() ([]*statsApi.ContainerStats, error) {
	opts := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2, // 2 samples are needed to compute "instantaneous" CPU
		Recursive: true,
	}
	infos, err := cc.GetContainerInfoV2("/", opts)
	if err != nil {
		return nil, err
	}

	result := []*statsApi.ContainerStats{}

	for key, cinfo := range infos {
		// on systemd using devicemapper each mount into the container has an associated cgroup.
		// we ignore them to ensure we do not get duplicate entries in our summary.
		// for details on .mount units: http://man7.org/linux/man-pages/man5/systemd.mount.5.html
		if strings.HasSuffix(key, ".mount") {
			continue
		}
		// Build the Pod key if this container is managed by a Pod
		if isPodManagedContainer(&cinfo) {
			continue
		}

		name := types.GetContainerName(cinfo.Spec.Labels)
		result = append(result, containerInfoV2ToStats(name, &cinfo))
	}
	return result, nil
}

func (cc *cadvisorClient) GetContainerStats(id string) (*statsApi.ContainerStats, error) {
	req := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2, // 2 samples are needed to compute "instantaneous" CPU
		Recursive: false,
	}
	infos, err := cc.GetContainerInfoV2(id, req)
	if err != nil {
		return nil, err
	}

	for _, cinfo := range infos {
		// For non-recursive request, there should only be one info.
		name := types.GetContainerName(cinfo.Spec.Labels)
		return containerInfoV2ToStats(name, &cinfo), nil
	}
	return nil, fmt.Errorf("no such container found")
}

// isPodManagedContainer returns true if the cinfo container is managed by a Pod
func isPodManagedContainer(cinfo *cadvisorapiv2.ContainerInfo) bool {
	podName := types.GetPodName(cinfo.Spec.Labels)
	podNamespace := types.GetPodNamespace(cinfo.Spec.Labels)
	managed := podName != "" && podNamespace != ""
	if !managed && podName != podNamespace {
		glog.Warningf(
			"Expect container to have either both podName (%s) and podNamespace (%s) labels, or neither.",
			podName, podNamespace)
	}
	return managed
}

func containerInfoV2ToStats(name string, info *cadvisorapiv2.ContainerInfo) *statsApi.ContainerStats {
	cStats := &statsApi.ContainerStats{
		StartTime: unversioned.NewTime(info.Spec.CreationTime),
		Name:      name,
	}
	cstat, found := latestContainerStats(info)
	if !found {
		return cStats
	}
	if info.Spec.HasCpu {
		cpuStats := statsApi.CPUStats{
			Time: unversioned.NewTime(cstat.Timestamp),
		}
		if cstat.CpuInst != nil {
			cpuStats.UsageNanoCores = &cstat.CpuInst.Usage.Total
		}
		if cstat.Cpu != nil {
			cpuStats.UsageCoreNanoSeconds = &cstat.Cpu.Usage.Total
		}
		cStats.CPU = &cpuStats
	}
	if info.Spec.HasMemory {
		pageFaults := cstat.Memory.ContainerData.Pgfault
		majorPageFaults := cstat.Memory.ContainerData.Pgmajfault
		cStats.Memory = &statsApi.MemoryStats{
			Time:            unversioned.NewTime(cstat.Timestamp),
			UsageBytes:      &cstat.Memory.Usage,
			WorkingSetBytes: &cstat.Memory.WorkingSet,
			RSSBytes:        &cstat.Memory.RSS,
			PageFaults:      &pageFaults,
			MajorPageFaults: &majorPageFaults,
		}
		// availableBytes = memory  limit (if known) - workingset
		if !isMemoryUnlimited(info.Spec.Memory.Limit) {
			availableBytes := info.Spec.Memory.Limit - cstat.Memory.WorkingSet
			cStats.Memory.AvailableBytes = &availableBytes
		}
	}
	// TODO: populate FS stats
	return cStats
}

// latestContainerStats returns the latest container stats from cadvisor, or nil if none exist
func latestContainerStats(info *cadvisorapiv2.ContainerInfo) (*cadvisorapiv2.ContainerStats, bool) {
	stats := info.Stats
	if len(stats) < 1 {
		return nil, false
	}
	latest := stats[len(stats)-1]
	if latest == nil {
		return nil, false
	}
	return latest, true
}

// Size after which we consider memory to be "unlimited". This is not
// MaxInt64 due to rounding by the kernel.
// TODO: cadvisor should export this https://github.com/google/cadvisor/blob/master/metrics/prometheus.go#L596
const maxMemorySize = uint64(1 << 62)

func isMemoryUnlimited(v uint64) bool {
	return v > maxMemorySize
}
